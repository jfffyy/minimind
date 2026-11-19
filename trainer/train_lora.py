"""
MiniMindB LoRA 微调脚本

LoRA 核心知识点：
- 什么是 LoRA：一种参数高效微调方法，只训练少量新增参数。
- 原理：冻结预训练权重 W，在旁边添加低秩分解矩阵 ΔW = BA。
  - A 的形状是 (d, r)，B 的形状是 (r, d)，其中 r 远小于 d。
  - 前向计算：output = Wx + BAx。
  - 原始权重 W 不参与梯度更新，只更新 A 和 B。
- 优势：
  - Full SFT：更新所有参数，效果好，但需要大显存和长时间。
  - LoRA：只更新少量参数，显存需求小、训练快，适合单卡或小显存环境。
  - 一个基座模型可以搭配多组 LoRA 权重，快速切换不同任务能力。

本脚本与当前项目的对应关系：
- 模型：model.model.MiniMindBForCausalLM，forward 只返回 logits，不返回 loss。
- 数据：dataset.lm_dataset.SFTDataset，返回 (input_ids, labels, attention_mask) 三元组。
- loss：在本脚本内用 F.cross_entropy 计算；SFTDataset 已把 prompt/pad 位置设为 -100。
- LoRA：复用 model.model_lora 中的 apply_lora / save_lora。
"""

import os
import sys

# 显式声明本模块所属的包，并把项目根目录加入模块搜索路径。
# 这样无论从哪个工作目录启动脚本，都能正确导入 model / dataset / trainer。
__package__ = "trainer"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
sys.path.append(PROJECT_ROOT)

# ========== 标准库 ==========
import argparse  # 命令行参数解析
import importlib  # 动态导入 swanlab，缺包时给出友好提示
import time  # 训练耗时统计
import warnings  # 警告控制

# ========== 第三方库 ==========
import torch
import torch.distributed as dist  # 分布式训练支持
import torch.nn.functional as F  # 交叉熵损失函数
from contextlib import nullcontext  # CPU 上用空上下文占位
from torch import optim  # AdamW 优化器
from torch.amp.autocast_mode import autocast  # 新版混合精度 API
from torch.amp.grad_scaler import GradScaler  # 新版梯度缩放器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# ========== 项目内部模块 ==========
from model.model import MiniMindBConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import apply_lora, save_lora
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

# 忽略无关警告，保持训练日志整洁。
warnings.filterwarnings("ignore")


def resolve_init_weight(from_weight, save_dir, hidden_size, use_moe):
    """
    把 --from_weight 解析成 init_model 需要的 (权重名, 目录)。

    支持三种写法：
    1. "none"：不加载任何权重，随机初始化。
    2. 完整的 .pth 文件路径：自动拆成文件名和所在目录。
    3. 权重名（如 "full_sft"）：相对于 save_dir 查找。
    """
    if from_weight == "none":
        return "none", save_dir

    # 用户给的是完整文件路径时，拆出权重名和目录。
    if os.path.isfile(from_weight):
        name = os.path.splitext(os.path.basename(from_weight))[0]
        # init_model 会自己拼 _{hidden_size}[_moe].pth，
        # 所以这里要把文件名末尾已有的维度后缀去掉，避免重复。
        suffix = f"_{hidden_size}" + ("_moe" if use_moe else "")
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        return name, os.path.dirname(from_weight) or save_dir

    # 否则视为权重名，交给 init_model 在 save_dir 下查找。
    return from_weight, save_dir


def save_lora_resume(model, optimizer, scaler, lm_config, lora_name, epoch, step, wandb):
    """
    保存 LoRA 专用的断点续训状态。

    与 lm_checkpoint() 不同，这里不保存完整基座模型，只保存：
    - LoRA 的 A/B 权重（用于恢复适配器本身）
    - 优化器状态、GradScaler 状态
    - epoch / step 训练进度、wandb_id、world_size

    保存格式刻意与 lm_checkpoint() 保持一致，
    因此可以直接用 lm_checkpoint(..., weight=lora_name) 读取。
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 文件名规则与 lm_checkpoint() 完全相同：
    # {lora_name}_{hidden_size}[_moe]_resume.pth
    moe_suffix = "_moe" if getattr(lm_config, "use_moe", False) else ""
    resume_path = os.path.join(
        CHECKPOINT_DIR, f"{lora_name}_{lm_config.hidden_size}{moe_suffix}_resume.pth"
    )

    # DDP 包装后真实模型在 model.module 里，取它的 state_dict 才没有 module. 前缀。
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    # 只保留名字里含 lora 的参数，基座权重不落盘。
    lora_state = {
        k: v for k, v in raw_model.state_dict().items() if "lora" in k
    }

    # 尝试取出实验 ID，续训时恢复到同一个 wandb/swanlab 实验。
    wandb_id = None
    if wandb:
        if hasattr(wandb, "get_run"):
            run = wandb.get_run()
            wandb_id = getattr(run, "id", None) if run else None
        else:
            wandb_id = getattr(wandb, "id", None)

    resume_data = {
        "model": lora_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        # GPU 数量变化时，lm_checkpoint() 会按 world_size 自动换算 step。
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "wandb_id": wandb_id,
    }

    # 先写临时文件再原子替换，避免训练中途退出留下损坏的检查点。
    tmp_path = resume_path + ".tmp"
    torch.save(resume_data, tmp_path)
    os.replace(tmp_path, resume_path)


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    执行一个 epoch 的 LoRA 训练。

    与普通全量 SFT 的流程相同，区别只有两点：
    1. 梯度裁剪只针对 lora_params。
    2. 保存时只落盘 LoRA adapter，不保存完整模型。

    Args:
        epoch: 当前 epoch 序号。
        loader: 数据加载器。
        iters: 该 epoch 的总步数（含已跳过的步数）。
        lora_params: 需要更新的 LoRA 参数列表。
        start_step: 断点续训时的起始步数。
        wandb: 实验跟踪对象。
    """
    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps 必须大于 0")

    model.train()
    # 每个 epoch 开始时清空梯度，避免残留梯度影响训练。
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    # 保存请求先挂起，等本组梯度累积完成、参数真正更新后再保存。
    save_pending = False

    # enumerate 的 start=start_step + 1 让续训时 step 编号保持连续。
    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start_step + 1
    ):
        # SFTDataset 返回三元组，直接解包；数据必须和模型放在同一设备。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        # 余弦退火学习率：随训练进度平滑下降。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 混合精度上下文：关键运算保持 float32，其余用 float16/bfloat16。
        with autocast_ctx:
            # 当前模型只返回 logits，不返回 loss 或 aux_loss。
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            if logits is None:
                raise RuntimeError("模型未返回 logits，无法计算 SFT 损失")

            # 用第 t 个位置预测第 t+1 个 token，所以 logits 和 labels 右移一位对齐。
            # SFTDataset 的 generate_labels() 已把 prompt 和 pad 设为 -100，
            # ignore_index=-100 会让这些位置不参与 loss 计算。
            raw_loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

            # 梯度累积：最后不足 accumulation_steps 的一组按实际批次数归一化，
            # 避免尾批梯度被人为缩小。
            group_start = start_step + (
                (step - start_step - 1) // args.accumulation_steps
            ) * args.accumulation_steps
            group_size = min(args.accumulation_steps, iters - group_start)
            loss = raw_loss / group_size

        # 放大 loss 再反向传播，防止 float16 下梯度下溢。
        scaler.scale(loss).backward()

        # 用相对 step 判断是否该更新参数，这样断点续训时不会因绝对 step 错位。
        is_update_step = (
            (step - start_step) % args.accumulation_steps == 0 or step == iters
        )
        if is_update_step:
            # 反缩放必须在裁剪之前，拿到真实梯度值。
            scaler.unscale_(optimizer)
            # 只裁剪 LoRA 参数；基座参数已冻结，没有梯度。
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 每隔 log_interval 步或最后一步打印日志。
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = raw_loss.item()
            current_lr = optimizer.param_groups[-1]["lr"]
            # 已用时间 / 已完成步数 × 剩余步数 = 预计剩余秒数。
            eta_min = (
                spend_time / max(step - start_step, 1) * (iters - step) / 60
            )
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        # 到保存点先挂起请求；只有参数更新完成后才真正保存。
        save_pending = save_pending or step % args.save_interval == 0 or step == iters
        if save_pending and is_update_step and is_main_process():
            model.eval()

            # LoRA adapter 单独保存，eval.py 会从 out/lora/ 下加载它。
            lora_save_path = os.path.join(
                args.save_dir, f"{args.lora_name}_{lm_config.hidden_size}.pth"
            )
            save_lora(model, lora_save_path)

            # 续训状态单独保存，不包含完整基座模型。
            save_lora_resume(
                model,
                optimizer,
                scaler,
                lm_config,
                args.lora_name,
                epoch,
                step,
                wandb,
            )

            model.train()
            save_pending = False


if __name__ == "__main__":
    # ========== 命令行参数 ==========
    parser = argparse.ArgumentParser(description="MiniMindB LoRA Fine-tuning")

    # ---- 保存相关 ----
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "out", "lora"),
        help="LoRA 权重保存目录",
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default="lora_identity",
        help="LoRA 权重名称，如 lora_identity / lora_medical",
    )
    # rank 越大，LoRA 容量越大，但参数量也越多。
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA 低秩矩阵的秩")

    # ---- 训练超参数 ----
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # LoRA 只训少量参数，学习率通常比全量 SFT 大（全量 SFT 常用 1e-6）。
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="初始学习率"
    )

    # ---- 硬件和精度 ----
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="混合精度类型",
    )
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # ---- 训练策略 ----
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="模型保存间隔"
    )

    # ---- 模型架构 ----
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument(
        "--num_hidden_layers", default=8, type=int, help="隐藏层数量"
    )
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ---- 数据与恢复 ----
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "sft_t2t_mini.jsonl"),
        help="SFT 数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default=os.path.join(
            PROJECT_ROOT,
            "out",
            "pretrain-20261110-190312",
            "pretrain_full_20261110_190312_512.pth",
        ),
        type=str,
        help="基座权重：完整 .pth 路径、权重名（相对 out），或 none 表示随机初始化",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测并断点续训（0=否，1=是）",
    )

    # ---- 实验跟踪 ----
    parser.add_argument("--use_wandb", action="store_true", help="是否使用实验记录")
    parser.add_argument(
        "--wandb_project", type=str, default="MiniMindB-LoRA", help="wandb项目名"
    )

    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    # 多卡时每个进程负责一张卡，local_rank 是本机 GPU 编号。
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 不同进程用不同种子，避免采样到完全相同的数据顺序。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindBConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # 如果开启续训，尝试读取 LoRA 检查点。
    # 这里走的是 lm_checkpoint 的“加载模式”（不传 model）。
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.lora_name, save_dir=CHECKPOINT_DIR)
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不支持 autocast，用 nullcontext 作为空操作占位。
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else autocast(device_type, dtype=dtype)
    )

    # ========== 4. 配置实验跟踪系统 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 用 importlib 动态导入，缺包时给出明确安装提示。
        try:
            wandb = importlib.import_module("swanlab")
        except ModuleNotFoundError as exc:
            if exc.name != "swanlab":
                raise
            raise RuntimeError(
                "启用 --use_wandb 前，请先在 Anaconda 环境执行：pip install swanlab"
            ) from exc

        # 续训时恢复到同一个实验，避免新建一条曲线。
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"MiniMindB-LoRA-{args.lora_name}-Epoch-{args.epochs}"
            f"-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 加载基座模型并注入 LoRA ==========
    # 解析权重路径，让 --from_weight 既能填完整路径也能填权重名。
    init_weight, init_dir = resolve_init_weight(
        args.from_weight,
        os.path.join(PROJECT_ROOT, "out"),
        args.hidden_size,
        args.use_moe,
    )
    model, tokenizer = init_model(
        lm_config, init_weight, save_dir=init_dir, device=args.device
    )

    # 在模型的线性层旁边挂上 LoRA 分支。
    # apply_lora 不会冻结原权重，冻结在下面单独完成。
    apply_lora(model, rank=args.lora_rank)

    # 统计参数量，确认 LoRA 只占很小比例。
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(
        p.numel() for name, p in model.named_parameters() if "lora" in name
    )
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # 冻结策略：名字含 lora 的参数可训练，其余全部冻结。
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # ========== 6. 数据、优化器、梯度缩放器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式采样器保证不同进程拿到不同数据子集。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # bfloat16 通常不需要 GradScaler；只有 float16 才启用。
    scaler = GradScaler(
        "cuda", enabled=(device_type == "cuda" and args.dtype == "float16")
    )
    # 优化器只接收 LoRA 参数，基座权重不会产生优化器状态，显存大幅下降。
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 7. 从检查点恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # ckp_data["model"] 只含 LoRA 权重；strict=False 让基座权重保持加载时的值。
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 8. DDP 包装 ==========
    if dist.is_initialized():
        # broadcast_buffers=False 避免每步广播 RoPE 缓存等不需要同步的 buffer。
        model = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )

    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每个 epoch 更换采样顺序，保证数据被充分打乱。
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            # 续训的第一个 epoch：用 SkipBatchSampler 跳过已训练过的 batch。
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，"
                f"从step {start_step + 1}开始"
            )
            train_epoch(
                epoch, loader, len(loader) + start_step, lora_params, start_step, wandb
            )
        else:
            # 普通 epoch：单卡用 shuffle 打乱，多卡由 sampler 控制。
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # ========== 10. 清理分布式进程组 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
