"""
MiniMindB DPO（Direct Preference Optimization）训练脚本。

本脚本由 MokioMind DPO 脚本适配到当前项目：
- 模型：MiniMindBConfig / MiniMindBForCausalLM
- 数据：dataset.lm_dataset.DPODataset（chosen / rejected 偏好对）
- 工具：沿用 trainer_utils 中的分布式、检查点、日志工具

DPO 核心：
- 不需要奖励模型，也不需要在线 PPO 采样。
- 直接用一个冻结的 Reference 模型作为基准。
- 目标是提高 chosen 相对 reference 的 logprob，降低 rejected 相对 reference 的 logprob。
"""

import os
import sys

__package__ = "trainer"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
sys.path.append(PROJECT_ROOT)

import argparse
import importlib
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model import MiniMindBConfig
from dataset.lm_dataset import DPODataset
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

warnings.filterwarnings("ignore")


def resolve_init_weight(from_weight, save_dir, hidden_size, use_moe):
    """把 --from_weight 解析成 init_model 需要的 (权重名, 目录)。"""
    if from_weight == "none":
        return "none", save_dir
    if os.path.isfile(from_weight):
        name = os.path.splitext(os.path.basename(from_weight))[0]
        suffix = f"_{hidden_size}" + ("_moe" if use_moe else "")
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        return name, os.path.dirname(from_weight) or save_dir
    return from_weight, save_dir


def logits_to_log_probs(logits, labels):
    """把 [B, S, V] 的 logits 转成 [B, S] 的 token log 概率。"""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    DPO loss。

    batch 的前半是 chosen，后半是 rejected。
    mask 只在回答部分为 1，prompt 和 pad 不参与 loss。
    """
    # 按每个样本的有效 token 数取平均 logprob。
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze(-1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze(-1)

    batch_size = ref_log_probs.shape[0]
    half = batch_size // 2

    chosen_ref = ref_log_probs[:half]
    rejected_ref = ref_log_probs[half:]
    chosen_policy = policy_log_probs[:half]
    rejected_policy = policy_log_probs[half:]

    # 策略模型和参考模型各自的 chosen-rejected logprob 差。
    pi_logratios = chosen_policy - rejected_policy
    ref_logratios = chosen_ref - rejected_ref

    # DPO 目标就是让这个差值越大越好。
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()


def train_epoch(
    epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        # 把 chosen 和 rejected 拼成一个 batch，前一半 chosen，后一半 rejected。
        x_chosen = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        am_chosen = batch["attention_mask_chosen"].to(args.device)
        am_rejected = batch["attention_mask_rejected"].to(args.device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        attention_mask = torch.cat([am_chosen, am_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # Reference 冻结，只用来算 baseline logprob。
            with torch.no_grad():
                ref_outputs = ref_model(x, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # Policy 是需要优化的模型。
            outputs = model(x, attention_mask=attention_mask)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val

            # 当前 MiniMindB 模型没有 aux_loss；兼容未来带 MoE aux_loss 的版本。
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is not None:
                loss = loss + aux_loss

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # 用相对 step 判断更新时机，断点续训时不会错位。
        is_update_step = (
            (step - start_step) % args.accumulation_steps == 0 or step == iters
        )
        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = (
                spend_time / max(step - start_step, 1) * (iters - step) / 60
            )
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.6f} lr:{current_lr:.12f} "
                f"epoch_Time:{eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = (
                f"{args.save_dir}/{args.save_weight}_"
                f"{lm_config.hidden_size}{moe_suffix}.pth"
            )
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            state_dict = {k: v.half() for k, v in raw_model.state_dict().items()}
            torch.save(state_dict, ckp)

            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=CHECKPOINT_DIR,
            )
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMindB DPO")

    parser.add_argument(
        "--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "out"), help="模型保存目录"
    )
    parser.add_argument("--save_weight", default="dpo", type=str, help="保存权重前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数，DPO 通常 1~2 轮")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    parser.add_argument("--learning_rate", type=float, default=4e-8, help="DPO 学习率")
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

    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")

    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=1024, type=int, help="最大序列长度")
    parser.add_argument(
        "--use_moe", default=0, type=int, choices=[0, 1], help="是否使用 MoE"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "dpo.jsonl"),
        help="DPO 偏好数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="full_sft",
        type=str,
        help="基座权重，通常是 full_sft；也可填完整 .pth 路径或 none",
    )
    parser.add_argument(
        "--from_resume", default=0, type=int, choices=[0, 1], help="是否断点续训"
    )
    parser.add_argument("--beta", default=0.1, type=float, help="DPO beta 参数")

    parser.add_argument("--use_wandb", action="store_true", help="是否启用实验记录")
    parser.add_argument(
        "--wandb_project", type=str, default="MiniMindB-DPO", help="wandb 项目名"
    )

    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindBConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=CHECKPOINT_DIR)
        if args.from_resume == 1
        else None
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else autocast(device_type, dtype=dtype)
    )

    wandb = None
    if args.use_wandb and is_main_process():
        try:
            wandb = importlib.import_module("swanlab")
        except ModuleNotFoundError as exc:
            if exc.name != "swanlab":
                raise
            raise RuntimeError(
                "启用 --use_wandb 前，请先在 Anaconda 环境执行：pip install swanlab"
            ) from exc
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"MiniMindB-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}"
            f"-LR-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    init_weight, init_dir = resolve_init_weight(
        args.from_weight,
        os.path.join(PROJECT_ROOT, "out"),
        args.hidden_size,
        args.use_moe,
    )
    model, tokenizer = init_model(
        lm_config, init_weight, save_dir=init_dir, device=args.device
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reference 模型和 Policy 初始权重相同，但完全冻结。
    ref_model, _ = init_model(
        lm_config, init_weight, save_dir=init_dir, device=args.device
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = GradScaler(
        "cuda", enabled=(device_type == "cuda" and args.dtype == "float16")
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
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
                epoch,
                loader,
                len(loader) + start_step,
                ref_model,
                lm_config,
                start_step,
                wandb,
                args.beta,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(
                epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta
            )

    if dist.is_initialized():
        dist.destroy_process_group()
