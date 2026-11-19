"""
MiniMindB PPO（Proximal Policy Optimization）训练脚本。

本脚本按照你项目现有的 trainer 风格改造：
- argparse 管理超参
- init_model 加载基座权重
- lm_checkpoint 管理断点续训
- 分节注释 + 关键逻辑说明

PPO 训练架构：
1. Actor：正在训练的策略模型，负责生成回答
2. Old Actor：上一轮的冻结副本，用来计算重要性采样比例 r_t
3. Reference：冻结的初始模型，用来算 KL 惩罚，防止策略漂移
4. Critic：价值模型，估计每个状态的价值，用来算优势
5. Reward Model：给生成的回答打分，提供强化学习奖励信号

重要前提：
- 奖励模型使用外部 InternLM2-1.8B-Reward（与 MokioMind 方案一致），
  默认位于 minimind 同级目录：../internlm2-1_8b-reward。
- 该目录需包含 config.json、tokenizer、safetensors 权重，并带有 get_score 方法。
- 不再需要训练本地 reward_model_512.pth。
"""

import os
import sys

__package__ = "trainer"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
sys.path.append(PROJECT_ROOT)

import argparse
import importlib
import re
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoConfig, AutoModel, AutoTokenizer

from model.model import MiniMindBConfig, MiniMindBForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
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


# ========== Critic Model ==========
class CriticModel(MiniMindBForCausalLM):
    """
    价值模型：在 MiniMindB 底座上加一个 value_head。

    与 Reward Model 的区别：
    - Reward Model 输出整段回答的一个标量总分
    - Critic 输出每个 token 位置的价值估计 V(s_t)，供 GAE / 优势计算使用
    """

    def __init__(self, config):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # MiniMindBModel.forward 返回 (hidden_states, presents)，
        # 且 hidden_states 已经过 layer_norm，不需要再额外 norm。
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = outputs[0]
        return self.value_head(hidden_states).squeeze(-1)


# ========== 奖励计算 ==========
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    计算每条回答的总奖励。

    奖励可以由两部分组成：
    1. 格式奖励（可选）：推理模式下检查 <think>/<answer> 结构是否完整
    2. Reward Model 打分：对回答内容给出标量分数
    """

    def reasoning_model_reward(rewards):
        """检查推理格式是否合规，给格式分和标签分。"""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [
            re.match(pattern2, response, re.S) for response in responses
        ]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            """四个标签各出现一次，各给 0.25 分。"""
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # Reward Model 打分：把 prompt 解析成 messages，再拼上模型回答。
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [
                {"role": role, "content": content.strip()} for role, content in matches
            ]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 限制分数范围，防止极端奖励破坏训练。
            scale = 3.0
            score = max(min(score, scale), -scale)

            if args.reasoning == 1:
                # 推理模式下，额外对 <answer> 里的内容单独打分并加权。
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat = messages + [
                        {"role": "assistant", "content": answer_content}
                    ]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6

            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


# ========== PPO 训练一个 Epoch ==========
def ppo_train_epoch(
    epoch,
    loader,
    iters,
    old_actor_model,
    ref_model,
    actor_scheduler,
    critic_scheduler,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        # tokenizer 在线编码 prompt；左 padding 是 decoder-only 生成的标准做法。
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        ).to(args.device)

        # 真实 prompt 长度，用于构造 response mask。
        prompt_lengths = enc.attention_mask.sum(dim=1)
        # 左 padding 下，生成结果的 response 起点是补齐后的 prompt 长度，
        # 而不是真实 token 数。这里用 enc.input_ids.shape[1] 才对。
        resp_start = enc.input_ids.shape[1]

        # ========== 1. 用当前 Policy 采样生成回答 ==========
        with torch.no_grad():
            model_for_gen = (
                actor_model.module
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model
            )
            generate_fn = getattr(model_for_gen, "generate")
            gen_out = generate_fn(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解码生成部分：从 resp_start 开始，而不是 prompt_lengths。
        responses_text = [
            tokenizer.decode(gen_out[i, resp_start:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]

        # ========== 2. 计算奖励 ==========
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)

        # 有效 token mask，pad 位置为 0。
        full_mask = (gen_out != tokenizer.pad_token_id).long()

        # ========== 3. Critic 估价值，算优势 ==========
        with autocast_ctx:
            value_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)
        last_indices = full_mask.sum(dim=1) - 1
        values = value_seq[torch.arange(len(last_indices), device=gen_out.device), last_indices]
        # 优势 = 实际奖励 - 预估价值；detach 阻止梯度回传到 Critic 的这一项。
        advantages = rewards - values.detach()

        # ========== 4. 计算当前 Policy 对生成序列的 log 概率 ==========
        with autocast_ctx:
            logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits
        labels = gen_out[:, 1:].clone()
        logp_tokens = (
            F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            .gather(2, labels.unsqueeze(-1))
            .squeeze(-1)
        )  # [B, L-1]

        seq_len = gen_out.size(1) - 1
        # 只统计 response 部分的概率，prompt 部分置 0。
        # logp_tokens 的第 j 位对应预测 gen_out[j+1]，所以 response 起点要减 1。
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= (
            resp_start - 1
        )
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))
        actor_logp = (logp_tokens * final_mask).sum(dim=1)

        # ========== 5. Old Policy 和 Reference 的 log 概率 ==========
        with torch.no_grad():
            with autocast_ctx:
                old_logits = old_actor_model(
                    input_ids=gen_out, attention_mask=full_mask
                ).logits
                ref_logits = ref_model(
                    input_ids=gen_out, attention_mask=full_mask
                ).logits

            old_logp_tokens = (
                F.log_softmax(old_logits[:, :-1, :].float(), dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )
            ref_logp_tokens = (
                F.log_softmax(ref_logits[:, :-1, :].float(), dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)

        # ========== 6. PPO 裁剪目标与损失 ==========
        # kl：新旧策略差异；kl_ref：当前策略偏离初始模型的程度。
        kl = (actor_logp - old_logp).mean()
        kl_ref = (actor_logp - ref_logp).mean()
        ratio = torch.exp(actor_logp - old_logp)

        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, rewards)
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref

        loss.backward()

        # ========== 7. 参数更新 ==========
        if step % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ========== 8. 日志记录 ==========
        if is_main_process() and (step % args.log_interval == 0 or step == iters):
            response_ids = gen_out[:, resp_start:]
            is_eos = response_ids == tokenizer.eos_token_id
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(
                has_eos,
                eos_indices + 1,
                torch.tensor(response_ids.shape[1], device=is_eos.device),
            )
            avg_len = lengths.float().mean()

            metrics = {
                "actor_loss": policy_loss.item(),
                "critic_loss": value_loss.item(),
                "reward": rewards.mean().item(),
                "kl": kl.item(),
                "kl_ref": kl_ref.item(),
                "avg_response_len": avg_len.item(),
                "actor_lr": actor_optimizer.param_groups[0]["lr"],
            }
            if wandb is not None:
                wandb.log(metrics)

            Logger(
                f"Epoch: {epoch + 1}, Step: {step}/{iters}, "
                f"Actor Loss: {metrics['actor_loss']:.6f}, "
                f"Critic Loss: {metrics['critic_loss']:.6f}, "
                f"Reward: {metrics['reward']:.6f}, "
                f"KL: {metrics['kl']:.6f}, KL_ref: {metrics['kl_ref']:.6f}, "
                f"Avg Response Len: {metrics['avg_response_len']:.2f}, "
                f"Actor LR: {metrics['actor_lr']:.2e}"
            )

        # ========== 9. 定期同步 Old Actor ==========
        if step % args.update_old_actor_freq == 0:
            state_dict = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            old_actor_model.load_state_dict(
                {k: v.detach().cpu() for k, v in state_dict.items()}
            )
            old_actor_model = nn.Module.to(old_actor_model, args.device)

        # ========== 10. 模型保存 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = (
                f"{args.save_dir}/{args.save_weight}_"
                f"{lm_config.hidden_size}{moe_suffix}.pth"
            )
            actor_state = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)

            # lm_checkpoint 的 **kwargs 会把带 state_dict 的对象一起存进续训文件。
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=actor_model,
                optimizer=actor_optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=CHECKPOINT_DIR,
                scheduler=actor_scheduler,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )
            actor_model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMindB PPO (Proximal Policy Optimization)")

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "out"), help="模型保存目录")
    parser.add_argument("--save_weight", default="ppo_actor", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size（PPO batch 通常较小）")

    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor 学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic 学习率")

    # ========== 硬件配置 ==========
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

    # ========== 训练策略 ==========
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")

    # ========== 模型架构 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe", default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构"
    )

    # ========== PPO 生成参数 ==========
    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt 最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")

    # ========== 数据和权重 ==========
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "sft_t2t_mini.jsonl"),
        help="RLAIF 数据路径",
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
        help="Actor 初始权重：完整 .pth 路径、权重名（相对 out），或 none",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=os.path.join(
            os.path.dirname(PROJECT_ROOT), "internlm2-1_8b-reward"
        ),
        help="外部奖励模型目录（InternLM2-1.8B-Reward）",
    )

    # ========== PPO 超参数 ==========
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO 裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value loss 系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL 惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help="是否启用推理格式奖励")
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="同步 Old Actor 的频率")

    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否断点续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否启用实验记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMindB-PPO", help="wandb 项目名")

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
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

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else autocast(device_type, dtype=dtype)
    )

    # ========== 4. 实验跟踪 ==========
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
            f"MiniMindB-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化 5 个模型 ==========
    init_weight, init_dir = resolve_init_weight(
        args.from_weight,
        os.path.join(PROJECT_ROOT, "out"),
        args.hidden_size,
        args.use_moe,
    )

    # Actor：正在训练的策略模型。
    actor_model, tokenizer = init_model(
        lm_config, init_weight, save_dir=init_dir, device=args.device
    )
    # 生成时左 padding 是 decoder-only 模型的标准做法。
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Old Actor：上一版权重快照，冻结不训练。
    old_actor_model, _ = init_model(
        lm_config, init_weight, save_dir=init_dir, device=args.device
    )
    old_actor_model = old_actor_model.eval().requires_grad_(False)

    # Reference：初始 SFT 模型快照，冻结不训练，只用来算 KL。
    ref_model, _ = init_model(
        lm_config, init_weight, save_dir=init_dir, device=args.device
    )
    ref_model = ref_model.eval().requires_grad_(False)

    # Critic：在底座上加 value_head，预估每个 token 位置的价值。
    moe_suffix = "_moe" if lm_config.use_moe else ""
    base_ckp = os.path.join(
        init_dir, f"{init_weight}_{args.hidden_size}{moe_suffix}.pth"
    )
    critic_model = CriticModel(lm_config)
    if init_weight != "none" and os.path.isfile(base_ckp):
        critic_model.load_state_dict(torch.load(base_ckp, map_location=args.device), strict=False)
    critic_model = nn.Module.to(critic_model, args.device)

    # Reward Model：使用外部 InternLM2-1.8B-Reward，和 MokioMind 方案一致。
    if not os.path.isdir(args.reward_model_path):
        raise FileNotFoundError(
            f"未找到奖励模型目录: {args.reward_model_path}，"
            "请确认已下载 internlm2-1_8b-reward 到 minimind 同级目录。"
        )
    reward_config = AutoConfig.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )
    # 新版 transformers 会把 rope_scaling 自动补成 {"rope_type": "default"}，
    # 但 InternLM2 远程代码仍按旧版字段 "type" 读取；这里还原成 None，
    # 让它走默认 RoPE 分支，避免 KeyError: 'type'。
    reward_config.rope_scaling = None
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path,
        config=reward_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    reward_model = nn.Module.to(reward_model, args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )

    # ========== 6. 数据集和优化器 ==========
    train_ds = RLAIFDataset(
        args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len)
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(
        critic_model.parameters(), lr=args.critic_learning_rate
    )

    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler
    )
    iters = len(loader_for_count)
    total_optimizer_steps = max(1, (iters // args.accumulation_steps) * args.epochs)
    actor_scheduler = CosineAnnealingLR(
        actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    critic_scheduler = CosineAnnealingLR(
        critic_optimizer,
        T_max=total_optimizer_steps,
        eta_min=args.critic_learning_rate / 10,
    )

    # ========== 7. 从检查点恢复 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 8. DDP 包装 ==========
    if dist.is_initialized():
        actor_model = DistributedDataParallel(
            actor_model, device_ids=[local_rank], broadcast_buffers=False
        )
        critic_model = DistributedDataParallel(
            critic_model, device_ids=[local_rank], broadcast_buffers=False
        )
        old_actor_model = nn.Module.to(old_actor_model, args.device)

    # ========== 9. 开始训练 ==========
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
            ppo_train_epoch(
                epoch, loader, len(loader) + start_step, old_actor_model, ref_model,
                actor_scheduler, critic_scheduler, reward_model, reward_tokenizer,
                start_step, wandb,
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
            ppo_train_epoch(
                epoch, loader, len(loader), old_actor_model, ref_model,
                actor_scheduler, critic_scheduler, reward_model, reward_tokenizer,
                0, wandb,
            )

    if dist.is_initialized():
        dist.destroy_process_group()
