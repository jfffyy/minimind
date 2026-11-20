import argparse
from contextlib import nullcontext
from importlib import import_module
import json
from math import ceil, isfinite
from pathlib import Path
import re
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

# 以脚本的位置定位项目，避免从不同目录启动时读错权重或数据。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model import MiniMindBConfig, MiniMindBForCausalLM
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, init_distributed_mode, is_main_process, lm_checkpoint, setup_seed,
)


class GRPOPromptDataset(Dataset):
    """复用 RLAIF 的 conversations 格式，但直接保留角色和消息边界。"""

    def __init__(self, path):
        self.samples = load_dataset("json", data_files=str(path), split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        turns = self.samples[index]["conversations"]
        # 新版语料已有 role，直接保留；只有缺失时才按 user/assistant 交替补上。
        messages = [
            {"role": turn.get("role") or ("user" if i % 2 == 0 else "assistant"),
             "content": turn.get("content") or ""}
            for i, turn in enumerate(turns)
        ]
        # 从最后一个用户问题处开始续写，不把原始参考回答泄漏给策略或奖励模型。
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(f"RLAIF 第 {index + 1} 条样本必须以 assistant 参考答案结束")
        for i in range(len(messages) - 2, -1, -1):
            if messages[i]["role"] == "user":
                return messages[:i + 1]
        raise ValueError(f"RLAIF 第 {index + 1} 条样本没有 user 消息")


def fit_prompt(tokenizer, messages, max_length):
    """先移除过早的完整对话轮，再截短最后的用户消息，保留生成标记。"""
    # 复制消息，下面会删历史、改最后一个问题的文本，不应污染原始样本。
    messages = [dict(message) for message in messages]

    def render():
        # add_generation_prompt=True：末尾补 assistant 起始标记，让模型从这里开始回答。
        # BOS 等标记由模板负责，统计 token 数时不要再重复添加。
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))

    prompt, length = render()
    # 有 system 消息时保留它；从下一条 user 开始切，避免截断在一轮对话中间。
    keep_system = int(messages[0]["role"] == "system")
    while length > max_length:
        next_user = next(
            (i for i in range(keep_system + 1, len(messages)) if messages[i]["role"] == "user"),
            None,
        )
        if next_user is None:
            break
        del messages[keep_system:next_user]
        prompt, length = render()
    if length > max_length:
        # 历史都删完还超长，只缩短当前问题的正文，不裁掉模板和 assistant 标记。
        user_ids = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        count = max(0, len(user_ids) - (length - max_length))
        # 渲染后重新分词，避免文本边界处的 BPE 合并影响长度。
        while True:
            messages[-1]["content"] = tokenizer.decode(user_ids[:count])
            prompt, length = render()
            if length <= max_length:
                break
            if count == 0:
                raise ValueError("max_seq_len 太小，放不下系统提示和聊天模板")
            count = max(0, count - max(1, length - max_length))
    # 文本交给 Policy；同一份裁剪后的结构化消息交给奖励模型，保证两者看到同一个问题。
    return prompt, messages


def completion_mask(completion_ids, pad_id, eos_id):
    """只计入第一个 EOS（含）之前的实际生成 token；不让补位参与 loss。"""
    is_eos = completion_ids.eq(eos_id)
    # 用累计 EOS 数减去当前位置的 EOS，得到“之前是否已结束”。
    # 例如 [A, EOS, EOS, PAD] → [True, True, False, False]，第一个 EOS 参与策略优化。
    past_eos = (is_eos.long().cumsum(dim=1) - is_eos.long()).gt(0)
    mask = ~past_eos
    return mask & completion_ids.ne(pad_id) if pad_id != eos_id else mask


def get_per_token_logps(model, ids, attention_mask, completion_length):
    # 最后 N+1 个 logits 去掉结尾，正好预测末尾 N 个回答 token。
    # 第一个回答 token 由 prompt 的最后一个位置预测，所以不是直接取最后 N 个 logits。
    logits = model(
        input_ids=ids, attention_mask=attention_mask, logits_to_keep=completion_length + 1,
    ).logits
    if logits is None:
        raise RuntimeError("策略模型没有返回 logits")
    # [B*G, N, V] 在词表维做 log_softmax，再用实际采样的 token ID gather。
    # 返回 [B*G, N]，每一项是该位置选中 token 的 log 概率，而不是整张词表的分数。
    return F.log_softmax(logits[:, :-1, :].float(), dim=-1).gather(
        dim=-1, index=ids[:, -completion_length:].unsqueeze(-1),
    ).squeeze(-1)


def calculate_rewards(contexts, responses, score, num_generations, reasoning):
    """每个用户问题的回答按生成顺序分在同一组。"""
    rewards = []
    for i, context in enumerate(contexts):
        # 生成结果按 prompt 排列：第 i 个问题的 G 个回答占据一段连续的位置。
        for response in responses[i * num_generations:(i + 1) * num_generations]:
            value = score(context + [{"role": "assistant", "content": response}])
            if not isfinite(value):
                raise RuntimeError("奖励模型返回了非有限分数")
            # 限制单次外部模型打分的幅度；这个分数是后续组内比较的依据，不是标签 ID。
            value = max(-3.0, min(value, 3.0))
            if reasoning:
                # 可选实验：最终答案单独打分，并奖励约定的标签格式。普通对话默认不走这里。
                answer = re.search(r"<answer>(.*?)</answer>", response, re.S)
                if answer:
                    answer_score = score(context + [{"role": "assistant", "content": answer[1].strip()}])
                    if not isfinite(answer_score):
                        raise RuntimeError("奖励模型返回了非有限答案分数")
                    value = 0.4 * value + 0.6 * max(-3.0, min(answer_score, 3.0))
                value += 0.25 * sum(
                    response.count(tag) == 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")
                )
                if re.fullmatch(r"<think>\n.*?\n</think>\n\n?<answer>\n.*?\n</answer>", response, re.S):
                    value += 0.5
            rewards.append(value)
    return rewards


def rollout_loss(model, reference, tokenizer, score, batch, args):
    # B：问题数；G：每题生成数；P：左补齐后的 prompt 长度；N：本批生成的回答宽度；
    # V：策略模型词表大小。默认 B=2、G=4，一批生成 8 个回答。
    # 1. 准备 prompt：[B, P]。main 已设置左 padding，使最后一列是真正的 prompt 结尾。
    fitted = [fit_prompt(tokenizer, messages, args.max_seq_len) for messages in batch]
    prompts, contexts = zip(*fitted)
    inputs = tokenizer(
        list(prompts), return_tensors="pt", padding=True, add_special_tokens=False,
        return_token_type_ids=False,
    ).to(args.device)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    # 2. rollout：在线采样，生成过程不构建反向图；输出 ids 为 [B*G, P+N]。
    # 有 DDP 包装时从内部模型调用 generate，稍后用外层 model 做带梯度的前向。
    with torch.no_grad():
        generate = getattr(raw_model, "generate")
        # 不用 temperature/top-k 改写分布，使 rollout 和下方重算概率一致。
        ids = generate(
            **inputs, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
            max_new_tokens=args.max_gen_len, num_return_sequences=args.num_generations,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成部分 [B*G, N]。同一个 prompt 的 mask 复制 G 份，再接有效回答范围。
    # full_mask 为 [B*G, P+N]，Policy 与 Reference 重算概率时必须看到相同的上下文。
    completion_ids = ids[:, inputs.input_ids.size(1):]
    mask = completion_mask(completion_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
    full_mask = torch.cat(
        (inputs.attention_mask.repeat_interleave(args.num_generations, dim=0), mask.long()), dim=1,
    )
    responses = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    # 3. 每个回答一个标量奖励。reshape 成 [B, G]，只和同一问题的其他回答比较。
    rewards = torch.tensor(
        calculate_rewards(contexts, responses, score, args.num_generations, args.reasoning),
        device=args.device, dtype=torch.float32,
    ).view(-1, args.num_generations)
    # A = (分数 - 同组均值) / (同组标准差 + ε)：正值表示比同题其他回答好，负值反之。
    # 同组分数都相等时 A 全为 0；没有来自该组的相对偏好梯度。
    advantages = (rewards - rewards.mean(1, keepdim=True)) / (
        rewards.std(1, keepdim=True, unbiased=False) + 1e-4
    )
    # 4. 对同一批已采样 token 重算 logp。Policy 有梯度；冻结的 Reference 只提供基准。
    n = completion_ids.size(1)
    logp = get_per_token_logps(model, ids, full_mask, n)
    with torch.no_grad():
        ref_logp = get_per_token_logps(reference, ids, full_mask, n)
    # 对每个有效 token：d = log p_ref - log p_policy，KL 估计项 = exp(d) - d - 1。
    # 两模型给该 token 相同概率时，d=0、惩罚为 0；beta 决定约束强度。
    log_ratio = (ref_logp - logp).masked_fill(~mask, 0.0)
    kl = torch.expm1(log_ratio) - log_ratio
    # 每份 rollout 只用一次；detach 固定采样策略，不是多轮 PPO 裁剪。
    # exp(logp - logp.detach()) 前向为 1，但反向只经过左边的 logp，对它的导数为 1。
    # 因此，组内 A 正负抵消、KL 很小导致日志 loss≈0 时，参数仍可能更新。
    # 忽略 KL 时，A>0 的回答提供增大所采样 token 的 logp 的梯度，A<0 的方向相反。
    loss_per_token = args.beta * kl - (logp - logp.detach()).exp() * advantages.reshape(-1, 1)
    # 5. 对每条回答先按有效 token 数取平均，再对所有回答取平均；padding 不计入。
    lengths = mask.sum(1)
    loss = (loss_per_token.masked_fill(~mask, 0.0).sum(1) / lengths.clamp_min(1)).mean()
    return loss, rewards.mean().item(), lengths.float().mean().item()


def train_epoch(epoch, loader, start_step, model, reference, tokenizer, score,
                optimizer, scheduler, scaler, args, checkpoint, wandb=None):
    # 恢复时 loader 已跳过前 start_step 批，把它加回来得到本轮总批数。
    iters = len(loader) + start_step
    model.eval()  # 采样和梯度重算都不启用 dropout；eval 不会关闭梯度。
    optimizer.zero_grad(set_to_none=True)
    save_pending = False
    for step, batch in enumerate(loader, start_step + 1):
        device_type = torch.device(args.device).type
        amp = autocast("cuda", dtype=getattr(torch, args.dtype)) if device_type == "cuda" else nullcontext()
        with amp:
            loss, reward, response_length = rollout_loss(model, reference, tokenizer, score, batch, args)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"第 {step} 批的 GRPO loss 非有限，已停止更新")
        # 每个微批贡献 loss / 组内实际批数。例：5 批、累积 2，更新为 2+2+1。
        group_start = start_step + (step - start_step - 1) // args.accumulation_steps * args.accumulation_steps
        group_size = min(args.accumulation_steps, iters - group_start)
        scaler.scale(loss / group_size).backward()
        update = (step - start_step) % args.accumulation_steps == 0 or step == iters
        if update:
            # FP16 梯度先反缩放再裁剪；GradScaler 因溢出跳过更新时，不推进学习率计划。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if is_main_process() and (step % args.log_interval == 0 or step == iters):
            metrics = dict(loss=loss.item(), reward=reward, response_length=response_length,
                           lr=optimizer.param_groups[0]["lr"])
            Logger(f"GRPO epoch={epoch + 1} step={step}/{iters} {metrics}")
            if wandb is not None:
                wandb.log(metrics)
        # 保存请求可以出现在累积中途，但只在更新边界落盘，避免断点丢失未保存的梯度。
        save_pending |= step % args.save_interval == 0 or step == iters
        if update and save_pending:
            if is_main_process():
                checkpoint(epoch, step)
            save_pending = False


def project_path(value):
    return (PROJECT_ROOT / value).resolve()


def resolve_base_weight(value, hidden_size):
    # 支持权重名前缀（如 full_sft → out/full_sft_512.pth）和明确的 .pth 路径。
    if value.endswith(".pth"):
        return project_path(value)
    return project_path(Path("out") / f"{value}_{hidden_size}.pth")


def encode_reward_chat(tokenizer, messages, token_remap, reward_id, vocab_size):
    """按 InternLM2 的聊天模板编码，并用训练时的特殊 token ID 打分。"""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    source_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = source_ids.clone()
    for runtime_id, trained_id in token_remap.items():
        # 用原张量做匹配，避免一个 token 改完后又被后续规则二次替换。
        ids.masked_fill_(source_ids == runtime_id, trained_id)
    # 序列末尾是奖励模型读出分数的专用位置；如果模板已经加过，就不重复追加。
    if ids.size(1) == 0 or ids[0, -1].item() != reward_id:
        ids = torch.cat((ids, ids.new_tensor([[reward_id]])), dim=1)
    # 在索引 embedding 前检查范围，避免错误 ID 到 GPU 上才触发设备断言。
    if ids.min().item() < 0 or ids.max().item() >= vocab_size:
        raise ValueError("奖励分词器生成了超出权重词表的 ID，请检查对应的 tokenizer_config.json")
    return ids


def load_reward_scorer(path, device):
    """加载本地 InternLM2 评分器，只使用它已有的权重和代码。"""
    # 奖励模型与 MiniMindB 各有自己的分词器和词表，不能共用输入 ID。
    # trust_remote_code=True 会运行该本地目录中的模型实现，只使用可信来源的模型文件。
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    config.use_cache = False  # 旧版实现默认会调用已移除的 DynamicCache.from_legacy_cache。
    if json.loads((Path(path) / "config.json").read_text(encoding="utf-8")).get("rope_scaling") is None:
        # 旧 InternLM2 将 None 解释为普通 RoPE，而不认识 5.x 的默认 rope_parameters 字典。
        config.rope_parameters = None
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    reward = AutoModel.from_pretrained(
        path, config=config, dtype=dtype, trust_remote_code=True, local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    if not callable(getattr(reward, "get_score", None)):
        raise TypeError(f"{path} 不是带 get_score 接口的 InternLM2 奖励模型")
    vocab_size = reward.get_input_embeddings().weight.size(0)
    reward_id = int(getattr(config, "reward_token_id"))
    declared = json.loads((Path(path) / "tokenizer_config.json").read_text(encoding="utf-8"))
    token_remap = {}
    # Transformers 5 转换旧 SentencePiece 时，可能把 ChatML 特殊 token
    # 追加到词表末尾。用原模型随附配置中的 ID 恢复，不扩张或修改模型权重。
    for trained_id_text, item in declared.get("added_tokens_decoder", {}).items():
        content = item["content"]
        runtime_id = tokenizer.convert_tokens_to_ids(content)
        trained_id = int(trained_id_text)
        if (isinstance(runtime_id, int) and runtime_id != tokenizer.unk_token_id
                and 0 <= trained_id < vocab_size and runtime_id != trained_id):
            token_remap[runtime_id] = trained_id

    @torch.no_grad()
    def score(messages):
        ids = encode_reward_chat(tokenizer, messages, token_remap, reward_id, vocab_size).to(device)
        # 等价于 get_score 的前向调用，但先校正 token 并禁用旧版 KV cache。
        logits = reward(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False, return_dict=True).logits
        return float(logits.item())

    return score


def main():
    # 命令行参数：默认训练普通对话，不启用 reasoning 格式奖励。
    parser = argparse.ArgumentParser(description="MiniMindB GRPO")
    parser.add_argument("--data_path", default="dataset/rlaif.jsonl")
    parser.add_argument("--from_weight", help="初始权重前缀或完整 .pth 路径（默认 full_sft）")
    parser.add_argument("--reward_model_path", default="../internlm2-1_8b-reward")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--save_weight", default="grpo")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=8e-8)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning", type=int, choices=[0, 1], default=0)
    parser.add_argument("--from_resume", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MiniMindB-GRPO")
    args = parser.parse_args()
    if args.num_generations < 2:
        parser.error("GRPO 每个问题至少需要两份回答（--num_generations >= 2）")
    if min(args.batch_size, args.epochs, args.max_seq_len, args.max_gen_len,
           args.accumulation_steps, args.log_interval, args.save_interval) < 1:
        parser.error("批次、轮数、长度、累积步数和日志/保存间隔必须大于 0")
    if args.num_workers < 0 or args.learning_rate <= 0 or args.grad_clip <= 0 or args.beta < 0:
        parser.error("请检查 num_workers、learning_rate、grad_clip 和 beta")
    # 初始权重不是续训断点：普通 GRPO 从 SFT 开始，reasoning 模式才默认找 reason 权重。
    base = resolve_base_weight(args.from_weight or ("reason" if args.reasoning else "full_sft"), args.hidden_size)
    if not base.is_file():
        parser.error(f"找不到初始权重 {base}。先完成 SFT，或用 --from_weight 指定可用的 .pth。")
    data_path, reward_path, save_dir = map(project_path, (args.data_path, args.reward_model_path, args.save_dir))
    if not data_path.is_file() or not (reward_path / "config.json").is_file():
        parser.error(f"缺少 RLAIF 数据或本地奖励模型：{data_path}；{reward_path}")
    if Path(args.save_weight).name != args.save_weight or not args.save_weight:
        parser.error("--save_weight 必须是文件名前缀，不是路径")
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    output_name = f"{args.save_weight}_{args.hidden_size}.pth"
    if base in ((save_dir / output_name).resolve(), (checkpoint_dir / output_name).resolve()):
        parser.error("GRPO 输出不能覆盖初始权重，请修改 --save_weight")

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    config = MiniMindBConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "model"), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("GRPO tokenizer 必须有 PAD 和 EOS")
    # 两份模型加载同一个初始权重。只有 model 的参数交给优化器，ref 始终冻结。
    model = MiniMindBForCausalLM(config)
    ref = MiniMindBForCausalLM(config)
    state = torch.load(base, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    ref.load_state_dict(state, strict=True)
    del state
    model = MiniMindBForCausalLM.to(model, args.device)
    ref = MiniMindBForCausalLM.to(ref, args.device).eval().requires_grad_(False)
    data = GRPOPromptDataset(data_path)
    if not len(data):
        raise ValueError("RLAIF 数据为空")
    sampler = DistributedSampler(data, seed=args.seed) if dist.is_initialized() else None
    # scheduler 按优化器更新次数计数，不是按微批计数；最后不满组也有一次更新。
    batches_per_epoch = ceil((len(sampler) if sampler is not None else len(data)) / args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=ceil(batches_per_epoch / args.accumulation_steps) * args.epochs,
        eta_min=args.learning_rate / 10,
    )
    scaler = GradScaler("cuda", enabled=args.device.startswith("cuda") and args.dtype == "float16")
    start_epoch, start_step = 0, 0
    saved = None
    if args.from_resume:
        # 只恢复正在训练的 Policy/优化器等状态，Reference 仍从初始权重加载。
        saved = lm_checkpoint(config, weight=args.save_weight, save_dir=str(checkpoint_dir))
        if saved is None:
            raise FileNotFoundError("未找到 GRPO 断点，请确认 --save_weight")
        if saved.get("base_weight") != str(base):
            raise ValueError("续训必须使用与原训练相同的 --from_weight，保证 Reference 不变")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, start_step = saved["epoch"], saved["step"]
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    # 先试真实打分，及早发现奖励模型、分词器或 Transformers 版本不兼容。
    score = load_reward_scorer(str(reward_path), args.device)
    if not isfinite(score([{"role": "user", "content": "1 + 1?"}, {"role": "assistant", "content": "2"}])):
        raise RuntimeError("本地奖励模型打分预检失败")
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            wandb = import_module("swanlab")
        except ModuleNotFoundError as exc:
            if exc.name != "swanlab":
                raise
            raise RuntimeError("--use_wandb 需要在当前 Python 环境安装 swanlab") from exc
        run_id = saved.get("wandb_id") if saved else None
        wandb.init(project=args.wandb_project, id=run_id, resume="must" if run_id else None)
    save_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(epoch, step):
        # 推理用的半精度权重先写临时文件再替换；续训另存优化器、调度器等完整状态。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        destination = save_dir / output_name
        tmp_path = destination.with_suffix(".pth.tmp")
        torch.save({k: v.detach().cpu().half() for k, v in raw_model.state_dict().items()}, tmp_path)
        tmp_path.replace(destination)
        lm_checkpoint(
            config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch,
            step=step, wandb=wandb, save_dir=str(checkpoint_dir), scheduler=scheduler,
            scaler=scaler, base_weight=str(base),
        )

    for epoch in range(start_epoch, args.epochs):
        # 用同一个 seed + epoch 生成单卡排列，恢复后跳过的是同一顺序的前若干批。
        # 这能复现数据顺序，但脚本没有保存采样 RNG 状态，不保证每次续训生成逐 token 相同。
        if sampler is not None:
            sampler.set_epoch(epoch)
            indices = sampler
        else:
            indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        skip = start_step if epoch == start_epoch else 0
        loader = DataLoader(data, batch_sampler=SkipBatchSampler(indices, args.batch_size, skip),
                            collate_fn=list, num_workers=args.num_workers)
        train_epoch(epoch, loader, skip, model, ref, tokenizer, score, optimizer,
                    scheduler, scaler, args, checkpoint, wandb)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
