"""
本地奖励模型实现。

原 PPO 脚本依赖外部 InternLM2 奖励模型（带 get_score 方法），
当前项目没有这个外部模型，因此这里用 MiniMindBForCausalLM 做底座，
在最后一层 hidden state 上接一个标量打分头，得到 Reward Model。

注意：
- 这里的 score_head 默认是随机初始化的，直接用于 PPO 只能跑通流程，
  奖励信号没有意义。
- 正式训练前需要先准备 chosen/rejected 偏好数据，训练出 reward_model_512.pth。
"""

import torch
from torch import nn

from model.model import MiniMindBConfig, MiniMindBForCausalLM


class RewardModel(nn.Module):
    """基于 MiniMindB 的序列级奖励模型，给整段对话输出一个分数。"""

    def __init__(self, config: MiniMindBConfig):
        super().__init__()
        # 底座复用当前项目的因果语言模型。
        self.backbone = MiniMindBForCausalLM(config)
        # 打分头：把最后一个 token 的 hidden state 映射成一个标量分数。
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """返回批次中每条序列的奖励分数，形状为 [B]。"""
        # MiniMindBModel 已经做过 layer_norm，返回的 hidden_states 可直接使用。
        hidden_states, _ = self.backbone.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 取每条序列最后一个有效 token 的隐藏状态作为整句表征。
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            hidden_states.size(0), device=hidden_states.device
        )
        last_hidden = hidden_states[batch_indices, last_indices]

        return self.score_head(last_hidden).squeeze(-1)

    @torch.no_grad()
    def get_score(self, tokenizer, messages, max_length=1024):
        """
        对一段 chat messages 打分，返回 Python float。

        这里用 tokenizer 的 chat template 把 messages 渲染成字符串，
        再走一次 forward 拿到标量分数。
        """
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(self.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        return float(self.forward(**enc).item())
