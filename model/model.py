import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# 每层是 (K, V) 两个张量；外层列表按 Transformer 层的顺序保存缓存。
KVCache = Tuple[torch.Tensor, torch.Tensor]
PastKeyValues = list[Optional[KVCache]]


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int | None = None,  # pyright: ignore[reportArgumentType]
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        # 未启用长文本外推时为 None；读取字段处需先判空，避免 None.get(...)
        self.rope_scaling: dict[str, str | float | int] | None = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-5) -> None:
        super().__init__()
        self.dim= dim
        self.eps= eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        return self.weight * self._norm(x.float()).type_as(x) * x

def precompute_freqs(
    dim: int,
    end: int = 32 * 1024,
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos、sin 表，之后旋转 Q、K 时按 token 位置取用。

    参数：
        dim：单个注意力头参与旋转的维度 d，必须为正偶数。
            当前配置为 hidden_size / num_attention_heads = 512 / 8 = 64。
        end：缓存的位置数，位置编号为 0, 1, ..., end - 1。
            它是预计算表的长度，不是每次输入文本的实际长度。
        rope_base：频率基数；不同维度对由此得到不同的旋转速度。
        rope_scaling：None 表示普通 RoPE；非空时按下面的 YaRN 逻辑处理。

    核心公式（i 是频率组索引，范围为 0 到 d/2 - 1）：
        频率 omega_i = 1 / rope_base ** (2*i/d)
        角度 angle[m, i] = 位置 m * 频率 omega_i
        一对分量旋转后：x_a' = x_a*cos(angle) - x_b*sin(angle)
                        x_b' = x_a*sin(angle) + x_b*cos(angle)
    本函数准备上述 cos、sin，实际旋转在应用 RoPE 的函数中完成。

    返回：
        freqs_cos、freqs_sin：形状均为 [end, dim]，采用前后半段配对布局。
    """
    # 1. 基础频率：[dim // 2]，每对分量共用一个频率。
    # 例如 dim=8：arange -> [0, 2, 4, 6]，除以 dim -> [0, .25, .5, .75]。
    # 再取 base 的相应幂的倒数：索引越大，频率越小，旋转越慢。
    # dim 为偶数时，arange 本来就有 dim//2 个元素，这里的切片可省略。
    freqs, attn_factor = (
        1.0 / rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim),
        1.0,
    )

    # 2. 只有传入配置才读取 YaRN 参数；None 时保留上面的基础频率。
    if rope_scaling is not None:
        # orig_max：原始训练窗口长度，应按模型实际训练配置填写。
        # factor：扩展倍数 s，例如从 2048 扩展至 32768 时 s=16。
        # beta_fast=32：高频边界，对应论文的 beta，表示原始窗口内约转 32 圈。
        # beta_slow=1：低频边界，对应论文的 alpha，表示原始窗口内约转 1 圈。
        # attn_factor：cos、sin 的幅度系数 a，默认 1.0 表示不调整幅度。
        # 这里直接读取配置，不会自动计算论文的经验值 1 + 0.1*ln(factor)。
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 3. 缓存长度超过原始窗口时，用缩放频率生成整张表。
        # 此判断必须在配置判空内部：未传配置时，orig_max 等变量尚未定义。
        # 例如 end=32768、orig_max=2048，即使之后只取前 100 个位置，
        # 取到的也是这张采用缩放频率的缓存，不会按实际输入长度自动切换。
        if end > orig_max:
            # 一圈为 2*pi 弧度，因此原始窗口内的圈数 b = orig_max*omega_i/(2*pi)。
            # 代入 omega_i = base**(-2*i/d)，反解频率组索引 i：
            # i = d * ln(orig_max/(2*pi*b)) / (2*ln(base))。
            # lambda 只是在定义函数；调用 inv_dim(b) 时才计算。
            # b 越大，索引越小，所以 beta_fast 对应较小索引。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )
            # low、high 是索引的下界、上界；floor 向下取整，ceil 向上取整。
            # 默认 dim=64、orig_max=2048、base=1e6 时：low=5，high=14。
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 4. 每组频率的混合权重 ramp，形状仍为 [dim // 2]。
            # ramp_i = clamp((i-low)/(high-low), 0, 1)。
            # i<=low 时为 0；i>=high 时为 1；中间随索引线性增长。
            # clamp 把结果限制在 [0, 1]；分母至少为 0.001，避免除以 0。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 5. 逐元素混合原频率与降低后的频率：
            # new_freq = (1-ramp)*old_freq + ramp*(old_freq/factor)。
            # ramp=0：高频保持不变；ramp=1：低频降为原来的 1/factor。
            # 例如 factor=16、ramp=0.5：倍率为 0.5 + 0.5/16 = 0.53125。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 6. 位置编号 t：[end]，例如 end=4 时为 [0, 1, 2, 3]。
    t = torch.arange(end, device=freqs.device).float()

    # 7. 外积得到每个位置、每组频率对应的角度：angle[m, i] = t[m]*omega_i。
    # 注意：这行之后 freqs 的含义从“频率”变成“角度”，形状变为 [end, dim//2]。
    freqs = torch.outer(t, freqs).float()

    # 8. 将整组 cos、sin 各重复一次，扩展为 [end, dim]。
    # dim=8 时一行 cos 为 [c0,c1,c2,c3,c0,c1,c2,c3]。
    # 对应前后半段配对 (x0,x4)、(x1,x5)、(x2,x6)、(x3,x7)，
    # 应配合 rotate_half(x)=cat([-x[..., dim//2:], x[..., :dim//2]]) 使用。
    # 再按 q*cos + rotate_half(q)*sin 即可完成每对分量的二维旋转。
    # 两张表都乘 a=attn_factor 后，旋转后的 Q、K 各乘 a，点积整体乘 a**2。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """将预计算的 cos、sin 应用到 Q、K，按各 token 的位置旋转向量。

    默认输入布局：
        q：[B, T, Hq, D]，k：[B, T, Hkv, D]。
        B 为 batch 大小，T 为当前序列长度，H 为头数，D 为单个头的维度。
        当前配置 Hq=8、Hkv=2、D=64；Q、K 的头数可以不同。
        cos、sin：[T, D]，应已从预计算表中选出当前 token 对应的位置。

    参数：
        position_ids：预留参数，当前实现未读取它，也不会据此索引 cos、sin。
            调用前需选好位置；例如从缓存位置 start_pos 开始时，
            传入 cos[start_pos:start_pos+T]、sin[start_pos:start_pos+T]。
        unsqueeze_dim：在 cos、sin 的哪个维度插入大小为 1 的轴。
            上述布局使用默认值 1，得到 [T, 1, D]，在各头之间共享角度。

    对每对分量 (a, b)，角度为 theta 时：
        a' = a*cos(theta) - b*sin(theta)
        b' = a*sin(theta) + b*cos(theta)
    将各对分量拼回向量后，可统一写为：
        x_rotated = x*cos + rotate_half(x)*sin。
    返回旋转后的 Q、K，形状分别与输入 q、k 相同。
    """

    def rotate_half(x):
        # 1. 沿最后一维 D 拆成前后两半，要求 D 为偶数。
        # ... 保留 batch、位置和头这几维，切片只处理每个头内部的分量。
        # 这里的配对为 (x_i, x_{i+D/2})，与 precompute_freqs 重复整组
        # cos、sin 的布局一致。例如 D=4 时，配对为 (x0,x2)、(x1,x3)。
        # cat((-后半, 前半))：[x0,x1,x2,x3] -> [-x2,-x3,x0,x1]。
        # 对每一对 (a,b)，这相当于先变成 (-b,a)，即旋转 90 度。
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
            dim=-1,
        )

    # 2. 在 cos、sin 上插入相同的轴，确保两者采用同一种广播方式。
    # 默认：[T,D] -> [T,1,D]；与 [B,T,H,D] 运算时，
    # PyTorch 将其视为 [1,T,1,D]，自动扩展到每个 batch、每个注意力头。
    # 因此同一位置的各个头共用角度，不同位置使用不同角度。
    # 下面的 * 是逐元素乘法，每一项最后仍为 [B,T,H,D]。
    # 前半段：a*cos + (-b)*sin；后半段：b*cos + a*sin。
    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)

    # 3. 分别返回 Q、K 的旋转结果；V 在注意力模块中直接用于加权求和。
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x

    # 沿第 2 维（KV头维度），把每个头连续重复 n_rep 次。
    return torch.repeat_interleave(x, repeats=n_rep, dim=2)


class Attention(nn.Module):
    """定义注意力层使用的投影、头数和 Dropout 配置。

    后续输入 x 的形状为 [B, T, hidden_size]：B 是批次大小，T 是序列长度。
    Q、K 用于计算注意力权重，V 提供按这些权重汇总的内容。
    多个 Q 头共享较少的 KV 头，就是 GQA（分组查询注意力）。
    """

    def __init__(self, args: MokioMindConfig):
        super().__init__()

        # 1. 未单独指定 KV 头数时，令其等于 Q 头数，使用普通多头注意力。
        # 显式指定较少的 KV 头数时，每组 Q 头共享一个 K 头和一个 V 头。
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # 配置中的头数应为正整数；这里检查 Q 头数能否被 KV 头数整除。
        # 只有整除，才能让每个 KV 头恰好对应同样数量的 Q 头。
        assert args.num_attention_heads % self.num_key_value_heads == 0

        # 2. 保存头数，并计算每个 KV 头要供多少个 Q 头共享。
        # 默认 Hq=8、Hkv=2，所以 n_rep=8//2=4。
        # 后续 repeat_kv 会把 KV 头按 [A,A,A,A,B,B,B,B] 的顺序展开。
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 每个头的维度 D = hidden_size / Hq；这里要求 hidden_size 能被 Hq 整除。
        # 默认 D=512//8=64，因此 Hq*D=hidden_size，Q 和 O 的维度写法可以简化。
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 3. 为下面的线性层定义维度别名，减少重复的长表达式。
        # dim 是模型隐藏维度；kv_dim 是全部 KV 头尚未重复时的总维度。
        # 默认 dim=512，kv_dim=2*64=128。
        dim = args.hidden_size
        kv_dim = self.num_key_value_heads * self.head_dim

        # nn.Linear(in_dim, out_dim, bias=False) 的计算为 y = x @ weight.T。
        # weight 形状为 [out_dim, in_dim]；bias=False 表示不额外加偏置向量。
        # 下面四次构造各自创建独立的可学习权重。

        # Q 投影：[B,T,512] -> [B,T,512]，之后拆成 [B,T,8,64]。
        # 输出维度为 Hq*D=dim，每个 Q 头有各自的查询表示。
        self.q_proj = nn.Linear(dim, dim, bias=False)

        # K 投影：[B,T,512] -> [B,T,128]，之后拆成 [B,T,2,64]。
        # 先只生成 Hkv 个 K 头，后续再通过 repeat_kv 与 Q 的头数对齐。
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)

        # V 投影与 K 的形状相同，但使用独立权重，学习用于加权汇总的内容。
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)

        # O 投影：将各 Q 头的注意力输出拼接为 [B,T,Hq*D]=[B,T,512] 后，
        # 再映射回模型隐藏维度 [B,T,512]，学习混合来自不同头的信息。
        # 输入输出维度虽然相同，仍然会通过可学习矩阵改变向量内容。
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # 4. 为后续前向计算准备 Dropout。
        # attn_dropout 用于 softmax 后的注意力权重；
        # resid_dropout 用于 O 投影之后、加入残差之前的输出。
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 保存概率数值，供后续调用 scaled_dot_product_attention 的 dropout_p 使用。
        # 该接口在推理时需要显式传入 0.0，例如 self.dropout if self.training else 0.0。
        self.dropout = args.dropout

        # 5. 检查 PyTorch 是否提供 SDPA 接口，以及配置是否允许使用它。
        # self.flash 是使用该接口的开关；实际选择哪种注意力内核由 PyTorch 决定。
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """计算因果自注意力，返回当前 token 的输出和可选的 KV 缓存。

        维度记号：
            B：batch 大小；T：本次输入的 token 数；P：历史缓存中的 token 数。
            S=P+T：Key/Value 总长度；C：hidden_size；D：单个头的维度。
            Hq：Q 头数；Hkv：KV 头数。默认 C=512、D=64、Hq=8、Hkv=2。

        参数：
            x：[B,T,C]，只传本次要计算的 token 的隐藏状态。
            position_embeddings：(cos, sin)，分别为当前 token 对应的 [T,D]。
                调用前从整张 RoPE 表选取正确的绝对位置。例如 P=10、T=2，
                应传 all_cos[10:12]、all_sin[10:12]，而不是再次从位置 0 开始。
            past_key_value：历史 K、V，均为 [B,P,Hkv,D]；首次计算时为 None。
                历史 K 已经应用 RoPE，后续只拼接，不再次旋转。
            use_cache：为 True 时，返回拼接后的 K、V，供下次调用继续使用。
            attention_mask：[B,S] 的 0/1 掩码，1 表示有效 Key，0 表示 PAD。
                有历史缓存时，也要包含历史 Key 的有效性标记。

        返回：
            output：[B,T,C]，与本次输入的形状一致。
            past_kv：use_cache=True 时为两个 [B,S,Hkv,D] 张量，否则为 None。
        """
        # 1. 读取输入形状。x 有三维，因此需要用 _ 接收最后的隐藏维度 C。
        bsz, seq_len, _ = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].shape[1]

        # 2. 用三个独立的线性映射生成 Q、K、V。
        # Q：[B,T,C] -> [B,T,Hq*D]；K/V：[B,T,C] -> [B,T,Hkv*D]。
        # 再把最后一维拆为“头数、每头维度”，此时 token 位置仍在第 1 维。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3. 按当前 token 的位置旋转 Q、K，使 QK 点积带上相对位置关系。
        # 一对分量 (a,b) -> (a*cos-b*sin, b*cos+a*sin)，V 保留投影后的内容。
        # cos/sin 应已由调用方选好绝对位置；[:seq_len] 取本次需要的 T 行。
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 4. 将当前 K/V 接到历史 K/V 后面：[B,P,Hkv,D] + [B,T,Hkv,D]
        # -> [B,S,Hkv,D]。Q 只保留当前 T 个，因为只需计算当前 token 的输出。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 在重复 KV 头之前保存缓存，每个 token 只存 Hkv 个头，减少缓存占用。
        # use_cache 控制是否返回新缓存；即使为 False，传入的历史缓存仍会参与计算。
        past_kv = (xk, xv) if use_cache else None

        # 5. 对齐注意力头数，并把“头”移到“序列”前面，方便每个头独立做矩阵乘法。
        # repeat_kv：例如 Hkv=2、n_rep=4，头顺序变成 [A,A,A,A,B,B,B,B]。
        # Q：[B,T,Hq,D] -> [B,Hq,T,D]。
        # K/V：[B,S,Hkv,D] -> [B,S,Hq,D] -> [B,Hq,S,D]。
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        kv_len = xk.shape[-2]

        # 6. 无历史缓存、输入含多个 token、没有 PAD 时，使用 SDPA 接口。
        # 这一分支中 T=S，is_causal=True 正好对应普通的下三角因果掩码。
        # 全 1 的 padding 掩码与 None 的效果相同，因此无需再传显式 attn_mask。
        # 缓存使 T 和 S 不同时，转到下面的手动分支，按历史长度正确对齐掩码。
        if (
            self.flash
            and seq_len > 1
            and past_len == 0
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # SDPA 合并完成 softmax(QK^T/sqrt(D) + mask) @ V。
            # 该接口不根据 self.training 自动关闭 Dropout，推理时需显式传 0.0。
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 7. 手动计算每个 Query 对每个 Key 的匹配分数。
            # [B,Hq,T,D] @ [B,Hq,D,S] -> [B,Hq,T,S]。
            # 除以 sqrt(D)，缓和维度增大导致的点积尺度增大；后续 softmax 用 float32。
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)

            # 8. 构造因果规则：当前第 i 个 Query 的绝对位置是 P+i，
            # 因而只允许关注 Key j <= P+i，禁止查看未来位置。
            # 例如 P=3、T=2、S=5，允许矩阵为：
            # [[1,1,1,1,0], [1,1,1,1,1]]。
            # tril(diagonal=P) 保留下三角及偏移后的对角线；True 表示允许关注。
            causal_mask = torch.ones(
                seq_len, kv_len, dtype=torch.bool, device=xq.device
            ).tril(diagonal=past_len)
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,S]

            # 9. 有 PAD 时，再与 padding 规则取交集：既不是未来，也不是 PAD，才能看。
            if attention_mask is not None:
                padding_mask = (
                    attention_mask
                    # [B,S] -> [B,1,1,S]：插入头维度和 Query 维度。
                    .view(bsz, 1, 1, -1)
                    # -> [B,Hq,T,S]：每个头、每个 Query 共用这条 Key 有效性规则。
                    # 最后的 -1 保持 S 不变；expand 本身通过视图共享数据。
                    .expand(bsz, self.n_local_heads, seq_len, -1)
                    # 1 -> True（有效），0 -> False（PAD）。
                    .bool()
                )
                attn_mask = attn_mask & padding_mask

            # 10. 屏蔽位置的分数设为 -inf，使其 softmax 权重为 0。
            scores = scores.masked_fill(~attn_mask, float("-inf"))

            # 左侧补齐可能让某些 PAD Query 没有任何可见 Key。
            # 先将这种整行设为 0，避免对全 -inf 做 softmax；再将被屏蔽权重清零。
            # 因此无可见 Key 的 Query 得到零注意力输出，正常 Query 的分数不变。
            has_valid_key = attn_mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~has_valid_key, 0.0)
            attn_weights = F.softmax(scores, dim=-1).masked_fill(~attn_mask, 0.0)
            attn_weights = attn_weights.type_as(xq)
            attn_weights = self.attn_dropout(attn_weights)

            # 11. 按注意力权重汇总 V：[B,Hq,T,S] @ [B,Hq,S,D] -> [B,Hq,T,D]。
            output = attn_weights @ xv

        # 12. 两个计算分支都要做输出投影并返回，不能只放在 else 内部。
        # 先恢复 [B,T,Hq,D]，再把各头拼成 [B,T,Hq*D]=[B,T,C]。
        # transpose 改变步长，contiguous 确保内存布局适合后面的 view。
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(output)  # 用 W_o 学习混合不同头的信息。
        output = self.resid_dropout(output)

        # 残差相加通常由外面的 Transformer block 完成；这里返回注意力分支的输出。
        return output, past_kv


class FeedForward(nn.Module):
    """对每个 token 的隐藏向量做门控非线性变换，所有位置共享同一组参数。

    输入与输出均为 [B,T,C]；中间维度记为 I。
    默认 hidden_act="silu" 时是 SwiGLU：
        gate = SiLU(x @ W_gate.T)，其中 SiLU(z) = z * sigmoid(z)
        up = x @ W_up.T
        output = Dropout((gate * up) @ W_down.T)
    * 表示逐元素相乘，门控分支对内容分支的各个特征分别进行调制。
    """

    def __init__(self, args: MokioMindConfig):
        super().__init__()

        # 1. 确定中间维度 I；配置中已指定时，直接使用指定值。
        intermediate_size = args.intermediate_size
        if intermediate_size is None:
            # 普通 C->4C->C 的两层 FFN 约有 8*C^2 个权重。
            # SwiGLU 有 gate、up、down 三个矩阵，约有 3*C*I 个权重。
            # 令两者参数规模接近：3*C*I ≈ 8*C^2，得到 I ≈ 8*C/3。
            intermediate_size = int(args.hidden_size * 8 / 3)

            # 将 I 向上取整到 64 的倍数：ceil(I/64)*64。
            # 对整数而言 ceil(I/64) = (I+63)//64；对齐有利于矩阵计算的布局。
            # 默认 C=512：先得到 1365，再向上对齐成 1408。
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            args.intermediate_size = intermediate_size

        # 2. up、gate 分别把 C 维投影到 I 维，使用两组独立的权重。
        # 默认：[B,T,512] -> [B,T,1408]。
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)

        # down 将逐元素相乘后的特征映射回 C 维，方便与输入做残差相加。
        # 默认：[B,T,1408] -> [B,T,512]。
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)

        # ACT2FN 将字符串名称映射为激活函数，例如 "silu" 对应 SiLU。
        # 选择其他激活时，仍是门控 FFN，但不再称为 SwiGLU。
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3. 内容分支 up 和门控分支 gate 都读取原输入 x，形状均为 [B,T,I]。
        # 激活函数先作用于 gate 分支，再与 up 相乘。
        # SiLU 输出可为负数或大于 1，因此这里的门控值不是概率。
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        hidden = gate * up

        # 4. 将门控后的特征投影回隐藏维度；本层不改变 batch 或序列长度。
        output = self.down_proj(hidden)
        return self.dropout(output)


class MokioMindBlock(nn.Module):
    """一个使用 Pre-Norm 的 Transformer block：注意力子层 + 门控 FFN 子层。

    两个子层都先归一化输入，再计算变换，最后加回该子层的原始输入：
        h = x + Attention(RMSNorm_1(x))
        y = h + FeedForward(RMSNorm_2(h))
    这里 Attention(...) 表示注意力输出张量，KV 缓存另外传回。
    x、h、y 的形状均为 [B,T,C]，因此多个 block 可以顺序堆叠。
    """

    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()

        # 保存层编号和维度信息；layer_id 用于标识第几层，本实现不参与数值计算。
        self.layer_id = layer_id
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 注意力负责汇总不同 token 的信息，输入输出均为 [B,T,C]。
        self.self_attn = Attention(config)

        # 两个独立的 RMSNorm 都沿最后一维 C 归一化，每个模块有自己的缩放权重。
        # input_layernorm 用在 Attention 之前；post_attention_layernorm 用在 FFN 之前。
        # 后者的名字表示其处在注意力子层之后，而这个 block 整体仍采用 Pre-Norm。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN 对每个 token 的特征向量进行门控变换，本模块使用稠密 FeedForward。
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """返回本层更新后的 [B,T,C] 隐藏状态，以及本层可选的 KV 缓存。

        position_embeddings 是当前 token 对应的 cos、sin；缓存和掩码的格式
        与 Attention.forward 相同。不同 block 应分别保存、传入各自的 KV 缓存。
        """
        # 1. 第一条残差路径：保存注意力子层的原始输入 x。
        # 只把归一化后的输入送给 Attention，旁路保留原始信息及其梯度路径。
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            normed_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )

        # 2. h = x + Attention(RMSNorm_1(x))；两项形状相同，可逐元素相加。
        hidden_states = residual + attn_output

        # 3. 第二条残差路径：以注意力子层的输出 h 作为 FFN 子层的原始输入。
        residual = hidden_states
        normed_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(normed_states)

        # 4. y = h + FeedForward(RMSNorm_2(h))，输出仍为 [B,T,C]。
        hidden_states = residual + ffn_output

        # 本层 KV 来自 Attention；FFN 逐 token 计算，不新增历史 KV 缓存。
        return hidden_states, present_key_value

class MokioMindModel(nn.Module):
    # register_buffer 动态注册属性；声明类型后，编辑器可确定它们支持张量切片。
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.vocab_size,self.num_hidden_layers=(
            config.vocab_size,config.num_hidden_layers
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MokioMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos,freqs_sin =precompute_freqs(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )

        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[PastKeyValues] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, PastKeyValues]:
        if input_ids is None:
            raise ValueError("必须传入 input_ids")
        batch_size, seq_len = input_ids.shape
        if not past_key_values:
            # 显式指定元素类型，避免空缓存被推断成只能存放 None 的列表。
            past_key_values = list[Optional[KVCache]](
                None for _ in self.layers
            )
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values 必须为每个 Transformer 层提供一项缓存")

        # 每层缓存为 (K, V) 或 None；K 的布局是 [B, 历史长度, Hkv, D]。
        # 先取到局部变量再判空，类型检查器便能确定非空分支中的 K 是张量。
        first_cache = past_key_values[0] if past_key_values else None
        start_pos = first_cache[0].shape[1] if first_cache is not None else 0
        if start_pos + seq_len > self.freqs_cos.shape[0]:
            raise ValueError("输入与历史缓存的总长度超过了 RoPE 位置表长度")
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents: PastKeyValues = []

        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states,present = layer(
                hidden_states,position_embeddings,
                past_key_value = past_key_value,use_cache = use_cache,attention_mask = attention_mask,
            )

            presents.append(present)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states,presents

class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    # PreTrainedModel 提供模型基类功能，GenerationMixin 提供 generate() 接口。
    # config_class 指明这个模型使用哪种配置类。
    config_class = MokioMindConfig

    def __init__(self,config:MokioMindConfig):
        # 父类初始化会保存 config，并准备 Hugging Face 模型需要的基础属性。
        super().__init__(config)
        # model 是 Transformer 主干，输出每个 token 的 C 维隐藏向量。
        self.model = MokioMindModel(config)
        # 输出层把 C 维特征映射成 V 个词表分数：h @ W.T；False 表示不使用偏置。
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size, False)
        # 输入嵌入与输出投影共享同一个 [V,C] 权重参数。
        self.model.embed_tokens.weight = self.lm_head.weight


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: PastKeyValues | DynamicCache | None = None,
        use_cache: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **args,
    ):
        # 维度约定：B=批次大小，T=本次输入长度，P=历史缓存长度，
        # C=hidden_size，V=vocab_size，H=KV 头数，D=每个注意力头的维度。
        # input_ids：[B,T]；attention_mask 通常为 [B,P+T]，1 表示有效 Key。

        # 1. 将接口传入的缓存转换为主干认识的格式。
        # generate() 的缓存布局是 [B,H,P,D]，本模型内部使用 [B,P,H,D]。
        # transpose(1,2) 交换“头数”和“历史长度”两个轴，不改变张量中的数值。
        # 首次生成时，某层可能是 (None,None)，对应主干中的 None。
        # 如果调用者传来的已经是本项目的列表缓存，就直接沿用。
        if isinstance(past_key_values, DynamicCache):
            past_key_values = [
                None if key is None and value is None
                else (key.transpose(1, 2), value.transpose(1, 2))
                for key, value in past_key_values.to_legacy_cache()
            ]

        # 2. 主干依次完成 Embedding、各层 Attention/FFN 和最终 RMSNorm。
        # hidden_states：[B,T,C]，表示本次输入的所有 token 的最终特征。
        # 返回的 past_key_values 是新的逐层缓存列表，覆盖这里同名的输入变量；
        # use_cache=True 时，每层保存两个 [B,P+T,H,D] 张量，否则每项为 None。
        # **args 将额外的关键字参数继续传给主干。
        hidden_states,past_key_values = self.model(
            input_ids = input_ids,attention_mask = attention_mask,
            past_key_values = past_key_values,use_cache = use_cache,**args
        )

        # 3. 决定返回哪些 token 位置的词表分数。
        # 整数 n>0 表示保留末尾 n 个位置；0 表示保留全部位置。
        # 负数不是有效的“保留数量”，因此抛出明确的异常。
        if isinstance(logits_to_keep, int) and logits_to_keep < 0:
            raise ValueError("logits_to_keep 不能为负数")
        # slice(-n,None) 相当于 [-n:]；n=0 时 -0=0，即 [0:]，保留整段。
        # 若传入整数索引张量，例如 tensor([0,2])，则选择第 0、2 个位置。
        slice_indices = (
            slice(-logits_to_keep,None)
            if isinstance(logits_to_keep,int) else logits_to_keep
        )
        # 4. 先将完整隐藏状态投影为 logits：[B,T,C] -> [B,T,V]。
        # 再沿第 1 维选取位置，得到 [B,N,V]；N 是实际保留的位置数。
        # 当前顺序只减少返回的位置数，lm_head 仍计算全部 T 个位置。
        # logits 是未经过 softmax 的原始分数，不能直接当作概率。
        logits = self.lm_head(hidden_states)[:,slice_indices,:]

        return CausalLMOutputWithPast(
            logits = logits,past_key_values = past_key_values,
            hidden_states = hidden_states,
        )
