import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class MiniMindBConfig(PretrainedConfig):
    model_type = "minimindb"

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

    def __init__(self, args: MiniMindBConfig):
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
