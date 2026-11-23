# MiniMindB 学习项目

MiniMindB 是一个从零实现和训练的小型中文语言模型。模型为 Dense Decoder-only Transformer，约 **25.83M 参数**，隐藏维度 512，共 8 层，使用 8 个 Query 头、2 个 KV 头和 6400 词表。已跑通预训练、监督微调（SFT）、DPO 和 GRPO；PPO 脚本留作学习参考，未完成正式训练验证。本项目是单卡学习复现，不是官方 MiniMind 成品模型。

模型实现见 [`model/model.py`](model/model.py)，分词器出处见 [`model/TOKENIZER_SOURCE.md`](model/TOKENIZER_SOURCE.md)。学习参考：[MiniMind](https://github.com/jingyaogong/minimind) 和 [MokioMind](https://github.com/Wood-Q/MokioMind)。

## 已完成的实验

| 实验 | 起点 | 训练记录 | 推理权重 |
| --- | --- | --- | --- |
| 预训练 | 随机初始化 | `pretrain_t2t_mini.jsonl`，1 轮，39,695 批 | `out/pretrain-20261110-190312/pretrain_full_20261110_190312_512.pth` |
| SFT | 预训练权重 | `sft_t2t_mini.jsonl`，1 轮，56,608 批 | `out/full_sft_512.pth` |
| DPO | SFT | `dpo.jsonl`，17,166 对偏好样本，1 轮，4,292 批 | `out/dpo_512.pth` |
| GRPO | SFT | `rlaif.jsonl`，19,502 个问题，每题生成 4 个回答，1 轮，9,751 批 | `out/grpo_512.pth` |
| GRPO → DPO | GRPO | 104 对定向偏好样本，1 轮，52 次更新 | `out/grpo_then_dpo_512.pth` |

DPO 和 GRPO 是从同一份 SFT 权重出发的两条独立路线。`GRPO → DPO` 是后续的单独小实验，不是必经步骤。

## 对比结果

四份后训练权重用同样的 8 道自动测试题、随机种子 `2026`、`temperature=0.85` 和 `top_p=0.85` 评测，每题最多生成 256 个新 token。单独做 DPO 后有 4 道题与 SFT 的回答完全相同：自我介绍较贴切，但美食题重复更严重，不能说整体变好。GRPO 的回答与两份基线均不同，仍有事实错误、重复和指令未完成的情况。

在另留的 24 道题上，原 GRPO 与 `GRPO → DPO` 每题各采样两次。连续短语循环由 **5/48 降到 3/48**；指定函数名与 `return` 的静态检查由 **0/12 变为 2/12**，人工核对后这两次仍未完成题目要求；短答案严格匹配均为 **0/12**。小样本结果不足以证明整体能力提升，也不是与官方模型的性能比较。

四栏报告在 `runs/grpo-20261123/MiniMindB_SFT_DPO_GRPO_对比.html`，原始回答、留出题、指标与口径保存在 `runs/` 下。Git 不跟踪训练产物，需从本机归档恢复后才能在新克隆中打开报告或加载权重。

## 直接测试已有权重

从项目根目录执行。输入 `0` 跑固定测试题，输入 `1` 自己提问；可把 `full_sft` 换成 `dpo`、`grpo` 或 `grpo_then_dpo`。

```powershell
Set-Location 'D:\sdxx\minimind'
$python = 'D:\anaconda\python.exe'
& $python -X utf8 -B eval.py --weight full_sft --max_new_tokens 256 --temperature 0.85 --top_p 0.85 --historys 0
```

只测试预训练权重时使用它原来的目录和前缀，`eval.py` 会用纯文本续写，不套聊天模板：

```powershell
& $python -X utf8 -B eval.py --save_dir out/pretrain-20261110-190312 --weight pretrain_full_20261110_190312 --max_new_tokens 256
```

## 训练命令

本机验证环境：Windows、RTX 5070 Ti（16 GB）、Python 3.13.9、PyTorch 2.12.1+cu132、Transformers 5.22.2、datasets 5.2.0、tokenizers 0.23.3。使用上面的 Anaconda 解释器；[environment-local.json](environment-local.json) 是早期环境快照，不代表这里的最终版本。

把 `pretrain_t2t_mini.jsonl` 和 `sft_t2t_mini.jsonl` 放在 `dataset/` 下。GRPO 还需要可信来源的本地 `InternLM2-1.8B-Reward`，默认路径是项目同级的 `../internlm2-1_8b-reward`。DPO 不需要奖励模型。数据版本、哈希和训练日志见归档。

下列命令用于复现训练，**会写入正式权重的路径**。已有结果时请在单独的恢复副本中运行；只想推理不要执行它们。

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_UPDATE_DOWNLOAD_COUNTS = '0'
$env:HF_DATASETS_CACHE = Join-Path (Get-Location) '.cache/datasets'

# 预训练：单卡 BF16，完整 mini 数据 1 轮。
& $python -X utf8 -u -B trainer/trainer_pretrain.py --data_path dataset/pretrain_t2t_mini.jsonl --epochs 1 --max_seq_len 512 --batch_size 32 --accumulation_steps 8 --learning_rate 5e-4 --num_workers 2 --log_interval 20 --save_interval 1000 --dtype bfloat16 --device cuda:0 --save_dir out/pretrain-20261110-190312 --save_weight pretrain_full_20261110_190312 --from_weight none

# 从预训练权重做 SFT。
& $python -X utf8 -u -B trainer/train_full_sft.py --from_weight out/pretrain-20261110-190312/pretrain_full_20261110_190312_512.pth --epochs 1 --device cuda:0 --num_workers 0

# DPO、GRPO 从 SFT 分支；单卡请分别运行。
& $python -X utf8 -u -B trainer/train_dpo.py --from_weight full_sft --epochs 1 --device cuda:0 --num_workers 0 --batch_size 4 --accumulation_steps 1 --log_interval 20
& $python -X utf8 -u -B trainer/train_grpo.py --from_weight full_sft --data_path dataset/rlaif.jsonl --reward_model_path ../internlm2-1_8b-reward --device cuda:0 --epochs 1 --batch_size 2 --num_generations 4 --max_gen_len 128 --log_interval 10
```

`GRPO → DPO` 的数据、精确启动参数和留出题见归档中的 `runs/grpo-then-dpo-20261123-203814/`。预训练的原始命令与源码快照见 `runs/pretrain-20261110-190312/`；使用当前源码重跑不保证产生逐位相同的权重。

## 验证与归档

```powershell
& $python -X utf8 -B -m unittest discover -s tests
```

`out/` 存放推理权重，`checkpoints/` 存放续训状态，`runs/` 存放训练日志、评测输出和报告。这三个目录被 Git 忽略；GitHub 仓库不是训练产物备份。本机结项归档保留这些最终产物、源码快照和 SHA-256 清单，训练大语料和奖励模型另按清单恢复。
