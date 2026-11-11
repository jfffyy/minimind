# 分词器资源来源

这两份资源来自你参考的[上游项目固定版本](https://github.com/Wood-Q/MokioMind/tree/125dc2d245ee35f4940855efabb915a0ad02428c/model)，没有同步或覆盖该仓库的 Python 模型、训练代码。

- 固定提交：`125dc2d245ee35f4940855efabb915a0ad02428c`。
- `tokenizer.json`：439547 字节，Git blob `b255122b500f7bafb3efaa10c22359abc6a84e8a`。
- `tokenizer_config.json`：4506 字节，Git blob `fc4e726fcde7779578a8fd89ef2f1c6321a28ece`。
- BPE 词表大小：6400；BOS / EOS / PAD ID：1 / 2 / 0。

下载内容已与该提交的 Git blob 核对，并通过当前 Transformers 5.22.2 的本地加载与中文编解码验证，不需要为这两份 JSON 安装 sentencepiece 或 tiktoken。

这里只提供分词器，不包含预训练模型权重，也不包含 `dataset/pretrain_hq.jsonl` 训练语料。
