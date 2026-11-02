# MiniMind 学习项目

当前按 MiniMind 视频学习并补齐自己的模型实现；源码入口为 `model/model.py`。学习进度按实际代码与输出记录推进。

## 本机运行环境

当前统一使用 **`D:\anaconda\python.exe`**，支持 RTX 5070 Ti。已验证 Python 3.13.9、PyTorch 2.12.1+cu132、CUDA 13.2、Transformers 4.57.6、NumPy 2.2.0、Pandas 3.0.3；具体版本与检查范围见 [environment-local.json](environment-local.json)。

GPU 上的 RMSNorm、Q/K 投影、RoPE 和 KV 重复前后向检查通过；完整 `Attention.forward` 仍待自己实现。这次环境切换不代表训练链路已完成。

在 PowerShell 里运行自己的脚本（先创建相应练习文件，再替换下列文件名）：

```powershell
Set-Location 'D:\sdxx\minimind'
& 'D:\anaconda\python.exe' '.\lesson_attention.py'
```

查看当前解释器与 GPU：

```powershell
& 'D:\anaconda\python.exe' -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 编辑器运行与调试

- 项目默认解释器指向 `D:\anaconda\python.exe`。
- 新开项目终端使用本地 Anaconda 路径；修改配置前已经打开的终端请关闭后重开。
- 打开要运行的 `.py` 文件，`Ctrl+Shift+B` 使用“运行当前 Python（Anaconda GPU）”任务。
- `F5` 选择“Python: 当前文件（Anaconda GPU）”调试，使用明确的本地解释器。
- 若编辑器之前缓存了 `.venv`，执行 `Python: Select Interpreter`，选择或输入 `D:\anaconda\python.exe`；仅修改默认路径不保证覆盖旧选择。

## 依赖管理

当前直接使用已有 Anaconda 环境。需要补包时指定解释器，例如 `& 'D:\anaconda\python.exe' -m pip install 包名`，先检查安装计划，保留现有 CUDA PyTorch。

`pyproject.toml` 声明项目所需依赖；本机实际已验证版本记录在 `environment-local.json`。已有 `.venv` 和 `uv.lock` 属于独立的 uv 环境，普通 `uv run` / `uv sync` 会使用那套环境。当前学习请使用上面的显式 Python 命令或编辑器任务；不要把 uv 环境版本与本机 Anaconda 版本混为一份记录。

## 当前结构

- `model/model.py`：Config、RMSNorm、基础 RoPE、KV 重复、Attention 初始化。
- `dataset/lm_dataset.py`：待实现。
- `trainer/trainer_pretrain.py`：待实现。

下一步补 `Attention.forward`，验证形状、参考结果、因果性与反向梯度，再接 FFN、Block、完整 LM 和 loss。
