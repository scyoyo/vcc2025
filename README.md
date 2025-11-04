# VCC2025 - STATE Training Scripts

Virtual Cell Challenge 2025 训练脚本，基于 STATE 框架的高性能优化版本。

## 📋 主要特性

- High MFU 优化：GPU 利用率 1.5% → 10%+
- 多 GPU 支持：自动检测并使用多个 GPU
- RTX 5090 优化：针对 32GB 显存优化
- 验证频率：每 2000 步
- WandB 在线监控

## 🚀 安装

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/scyoyo/vcc2025.git
cd vcc2025
pip install -r requirements.txt
```

### 2. 安装 STATE 框架

```bash
git clone https://github.com/ArcInstitute/state.git
cd state
pip install -e .
cd ..
```

### 3. 配置 WandB（可选）

```bash
# 如果 wandb 命令找不到，先添加到 PATH
export PATH="$HOME/.local/bin:$PATH"

# 然后登录 WandB
wandb login

# 或者使用 Python API 登录（如果命令不可用）
python -c "import wandb; wandb.login()"
```

## ⚙️ 配置

### 环境变量（可选）

```bash
# 数据目录（默认：当前目录）
export VCC_DATA_DIR=/path/to/data

# STATE 仓库路径（默认：当前目录/state）
export STATE_REPO_DIR=/path/to/state

# GPU 数量（默认：使用所有 GPU）
export NUM_GPUS=4

# WandB 配置（可选，默认使用当前登录用户）
export WANDB_ENTITY=your_wandb_username  # 如果不设置，会自动使用当前登录用户
export WANDB_PROJECT=vcc                 # 项目名称（默认：vcc）
```

### GPU 配置

```bash
# 使用 4 个 GPU
export NUM_GPUS=4

# 使用 1 个 GPU
export NUM_GPUS=1

# 使用所有 GPU（默认）
# 不设置 NUM_GPUS 即可
```

## 📖 使用方法

```bash
python state_highmfu_v2.py
```

脚本会自动：
- 检测 GPU 并优化配置
- 下载训练数据（首次运行，约 5GB）
- 开始训练并记录到 WandB

## 📊 训练监控

- WandB Dashboard: 自动使用当前登录用户的项目（https://wandb.ai/your_username/vcc）
- 预期训练时间：1-2 小时（多 GPU）/ 2-2.5 小时（单 GPU）

## 🔧 常见问题

**OOM 错误**: 减少 `num_workers` 或增加 `gradient_accumulation_steps`

**数据未找到**: 检查 `VCC_DATA_DIR` 环境变量，或手动下载数据到 `competition_support_set/`

**STATE 未找到**: 确保已安装 STATE 框架，或设置 `STATE_REPO_DIR` 环境变量

**wandb 命令找不到**: 
```bash
# 添加到 PATH
export PATH="$HOME/.local/bin:$PATH"
# 或者使用 Python API
python -c "import wandb; wandb.login()"
```

**hydra 模块未找到**: 重新安装 STATE 框架：`cd state && pip install -e .`

**transformers 导入错误 (`ImportError: cannot import name 'GPT2Model'`)**:
```bash
# STATE 框架需要 transformers 4.x 版本
pip install "transformers>=4.30.0,<4.42.0"
# 或者安装特定版本
pip install transformers==4.41.2
```

**多GPU DDP错误 (`RuntimeError: It looks like your LightningModule has parameters that were not used`)**:
```bash
# 代码已自动处理，使用 ddp_find_unused_parameters_true 策略
# 如果仍有问题，检查代码中的 strategy 配置
```

## 📚 相关资源

- [STATE Paper](https://www.biorxiv.org/content/10.1101/2025.06.26.661135)
- [Virtual Cell Challenge](https://virtualcellchallenge.org/)
- [STATE GitHub](https://github.com/ArcInstitute/state)
