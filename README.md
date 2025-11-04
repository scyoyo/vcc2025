# VCC2025 - STATE Training Scripts

Virtual Cell Challenge 2025 训练脚本，基于 STATE 框架的高性能优化版本。

## 📋 项目简介

本项目包含优化的 STATE 训练脚本，专门针对服务器环境进行了优化，支持多 GPU 训练和 RTX 5090 等高端 GPU。

### 主要特性

- ✅ **High MFU 优化**: GPU 利用率从 1.5% 提升到 10%+
- ✅ **多 GPU 支持**: 自动检测并使用多个 GPU，支持分布式训练
- ✅ **RTX 5090 优化**: 针对 32GB 显存的高端 GPU 特别优化
- ✅ **验证频率优化**: 每 2000 步进行验证，更及时发现问题
- ✅ **WandB 在线监控**: 实时查看训练曲线和指标
- ✅ **智能配置**: 根据 GPU 类型自动调整参数

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch with CUDA support
- STATE 框架
- WandB 账号（可选，用于训练监控）

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/scyoyo/vcc2025.git
cd vcc2025
```

2. 安装依赖（根据 STATE 框架要求）：
```bash
# 安装 STATE 框架
git clone https://github.com/ArcInstitute/state.git
cd state
pip install -e .
```

## ⚙️ 环境配置

### 路径配置

在运行脚本前，可以设置环境变量自定义路径：

```bash
# 数据目录（可选，默认使用当前目录）
export VCC_DATA_DIR=/path/to/your/data/directory

# STATE 仓库路径（可选，默认使用当前目录/state）
export STATE_REPO_DIR=/path/to/state/repo
```

如果不设置，将使用当前工作目录。

### GPU 数量配置

**方法 1: 环境变量（推荐）**

```bash
# 使用 4 个 GPU
export NUM_GPUS=4
python state_highmfu_v2.py

# 使用 1 个 GPU（单 GPU 训练）
export NUM_GPUS=1
python state_highmfu_v2.py

# 使用所有 GPU（默认，不需要设置）
python state_highmfu_v2.py
```

**方法 2: 在代码中设置**

编辑 `state_highmfu_v2.py`，找到约第 141 行：
```python
# use_num_gpus = 4  # Example: use 4 GPUs instead of all
```

取消注释并修改：
```python
use_num_gpus = 4  # 使用 4 个 GPU
```

**方法 3: 使用 CUDA_VISIBLE_DEVICES（指定特定 GPU）**

```bash
# 只使用 GPU 0, 1, 2
export CUDA_VISIBLE_DEVICES=0,1,2
python state_highmfu_v2.py

# 只使用 GPU 0 和 GPU 1
export CUDA_VISIBLE_DEVICES=0,1
python state_highmfu_v2.py
```

### 推荐配置

对于拥有多个 RTX 5090（32GB）的服务器：

- **测试/调试**: `export NUM_GPUS=1` 或 `export NUM_GPUS=2`
- **正常训练**: `export NUM_GPUS=4` 或 `export NUM_GPUS=8`
- **最大性能**: 不设置（使用所有可用 GPU）

## 📊 性能优化说明

### GPU 配置自动检测

脚本会自动检测 GPU 类型和数量，并应用最优配置：

| GPU 类型 | 显存 | 模型 | Workers | 说明 |
|---------|------|------|---------|------|
| RTX 5090 / A100 | 30GB+ | `state_lg` | 12 | 高端 GPU，使用大模型 |
| L4/T4 / RTX 3090 | 20-30GB | `state_sm` | 8 | 中端 GPU |
| 标准 GPU | <20GB | `state_sm` | 4 | 基础配置 |

### 梯度累积策略

根据 GPU 数量自动调整梯度累积：

- **单 GPU**: 8x 梯度累积
- **2-4 个 GPU**: 4x 梯度累积
- **5+ 个 GPU**: 2x 梯度累积

总有效 batch size = GPU 数量 × 梯度累积步数

例如：9 个 GPU × 2x = 18x 有效 batch size

### 性能优化特性

1. **HDF5 缓存**: 256MB（vs 1MB 基线）
2. **Persistent Workers**: 数据加载器持久化
3. **Prefetch Factor**: 4x 预取
4. **Pin Memory**: 加速 CPU-GPU 数据传输
5. **Mixed Precision (FP16)**: 2-3x 训练加速
6. **DDP 策略**: 多 GPU 分布式训练

## 📈 训练监控

### 训练指标

运行后观察以下指标：

- `mfu (%)=8-12%` ← GPU 利用率（单 GPU）
- `mfu (%)=10-15%` ← GPU 利用率（多 GPU）
- Step 2000: `val/pearson=0.XX` ← 第一次验证
- WandB Dashboard: 实时查看训练曲线

### WandB 监控

训练会自动记录到 WandB，查看地址：
- https://wandb.ai/cyshen/vcc

跟踪的指标：
- `val/pearson` - Pearson 相关系数（VCC2025 主指标）
- `val/spearman` - Spearman 相关系数
- `val/mse` - 均方误差
- `train/loss` - 训练损失
- `val/loss` - 验证损失

### 预期性能

- **单 GPU (RTX 5090)**: 2-2.5 小时（40k 步）
- **多 GPU (4-9 × RTX 5090)**: 1-2 小时（40k 步）
- **原始版本**: 8-10 小时（40k 步）

## 📖 使用方法

### 基本使用

```bash
# 1. 设置环境变量（可选）
export VCC_DATA_DIR=/path/to/data
export NUM_GPUS=4

# 2. 运行训练脚本
python state_highmfu_v2.py
```

### 重要提示

⚠️ **必须按顺序运行所有代码块！**

如果直接运行后面的代码，会出现错误：
```python
NameError: name 'OUTPUT_DIR' is not defined
NameError: name 'num_workers' is not defined
```

### 训练流程

1. **环境配置**: 检测 GPU 和配置路径
2. **数据准备**: 下载或使用本地数据
3. **模型配置**: 根据 GPU 自动选择模型大小
4. **训练执行**: 开始训练并记录指标
5. **推理验证**: 使用训练好的模型进行推理
6. **提交准备**: 生成提交文件

## 🔧 常见问题

### 问题：Out of Memory (OOM) 错误

**解决方案：**
```python
# 在代码中修改 num_workers（约第 376 行）
num_workers = 4  # 减少 workers

# 或增加梯度累积
gradient_accumulation_steps = 16
```

### 问题：训练数据未找到

**解决方案：**
- 验证 `competition_support_set` 文件夹是否存在
- 重新运行数据下载部分
- 检查磁盘空间（约需 5GB）
- 确认 `VCC_DATA_DIR` 环境变量设置正确

### 问题：训练速度慢

**解决方案：**
- 确认 GPU 已启用：`nvidia-smi`
- 检查 GPU 使用率：`watch -n 1 nvidia-smi`
- 确认 `precision="16-mixed"` 已设置
- 检查是否使用了多 GPU（设置 `NUM_GPUS`）

### 问题：WandB 登录问题

**解决方案：**
- 运行 `wandb login` 输入 API key
- 或设置离线模式：在代码中取消注释 `os.environ['WANDB_MODE'] = 'offline'`

### 问题：Checkpoint 未找到

**解决方案：**
- 检查 checkpoint 目录：`{OUTPUT_DIR}/{RUN_NAME}/checkpoints/`
- 训练可能仍在进行中，等待下一个 checkpoint
- 使用较早的 checkpoint（如 step=30000 或 step=20000）

## 📁 文件说明

- `state_highmfu_v2.py` - 服务器优化版本的训练脚本（推荐使用）
- `state_optimized_v1.py` - 基础优化版本（已弃用）
- `state_optimized_addval.py` - 带验证的优化版本（已弃用）

## 🎯 版本对比

### state_highmfu_v2.py（推荐）

- ✅ 验证频率：每 2000 步
- ✅ WandB 在线监控
- ✅ 256MB H5 缓存
- ✅ 8x 梯度累积（单 GPU）
- ✅ 多 GPU 支持
- ✅ RTX 5090 优化
- ✅ 服务器环境适配

## 📚 相关资源

- [STATE Paper](https://www.biorxiv.org/content/10.1101/2025.06.26.661135)
- [Virtual Cell Challenge](https://virtualcellchallenge.org/)
- [STATE GitHub](https://github.com/ArcInstitute/state)

## 📝 许可证

本项目遵循原始 STATE 框架的许可证。

---

**祝训练顺利！🚀**
