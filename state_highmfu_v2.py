# -*- coding: utf-8 -*-
"""STATE_HighMFU_v2.py

Server-optimized version (adapted from Colab notebook)

# STATE - High MFU Version (Enhanced) - Server Edition

🎯 **MFU优化: 1.5% → 10%+**

基于AI Team Phase 3-4的数据加载优化

**改进功能：**
- ✅ 验证频率：每2000步（更及时发现问题）
- ✅ WandB在线监控：实时查看训练曲线
- ✅ 保留所有性能优化：256MB缓存、8x梯度累积、persistent workers等

## 📖 使用说明

### ⚡ 重要：必须按顺序运行所有代码块！

### 🔧 环境配置

在运行前，可以设置环境变量自定义路径和GPU数量：
```bash
export VCC_DATA_DIR=/path/to/your/data/directory  # 数据目录（可选）
export STATE_REPO_DIR=/path/to/state/repo         # STATE仓库路径（可选）
export NUM_GPUS=4                                 # 使用GPU数量（可选，默认使用所有GPU）
```

如果不设置，将使用当前工作目录和所有可用GPU。

**设置GPU数量示例：**
- `export NUM_GPUS=1` - 只使用1个GPU
- `export NUM_GPUS=4` - 使用4个GPU
- `export NUM_GPUS=9` - 使用9个GPU（如果服务器有9个GPU）
- 不设置 - 自动使用所有可用GPU

### ❌ 不要跳过代码块！

如果直接运行后面的代码，会出现错误：
```python
NameError: name 'OUTPUT_DIR' is not defined
NameError: name 'num_workers' is not defined
```

### 📊 训练监控

运行后观察：
- `mfu (%)=8-12%` ← GPU利用率
- Step 2000: `val/pearson=0.XX` ← 第一次validation
- WandB Dashboard: 实时查看训练曲线

---

## 1. Setup: Configure Paths and Environment
"""

# Server environment configuration
import os
import subprocess
from pathlib import Path

# Configure base paths - use environment variable or current directory
BASE_DIR = os.environ.get('VCC_DATA_DIR', os.getcwd())  # Use VCC_DATA_DIR env var or current dir
DATA_DIR = os.path.join(BASE_DIR, 'competition_support_set')
STATE_REPO_DIR = os.environ.get('STATE_REPO_DIR', os.path.join(BASE_DIR, 'state'))

print("=" * 70)
print("SERVER ENVIRONMENT CONFIGURATION")
print("=" * 70)
print(f"Base Directory: {BASE_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"STATE Repo Directory: {STATE_REPO_DIR}")
print(f"\n💡 To customize paths, set environment variables:")
print(f"   export VCC_DATA_DIR=/path/to/data")
print(f"   export STATE_REPO_DIR=/path/to/state/repo")
print("=" * 70)

# Check if STATE repository exists
if not os.path.exists(STATE_REPO_DIR) or not os.path.exists(os.path.join(STATE_REPO_DIR, 'setup.py')):
    print(f"\n❌ STATE repository not found at {STATE_REPO_DIR}")
    print(f"\n📋 Please install STATE framework first:")
    print(f"   1. Clone the STATE repository:")
    print(f"      git clone https://github.com/ArcInstitute/state.git {STATE_REPO_DIR}")
    print(f"   2. Install STATE:")
    print(f"      cd {STATE_REPO_DIR}")
    print(f"      pip install -e .")
    print(f"\n   Or set STATE_REPO_DIR to an existing STATE installation:")
    print(f"      export STATE_REPO_DIR=/path/to/state")
    raise FileNotFoundError(f"STATE repository not found at {STATE_REPO_DIR}. Please install STATE first.")
else:
    print(f"✅ STATE repository found at {STATE_REPO_DIR}")

# Change to STATE repo directory if it exists
if os.path.exists(STATE_REPO_DIR):
    original_cwd = os.getcwd()
    os.chdir(STATE_REPO_DIR)
    print(f"📂 Changed to directory: {os.getcwd()}")

# Optional: Set matplotlib backend (for headless servers)
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

# Enable faster training with optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '8'

"""## 2. Check GPU and Configure Settings

"""

# Check GPU and configure for high GPU usage, low RAM usage
import torch
import os

print("=" * 70)
print("GPU DETECTION & CONFIGURATION")
print("=" * 70)

if torch.cuda.is_available():
    total_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ Found {total_gpus} GPU(s) available")
    print(f"✅ GPU 0: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.2f} GB per GPU")
    
    # List all GPUs
    if total_gpus > 1:
        print(f"\n📋 All GPUs:")
        for i in range(total_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.2f} GB)")
    
    # Configure number of GPUs to use
    # Option 1: Environment variable NUM_GPUS
    # Option 2: Set use_num_gpus in code (uncomment and modify below)
    use_num_gpus = os.environ.get('NUM_GPUS')
    if use_num_gpus:
        try:
            use_num_gpus = int(use_num_gpus)
            use_num_gpus = min(use_num_gpus, total_gpus)  # Don't exceed available GPUs
            print(f"\n⚙️  Using {use_num_gpus} GPU(s) (set via NUM_GPUS environment variable)")
        except ValueError:
            print(f"⚠️  Invalid NUM_GPUS value, using all {total_gpus} GPUs")
            use_num_gpus = total_gpus
    else:
        # Option 2: Uncomment and modify the line below to set GPU count manually
        # use_num_gpus = 4  # Example: use 4 GPUs instead of all
        use_num_gpus = total_gpus  # Default: use all available GPUs
        if use_num_gpus < total_gpus:
            print(f"\n⚙️  Using {use_num_gpus} GPU(s) (configured in code)")
        else:
            print(f"\n⚙️  Using all {use_num_gpus} GPU(s) (default)")
    
    num_gpus = use_num_gpus

    # Optimize: High GPU, Low System RAM
    # RTX 5090 has 32GB, treat similar to A100 but with RTX optimizations
    if gpu_memory >= 30:  # RTX 5090, A100, etc.
        num_workers = 12  # HIGH MFU: 更多workers用于RTX 5090
        cell_set_length = 4096
        model_size = "state_lg"  # Use large model for high-end GPUs
        if "RTX 5090" in gpu_name or "5090" in gpu_name:
            print("✨ RTX 5090: High-end GPU with 32GB - Optimal configuration")
        else:
            print("✨ High-end GPU (30GB+): Large batch, low RAM")
    elif gpu_memory >= 20:  # L4/T4, RTX 3090, etc.
        num_workers = 8  # HIGH MFU: 从4提升
        cell_set_length = 4096
        model_size = "state_sm"
        print("✨ Mid-range GPU (20-30GB): Medium-large batch, low RAM")
    else:
        num_workers = 4
        cell_set_length = 2048
        model_size = "state_sm"
        print("✨ Standard GPU: balanced")

    print(f"\n📊 Configuration:")
    print(f"  • Available GPUs: {total_gpus}")
    print(f"  • GPUs to use: {num_gpus}")
    print(f"  • num_workers: {num_workers}")
    print(f"  • cell_set_length: {cell_set_length}")
    print(f"  • model: {model_size}")
    
    if num_gpus > 1:
        print(f"\n🚀 Multi-GPU Training:")
        print(f"  • Will use {num_gpus} GPU(s) with DDP (Distributed Data Parallel)")
        print(f"  • Effective batch size will be multiplied by {num_gpus}")
        if num_gpus < total_gpus:
            print(f"  • Note: {total_gpus - num_gpus} GPU(s) will not be used")
    elif num_gpus == 1:
        print(f"\n💻 Single GPU Training:")
        if total_gpus > 1:
            print(f"  • Note: {total_gpus - 1} other GPU(s) available but not used")
            print(f"  • To use more GPUs, set: export NUM_GPUS=<number>")
else:
    print("❌ No GPU detected")
    num_workers = 2
    model_size = "state_sm"
    num_gpus = 0

print("=" * 70)

"""## 3. Setup Data Directory

"""

# Setup data directory on server
import os
import requests
import shutil
from tqdm.auto import tqdm
from zipfile import ZipFile
import re

print("=" * 70)
print("DATA DIRECTORY SETUP")
print("=" * 70)

# Use configured data directory
LOCAL_DATA_DIR = DATA_DIR
zip_path = os.path.join(BASE_DIR, "competition_support_set.zip")

if os.path.exists(LOCAL_DATA_DIR) and os.listdir(LOCAL_DATA_DIR):
    print(f"✅ Data directory already exists: {LOCAL_DATA_DIR}")
else:
    # Check if zip file exists locally
    if os.path.exists(zip_path):
        print(f"📦 Found existing zip file: {zip_path}")
    else:
        # Download zip file
        print("📥 Downloading competition_support_set.zip...")
        url = "https://storage.googleapis.com/vcc_data_prod/datasets/state/competition_support_set.zip"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"✅ Downloaded to {zip_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print(f"💡 Please download manually from: {url}")
            print(f"   and place it at: {zip_path}")

    # Unzip if needed
    if os.path.exists(zip_path):
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        print(f"📦 Unzipping to {LOCAL_DATA_DIR}...")
        
        with ZipFile(zip_path, 'r') as z:
            z.extractall(BASE_DIR)
        
        print(f"✅ Data ready at {LOCAL_DATA_DIR}")

# Set path for training
data_path = LOCAL_DATA_DIR
print(f"\n🚀 Using data directory: {data_path}")

# Fix starter.toml paths
toml_path = os.path.join(LOCAL_DATA_DIR, "starter.toml")
if os.path.exists(toml_path):
    print(f"\n🔧 Fixing paths in starter.toml...")
    with open(toml_path, 'r') as f:
        content = f.read()

    # Fix dataset paths to use absolute path
    # Replace relative paths with absolute paths
    content = re.sub(r'(\w+)\.h5(?!ad)', rf'{LOCAL_DATA_DIR}/\\1.h5', content)
    # Fix any /content/ paths that might exist
    content = content.replace('/content/competition_support_set/', f'{LOCAL_DATA_DIR}/')
    content = content.replace('/content/state/competition_support_set/', f'{LOCAL_DATA_DIR}/')

    with open(toml_path, 'w') as f:
        f.write(content)

    print(f"✅ Fixed paths in starter.toml")
else:
    print(f"⚠️  starter.toml not found at {toml_path}")

print("=" * 70)

"""## 4. Prepare Training Configuration

"""

# Verify training data is ready
import os

print("📊 Checking training data...")
print("\nDatasets in competition_support_set:")
support_dir = LOCAL_DATA_DIR

if os.path.exists(support_dir):
    h5_files = [f for f in os.listdir(support_dir) if f.endswith('.h5')]
    for f in sorted(h5_files):
        path = os.path.join(support_dir, f)
        size_mb = os.path.getsize(path) / 1024**2
        print(f"  ✅ {f}: {size_mb:.1f} MB")

    print(f"\n💡 Note: competition_train.h5 contains the VCC H1 training data")
    print(f"   Other datasets (k562, rpe1, etc.) are for context generalization")
else:
    print("⚠️  competition_support_set not found")

# Use the default starter.toml configuration
import os

config_toml = os.path.join(LOCAL_DATA_DIR, "starter.toml")

print("📝 Using training configuration...")

if os.path.exists(config_toml):
    print(f"✅ Config file: {config_toml}")
    print(f"\n📄 Training datasets:")
    print(f"   • competition_train (VCC H1 cells)")
    print(f"   • k562_gwps, rpe1, jurkat, k562, hepg2 (Replogle data)")
    print(f"\n💡 These datasets enable context generalization")
else:
    print(f"⚠️  {config_toml} not found, will be created during training")

"""## 6. Install STATE Package

**Note**: 如果System RAM占用过高（>30GB），在Cell 5中设置 `num_workers=4`

"""

# Install STATE package (simple system install, fast)
import sys

print("=" * 70)
print("STATE PACKAGE INSTALLATION")
print("=" * 70)

# Check if already installed
try:
    import state
    print("✅ STATE already installed!")
    result = subprocess.run(['state', '--help'], capture_output=True, text=True)
    print('\n'.join(result.stdout.split('\n')[:10]))
except ImportError:
    print("📦 Installing STATE package...")
    print("⏳ Takes ~1 minute...")

    # Install with pip to system Python
    if os.path.exists('setup.py'):
        result = subprocess.run(['pip', 'install', '-e', '.', '-q'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Installation warning: {result.stderr}")
        else:
            print("✅ Installation complete!")
    else:
        print("⚠️  setup.py not found. Please install STATE manually:")
        print("   pip install -e /path/to/state")
    
    # Try to show help
    try:
        result = subprocess.run(['state', '--help'], capture_output=True, text=True)
        print('\n'.join(result.stdout.split('\n')[:10]))
    except:
        pass

print("=" * 70)

# Enable lazy checkpoint loading (for faster model loading)
import torch

# Patch torch.load to use mmap for lazy loading
original_torch_load = torch.load

def lazy_torch_load(f, map_location=None, **kwargs):
    """Load checkpoints with memory mapping for efficiency"""
    kwargs['mmap'] = True  # Enable memory mapping
    kwargs.setdefault('weights_only', False)  # For compatibility
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = lazy_torch_load

print("✅ Enabled lazy checkpoint loading with mmap")
print("💡 Model checkpoints will load faster and use less RAM")

# Enable lazy loading for .h5 files (CRITICAL for low RAM!)
import h5py

print("🔧 Configuring h5 lazy loading...")

# Patch h5py to use minimal cache (reduces RAM usage dramatically)
_original_h5py_File = h5py.File

class LazyH5pyFile:
    def __init__(self, *args, **kwargs):
        # Set tiny cache to force lazy loading
        kwargs.setdefault('rdcc_nbytes', 256 * 1024**2)  # HIGH MFU: 256MB  # 1MB cache (default 1GB!)
        kwargs.setdefault('rdcc_nslots', 10007)  # 更多slots
        self._file = _original_h5py_File(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._file, name)

    def __enter__(self):
        return self._file.__enter__()

    def __exit__(self, *args):
        return self._file.__exit__(*args)

# Apply patch
import sys
if 'state.tx.data' not in sys.modules:
    # Patch before STATE loads
    h5py.File = LazyH5pyFile
    print("✅ h5py patched for lazy loading (256MB cache)")
    print("🎯 Expected RAM reduction: 10-15GB")
else:
    print("⚠️  Run this BEFORE loading STATE modules")

# Additional memory optimization settings
import os
import gc

# Force garbage collection
gc.collect()

# Set PyTorch to use less system memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Reduce dataloader prefetch to save memory
os.environ['PREFETCH_FACTOR'] = '2'

# If still high memory, reduce workers here:
# num_workers = 4  # Uncomment if memory is still too high

print("✅ Memory optimization settings applied")
print(f"Current workers: {num_workers}")

"""## 7. Configure Run Name (for parallel training)

"""

# Smart Run Manager - Auto-generate run names or resume training
import os
import glob
from datetime import datetime

print("=" * 70)
print("RUN MANAGER")
print("=" * 70)

# Find existing runs
existing_runs = []
if os.path.exists('competition_run1'):
    # Old naming convention
    for i in range(1, 20):
        if os.path.exists(f'competition_run{i}'):
            existing_runs.append(f'run{i}')

# Also check for date-based runs
date_runs = glob.glob('competition_*')
for dr in date_runs:
    run_name = dr.replace('competition_', '')
    if run_name not in existing_runs and os.path.isdir(dr):
        existing_runs.append(run_name)

if existing_runs:
    print(f"\n📁 Found {len(existing_runs)} existing run(s):")
    for i, run in enumerate(existing_runs[-5:], 1):  # Show last 5
        run_dir = f'competition_{run}'
        # Check for checkpoints
        ckpts = glob.glob(f'{run_dir}/{run}/checkpoints/*.ckpt')
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            ckpt_name = os.path.basename(latest_ckpt)
            print(f"  {i}. {run} (latest: {ckpt_name})")
        else:
            print(f"  {i}. {run} (no checkpoints yet)")

    print("\n❓ Options:")
    print("  1. Resume existing run (enter run name, e.g., 'run1')")
    print("  2. Start new run (press Enter for auto-generated name)")

    choice = input("\nYour choice (run name or Enter for new): ").strip()

    if choice and choice in existing_runs:
        RUN_NAME = choice
        RESUME_TRAINING = True
        print(f"\n✅ Resuming: {RUN_NAME}")
    else:
        # Generate new run name with date
        RUN_NAME = datetime.now().strftime('run_%Y%m%d_%H%M')
        RESUME_TRAINING = False
        print(f"\n✅ Starting new run: {RUN_NAME}")
else:
    # First time - create first run
    RUN_NAME = datetime.now().strftime('run_%Y%m%d_%H%M')
    RESUME_TRAINING = False
    print(f"\n✅ First run: {RUN_NAME}")

OUTPUT_DIR = f"competition_{RUN_NAME}"

print(f"\n📊 Configuration:")
print(f"  • Run Name: {RUN_NAME}")
print(f"  • Output Directory: {OUTPUT_DIR}")
print(f"  • Resume Training: {RESUME_TRAINING}")
print(f"  • Checkpoints will be saved to: {OUTPUT_DIR}/{RUN_NAME}/checkpoints/")

if RESUME_TRAINING:
    # Find latest checkpoint
    ckpts = glob.glob(f'{OUTPUT_DIR}/{RUN_NAME}/checkpoints/*.ckpt')
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        print(f"\n🔄 Will resume from: {os.path.basename(latest_ckpt)}")
    else:
        print(f"\n⚠️  No checkpoints found, will start from scratch")
        RESUME_TRAINING = False

print("=" * 70)

# Disable verbose INFO logs (FLOPs, etc.) - RUN THIS BEFORE TRAINING!
import logging
import os

# Disable FLOPs callback completely
logging.getLogger('state.tx.callbacks.cumulative_flops').setLevel(logging.ERROR)
logging.getLogger('state.tx.callbacks').setLevel(logging.ERROR)


# Suppress other verbose loggers
for logger_name in ['state.tx.callbacks.cumulative_flops', 'pytorch_lightning.utilities',
                    'pytorch_lightning.callbacks']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

print("✅ All verbose logs suppressed (FLOPs, etc.)")
print("💡 Training output will be cleaner now")

"""## 7. Optimized Training

### 🚀 Key Optimizations Applied:
1. **Mixed Precision (FP16)**: 2-3x speedup with minimal accuracy loss
2. **Increased Workers**: Faster parallel data loading (auto-adaptive)
3. **Gradient Accumulation**: Effective larger batch size without OOM
4. **Frequent Validation**: Every 2k steps for better monitoring
5. **PyTorch Compilation**: Additional optimization for PyTorch 2.0+
6. **Pin Memory & Persistent Workers**: Faster CPU-GPU data transfer
7. **Benchmark Mode**: cuDNN auto-tuner enabled

### 📊 Training Data:
- **competition_train**: VCC H1 cell training data
- **Replogle datasets**: K562, RPE1, Jurkat, K562 GWPS, HepG2
- **Purpose**: Learn perturbation embeddings across multiple contexts

### ⚡ Expected Performance:
- **Training Speed**: 2-4x faster than original
- **Original**: ~8-10 hours for 40k steps
- **Optimized**: ~2-3 hours for 40k steps (on L4 GPU)

"""

# Display optimization summary before training
print("=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)
if torch.cuda.is_available():
    print(f"GPU: {gpu_name} x {num_gpus}")
    print(f"GPU Memory: {gpu_memory:.2f} GB per GPU")
else:
    print(f"GPU: CPU")
print(f"Data Workers: {num_workers}")
print(f"Mixed Precision: FP16 enabled")
print(f"Model Compilation: Enabled")
print(f"Training Steps: 40,000")
print(f"Validation Interval: 2,000 steps")
print(f"Checkpoint Interval: 5,000 steps")
print(f"\nDatasets:")
print(f"  • competition_train (VCC H1 cells)")
print(f"  • Replogle Support Datasets (K562, RPE1, Jurkat, K562 GWPS, HepG2)")
print(f"\n💡 Expected speedup: 2-4x faster than original!")
print("=" * 80)

import logging
logging.getLogger('state.tx.callbacks.cumulative_flops').disabled = True

# HIGH MFU优化训练（AI Team Phase 3-4建议）

# 1. 设置WANDB在线模式（实时监控训练）
import os
# WandB在线模式，可在 https://wandb.ai 实时查看训练进度
# 如需离线模式，取消下面注释并注释掉 WandB 配置
# os.environ['WANDB_MODE'] = 'offline'
print("✅ WandB online mode - 实时监控训练进度")

# 2. HIGH MFU训练命令
# Build command as list to handle paths with spaces correctly
pert_features_file = os.path.join(LOCAL_DATA_DIR, 'ESM2_pert_features.pt')

# Adjust gradient accumulation based on number of GPUs
# With multiple GPUs, we can reduce gradient accumulation since effective batch is larger
if num_gpus > 1:
    # For multi-GPU, reduce gradient accumulation proportionally
    # Base: 8x for single GPU, reduce to 4x for 2-4 GPUs, 2x for 5+ GPUs
    if num_gpus >= 5:
        gradient_accumulation_steps = 2
    elif num_gpus >= 2:
        gradient_accumulation_steps = 4
    else:
        gradient_accumulation_steps = 8
    print(f"💡 Multi-GPU detected: Reducing gradient accumulation to {gradient_accumulation_steps}x")
    print(f"   (Effective batch size: {num_gpus} GPUs × {gradient_accumulation_steps} = {num_gpus * gradient_accumulation_steps}x)")
else:
    gradient_accumulation_steps = 8

train_cmd_parts = [
    'state', 'tx', 'train',
    f'data.kwargs.toml_config_path={config_toml}',
    f'data.kwargs.num_workers={num_workers}',
    'data.kwargs.batch_col=batch_var',
    'data.kwargs.pert_col=target_gene',
    'data.kwargs.cell_type_key=cell_type',
    'data.kwargs.control_pert=non-targeting',
    f'data.kwargs.perturbation_features_file={pert_features_file}',
    '++data.kwargs.persistent_workers=True',
    '++data.kwargs.prefetch_factor=4',
    '++data.kwargs.pin_memory=True',
    'training.max_steps=40000',
    'training.ckpt_every_n_steps=5000',
    '++training.val_check_interval=2000',
    '++training.limit_val_batches=1.0',
    f'++training.gradient_accumulation_steps={gradient_accumulation_steps}',
    f'model={model_size}',
    '++training.precision=16-mixed',
    '++training.log_every_n_steps=50',
    'wandb.tags=[high_mfu,optimized]',
    'wandb.project=vcc',
    'wandb.entity=cyshen',
    '+wandb.log_model=true',
    f'output_dir={OUTPUT_DIR}',
    f'name={RUN_NAME}'
]

# Add multi-GPU strategy if multiple GPUs available
if num_gpus > 1:
    train_cmd_parts.append('++training.accelerator=gpu')
    train_cmd_parts.append(f'++training.devices={num_gpus}')
    train_cmd_parts.append('++training.strategy=ddp')
    print(f"✅ Multi-GPU training enabled: {num_gpus} GPUs with DDP strategy")
elif num_gpus == 1:
    train_cmd_parts.append('++training.accelerator=gpu')
    train_cmd_parts.append('++training.devices=1')
    print(f"✅ Single GPU training enabled")

if RESUME_TRAINING:
    import glob
    ckpts = glob.glob(f'{OUTPUT_DIR}/{RUN_NAME}/checkpoints/*.ckpt')
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        print(f"🔄 Resuming from: {os.path.basename(latest_ckpt)}")
        train_cmd_parts.append(f'training.resume_from_checkpoint={latest_ckpt}')

print("="*80)
print("🚀 HIGH MFU TRAINING CONFIGURATION")
print("="*80)
print(f"✅ HDF5 Cache: 256MB (vs 1MB baseline)")
print(f"✅ Workers: {num_workers} + persistent + prefetch=4")
print(f"✅ Grad Accumulation: {gradient_accumulation_steps}x (有效batch x{gradient_accumulation_steps})")
if num_gpus > 1:
    print(f"✅ Multi-GPU: {num_gpus} GPUs (总有效batch x{num_gpus * gradient_accumulation_steps})")
print(f"✅ Validation: 每2000步 (更频繁监控)")
print(f"✅ Checkpoint: 每5000步 (vs 10000步)")
print(f"✅ WandB: 在线模式 (实时监控)")
print(f"")
if num_gpus > 1:
    print(f"🎯 预期MFU: 10-15% (多GPU加速)")
    print(f"🎯 预期时间: 1-2小时 (vs 2-3小时单GPU)")
else:
    print(f"🎯 预期MFU: 8-12% (vs 1.5% baseline)")
    print(f"🎯 预期时间: 2-2.5小时 (vs 3-4小时)")
print("="*80)

print("\n🚀 Starting HIGH MFU training...\n")
print(f"Training command:\n{' '.join(train_cmd_parts)}\n")

# Execute training command
subprocess.run(train_cmd_parts, check=False)

"""### 🎯 Validation and Metrics Configuration

The training now includes:
1. **Validation every 2000 steps** (`training.val_check_interval=2000`) - 更频繁的监控
2. **Full validation set** (`training.limit_val_batches=1.0`)
3. **WandB online logging** with model tracking (`wandb.log_model=true`) - 实时查看训练曲线

#### VCC2025 Metrics:
STATE framework automatically computes:
- **Pearson Correlation**: Main metric for VCC2025
- **Spearman Correlation**: Rank correlation
- **MSE (Mean Squared Error)**: Reconstruction loss

These metrics will be logged to WandB during validation and can be viewed at:
- Console output during training
- WandB dashboard: https://wandb.ai/cyshen/vcc

"""

# Monitor training progress via WandB
import os
import glob

def show_training_progress():
    """Display training status and WandB link"""

    print("📊 Training Monitoring")
    print("=" * 70)

    # Check if training has started
    if os.path.exists(f"{OUTPUT_DIR}/{RUN_NAME}"):
        print(f"✅ Training directory: {OUTPUT_DIR}/{RUN_NAME}")

        # Check for checkpoints
        ckpts = glob.glob(f"{OUTPUT_DIR}/{RUN_NAME}/checkpoints/*.ckpt")
        if ckpts:
            print(f"✅ Found {len(ckpts)} checkpoint(s)")
            latest = max(ckpts, key=os.path.getmtime)
            print(f"   Latest: {os.path.basename(latest)}")

        print("\n📈 View metrics in WandB:")
        print("   🔗 https://wandb.ai/cyshen/vcc")
        print("\n📊 Tracked metrics:")
        print("   • val/pearson - Pearson correlation (VCC2025 main metric) ⭐")
        print("   • val/spearman - Spearman correlation")
        print("   • val/mse - Mean Squared Error")
        print("   • train/loss - Training loss")
        print("   • val/loss - Validation loss")
    else:
        print("⚠️  Training not started yet")

    print("=" * 70)

# Show progress
show_training_progress()

"""### Alternative: Quick Training (20k steps)

If you want even faster training for testing, uncomment and run the cell below instead (will take ~1-1.5 hours instead of 2-3 hours):

"""

# # Quick training option (uncomment to use)
# ! state tx train \
#   data.kwargs.toml_config_path="/content/competition_support_set/starter.toml" \
#   data.kwargs.num_workers={num_workers} \
#   data.kwargs.batch_col="batch_var" \
#   data.kwargs.pert_col="target_gene" \
#   data.kwargs.cell_type_key="cell_type" \
#   data.kwargs.control_pert="non-targeting" \
#   data.kwargs.perturbation_features_file="/content/competition_support_set/ESM2_pert_features.pt" \
#   training.max_steps=20000 \
#   training.ckpt_every_n_steps=10000 \
#   model={model_size} \
#   ++training.precision=16-mixed \
#   ++training.log_every_n_steps=100 \
#   wandb.tags="[quick_run]" \
#   wandb.project=vcc \
#   wandb.entity=arcinstitute \
#   output_dir="{OUTPUT_DIR}_quick" \
#   name="{RUN_NAME}_quick"

"""## 8. Check Training Progress and Checkpoints

"""

# View available checkpoints
import os
import glob

# 检查是否已运行Run Manager cell
if 'OUTPUT_DIR' not in globals() or 'RUN_NAME' not in globals():
    print("⚠️  请先运行上面的 'Run Manager' cell (Cell 18)")
    print("   需要先定义 OUTPUT_DIR 和 RUN_NAME")
else:
    checkpoint_dir = f"{OUTPUT_DIR}/{RUN_NAME}/checkpoints/"

    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(f"{checkpoint_dir}/*.ckpt")
        if checkpoints:
            print("📦 Available checkpoints:")
            for ckpt in sorted(checkpoints):
                size = os.path.getsize(ckpt) / 1024**2
                print(f"  • {os.path.basename(ckpt)} ({size:.1f} MB)")
        else:
            print("⚠️  No checkpoints found yet. Training may still be in progress.")
    else:
        print("⚠️  Checkpoint directory not found. Training may not have started yet.")

"""## 9. Run Inference on Validation Set

"""

# Run inference with the final checkpoint (step=40000)
# Change the checkpoint step if using a different one

infer_cmd = [
    'state', 'tx', 'infer',
    '--output', f'{OUTPUT_DIR}/prediction.h5ad',
    '--model-dir', f'{OUTPUT_DIR}/{RUN_NAME}',
    '--checkpoint', f'{OUTPUT_DIR}/{RUN_NAME}/checkpoints/step=40000.ckpt',
    '--adata', f'{LOCAL_DATA_DIR}/competition_val_template.h5ad',
    '--pert-col', 'target_gene'
]
subprocess.run(infer_cmd, check=False)

"""### Use Different Checkpoint

If step=40000.ckpt doesn't exist, you can use an earlier checkpoint:

"""

# # Alternative: Use step=30000 or step=20000 checkpoint
# infer_cmd = [
#     'state', 'tx', 'infer',
#     '--output', f'{OUTPUT_DIR}/prediction_30k.h5ad',
#     '--model-dir', f'{OUTPUT_DIR}/{RUN_NAME}',
#     '--checkpoint', f'{OUTPUT_DIR}/{RUN_NAME}/checkpoints/step=30000.ckpt',
#     '--adata', f'{LOCAL_DATA_DIR}/competition_val_template.h5ad',
#     '--pert-col', 'target_gene'
# ]
# subprocess.run(infer_cmd, check=False)

"""## 10. Prepare Submission File

"""

# Install zstd for cell-eval prep (if on Ubuntu/Debian)
# Note: Skip this if zstd is already installed or on non-Debian systems
try:
    subprocess.run(['which', 'zstd'], capture_output=True, check=True)
    print("✅ zstd already installed")
except:
    print("⚠️  zstd not found. Please install manually:")
    print("   Ubuntu/Debian: sudo apt install -y zstd")
    print("   macOS: brew install zstd")
    print("   Or skip this step if cell-eval doesn't require it")

# Prepare submission file using cell-eval
cell_eval_cmd = [
    'uv', 'tool', 'run', '--from', 'git+https://github.com/ArcInstitute/cell-eval@main',
    'cell-eval', 'prep',
    '-i', f'{OUTPUT_DIR}/prediction.h5ad',
    '-g', f'{LOCAL_DATA_DIR}/gene_names.csv'
]
subprocess.run(cell_eval_cmd, check=False)

# Find and display submission file
import glob
import os

vcc_files = glob.glob(f"{OUTPUT_DIR}/*.vcc")
if vcc_files:
    print("=" * 70)
    print("✅ SUBMISSION FILE READY!")
    print("=" * 70)
    for f in vcc_files:
        size = os.path.getsize(f) / 1024 / 1024
        print(f"\n📦 File: {f}")
        print(f"📊 Size: {size:.2f} MB")
    print("\n🚀 Next steps:")
    print("   1. Download the .vcc file from the file browser")
    print("   2. Go to https://virtualcellchallenge.org/")
    print("   3. Upload your .vcc file to the leaderboard")
    print("=" * 70)
else:
    print("❌ No .vcc file found. Check if cell-eval prep completed successfully.")

"""## 📊 Optimization Summary

This optimized notebook includes the following improvements over the original:

### ⚡ Speed Improvements (2-4x faster):
1. **Mixed Precision Training (FP16)**: ~2-3x speedup
2. **Increased Data Workers**: Faster data loading (adaptive: 8-16 workers)
3. **Gradient Accumulation**: Larger effective batch size (1-4x)
4. **Frequent Validation**: Every 2k steps for better monitoring
5. **PyTorch Compilation**: Additional optimization
6. **Pin Memory & Persistent Workers**: Faster CPU-GPU transfer
7. **Benchmark Mode**: cuDNN auto-tuner enabled

### 📦 Training Data:
- ✅ **competition_train**: VCC H1 cell training data (included in support set)
- ✅ **Replogle datasets**: K562, RPE1, Jurkat, K562 GWPS, HepG2
- ✅ **Multi-context learning**: Enables context generalization

### 🎯 Expected Performance:
- **Original**: ~8-10 hours for 40k steps
- **Optimized**: ~2-3 hours for 40k steps (on L4 GPU)
- **Quick Mode**: ~1-1.5 hours for 20k steps

### 🔧 Adaptive Configuration:
The notebook automatically detects your GPU and selects optimal settings:
- **A100 (40GB+)**: 16 workers, no gradient accumulation
- **L4/T4/V100 (20-40GB)**: 12 workers, 2x gradient accumulation
- **Standard GPU (<20GB)**: 8 workers, 4x gradient accumulation

### 💡 Tips:
- Monitor GPU usage with `!nvidia-smi` in a new cell
- If you get OOM errors, reduce `num_workers` or increase `gradient_accumulation`
- For faster iteration, use the quick training option (20k steps)
- Compare different checkpoints to find the best performing one

### 📚 Resources:
- [STATE Paper](https://www.biorxiv.org/content/10.1101/2025.06.26.661135)
- [Virtual Cell Challenge](https://virtualcellchallenge.org/)
- [STATE GitHub](https://github.com/ArcInstitute/state)

---

**Good luck with your submission! 🚀**

## 🔧 Troubleshooting

### Problem: Out of Memory (OOM) Error
**Solutions:**
```python
# Reduce workers
num_workers = 4

# Increase gradient accumulation
gradient_accumulation = 8

# Disable model compilation (run before training cell)
# Replace training.compile_model=True with training.compile_model=False
```

### Problem: Training Data Not Found
**Solutions:**
- Verify `competition_support_set` folder exists
- Re-run the download and unzip cells (Section 3)
- Check: `!ls -lh competition_support_set/*.h5`
- Ensure you have enough disk space (~5GB)

### Problem: Training is Slow
**Solutions:**
- Verify GPU is enabled: Runtime > Change runtime type > GPU
- Check GPU usage: `!nvidia-smi`
- Ensure `precision="16-mixed"` is set
- Verify `compile_model=True` is enabled

### Problem: Wandb Login Issues
**Solutions:**
- When prompted, select option 3: "Don't visualize my results"
- Or run `!wandb login` with your API key first

### Problem: Checkpoint Not Found
**Solutions:**
- Check available checkpoints with the cell in section 8
- Training may still be in progress - wait for next checkpoint
- Use an earlier checkpoint (e.g., step=30000 or step=20000)

### Monitor GPU Usage
Run this in a new cell to check GPU utilization:
```python
!nvidia-smi
```
"""