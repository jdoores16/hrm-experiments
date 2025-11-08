# Pre-Training Checklist

Before spending money on Lambda Cloud, verify everything works:

## ✅ Local Testing (Free, 10 minutes)

### 1. Test Dataset Generation
```bash
# Generate small test dataset
python dataset/build_panel_optimization_dataset.py \
  --num-examples 10 \
  --output-dir data/panel_test

# Expected: Creates train.json, val.json, test.json
```

### 2. Test Format Conversion
```bash
# Convert to HRM format
python dataset/panel_to_hrm_format.py

# Expected output:
# Converting train split from data/panel_optimization/train.json...
# ✓ Saved train_inputs.npy and train_labels.npy
# ...
# ✅ Conversion Complete!

# Verify files exist
ls -lh data/panel_optimization_hrm/
```

### 3. Test Data Loading
```python
# test_data_loading.py
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

config = PuzzleDatasetConfig(
    seed=42,
    dataset_path="data/panel_optimization_hrm",
    rank=0,
    num_replicas=1
)

dataset = PuzzleDataset(config, split="train")
print(f"✓ Dataset loaded: {len(dataset)} examples")

# Get one example
example = dataset[0]
print(f"✓ Example shape: {example['input'].shape}")
```

### 4. Test Model Initialization
```python
# test_model_init.py
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import torch

config = {
    "batch_size": 4,
    "seq_len": 86,
    "vocab_size": 101,
    "num_puzzle_identifiers": 1000,
    "hidden_size": 256,
    "num_heads": 8,
    "H_layers": 2,
    "L_layers": 4,
    "H_cycles": 8,
    "L_cycles": 8,
    "halt_max_steps": 8,
    "expansion": 2.0,
    "pos_encodings": "learned",
    "puzzle_emb_ndim": 256,
    "halt_exploration_prob": 0.1,
}

model = HierarchicalReasoningModel_ACTV1(config)
print(f"✓ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 5. Test Model Loader (after training)
```python
# test_model_loader.py
from app.ai.hrm_model_loader import get_model_loader

loader = get_model_loader("hrm_checkpoints")
print(f"✓ Loader created")
print(f"  Checkpoint dir: {loader.checkpoint_dir}")
print(f"  Device: {loader.device}")

# Will show warning if no checkpoints yet - that's OK
models = loader.load_all_available()
print(f"✓ Found {len(models)} models")
```

## ✅ GitHub Push (Required)

```bash
# Add new files
git add dataset/panel_to_hrm_format.py
git add app/ai/hrm_model_loader.py
git add train_on_lambda_fixed.sh
git add FIXES_APPLIED.md

# Commit fixes
git commit -m "Fix dataset format and model loading for Lambda training"

# Push to GitHub
git push origin main
```

## ✅ Lambda Preparation

### Before Launching Instance:

- [ ] GitHub repo is updated with fixes
- [ ] Weights & Biases account created (free: https://wandb.ai)
- [ ] Lambda Cloud account has credits
- [ ] SSH key added to Lambda Cloud

### Launch Configuration:
- Instance: **1x A100 (80GB)** - $1.10/hour
- Region: Closest to you
- Filesystem: **PyTorch 2.4.1** (Ubuntu 22.04)

## ✅ On Lambda Instance

### Initial Setup (10 minutes):
```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/hrm-experiments.git
cd hrm-experiments

# 2. Verify Python/CUDA
python --version  # Should be 3.10+
python -c "import torch; print(torch.cuda.is_available())"  # Should be True

# 3. Quick dependency test
pip list | grep torch  # Should show PyTorch installed
```

### Training Run (~2 hours):
```bash
# Use fixed training script
./train_on_lambda_fixed.sh

# This will:
# - Install dependencies
# - Generate dataset
# - Convert to HRM format
# - Setup W&B
# - Train model
# - Evaluate results
# - Package for download
```

### Monitoring:
```bash
# In separate terminal:
ssh ubuntu@<LAMBDA_IP>

# Monitor GPU
watch -n 1 nvidia-smi

# Check training progress
tmux attach -t hrm_training
```

## ✅ After Training

### Verify Success:
- [ ] Training completed (10,000 epochs)
- [ ] Validation accuracy > 90%
- [ ] Checkpoint file exists and is ~110MB
- [ ] Evaluation shows good phase balance

### Download:
```bash
# On local machine
scp ubuntu@<LAMBDA_IP>:~/hrm-experiments/hrm_panel_optimization_trained.tar.gz .
```

### Terminate Instance:
- [ ] **CRITICAL**: Terminate Lambda instance immediately
- [ ] Verify instance is stopped (no more charges)

## ✅ Deploy to Replit

```bash
# Extract checkpoint
tar -xzf hrm_panel_optimization_trained.tar.gz

# Move to Replit
mkdir -p hrm_checkpoints
mv deployment/panel_optimization.pt hrm_checkpoints/

# Test loading
python -c "from app.ai.hrm_orchestrator import get_orchestrator; o = get_orchestrator()"
```

## 🎯 Success Criteria

✅ All local tests pass  
✅ Code pushed to GitHub  
✅ Dataset converts correctly  
✅ Training completes on Lambda  
✅ Model achieves >90% accuracy  
✅ Checkpoint downloads successfully  
✅ Model loads in Replit  
✅ Instance terminated  

**Total Time**: ~3 hours  
**Total Cost**: ~$2-3  
**Result**: Working HRM AI design engineer ✅
