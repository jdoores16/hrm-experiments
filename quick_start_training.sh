#!/bin/bash
# Quick Start: Train HRM as AI Design Engineer

set -e

echo "=================================="
echo "HRM Training Quick Start"
echo "=================================="
echo ""

# Step 1: Generate Panel Optimization Dataset
echo "Step 1: Generating panel optimization dataset..."
python dataset/build_panel_optimization_dataset.py \
  --num-examples 200 \
  --output-dir data/panel_optimization

echo ""
echo "✓ Dataset created in data/panel_optimization/"
echo ""

# Step 2: Check Python dependencies
echo "Step 2: Checking dependencies..."
pip install -q torch einops tqdm pydantic omegaconf hydra-core wandb adam-atan2

echo "✓ Dependencies installed"
echo ""

# Step 3: Train HRM (single GPU)
echo "Step 3: Starting HRM training..."
echo ""
echo "Training configuration:"
echo "  Task: Panel Optimization"
echo "  Model: 27M parameters"
echo "  Epochs: 10,000"
echo "  Batch size: 256"
echo "  Learning rate: 1e-4"
echo ""
echo "Expected time: ~8 hours on RTX 4070"
echo ""

# Uncomment to start actual training
# python pretrain.py \
#   data_path=data/panel_optimization \
#   epochs=10000 \
#   eval_interval=1000 \
#   global_batch_size=256 \
#   lr=1e-4 \
#   puzzle_emb_lr=1e-4 \
#   weight_decay=1.0 \
#   puzzle_emb_weight_decay=1.0 \
#   arch.loss.loss_type=softmax_cross_entropy \
#   arch.L_cycles=8 \
#   arch.halt_max_steps=8

echo "ℹ️  Training command ready (uncommented in script)"
echo ""
echo "=================================="
echo "Next Steps:"
echo "=================================="
echo ""
echo "1. Review dataset: data/panel_optimization/"
echo "2. Uncomment training command in this script"
echo "3. Run: ./quick_start_training.sh"
echo "4. Monitor: Weights & Biases (wandb)"
echo "5. Evaluate: Use evaluate.py on test set"
echo ""
echo "For multi-GPU training (8 GPUs):"
echo "  OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py [...]"
echo ""
