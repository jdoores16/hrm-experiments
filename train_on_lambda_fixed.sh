#!/bin/bash
# Fixed Lambda Cloud Training Script
# This version uses correct HRM dataset format

set -e

echo "=================================="
echo "HRM Training on Lambda Cloud GPU"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Install dependencies
echo -e "${BLUE}Step 1: Installing dependencies...${NC}"
pip install --quiet --upgrade pip
pip install --quiet \
  adam-atan2 \
  einops \
  tqdm \
  coolname \
  pydantic \
  argdantic \
  wandb \
  omegaconf \
  hydra-core \
  huggingface_hub \
  numpy

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Install FlashAttention
echo -e "${BLUE}Step 2: Installing FlashAttention...${NC}"
pip install --quiet flash-attn --no-build-isolation
echo -e "${GREEN}✓ FlashAttention installed${NC}"
echo ""

# Step 3: Verify GPU
echo -e "${BLUE}Step 3: Verifying GPU...${NC}"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo -e "${GREEN}✓ GPU verified${NC}"
echo ""

# Step 4: Generate training data (JSON format)
echo -e "${BLUE}Step 4: Generating training dataset...${NC}"
python dataset/build_panel_optimization_dataset.py \
  --num-examples 200 \
  --output-dir data/panel_optimization

echo -e "${GREEN}✓ JSON dataset generated${NC}"
echo ""

# Step 5: Convert to HRM binary format
echo -e "${BLUE}Step 5: Converting to HRM binary format...${NC}"
python dataset/panel_to_hrm_format.py

echo -e "${GREEN}✓ HRM dataset ready${NC}"
echo ""

# Step 6: Verify dataset
echo -e "${BLUE}Step 6: Verifying dataset structure...${NC}"
ls -lh data/panel_optimization_hrm/
python -c "
import json
with open('data/panel_optimization_hrm/dataset.json') as f:
    meta = json.load(f)
    print(f\"Dataset: {meta['name']}\")
    print(f\"Train: {meta['splits']['train']} examples\")
    print(f\"Val: {meta['splits']['val']} examples\")
    print(f\"Test: {meta['splits']['test']} examples\")
"
echo -e "${GREEN}✓ Dataset verified${NC}"
echo ""

# Step 7: Setup W&B
echo -e "${BLUE}Step 7: Setting up Weights & Biases...${NC}"
echo "Please login to Weights & Biases:"
wandb login
echo -e "${GREEN}✓ W&B configured${NC}"
echo ""

# Step 8: Start training
echo -e "${BLUE}Step 8: Starting HRM training...${NC}"
echo ""
echo "Training configuration:"
echo "  - Task: Panel Optimization"
echo "  - Model: 27M parameters (HRM)"
echo "  - Epochs: 10,000"
echo "  - Batch size: 512"
echo "  - GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "  - Dataset: data/panel_optimization_hrm"
echo ""
echo "Expected time: 1-2 hours"
echo "Expected cost: ~\$2 on A100"
echo ""
echo "Training starting in 5 seconds..."
sleep 5

# Use correct dataset path
python pretrain.py \
  data_path=data/panel_optimization_hrm \
  epochs=10000 \
  eval_interval=1000 \
  global_batch_size=512 \
  lr=1e-4 \
  lr_min_ratio=0.1 \
  lr_warmup_steps=100 \
  weight_decay=1.0 \
  puzzle_emb_lr=1e-4 \
  puzzle_emb_weight_decay=1.0 \
  project_name=hrm_elect_engin \
  run_name=panel_optimization_v1

echo ""
echo -e "${GREEN}✓ Training complete!${NC}"
echo ""

# Step 9: Find best checkpoint
echo -e "${BLUE}Step 9: Finding best checkpoint...${NC}"

# Hydra creates outputs in outputs/ directory
CHECKPOINT_DIR="outputs/*/checkpoints"
if [ -d "outputs" ]; then
    LATEST_RUN=$(ls -td outputs/*/ | head -1)
    if [ -d "${LATEST_RUN}checkpoints" ]; then
        BEST_CHECKPOINT=$(ls -t ${LATEST_RUN}checkpoints/*.pt | head -1)
        echo "Best checkpoint: $BEST_CHECKPOINT"
        
        # Evaluate
        echo -e "${BLUE}Step 10: Evaluating model...${NC}"
        python evaluate.py \
          checkpoint=$BEST_CHECKPOINT \
          data_path=data/panel_optimization_hrm
        
        # Package for deployment
        echo -e "${BLUE}Step 11: Creating deployment package...${NC}"
        mkdir -p deployment
        cp $BEST_CHECKPOINT deployment/panel_optimization.pt
        
        cat > deployment/model_info.json << EOF
{
  "model": "HRM Panel Optimization",
  "task": "panel_schedule_optimization",
  "training_date": "$(date -u +%Y-%m-%d)",
  "checkpoint_path": "$BEST_CHECKPOINT",
  "gpu": "$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
}
EOF
        
        tar -czf hrm_panel_optimization_trained.tar.gz deployment/
        
        echo -e "${GREEN}✓ Deployment package created${NC}"
        echo ""
        
        # Final summary
        echo "=================================="
        echo "🎉 Training Complete!"
        echo "=================================="
        echo ""
        echo "Checkpoint: $BEST_CHECKPOINT"
        echo "Package: hrm_panel_optimization_trained.tar.gz"
        echo ""
        echo "Download command:"
        echo "  scp ubuntu@<LAMBDA_IP>:~/hrm-experiments/hrm_panel_optimization_trained.tar.gz ."
        echo ""
    else
        echo -e "${YELLOW}⚠ Checkpoints not found in expected location${NC}"
        echo "Looking for checkpoints..."
        find . -name "*.pt" -type f
    fi
else
    echo -e "${YELLOW}⚠ Hydra outputs directory not found${NC}"
    echo "Checkpoints may be in current directory"
fi

echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Terminate Lambda instance to stop charges!${NC}"
echo ""
