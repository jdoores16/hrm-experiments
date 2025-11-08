#!/bin/bash
# Complete Lambda Cloud Training Script
# Run this on Lambda GPU instance after cloning your project

set -e

echo "=================================="
echo "HRM Training on Lambda Cloud GPU"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
  huggingface_hub

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

# Step 4: Generate training data
echo -e "${BLUE}Step 4: Generating training dataset...${NC}"
python dataset/build_panel_optimization_dataset.py \
  --num-examples 200 \
  --output-dir data/panel_optimization

echo -e "${GREEN}✓ Dataset generated${NC}"
echo ""

# Step 5: Create training config
echo -e "${BLUE}Step 5: Creating training configuration...${NC}"
mkdir -p config

cat > config/panel_optimization.yaml << 'EOF'
defaults:
  - _self_

arch:
  name: models.hrm.hrm_act_v1:HierarchicalReasoningModel_ACTV1
  hidden_size: 256
  num_heads: 8
  expansion: 2.0
  H_layers: 2
  L_layers: 4
  H_cycles: 8
  L_cycles: 8
  halt_max_steps: 8
  halt_exploration_prob: 0.1
  pos_encodings: learned
  puzzle_emb_ndim: 256
  loss:
    name: models.losses:SoftmaxCrossEntropyLoss
    loss_type: softmax_cross_entropy

data_path: data/panel_optimization
epochs: 10000
eval_interval: 1000
global_batch_size: 512
lr: 1e-4
lr_min_ratio: 0.1
lr_warmup_steps: 100
weight_decay: 1.0
beta1: 0.9
beta2: 0.999
puzzle_emb_lr: 1e-4
puzzle_emb_weight_decay: 1.0
checkpoint_every_eval: true
seed: 42
EOF

echo -e "${GREEN}✓ Configuration created${NC}"
echo ""

# Step 6: Setup W&B
echo -e "${BLUE}Step 6: Setting up Weights & Biases...${NC}"
echo "Please login to Weights & Biases:"
wandb login
echo -e "${GREEN}✓ W&B configured${NC}"
echo ""

# Step 7: Start training
echo -e "${BLUE}Step 7: Starting HRM training...${NC}"
echo ""
echo "Training configuration:"
echo "  - Task: Panel Optimization"
echo "  - Model: 27M parameters (HRM)"
echo "  - Epochs: 10,000"
echo "  - Batch size: 512"
echo "  - GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""
echo "Expected time: 1-2 hours"
echo "Expected cost: ~\$2 on A100"
echo ""
echo "Training starting in 5 seconds..."
sleep 5

python pretrain.py \
  --config-path config \
  --config-name panel_optimization \
  project_name=hrm_elect_engin \
  run_name=panel_optimization_v1

echo ""
echo -e "${GREEN}✓ Training complete!${NC}"
echo ""

# Step 8: Evaluate
echo -e "${BLUE}Step 8: Evaluating final model...${NC}"
python evaluate.py \
  checkpoint=checkpoints/checkpoint_epoch_10000.pt \
  data_path=data/panel_optimization

echo ""
echo -e "${GREEN}✓ Evaluation complete!${NC}"
echo ""

# Step 9: Package for deployment
echo -e "${BLUE}Step 9: Creating deployment package...${NC}"
mkdir -p deployment
cp checkpoints/checkpoint_epoch_10000.pt deployment/panel_optimization.pt

cat > deployment/model_info.json << EOF
{
  "model": "HRM Panel Optimization",
  "task": "panel_schedule_optimization",
  "training_date": "$(date -u +%Y-%m-%d)",
  "training_time_hours": 2,
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
echo "Download trained model:"
echo "  scp ubuntu@<LAMBDA_IP>:~/hrm-experiments/hrm_panel_optimization_trained.tar.gz ."
echo ""
echo "Model location: deployment/panel_optimization.pt"
echo "Package: hrm_panel_optimization_trained.tar.gz (~110MB)"
echo ""
echo "⚠️  IMPORTANT: Terminate Lambda instance to stop charges!"
echo ""
