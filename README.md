# HRM Experiments - Integration Project

This project integrates **Sapient's HRM (Hierarchical Reasoning Model)** architecture for advanced reasoning capabilities.

## What is HRM?

HRM is a brain-inspired neural network architecture that excels at complex reasoning tasks:

- **Tiny but powerful**: 27 million parameters (vs billions for LLMs)
- **Data efficient**: Trains on ~1,000 examples (vs millions for LLMs)  
- **Fast inference**: 100x faster than chain-of-thought reasoning
- **No pre-training**: Works without massive datasets
- **State-of-the-art performance**: Outperforms Claude, GPT-4, and other large models on reasoning tasks

### Performance Highlights

| Task | HRM (27M params) | Large LLMs |
|------|------------------|------------|
| Sudoku Extreme | 55% accuracy | 0% |
| Maze 30x30 | 74.5% success | 0% |
| ARC-AGI-2 | 5% accuracy | Lower |

## Project Structure

```
.
├── models/              # HRM model architecture
│   ├── hrm/            # Core HRM implementation
│   ├── layers.py       # Neural network layers
│   └── losses.py       # Loss functions
├── dataset/            # Dataset builders (Sudoku, Maze, ARC)
├── utils/              # Utility functions
├── config/             # Configuration files
├── hrm_wrapper.py      # Integration wrapper for your app
├── integration_example.py  # Usage examples
├── pretrain.py         # Training script
├── evaluate.py         # Evaluation script
└── requirements.txt    # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: HRM requires PyTorch with CUDA and FlashAttention. For full setup instructions, see the "Prerequisites" section below.

### 2. Use Pre-trained Checkpoints

```python
from hrm_wrapper import HRMInference, download_pretrained_checkpoint

# Download pre-trained model
checkpoint = download_pretrained_checkpoint('sudoku-extreme')

# Load model
model = HRMInference(checkpoint)

# Run inference
result = model.predict(input_data)
```

### 3. Integrate with Your App

```python
from hrm_wrapper import HRMAdapter

# Create adapter
adapter = HRMAdapter(model)

# Convert your app's data to HRM format
hrm_input = adapter.prepare_input(your_data)

# Get predictions
output = model.predict(hrm_input)

# Convert back to your app's format
result = adapter.process_output(output)
```

## Available Pre-trained Models

Download from HuggingFace:

- **Sudoku Extreme**: `sapientinc/HRM-checkpoint-sudoku-extreme`
- **Maze 30x30**: `sapientinc/HRM-checkpoint-maze-30x30-hard`
- **ARC-AGI-2**: `sapientinc/HRM-checkpoint-ARC-2`

## Training Your Own Model

### Build a Dataset

```bash
# Sudoku
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-1k \
  --subsample-size 1000 \
  --num-aug 1000

# Maze
python dataset/build_maze_dataset.py

# ARC
python dataset/build_arc_dataset.py
```

### Train

```bash
# Single GPU training (laptop-friendly)
python pretrain.py \
  data_path=data/sudoku-1k \
  epochs=20000 \
  eval_interval=2000 \
  global_batch_size=384 \
  lr=7e-5 \
  weight_decay=1.0
```

Runtime: ~10 hours on RTX 4070 laptop GPU

## Integration Guide

### For Your Application

1. **Customize `hrm_wrapper.py`**:
   - Implement `prepare_input()` to convert your data format to HRM tensors
   - Implement `process_output()` to convert HRM results back to your format

2. **Load a model**:
   - Use pre-trained checkpoint, OR
   - Train on your custom reasoning task

3. **Integrate**:
   - Use `HRMAdapter` for data conversion
   - Call `model.predict()` for inference
   - Process results in your application

See `integration_example.py` for detailed examples.

## Prerequisites

### CUDA & PyTorch

```bash
# Install CUDA 12.6
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
sudo sh cuda_*.run --silent --toolkit --override
export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### FlashAttention

For Ampere or earlier GPUs:
```bash
pip install flash-attn
```

For Hopper GPUs:
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

## Use Cases

### Ideal for:
- Complex puzzle solving (Sudoku, mazes, logic puzzles)
- Abstract reasoning tasks
- Planning and pathfinding
- Tasks requiring deep computational reasoning
- Edge deployment (small model size)
- Low-latency applications

### Integration Scenarios:
- Add reasoning to chatbots/agents
- Solve optimization problems
- Game AI (puzzles, strategy)
- Decision support systems
- Educational applications

## Resources

- **Official Repository**: [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)
- **Research Paper**: [arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)
- **Company**: [sapient.inc](https://sapient.inc)
- **Discord Community**: [discord.gg/sapient](https://discord.gg/sapient)

## License

Apache 2.0 (from Sapient Intelligence)

## Citation

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
  title={Hierarchical Reasoning Model}, 
  author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
  year={2025},
  eprint={2506.21734},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## Next Steps

1. Tell me about your other app - what does it do?
2. What kind of reasoning tasks do you want HRM to handle?
3. Do you have a custom dataset, or want to use pre-trained models?

Let's customize the integration for your specific use case!
