# HRM Experiments - Sapient Integration

## Overview
This project integrates **Sapient's HRM (Hierarchical Reasoning Model)** - a brain-inspired neural network architecture that excels at complex reasoning tasks with minimal training data.

HRM is a revolutionary approach to AI reasoning:
- **27 million parameters** (vs billions for LLMs)
- **Trains on ~1,000 examples** (vs millions for LLMs)
- **100x faster inference** than chain-of-thought reasoning
- **No pre-training required**
- **Outperforms large LLMs** on reasoning benchmarks

## Purpose
Set up HRM architecture for integration with other applications that require advanced reasoning capabilities (puzzle solving, planning, pathfinding, abstract reasoning, etc.)

## Recent Changes
- **2025-11-08**: Initial setup in Replit environment
  - Cloned Sapient HRM repository from GitHub
  - Organized project structure (models, dataset, utils)
  - Created integration wrapper (`hrm_wrapper.py`) for easy app integration
  - Added usage examples (`integration_example.py`)
  - Set up Python 3.11 environment
  - Added comprehensive documentation

## Project Architecture

### Core Components
- **models/**: HRM neural network architecture
  - `hrm/`: Core hierarchical reasoning implementation
  - `layers.py`: Neural network building blocks
  - `losses.py`: Training loss functions
  - `sparse_embedding.py`: Efficient embeddings

- **dataset/**: Dataset builders for training
  - `build_sudoku_dataset.py`: Sudoku puzzle generator
  - `build_maze_dataset.py`: Maze generation
  - `build_arc_dataset.py`: ARC-AGI benchmark

- **utils/**: Helper utilities
- **config/**: Configuration files

### Integration Files
- **hrm_wrapper.py**: Simplified API for integrating HRM with apps
  - `HRMInference`: Load and run HRM models
  - `HRMAdapter`: Convert between app data and HRM format
  - Pre-trained checkpoint downloader

- **integration_example.py**: Complete usage examples
  - Sudoku solver example
  - Custom task training
  - API integration
  - Batch processing

### Training & Evaluation
- **pretrain.py**: Training script for custom tasks
- **evaluate.py**: Model evaluation
- **puzzle_dataset.py**: Dataset loader
- **requirements.txt**: Python dependencies

## Technology Stack
- **Python 3.11**
- **PyTorch** (deep learning framework)
- **CUDA** (GPU acceleration)
- **FlashAttention** (efficient attention mechanism)
- **Weights & Biases** (experiment tracking)
- **HuggingFace Hub** (pre-trained models)

## Dependencies
Key packages:
- torch (PyTorch)
- adam-atan2 (optimizer)
- einops (tensor operations)
- wandb (experiment tracking)
- huggingface_hub (model downloads)
- hydra-core (configuration)

## Available Pre-trained Models
1. **Sudoku Extreme**: Solves expert-level Sudoku (55% accuracy)
2. **Maze 30x30**: Optimal pathfinding in complex mazes (74.5% success)
3. **ARC-AGI-2**: Abstract reasoning benchmark (beats GPT-4, Claude)

## Integration Strategy

### To integrate HRM with another app:
1. **Customize `hrm_wrapper.py`**
   - Implement `prepare_input()` - convert your data to HRM tensors
   - Implement `process_output()` - convert HRM results to your format

2. **Load a model**
   - Option A: Download pre-trained checkpoint
   - Option B: Train on your custom reasoning task

3. **Use the adapter**
   ```python
   from hrm_wrapper import HRMInference, HRMAdapter
   
   model = HRMInference('checkpoint.pt')
   adapter = HRMAdapter(model)
   
   result = adapter.batch_predict(your_data)
   ```

## Use Cases
Perfect for applications requiring:
- Complex puzzle solving
- Planning and pathfinding
- Abstract reasoning
- Optimization problems
- Decision support systems
- Game AI
- Educational tools

## Performance Characteristics
- **Model Size**: 27M parameters (~108 MB)
- **Training Time**: 2-24 hours (depending on task)
- **Inference Speed**: 100x faster than LLM CoT
- **Data Requirements**: ~1,000 training examples
- **Edge Deployment**: Yes (small model size)

## Resources
- **Official Repo**: https://github.com/sapientinc/HRM
- **Paper**: https://arxiv.org/abs/2506.21734
- **Company**: https://sapient.inc
- **Discord**: https://discord.gg/sapient

## Next Steps
User should specify:
1. What their other app does
2. What reasoning tasks HRM should handle
3. Whether to use pre-trained models or train custom models
4. Data format and integration requirements

## User Preferences
None specified yet.
