# Training HRM as an AI Design Engineer

## Overview

HRM learns from **~1,000 examples** per task (vs millions for LLMs). This makes it perfect for specialized electrical engineering tasks where you can create high-quality training data from your domain expertise.

## Training Philosophy

### What Makes HRM Special:
1. **Data Efficient**: Learns from 1,000 examples (not millions)
2. **Task Specific**: Train separate models for each engineering task
3. **No Pre-training**: Starts fresh for your domain
4. **Fast Training**: 2-24 hours depending on task complexity

### Training Strategy for Electrical Engineering:

```
Start with highest-value task → Train → Validate → Deploy → Next task
```

**Recommended Order:**
1. Panel Optimization (highest immediate value)
2. Circuit Routing (proven HRM capability)
3. Load Calculations (complex but valuable)
4. NEC Validation (most complex, highest long-term value)

---

## Phase 1: Panel Schedule Optimization

**Goal**: Train HRM to create optimally balanced panel schedules

### Step 1: Create Training Dataset

#### Data Format:
```python
# Input: Panel specification
{
    "voltage": "480Y/277V",
    "phase": 3,
    "num_circuits": 42,
    "main_bus_amps": 800,
    "circuits": [
        {"id": 1, "load_amps": 20, "poles": 1, "description": "Lighting"},
        {"id": 2, "load_amps": 15, "poles": 1, "description": "Receptacle"},
        # ... all circuits
    ]
}

# Output: Optimal phase assignment
{
    "circuit_1": {"phase": "A"},
    "circuit_2": {"phase": "B"},
    "circuit_3": {"phase": "C"},
    # ...
    "phase_balance": {
        "A_total": 267.5,
        "B_total": 265.0,
        "C_total": 267.5
    },
    "balance_score": 0.99  # Perfect balance = 1.0
}
```

#### Generate 1,000 Examples:

**Option A: From Historical Projects**
```python
# dataset/build_panel_dataset.py

import json
from pathlib import Path
from app.schemas.panel_ir import PanelScheduleIR

def extract_from_historical_projects():
    """Extract panel schedules from past projects"""
    historical_panels = []
    
    # Load your completed projects
    project_dir = Path("historical_projects")
    for panel_file in project_dir.glob("**/*.json"):
        panel_ir = PanelScheduleIR.parse_file(panel_file)
        
        # Extract input (unassigned circuits)
        input_data = {
            "voltage": panel_ir.header.voltage,
            "phase": panel_ir.header.phase,
            "circuits": [
                {
                    "id": c.ckt,
                    "load_amps": c.load_amps,
                    "poles": c.poles,
                    "description": c.description
                }
                for c in panel_ir.circuits
            ]
        }
        
        # Extract output (actual phase assignments)
        output_data = {
            f"circuit_{c.ckt}": {
                "phase": "A" if c.phA else ("B" if c.phB else "C")
            }
            for c in panel_ir.circuits
        }
        
        historical_panels.append({
            "input": input_data,
            "output": output_data
        })
    
    return historical_panels

# Generate dataset
panels = extract_from_historical_projects()
print(f"Extracted {len(panels)} panels from history")
```

**Option B: Generate Synthetic Examples**
```python
# dataset/build_panel_dataset.py

import random
import numpy as np

def generate_optimal_panel(num_circuits=42):
    """Generate a panel with optimal phase balance"""
    
    # Random circuit loads (realistic distribution)
    loads = [
        random.choice([15, 20, 30, 40, 50])  # Common breaker sizes
        for _ in range(num_circuits)
    ]
    
    # Optimal assignment using greedy algorithm
    phases = {"A": [], "B": [], "C": []}
    phase_totals = {"A": 0, "B": 0, "C": 0}
    
    # Sort loads descending for better balance
    circuit_loads = [(i+1, load) for i, load in enumerate(loads)]
    circuit_loads.sort(key=lambda x: x[1], reverse=True)
    
    for ckt_id, load in circuit_loads:
        # Assign to least loaded phase
        min_phase = min(phase_totals, key=phase_totals.get)
        phases[min_phase].append(ckt_id)
        phase_totals[min_phase] += load
    
    # Create input/output pair
    input_data = {
        "voltage": "480Y/277V",
        "phase": 3,
        "num_circuits": num_circuits,
        "circuits": [
            {"id": i+1, "load_amps": loads[i], "poles": 1}
            for i in range(num_circuits)
        ]
    }
    
    output_data = {}
    for phase, ckts in phases.items():
        for ckt in ckts:
            output_data[f"circuit_{ckt}"] = {"phase": phase}
    
    return {"input": input_data, "output": output_data}

# Generate 1,000 examples
dataset = [generate_optimal_panel() for _ in range(1000)]
print(f"Generated {len(dataset)} synthetic panels")

# Add variations
for i in range(200):
    dataset.append(generate_optimal_panel(num_circuits=24))  # Smaller panels
    dataset.append(generate_optimal_panel(num_circuits=84))  # Larger panels

# Save
with open("data/panel_optimization/train.json", "w") as f:
    json.dump(dataset[:900], f)

with open("data/panel_optimization/test.json", "w") as f:
    json.dump(dataset[900:], f)
```

### Step 2: Convert to HRM Format

HRM expects tokenized sequences. Convert panel data to HRM-compatible format:

```python
# dataset/panel_to_hrm.py

def panel_to_sequence(panel_data):
    """
    Convert panel specification to token sequence
    
    Format: [VOLTAGE] [PHASE] [CKT_1_LOAD] [CKT_2_LOAD] ... [SEP] [TARGET]
    """
    tokens = []
    
    # Encode voltage (480Y/277V → token 1, 208Y/120V → token 2, etc.)
    voltage_map = {
        "480Y/277V": 1,
        "208Y/120V": 2,
        "240V/120V": 3
    }
    tokens.append(voltage_map.get(panel_data["voltage"], 1))
    
    # Encode phase
    tokens.append(panel_data["phase"])
    
    # Encode each circuit load (normalize to 0-100 range)
    for circuit in panel_data["circuits"]:
        tokens.append(min(circuit["load_amps"], 100))
    
    # Pad to fixed length
    max_circuits = 84
    while len(tokens) < max_circuits + 2:
        tokens.append(0)  # PAD token
    
    return tokens

def phase_assignment_to_sequence(output_data):
    """Convert phase assignments to target sequence"""
    phase_map = {"A": 1, "B": 2, "C": 3}
    
    targets = []
    for ckt_id in sorted([int(k.split("_")[1]) for k in output_data.keys()]):
        phase = output_data[f"circuit_{ckt_id}"]["phase"]
        targets.append(phase_map[phase])
    
    return targets

# Convert dataset
hrm_train = []
for example in dataset:
    input_seq = panel_to_sequence(example["input"])
    target_seq = phase_assignment_to_sequence(example["output"])
    
    hrm_train.append({
        "input": input_seq,
        "target": target_seq
    })
```

### Step 3: Train HRM Model

```bash
# Training configuration
cat > config/panel_optimization.yaml << EOF
arch:
  name: models.hrm.hrm_act_v1:HierarchicalReasoningModel_ACTV1
  
  # Model size (27M parameters)
  hidden_size: 256
  num_heads: 8
  H_layers: 2    # High-level planning
  L_layers: 4    # Low-level execution
  
  H_cycles: 8    # Reasoning depth
  L_cycles: 8
  
  halt_max_steps: 8
  
  pos_encodings: learned
  
  loss:
    name: models.losses:SoftmaxCrossEntropyLoss

# Training hyperparameters  
data_path: data/panel_optimization
epochs: 10000
eval_interval: 1000

global_batch_size: 256
lr: 1e-4
weight_decay: 1.0

puzzle_emb_lr: 1e-4
puzzle_emb_weight_decay: 1.0
EOF

# Start training (single GPU)
python pretrain.py \
  --config-path config \
  --config-name panel_optimization

# Multi-GPU (8 GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py \
  --config-path config \
  --config-name panel_optimization
```

**Expected Training Time:**
- Single GPU (RTX 4070): ~8 hours
- 8 GPUs (A100): ~1 hour

### Step 4: Validate Performance

```python
# evaluate_panel_model.py

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import torch

# Load trained model
checkpoint = torch.load("checkpoints/panel_optimization_epoch_10000.pt")
model = HierarchicalReasoningModel_ACTV1(checkpoint["config"])
model.load_state_dict(checkpoint["model"])
model.eval()

# Test on new panel
test_panel = {
    "voltage": "480Y/277V",
    "phase": 3,
    "circuits": [/* 42 circuits */]
}

input_seq = panel_to_sequence(test_panel)
input_tensor = torch.tensor([input_seq])

# HRM predicts optimal assignments
with torch.no_grad():
    predictions = model(input_tensor)

# Convert predictions to phase assignments
phase_assignments = predictions_to_phases(predictions)

# Calculate balance quality
balance_score = calculate_phase_balance(test_panel, phase_assignments)
print(f"Phase balance score: {balance_score:.2%}")

# Compare with baseline (current GPT approach)
gpt_assignments = gpt_optimize_panel(test_panel)
gpt_balance = calculate_phase_balance(test_panel, gpt_assignments)

print(f"HRM balance: {balance_score:.2%}")
print(f"GPT balance: {gpt_balance:.2%}")
print(f"Improvement: {(balance_score - gpt_balance):.2%}")
```

---

## Phase 2: Circuit Routing

**Goal**: Train HRM to route circuits optimally through floor plans

### Step 1: Create Floor Plan Dataset

```python
# dataset/build_routing_dataset.py

def generate_floor_plan_routing():
    """Generate optimal circuit routing problem"""
    
    # Create floor plan (30x30 grid, similar to maze)
    floor_plan = np.zeros((30, 30))
    
    # Add walls/obstacles
    num_walls = random.randint(5, 15)
    for _ in range(num_walls):
        x, y = random.randint(0, 29), random.randint(0, 29)
        w, h = random.randint(2, 8), random.randint(2, 8)
        floor_plan[y:y+h, x:x+w] = 1  # Wall = 1
    
    # Panel location (source)
    panel_x, panel_y = random.randint(0, 29), random.randint(0, 29)
    while floor_plan[panel_y, panel_x] == 1:
        panel_x, panel_y = random.randint(0, 29), random.randint(0, 29)
    
    # Device location (destination)
    device_x, device_y = random.randint(0, 29), random.randint(0, 29)
    while floor_plan[device_y, device_x] == 1:
        device_x, device_y = random.randint(0, 29), random.randint(0, 29)
    
    # Find optimal path (A* algorithm)
    path = a_star_pathfinding(
        floor_plan, 
        (panel_x, panel_y), 
        (device_x, device_y)
    )
    
    return {
        "input": {
            "floor_plan": floor_plan.tolist(),
            "start": [panel_x, panel_y],
            "end": [device_x, device_y]
        },
        "output": {
            "path": path,
            "length": len(path)
        }
    }

# Generate 1,000 routing problems
routing_dataset = [generate_floor_plan_routing() for _ in range(1000)]
```

### Step 2: Train Circuit Routing Model

```bash
# Very similar to Sudoku/Maze training HRM already does well
python pretrain.py \
  data_path=data/circuit_routing \
  epochs=20000 \
  eval_interval=2000 \
  lr=1e-4 \
  weight_decay=1.0
```

**Expected Performance**: >70% optimal path success (HRM gets 74.5% on mazes)

---

## Phase 3: Load Calculations

**Goal**: Train HRM to perform complex NEC load calculations

### Training Data Example:

```python
{
    "input": {
        "loads": [
            {"type": "lighting", "va": 12000, "area_sq_ft": 3000},
            {"type": "receptacle", "va": 9600, "area_sq_ft": 3000},
            {"type": "hvac", "hp": 25, "voltage": 480, "phase": 3},
            {"type": "motor", "hp": 10, "voltage": 480, "phase": 3}
        ],
        "voltage": "480Y/277V",
        "phase": 3
    },
    "output": {
        "lighting_demand": 12000,  # No diversity for lighting
        "receptacle_demand": 4800,  # 50% demand factor
        "hvac_demand": 34100,  # From NEC table
        "motor_demand": 14000,  # Largest + 25% of others
        "total_va": 64900,
        "total_amps": 78.1,
        "feeder_size": "3#4 AWG + #8 GND"
    }
}
```

Train with NEC demand factors, conductor sizing tables, etc.

---

## Phase 4: NEC Code Validation

**Goal**: Multi-step code compliance reasoning

### Training Data Structure:

```python
{
    "input": {
        "panel": {/* panel specification */},
        "nec_version": "2020",
        "application": "commercial"
    },
    "output": {
        "violations": [
            {
                "rule": "NEC 210.19(A)(1)",
                "issue": "Circuit #5 conductor undersized",
                "reasoning": [
                    "Load = 25A",
                    "Continuous load requires 125% = 31.25A",
                    "#10 AWG rated 30A at 75°C",
                    "Violation: 30A < 31.25A"
                ],
                "fix": "Upsize to #8 AWG (40A rating)"
            }
        ],
        "compliant": false
    }
}
```

This requires **1,000 examples** of designs with violations and multi-step reasoning chains.

---

## Best Practices

### 1. Data Quality > Quantity
- **1,000 perfect examples** beats 10,000 mediocre ones
- Have PE review training examples
- Include edge cases and tricky scenarios

### 2. Augmentation
```python
# Multiply your data through variations
def augment_panel(panel):
    """Create variations of a panel"""
    variations = []
    
    # Voltage variations
    for voltage in ["480Y/277V", "208Y/120V", "240/120V"]:
        variant = panel.copy()
        variant["voltage"] = voltage
        # Recalculate optimal assignment
        variations.append(variant)
    
    # Load variations (±10%)
    variant = panel.copy()
    for circuit in variant["circuits"]:
        circuit["load_amps"] *= random.uniform(0.9, 1.1)
    variations.append(variant)
    
    return variations

# 200 base examples → 1,000+ with augmentation
```

### 3. Progressive Training
```python
# Start simple, add complexity
datasets = [
    "panels_18ckt.json",   # Simple cases
    "panels_42ckt.json",   # Medium complexity
    "panels_84ckt.json",   # Full complexity
    "panels_mixed.json"    # Mixed difficulty
]

# Train in stages
for dataset in datasets:
    train_hrm(dataset, epochs=5000)
```

### 4. Validation Metrics

```python
def evaluate_hrm_engineer(model, test_set):
    """Comprehensive evaluation"""
    
    metrics = {
        "phase_balance_quality": 0,
        "nec_compliance_rate": 0,
        "routing_efficiency": 0,
        "calculation_accuracy": 0
    }
    
    for example in test_set:
        prediction = model.predict(example["input"])
        
        # Phase balance (0-100%)
        balance = calculate_balance_score(prediction)
        metrics["phase_balance_quality"] += balance
        
        # NEC compliance (pass/fail)
        compliant = check_nec_compliance(prediction)
        metrics["nec_compliance_rate"] += int(compliant)
        
        # Routing (vs optimal path)
        efficiency = len(optimal_path) / len(prediction["path"])
        metrics["routing_efficiency"] += efficiency
    
    # Average across test set
    for key in metrics:
        metrics[key] /= len(test_set)
    
    return metrics
```

---

## Integration with Elect_Engin1

Once trained, integrate HRM models:

```python
# app/ai/hrm_orchestrator.py

def _load_hrm_models(self):
    """Load trained HRM models"""
    self.hrm_models = {
        TaskType.PANEL_OPTIMIZATION: load_checkpoint(
            "hrm_checkpoints/panel_optimization.pt"
        ),
        TaskType.CIRCUIT_ROUTING: load_checkpoint(
            "hrm_checkpoints/circuit_routing.pt"
        ),
        TaskType.LOAD_CALCULATION: load_checkpoint(
            "hrm_checkpoints/load_calculation.pt"
        ),
        TaskType.NEC_VALIDATION: load_checkpoint(
            "hrm_checkpoints/nec_validation.pt"
        )
    }
```

---

## Summary

### To Train HRM as Best AI Design Engineer:

1. **Start with Panel Optimization** (highest value, easiest)
   - Generate 1,000 examples (synthetic or historical)
   - Train for 8-10 hours
   - Validate phase balance quality

2. **Add Circuit Routing** (proven HRM capability)
   - Generate floor plans + optimal paths
   - Train for 10-20 hours
   - Achieve >70% optimal routing

3. **Expand to Load Calculations** (complex but valuable)
   - Encode NEC tables as training data
   - Train multi-step calculation reasoning

4. **Master NEC Validation** (ultimate goal)
   - Create violation examples with reasoning chains
   - Train deep compliance checking

### Key Success Factors:
- **Quality data**: PE-reviewed examples
- **Augmentation**: Multiply your dataset
- **Validation**: Test on real projects
- **Iteration**: Continuously improve training data

HRM will become an expert AI design engineer through focused, high-quality training on electrical engineering tasks!
