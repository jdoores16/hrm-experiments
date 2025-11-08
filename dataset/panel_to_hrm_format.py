#!/usr/bin/env python3
"""
Convert Panel Optimization JSON to HRM Binary Format

HRM's PuzzleDataset expects:
- dataset.json (metadata)
- train_inputs.npy, train_labels.npy (binary tensors)
- val_inputs.npy, val_labels.npy
- test_inputs.npy, test_labels.npy
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def panel_to_tokens(panel_input: Dict) -> List[int]:
    """
    Convert panel specification to token sequence for HRM.
    
    Format: [VOLTAGE_TOKEN] [PHASE] [CIRCUIT_LOADS...] [PAD...]
    """
    # Voltage encoding
    voltage_map = {
        "480Y/277V": 1,
        "208Y/120V": 2,
        "240/120V": 3,
        "240V/120V": 3
    }
    
    tokens = []
    
    # Add voltage token
    tokens.append(voltage_map.get(panel_input["voltage"], 1))
    
    # Add phase count
    tokens.append(panel_input["phase"])
    
    # Add circuit loads (one token per circuit)
    for circuit in sorted(panel_input["circuits"], key=lambda c: c["id"]):
        # Normalize load to 0-100 range
        load = min(int(circuit["load_amps"]), 100)
        tokens.append(load)
    
    # Pad to fixed length (max 84 circuits + 2 header tokens)
    max_length = 86
    while len(tokens) < max_length:
        tokens.append(0)  # PAD token
    
    return tokens[:max_length]


def phase_assignment_to_labels(output_data: Dict, num_circuits: int) -> List[int]:
    """
    Convert phase assignments to label sequence.
    
    Labels: 0=PAD, 1=Phase A, 2=Phase B, 3=Phase C
    """
    phase_map = {"A": 1, "B": 2, "C": 3}
    
    labels = []
    for ckt_id in range(1, num_circuits + 1):
        key = f"circuit_{ckt_id}"
        if key in output_data["assignments"]:
            phase = output_data["assignments"][key]["phase"]
            labels.append(phase_map[phase])
        else:
            labels.append(0)  # PAD
    
    # Pad to max circuits
    while len(labels) < 84:
        labels.append(0)
    
    return labels[:84]


def convert_json_to_hrm_format(
    json_file: Path,
    output_dir: Path,
    split_name: str
):
    """Convert JSON dataset to HRM binary format"""
    
    print(f"Converting {split_name} split from {json_file}...")
    
    # Load JSON data
    with open(json_file) as f:
        data = json.load(f)
    
    print(f"  Found {len(data)} examples")
    
    # Convert to token sequences
    inputs = []
    labels = []
    
    for example in data:
        # Convert input panel to tokens
        input_tokens = panel_to_tokens(example["input"])
        inputs.append(input_tokens)
        
        # Convert output assignments to labels
        num_circuits = example["input"]["num_circuits"]
        label_tokens = phase_assignment_to_labels(example["output"], num_circuits)
        labels.append(label_tokens)
    
    # Convert to NumPy arrays
    inputs_array = np.array(inputs, dtype=np.int32)
    labels_array = np.array(labels, dtype=np.int32)
    
    print(f"  Input shape: {inputs_array.shape}")
    print(f"  Label shape: {labels_array.shape}")
    
    # Save as NumPy binary files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / f"{split_name}_inputs.npy", inputs_array)
    np.save(output_dir / f"{split_name}_labels.npy", labels_array)
    
    print(f"  ✓ Saved {split_name}_inputs.npy and {split_name}_labels.npy")
    
    return len(data)


def create_dataset_metadata(
    output_dir: Path,
    train_size: int,
    val_size: int,
    test_size: int
):
    """Create HRM-compatible dataset.json metadata"""
    
    metadata = {
        "name": "panel_optimization",
        "description": "Panel schedule optimization with optimal phase balance",
        "vocab_size": 101,  # 0-100 for loads + special tokens
        "seq_len": 86,  # 2 header + 84 circuits
        "num_puzzle_identifiers": 1000,  # Unique panel IDs
        "splits": {
            "train": train_size,
            "val": val_size,
            "test": test_size
        },
        "task": "sequence_to_sequence",
        "input_format": "panel_specification",
        "output_format": "phase_assignments"
    }
    
    metadata_file = output_dir / "dataset.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created {metadata_file}")


def main():
    """Main conversion pipeline"""
    
    print("=" * 60)
    print("Converting Panel Optimization to HRM Binary Format")
    print("=" * 60)
    print()
    
    # Input: JSON files from build_panel_optimization_dataset.py
    json_dir = Path("data/panel_optimization")
    
    # Output: HRM binary format
    output_dir = Path("data/panel_optimization_hrm")
    
    if not json_dir.exists():
        print(f"❌ Error: {json_dir} not found")
        print("Run build_panel_optimization_dataset.py first")
        return
    
    # Convert each split
    train_size = convert_json_to_hrm_format(
        json_dir / "train.json",
        output_dir,
        "train"
    )
    
    val_size = convert_json_to_hrm_format(
        json_dir / "val.json",
        output_dir,
        "val"
    )
    
    test_size = convert_json_to_hrm_format(
        json_dir / "test.json",
        output_dir,
        "test"
    )
    
    # Create metadata
    create_dataset_metadata(output_dir, train_size, val_size, test_size)
    
    print()
    print("=" * 60)
    print("✅ Conversion Complete!")
    print("=" * 60)
    print()
    print(f"HRM dataset ready at: {output_dir}/")
    print(f"  - train: {train_size} examples")
    print(f"  - val:   {val_size} examples")
    print(f"  - test:  {test_size} examples")
    print()
    print("Use this path for training:")
    print(f"  python pretrain.py data_path={output_dir}")
    print()


if __name__ == "__main__":
    main()
