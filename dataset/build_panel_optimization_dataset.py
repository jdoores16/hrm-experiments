#!/usr/bin/env python3
"""
Panel Optimization Dataset Builder

Creates training data for HRM to learn optimal panel schedule phase assignments.
Generates 1,000+ examples of panels with perfect phase balance.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def calculate_phase_totals(circuits: List[Dict], assignments: Dict) -> Dict[str, float]:
    """Calculate total load per phase"""
    totals = {"A": 0.0, "B": 0.0, "C": 0.0}
    
    for circuit in circuits:
        ckt_id = circuit["id"]
        phase = assignments.get(f"circuit_{ckt_id}", {}).get("phase")
        if phase in totals:
            totals[phase] += circuit["load_amps"]
    
    return totals


def greedy_phase_assignment(circuits: List[Dict]) -> Tuple[Dict, float]:
    """
    Greedy algorithm to create optimal phase assignment.
    This creates the "ground truth" for training.
    """
    # Sort circuits by load (descending) for better balance
    sorted_circuits = sorted(circuits, key=lambda c: c["load_amps"], reverse=True)
    
    phases = {"A": [], "B": [], "C": []}
    phase_totals = {"A": 0.0, "B": 0.0, "C": 0.0}
    assignments = {}
    
    for circuit in sorted_circuits:
        # Assign to least loaded phase
        min_phase = min(phase_totals, key=phase_totals.get)
        
        phases[min_phase].append(circuit["id"])
        phase_totals[min_phase] += circuit["load_amps"]
        assignments[f"circuit_{circuit['id']}"] = {"phase": min_phase}
    
    # Calculate balance quality (0-1, where 1 is perfect)
    max_load = max(phase_totals.values())
    min_load = min(phase_totals.values())
    balance_score = 1.0 - ((max_load - min_load) / max_load) if max_load > 0 else 1.0
    
    return assignments, balance_score


def generate_panel_example(
    num_circuits: int = 42,
    voltage: str = "480Y/277V",
    main_bus_amps: int = 800
) -> Dict:
    """Generate a single panel optimization example"""
    
    # Realistic load distribution
    load_options = {
        15: 0.30,   # 30% are 15A circuits (lighting)
        20: 0.40,   # 40% are 20A circuits (receptacles)
        30: 0.15,   # 15% are 30A circuits (equipment)
        40: 0.10,   # 10% are 40A circuits (large equipment)
        50: 0.05,   # 5% are 50A circuits (motors)
    }
    
    # Generate circuits
    circuits = []
    for i in range(num_circuits):
        # Weighted random selection
        load_amps = random.choices(
            list(load_options.keys()),
            weights=list(load_options.values())
        )[0]
        
        circuits.append({
            "id": i + 1,
            "load_amps": load_amps,
            "poles": 1,
            "description": f"Circuit {i+1}"
        })
    
    # Generate optimal assignment
    optimal_assignment, balance_score = greedy_phase_assignment(circuits)
    phase_totals = calculate_phase_totals(circuits, optimal_assignment)
    
    # Create example
    example = {
        "input": {
            "voltage": voltage,
            "phase": 3,
            "num_circuits": num_circuits,
            "main_bus_amps": main_bus_amps,
            "circuits": circuits
        },
        "output": {
            "assignments": optimal_assignment,
            "phase_totals": phase_totals,
            "balance_score": balance_score
        },
        "metadata": {
            "generated": True,
            "total_load": sum(phase_totals.values()),
            "avg_phase_load": sum(phase_totals.values()) / 3
        }
    }
    
    return example


def augment_panel(base_example: Dict) -> List[Dict]:
    """Create variations of a panel example"""
    variations = [base_example]
    
    # Voltage variations
    for voltage in ["208Y/120V", "240/120V"]:
        variant = json.loads(json.dumps(base_example))  # Deep copy
        variant["input"]["voltage"] = voltage
        # Recalculate for new voltage
        optimal_assignment, balance_score = greedy_phase_assignment(
            variant["input"]["circuits"]
        )
        variant["output"]["assignments"] = optimal_assignment
        variant["output"]["balance_score"] = balance_score
        variations.append(variant)
    
    # Load variations (±10% random noise)
    variant = json.loads(json.dumps(base_example))
    for circuit in variant["input"]["circuits"]:
        noise = random.uniform(0.9, 1.1)
        circuit["load_amps"] = round(circuit["load_amps"] * noise, 1)
    
    # Recalculate optimal for varied loads
    optimal_assignment, balance_score = greedy_phase_assignment(
        variant["input"]["circuits"]
    )
    variant["output"]["assignments"] = optimal_assignment
    variant["output"]["balance_score"] = balance_score
    variant["metadata"]["augmented"] = "load_variation"
    variations.append(variant)
    
    return variations


def build_dataset(
    num_base_examples: int = 200,
    augment: bool = True,
    output_dir: str = "data/panel_optimization"
) -> Dict:
    """Build complete panel optimization dataset"""
    
    print(f"Generating {num_base_examples} base panel examples...")
    
    all_examples = []
    
    # Generate diverse panels
    circuit_counts = [18, 24, 30, 42, 60, 84]
    voltages = ["480Y/277V", "208Y/120V", "240/120V"]
    
    for i in range(num_base_examples):
        # Random panel configuration
        num_circuits = random.choice(circuit_counts)
        voltage = random.choice(voltages)
        main_bus_amps = random.choice([400, 600, 800, 1000, 1200, 1600, 2000])
        
        example = generate_panel_example(num_circuits, voltage, main_bus_amps)
        
        if augment:
            # Create variations
            variations = augment_panel(example)
            all_examples.extend(variations)
        else:
            all_examples.append(example)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{num_base_examples} base examples...")
    
    print(f"\nTotal examples (with augmentation): {len(all_examples)}")
    
    # Calculate dataset statistics
    avg_balance = np.mean([ex["output"]["balance_score"] for ex in all_examples])
    print(f"Average balance score: {avg_balance:.3f}")
    
    # Split into train/validation/test
    random.shuffle(all_examples)
    
    train_size = int(0.8 * len(all_examples))
    val_size = int(0.1 * len(all_examples))
    
    train_set = all_examples[:train_size]
    val_set = all_examples[train_size:train_size + val_size]
    test_set = all_examples[train_size + val_size:]
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "train.json", "w") as f:
        json.dump(train_set, f, indent=2)
    
    with open(output_path / "val.json", "w") as f:
        json.dump(val_set, f, indent=2)
    
    with open(output_path / "test.json", "w") as f:
        json.dump(test_set, f, indent=2)
    
    # Save metadata
    metadata = {
        "total_examples": len(all_examples),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "avg_balance_score": float(avg_balance),
        "circuit_counts": circuit_counts,
        "voltages": voltages
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset saved to {output_path}/")
    print(f"   Train: {len(train_set)} examples")
    print(f"   Val:   {len(val_set)} examples")
    print(f"   Test:  {len(test_set)} examples")
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate panel optimization training dataset")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of base examples")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--output-dir", type=str, default="data/panel_optimization", help="Output directory")
    
    args = parser.parse_args()
    
    build_dataset(
        num_base_examples=args.num_examples,
        augment=not args.no_augment,
        output_dir=args.output_dir
    )
