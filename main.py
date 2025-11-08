#!/usr/bin/env python3
"""
HRM Integration - Main Entry Point

This demonstrates the Sapient HRM (Hierarchical Reasoning Model) setup
and provides examples for integrating it with your applications.
"""

import sys


def show_hrm_info():
    """Display information about HRM and its capabilities"""
    print("=" * 70)
    print("  SAPIENT HRM - Hierarchical Reasoning Model Integration")
    print("=" * 70)
    print("\n🧠 What is HRM?")
    print("   A brain-inspired neural network architecture for complex reasoning")
    print("   - Only 27M parameters (vs billions for LLMs)")
    print("   - Trains on ~1,000 examples (vs millions for LLMs)")
    print("   - 100x faster than chain-of-thought reasoning")
    print("   - No pre-training required")
    
    print("\n📊 Performance Highlights:")
    print("   Task              HRM (27M)    Large LLMs")
    print("   -" * 50)
    print("   Sudoku Extreme    55%          0%")
    print("   Maze 30x30        74.5%        0%")
    print("   ARC-AGI-2         5%           Lower")
    
    print("\n📁 Project Structure:")
    print("   ├── models/              Core HRM architecture")
    print("   ├── dataset/             Dataset builders (Sudoku, Maze, ARC)")
    print("   ├── utils/               Utility functions")
    print("   ├── hrm_wrapper.py       Integration wrapper")
    print("   ├── integration_example.py   Usage examples")
    print("   ├── pretrain.py          Training script")
    print("   └── evaluate.py          Evaluation script")
    
    print("\n🚀 Available Pre-trained Models:")
    print("   1. Sudoku Extreme - Expert-level puzzle solving")
    print("   2. Maze 30x30 - Optimal pathfinding")
    print("   3. ARC-AGI-2 - Abstract reasoning benchmark")
    
    print("\n📝 Next Steps:")
    print("   1. Run: python integration_example.py")
    print("      See examples of how to use HRM")
    print()
    print("   2. Customize hrm_wrapper.py")
    print("      Add your data conversion logic")
    print()
    print("   3. Train or download a model:")
    print("      - Use pre-trained: download_pretrained_checkpoint()")
    print("      - Train custom: python pretrain.py")
    print()
    print("   4. Integrate with your app:")
    print("      - Load model with HRMInference()")
    print("      - Convert data with HRMAdapter()")
    print("      - Get predictions with model.predict()")
    
    print("\n" + "=" * 70)
    print("📖 For full documentation, see README.md")
    print("🔗 Official repo: https://github.com/sapientinc/HRM")
    print("=" * 70)
    print()


def show_integration_menu():
    """Show interactive integration options"""
    print("\nWhat would you like to do?\n")
    print("1. View integration examples")
    print("2. Information about pre-trained models")
    print("3. Training guide")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n→ Run: python integration_example.py")
        print("  This shows complete examples of HRM integration\n")
    elif choice == "2":
        print("\nPre-trained Models:")
        print("-" * 50)
        print("Sudoku Extreme:")
        print("  - HuggingFace: sapientinc/HRM-checkpoint-sudoku-extreme")
        print("  - Task: Solve expert-level 9x9 Sudoku puzzles")
        print("  - Accuracy: 55% (vs 0% for large LLMs)")
        print()
        print("Maze 30x30 Hard:")
        print("  - HuggingFace: sapientinc/HRM-checkpoint-maze-30x30-hard")
        print("  - Task: Find optimal paths in complex mazes")
        print("  - Success: 74.5% (vs 0% for large LLMs)")
        print()
        print("ARC-AGI-2:")
        print("  - HuggingFace: sapientinc/HRM-checkpoint-ARC-2")
        print("  - Task: Abstract visual reasoning (AGI benchmark)")
        print("  - Performance: Outperforms GPT-4, Claude")
        print()
    elif choice == "3":
        print("\nTraining Guide:")
        print("-" * 50)
        print("1. Build dataset:")
        print("   python dataset/build_sudoku_dataset.py --subsample-size 1000")
        print()
        print("2. Start training:")
        print("   python pretrain.py data_path=data/your-dataset")
        print()
        print("3. Training time:")
        print("   - ~10 minutes for Sudoku (8 GPUs)")
        print("   - ~10 hours for Sudoku (laptop GPU)")
        print("   - ~24 hours for ARC tasks")
        print()
        print("See README.md for detailed training instructions")
        print()


def main():
    """Main entry point"""
    show_hrm_info()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        show_integration_menu()
    else:
        print("💡 Tip: Run with --interactive for more options")
        print("   python main.py --interactive\n")


if __name__ == "__main__":
    main()
