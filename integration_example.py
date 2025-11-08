"""
Example: Integrating HRM with Your Application

This demonstrates how to use the HRM wrapper to add advanced reasoning
capabilities to your existing app.
"""

import torch
from hrm_wrapper import HRMInference, HRMAdapter, download_pretrained_checkpoint


def example_sudoku_solver():
    """Example: Using HRM to solve Sudoku puzzles"""
    print("\n=== Sudoku Solver Example ===")
    
    # Option 1: Use pre-trained checkpoint
    # checkpoint_path = download_pretrained_checkpoint('sudoku-extreme')
    # model = HRMInference(checkpoint_path)
    
    # Option 2: Load local checkpoint
    # model = HRMInference('checkpoints/your_checkpoint.pt')
    
    print("HRM can solve expert-level Sudoku with 55% accuracy")
    print("(compared to 0% for Claude 3.7 and GPT-4)")
    

def example_custom_reasoning_task():
    """Example: Training HRM on your custom reasoning task"""
    print("\n=== Custom Task Example ===")
    
    # Initialize model
    model = HRMInference(device='cuda')
    
    # Define training configuration
    config = {
        'epochs': 20000,
        'batch_size': 384,
        'learning_rate': 7e-5,
        'weight_decay': 1.0,
        'reasoning_steps': 8
    }
    
    # Train on your dataset
    # model.train('data/your-task-dataset', config)
    
    print("HRM can learn complex reasoning from just 1,000 examples")
    print("Training takes ~2 GPU hours for most tasks")


def example_api_integration():
    """Example: Integrating HRM into an API or service"""
    print("\n=== API Integration Example ===")
    
    # Load model
    model = HRMInference()
    adapter = HRMAdapter(model)
    
    # Your API receives data
    api_request_data = {
        'puzzle_type': 'sudoku',
        'grid': [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    }
    
    # Convert to HRM format
    # hrm_input = adapter.prepare_input(api_request_data)
    
    # Get prediction
    # result = model.predict(hrm_input)
    
    # Convert back to your format
    # api_response = adapter.process_output(result)
    
    print("HRM can be integrated into REST APIs, microservices, or agents")
    print("Inference is 100x faster than LLM chain-of-thought reasoning")


def example_batch_processing():
    """Example: Processing multiple items with HRM"""
    print("\n=== Batch Processing Example ===")
    
    model = HRMInference()
    adapter = HRMAdapter(model)
    
    # Process multiple puzzles/tasks at once
    batch_data = [
        {'task': 'puzzle_1'},
        {'task': 'puzzle_2'},
        {'task': 'puzzle_3'}
    ]
    
    # results = adapter.batch_predict(batch_data)
    
    print("HRM can efficiently process batches of reasoning tasks")
    print("With only 27M parameters, it fits easily on edge devices")


if __name__ == "__main__":
    print("HRM Integration Examples")
    print("=" * 60)
    
    print("\nHRM (Hierarchical Reasoning Model) Architecture:")
    print("  - 27 million parameters (vs billions for LLMs)")
    print("  - Trains on ~1,000 examples (vs millions for LLMs)")
    print("  - 100x faster inference than chain-of-thought")
    print("  - No pre-training required")
    print("  - Brain-inspired hierarchical processing")
    
    example_sudoku_solver()
    example_custom_reasoning_task()
    example_api_integration()
    example_batch_processing()
    
    print("\n" + "=" * 60)
    print("To integrate HRM with your app:")
    print("1. Customize hrm_wrapper.py for your data format")
    print("2. Train on your task OR use pre-trained checkpoints")
    print("3. Use HRMAdapter to convert between formats")
    print("4. Call model.predict() for inference")
