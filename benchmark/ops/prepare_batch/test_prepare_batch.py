#!/usr/bin/env python3
"""
Simple test script to verify the prepare_batch benchmark works correctly
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from prepare_batch import run_benchmark

def test_small_benchmark():
    """Run a small benchmark to test functionality"""
    print("Running small benchmark test...")
    
    # Test with small batch sizes and single sequence length
    batch_sizes = [16, 64, 128, 256]
    seq_len = 100
    num_iterations = 50  # Small number for quick test
    
    try:
        results = run_benchmark(batch_sizes, seq_len, num_iterations)
        print(f"\nTest completed successfully!")
        
        # Print summary of results
        print("\nResults summary:")
        print(f"Prefill-only results: {len(results['prefill_only']['batch_sizes'])} batch sizes tested")
        print(f"Decode-only results: {len(results['decode_only']['batch_sizes'])} batch sizes tested")
        print(f"Decode-only pack metadata results: {len(results['decode_only_pack_metadata']['batch_sizes'])} batch sizes tested")
        
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up CUDA
    torch.set_default_device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    
    success = test_small_benchmark()
    if success:
        print("\n✅ Test passed!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
