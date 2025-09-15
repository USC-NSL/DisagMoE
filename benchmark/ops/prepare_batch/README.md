# KV Cache Management Benchmark

This benchmark compares the performance of CPU-based and GPU-based KV cache management implementations in DisagMoE.

## Overview

The benchmark tests two main components:

1. **`_update_block_table`**: Updates the block table for KV cache management
2. **`_pack_flash_attn_metadata`**: Prepares FlashAttention metadata for attention computation

## Files

- `prepare_batch.py`: Main benchmark script
- `test_prepare_batch.py`: Simple test script to verify functionality
- `README.md`: This documentation

## Usage

### Quick Test

Run a small test to verify everything works:

```bash
cd /home1/wangshao/DisagMoE/benchmark/ops
python test_prepare_batch.py
```

### Full Benchmark

Run the complete benchmark with default parameters:

```bash
python prepare_batch.py
```

### Custom Benchmark

Run with custom parameters:

```bash
python prepare_batch.py --batch-sizes 16 32 64 --seq-lens 128 256 512 --iterations 50 --output results.json
```

### Command Line Options

- `--batch-sizes`: List of batch sizes to test (default: [16, 32, 64, 128, 256])
- `--seq-lens`: List of sequence lengths to test (default: [128, 256, 512, 1024])
- `--iterations`: Number of iterations per test (default: 100)
- `--output`: Output file to save results in JSON format (optional)

## Output

The benchmark outputs a table showing:

- Batch size and sequence length
- CPU and GPU execution times for both methods
- Speedup ratios (CPU time / GPU time)

Example output:
```
================================================================================
KV Cache Management Benchmark: CPU vs GPU
================================================================================
Batch Size   Seq Len   CPU Update (ms) GPU Update (ms) Speedup   CPU Pack (ms) GPU Pack (ms) Speedup   
--------------------------------------------------------------------------------
16           128       0.123           0.045           2.73      0.234         0.089         2.63      
16           256       0.145           0.052           2.79      0.267         0.102         2.62      
...
```

## Implementation Details

### CPU Implementation
- Uses `BlockManager_C` from C++ bindings
- Manages block table on CPU
- Uses `prepare_batch_infos` for batch information preparation

### GPU Implementation
- Uses `TokenToKVPoolAllocator` and `ReqToTokenPool` for GPU-based management
- Manages block table and token allocation on GPU
- Optimized for GPU memory access patterns

### Key Differences

1. **Memory Management**: CPU version uses CPU-based block manager, GPU version uses GPU-based token allocator
2. **Data Movement**: CPU version requires CPU-GPU data transfers, GPU version keeps data on GPU
3. **Parallelization**: GPU version can leverage GPU parallelism for token allocation and block table updates

## Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- DisagMoE dependencies installed
- Sufficient GPU memory for the test configurations

## Troubleshooting

If you encounter issues:

1. **CUDA out of memory**: Reduce batch sizes or sequence lengths
2. **Import errors**: Ensure DisagMoE is properly installed and PYTHONPATH is set
3. **C++ binding errors**: Ensure the C++ extensions are properly compiled

## Performance Expectations

The GPU implementation is expected to be faster due to:
- Reduced CPU-GPU data transfers
- Better memory access patterns
- GPU parallelization benefits

Typical speedups range from 2-5x depending on batch size and sequence length.
