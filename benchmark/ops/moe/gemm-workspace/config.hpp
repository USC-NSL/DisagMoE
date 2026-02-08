#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe_gemm_profiler {

namespace gpu_config {
    constexpr bool ENABLE_A100 = true;
    constexpr bool ENABLE_H200 = false;
    
    constexpr int A100_SM_COUNT = 108;
    constexpr int64_t A100_L2_CACHE_SIZE = 40 * 1024 * 1024;
    constexpr int64_t A100_HBM_BANDWIDTH_GBps = 2039;
    constexpr double A100_FP16_TFLOPS = 312.0;
    
    constexpr int H200_SM_COUNT = 132;
    constexpr int64_t H200_L2_CACHE_SIZE = 60 * 1024 * 1024;
    constexpr int64_t H200_HBM_BANDWIDTH_GBps = 4800;
    constexpr double H200_FP16_TFLOPS = 989.0;
}

namespace profiling_config {
    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int BENCHMARK_ITERATIONS = 50;
    constexpr int WORKSPACE_COUNT = 0;
    constexpr int BATCH_SIZE_MIN = 1;
    constexpr int BATCH_SIZE_MAX = 2000;
    constexpr int BATCH_SIZE_STEP = 1;
    constexpr int NUM_EXPERTS = 8;
    constexpr bool USE_CUDA_GRAPHS = true;
}

struct MoELayerConfig {
    const char* model_name;
    int hidden_size;
    int moe_intermediate_size;
    int num_experts;
    int num_experts_per_tok;
};

namespace models {
    constexpr MoELayerConfig QWEN3_30B_A3B = {"Qwen3-30B-A3B", 2048, 768, 128, 8};
    constexpr MoELayerConfig QWEN3_235B_A22B = {"Qwen3-235B-A22B", 4096, 1536, 128, 8};
    constexpr MoELayerConfig GPT_OSS_120B = {"GPT-OSS-120B", 2880, 2880, 128, 4};
}

inline std::vector<MoELayerConfig> get_all_models() {
    return {
        models::QWEN3_30B_A3B,
        models::QWEN3_235B_A22B,
        models::GPT_OSS_120B,
    };
}

namespace gemm_config {
    constexpr int ALIGNMENT_A = 8;
    constexpr int ALIGNMENT_B = 8;
    constexpr int ALIGNMENT_C = 8;
    constexpr int SM80_TILE_M = 128;
    constexpr int SM80_TILE_N = 128;
    constexpr int SM80_TILE_K = 32;
    constexpr int SM80_STAGES = 4;
}

namespace output_config {
    constexpr const char* PLOTS_DIR = "plots";
    constexpr const char* CSV_OUTPUT_DIR = "../../simulation/expert_costs_profiles";
}

}
