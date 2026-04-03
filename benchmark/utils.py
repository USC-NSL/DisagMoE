from argparse import ArgumentParser

def add_workload_arguments(parser: ArgumentParser):
    parser.add_argument("-n", "--num-requests", type=int, default=1000, help="number of requests to generate")
    parser.add_argument("-r", "--rate", type=float, default=0, help="rate of incoming requests, seconds per request")
    parser.add_argument("--generator-type", type=str, default="poisson", help="generator type, including 'poisson', 'uniform', 'incremental_poisson' and 'dataset'.")
    parser.add_argument("--min-input-len", type=int, default=30, help="minimum prefill length for each seqeunce")
    parser.add_argument("--max-input-len", type=int, default=70, help="initial prefill length for each seqeunce")
    parser.add_argument("--min-output-len", type=int, default=80, help="maximum prefill length for each seqeunce")
    parser.add_argument("--max-output-len", type=int, default=120, help="length of output sequence")
    parser.add_argument("--dataset-path", type=str, default=None, help="path to .npy dataset lengths file (shape [N,2]: input_len, output_len). Used with --generator-type=dataset")
    parser.add_argument("--dataset-max-context-len", type=int, default=None, help="max total context length (input+output) to filter dataset samples. Used with --generator-type=dataset")
    parser.add_argument("--gate-profile-file", type=str, default=None, help="path to gate profile file to upload and broadcast to workers")
    
def add_runtime_arguments(parser: ArgumentParser):
    parser.add_argument("--transport", type=str, default="zmq", choices=["zmq", "ucx"], help="inter-worker transport backend")
    parser.add_argument("--host-ifname", type=str, default="", help="network interface for inter-node IP and NCCL sockets")
    parser.add_argument("--nccl-ib-hca", type=str, default="", help="NCCL IB/RoCE HCA device (e.g. mlx5_1)")
    parser.add_argument("--nccl-ib-gid-index", type=str, default="", help="NCCL IB/RoCE GID index for correct subnet (e.g. 3)")
    parser.add_argument("-ca", "--cuda-graph-attn", action="store_true", default=False, help="enable cuda graph for attention")
    parser.add_argument("-ce", "--cuda-graph-expert", action="store_true", default=False, help="enable cuda graph for experts")
    parser.add_argument("--max-attn-graph-bsz", type=int, default=160, help="max batch size for attention cuda graph")
    parser.add_argument("--graph-stride", type=int, default=8, help="CUDA graph batch size stride")
    
    parser.add_argument("--max-batch-size-attn", type=int, default=160, help="max batch size for attention cuda graph")
    parser.add_argument("--max-batch-size-expert", type=int, default=512, help="max batch size for experts")
    parser.add_argument("--max-pending-sends", type=int, default=16, help="max concurrent NCCL sends per GPU to prevent SM exhaustion deadlock")
    
    parser.add_argument("--dp-policy", type=str, default="max", choices=["max", "RR", "cap_rr", "weighted"],
                        help="DP request scheduling policy: 'max' (greedy by free blocks), 'RR' (round-robin, no capacity check), 'cap_rr' (capacity-aware round-robin), 'weighted'")
    parser.add_argument("--dp-size", type=int, default=1, help="data parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="expert parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    
    parser.add_argument("-u", "--gpu-usage", type=float, default=0.7, help="GPU memory usage")
    parser.add_argument("--block-size", type=int, default=16, help="block size in cache")
    
    parser.add_argument("--serial-gemm", action="store_true", default=False, help="use serial gemm for experts")
    parser.add_argument("--less-than-sm90", action="store_true", default=False)
    parser.add_argument("--layer-scheduler-type", type=str, default="mbfs", help="layer scheduler type, including 'mbfs', 'flfs', and 'mbflfs'.")
    parser.add_argument("--layer-scheduler-step", type=int, default=1, help="layer scheduler block step, should be factor of num_layers")
    parser.add_argument("--expert-wise-schedule", action="store_true", default=False, help="enable expert-wise schedule")

    parser.add_argument("--unified-scheduler-type", type=str, default="flfs", choices=["flfs", "defrag"],
                        help="unified scheduler type for colocate mode: 'flfs' or 'defrag'")
    parser.add_argument("--defrag-weight-decay", type=float, default=0.8,
                        help="weight decay factor for unified defragging scheduler")
    parser.add_argument("--defrag-lookahead-steps", type=int, default=4,
                        help="lookahead steps for unified defragging scheduler")
    parser.add_argument("--defrag-lookback-steps", type=int, default=4,
                        help="lookback steps for unified defragging scheduler")

def add_placement_arguments(parser: ArgumentParser):
    parser.add_argument("--placement", type=str, default="colocate", help="placement strategy")
    parser.add_argument("--expert-allocation-path", type=str, default=None, help="path to JSON file specifying number of experts per node per GPU")
    parser.add_argument("--zigzag-attn", action="store_true", default=False, help="enable zigzag attention placment")
    parser.add_argument("--step-attn", type=int, default=1, help="number of steps in attention placement")
    parser.add_argument("--step-expert", type=int, default=1, help="number of steps in expert placement")
    
def add_model_arguments(parser: ArgumentParser):
    # model config
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mixtral", "qwen3_235b", "qwen3_30b", "gptoss_120b", "glm45air_106b", "glm45air_half"],
        help="model configuration to use for benchmarking",
    )
    parser.add_argument("-L", "--num-layers", type=int, default=None, help="number of layers")
    parser.add_argument("-E", "--num-experts", type=int, default=None, help="number of experts")
    parser.add_argument("-K", "--topk", type=int, default=None, help="top k")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="number of kv heads")
    
    parser.add_argument("--attn-qkv-quant", type=str, default="none", choices=["none", "fp8"], help="quantization method for attention QKV projection")
    parser.add_argument("--moe-linear-quant", type=str, default="none", choices=["none", "fp8"], help="quantization method for MoE experts linear (Serial path)")
    
    # Shared expert configuration
    parser.add_argument("--num-shared-experts", type=int, default=0, help="number of shared experts (process all tokens, no routing). 0 = disabled.")
    parser.add_argument("--shared-expert-intermediate-size", type=int, default=None, help="intermediate size for shared experts (default: hidden_size // num_experts * top_k)")
    
def add_cluster_arguments(parser: ArgumentParser):
    parser.add_argument("-N", "--num-nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("-g", "--num-gpus", type=int, default=4, help="number of gpus per node")
    
def add_analysis_arguments(parser: ArgumentParser):
    parser.add_argument("-p", "--profile-dir", type=str, default=None, help="directory to store torch profiler output")
    parser.add_argument("--nsys", action="store_true", help="enable nsys profiling")
    parser.add_argument("-f", "--file", type=str, default="reports/benchmark.csv", help="file to write benchmark results")
    parser.add_argument("--trace", action="store_true", default=False, help="generate trace")
    parser.add_argument("--enable-trace-detail", action="store_true", default=False, help="generate trace")
    parser.add_argument("--analyze-throughput", action="store_true", default=False, help="analyze throughput")
    parser.add_argument("--analyze-throughput-window", type=str, default=None,
                        help="peak-state window as START_S,END_S after benchmark start (e.g. '15,60'). "
                             "Overrides the default middle-60s selection.")
    parser.add_argument("--enable-advanced-logging", action="store_true", default=False, help="enable advanced logging for MoE diagnostics")
    parser.add_argument("--advanced-logging-dir", type=str, default="./advanced_logs", help="output directory for advanced logs")
    parser.add_argument("--advanced-logging-sample-rate", type=float, default=0.1, help="fraction of MoE steps to instrument (0.0-1.0)")

def get_parser_base():
    parser = ArgumentParser()
    add_workload_arguments(parser)
    add_runtime_arguments(parser)
    add_placement_arguments(parser)
    add_model_arguments(parser)
    add_cluster_arguments(parser)
    add_analysis_arguments(parser)
    return parser
