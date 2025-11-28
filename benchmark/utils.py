from argparse import ArgumentParser

def add_workload_arguments(parser: ArgumentParser):
    parser.add_argument("-n", "--num-requests", type=int, default=1000, help="number of requests to generate")
    parser.add_argument("-r", "--rate", type=float, default=0, help="rate of incoming requests, seconds per request")
    parser.add_argument("--generator-type", type=str, default="poisson", help="generator type, including 'poisson' and 'uniform'.")
    parser.add_argument("--min-input-len", type=int, default=30, help="minimum prefill length for each seqeunce")
    parser.add_argument("--max-input-len", type=int, default=70, help="initial prefill length for each seqeunce")
    parser.add_argument("--min-output-len", type=int, default=80, help="maximum prefill length for each seqeunce")
    parser.add_argument("--max-output-len", type=int, default=120, help="length of output sequence")
    
    parser.add_argument("--gate-profile-file", type=str, default=None, help="path to gate profile file to upload and broadcast to workers")
    
def add_runtime_arguments(parser: ArgumentParser):
    parser.add_argument("--transport", type=str, default="zmq", choices=["zmq", "ucx"], help="inter-worker transport backend")
    
    parser.add_argument("-ca", "--cuda-graph-attn", action="store_true", default=False, help="enable cuda graph for attention")
    parser.add_argument("--max-attn-graph-bsz", type=int, default=160, help="max batch size for attention cuda graph")
    parser.add_argument("--graph-stride", type=int, default=8, help="CUDA graph batch size stride")
    
    parser.add_argument("--max-batch-size-attn", type=int, default=160, help="max batch size for attention cuda graph")
    parser.add_argument("--max-batch-size-expert", type=int, default=512, help="max batch size for experts")
    
    parser.add_argument("--dp-size", type=int, default=1, help="data parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="expert parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    
    parser.add_argument("-u", "--gpu-usage", type=float, default=0.7, help="GPU memory usage")
    parser.add_argument("--block-size", type=int, default=16, help="block size in cache")
    
    parser.add_argument("--serial-gemm", action="store_true", default=False, help="use serial gemm for experts")
    parser.add_argument("--layer-scheduler-type", type=str, default="mbfs", help="layer scheduler type, including 'mbfs', 'flfs', and 'mbflfs'.")
    parser.add_argument("--layer-scheduler-step", type=int, default=1, help="layer scheduler block step, should be factor of num_layers")
    parser.add_argument("--expert-wise-schedule", action="store_true", default=False, help="enable expert-wise schedule")

def add_placement_arguments(parser: ArgumentParser):
    parser.add_argument("--placement", type=str, default="colocate", help="placement strategy")
    parser.add_argument("--zigzag-attn", action="store_true", default=False, help="enable zigzag attention placment")
    parser.add_argument("--step-attn", type=int, default=1, help="number of steps in attention placement")
    parser.add_argument("--step-expert", type=int, default=1, help="number of steps in expert placement")
    
def add_model_arguments(parser: ArgumentParser):
    # model config
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mixtral", "qwen3_235b"],
        help="model configuration to use for benchmarking",
    )
    parser.add_argument("-L", "--num-layers", type=int, default=None, help="number of layers")
    parser.add_argument("-E", "--num-experts", type=int, default=None, help="number of experts")
    parser.add_argument("-K", "--topk", type=int, default=None, help="top k")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="number of kv heads")
    
    parser.add_argument("--attn-qkv-quant", type=str, default="none", choices=["none", "fp8"], help="quantization method for attention QKV projection")
    parser.add_argument("--moe-linear-quant", type=str, default="none", choices=["none", "fp8"], help="quantization method for MoE experts linear (Serial path)")
    
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

def get_parser_base():
    parser = ArgumentParser()
    add_workload_arguments(parser)
    add_runtime_arguments(parser)
    add_placement_arguments(parser)
    add_model_arguments(parser)
    add_cluster_arguments(parser)
    add_analysis_arguments(parser)
    return parser