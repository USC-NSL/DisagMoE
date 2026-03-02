import os

_env_vars = {
    "DMOE_PROFILE_DIR": "",
    "CUDA_LAUNCH_BLOCKING": "0",
    "NCCL_DEBUG": "",
    "NCCL_LAUNCH_MODE": "",
    "RAY_DEDUP_LOGS": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "",
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "26500",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "LD_LIBRARY_PATH": "",
    "ENABLE_NVTX": "0",
    "DMOE_WEIGHTED_ROUTER_FILE": "",
    "VLLM_FLASH_ATTN_VERSION": "",
    "TMPDIR": "/tmp/disagmoe/",
}

ENV_VARS = {
    k: val for k, val in (
        (k, os.environ.get(k, v)) for k, v in _env_vars.items()
    ) if val # exclude empty strings
}
