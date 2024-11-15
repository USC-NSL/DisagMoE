import os

_env_vars = {
    "DMOE_PROFILE_DIR": "",
    "CUDA_LAUNCH_BLOCKING": "0",
    "NCCL_DEBUG": "",
    "NCCL_LAUNCH_MODE": ""
}

ENV_VARS = {
    k: os.environ.get(k, v) for k, v in _env_vars.items()
}