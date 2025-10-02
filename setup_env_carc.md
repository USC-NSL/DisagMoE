No sudo, no apt. So packages are installed via 1) module, 2) conda, 3) building from source.

## Python
Use 3.12. Even 3.11 won't work.

## Conda install
```
conda install -c conda-forge ray-all cereal cppzmq
```

## Module
```
module load cuda/12.6.3
```

## Build from source
NCCL.

## Set paths
```
export CPLUS_INCLUDE_PATH=/home1/<user_name>/nccl/build/include:~/miniconda3/envs/<conda_env_name>/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/home1/<user_name>/nccl/build/lib:~/miniconda3/envs/<conda_env_name>/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home1/<user_name>/nccl/build/lib:~/miniconda3/envs/<conda_env_name>/lib:$LD_LIBRARY_PATH
```

## `tmp` dir
Sometimes carc's tmp cause compilation to crash. So,
```
mkdir ~/my_tmp
export TMPDIR=~/my_tmp/
```

## start Ray
You may want to use a specific tmpdir.
```
export RAY_TMPDIR=/home1/yizhuoli/tmp_ray/
ray start --head   --node-ip-address=10.125.0.48   --port=0   --dashboard-port=0   --min-worker-port=30000 --max-worker-port=39999   --temp-dir="$RAY_TMPDIR"
```