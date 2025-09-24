No sudo, no apt. So packages are installed via 1) module, 2) conda, 3) building from source.

## Python
Use 3.12. Even 3.11 won't work.

## Conda install
```
conda install -c conda-forge ray-all cereal
```

## Module
```
module load cuda/12.6.3 libzmq/4.3.5
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