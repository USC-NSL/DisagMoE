# Disag MoE

## Install

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nccl conda-forge::cuda-cudart-dev nvidia/label/cuda-12.4.0::cuda-runtime
sudo apt-get install libzmq3-dev libcereal-dev
pip install -r requirements.txt
pip install .
```

## Tests

```bash
python tests/test_binding.py
```