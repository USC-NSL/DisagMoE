Sphere-3 asymmetric deployment test

Cluster:
- 3 nodes: sgpu0 (10.0.0.1), sgpu2 (10.0.0.2), sgpu3 (10.0.0.3)
- 2 L40S GPUs per node
- 6 GPUs total

Asymmetric deployment config:
- Expert distribution across 6 GPUs: 2:2:1:1:1:1
- Total experts: 8
- Total layers: 4

Setup:
1) Activate conda environment on all nodes:
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12

2) Start Ray head on sgpu0:
   ray start --head --node-ip-address=10.0.0.1 --port=6379 --dashboard-port=8265 --min-worker-port=30000 --max-worker-port=39999

3) Start Ray workers from sgpu0:
   for node in sgpu2 sgpu3; do ssh $node "source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12 && ray start --address='10.0.0.1:6379'"; done

4) Launch server from repository root on sgpu0:
   bash experiments/scripts/sphere-3/launch_server.sh

Send a 50-request test:
curl -X POST http://localhost:6699/run_once \
  -H "Content-Type: application/json" \
  -d '{
    "rate": 5,
    "time": 10,
    "distribution": "poisson",
    "min_input_len": 200,
    "max_input_len": 500,
    "min_output_len": 100,
    "max_output_len": 300
  }'

After C++/CUDA code changes, rebuild Python package:
cd ~/DisagMoE && make clean && make pip
