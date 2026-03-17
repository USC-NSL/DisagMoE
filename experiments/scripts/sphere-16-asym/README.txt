Sphere-16 Asymmetric Heterogeneous Experiment
==============================================

Cluster: 8 nodes × 2 L40S GPUs = 16 GPUs total
  Full-compute nodes:  sgpu0 (10.0.0.1), sgpu2 (10.0.0.2), sgpu3 (10.0.0.3), sgpu4 (10.0.0.4)
  Throttled nodes:     sgpu6 (10.0.0.5), sgpu7 (10.0.0.6), sgpu8 (10.0.0.7), sgpu9 (10.0.0.8)
Conda env: disag12 (Python 3.12)

Purpose: Compare throughput of asymmetric expert placement on a simulated
heterogeneous cluster (50% compute throttling via CUDA MPS on 8 of 16 GPUs).

Expert allocation: 128 total (gptoss_120b, 36 layers)
  - Full-compute GPUs:  10-11 experts each (84 total)
  - Throttled GPUs:      5-6 experts each (44 total)

Gate profile: gsm8k (gating_math_gsm8k_200.parquet)
Input lengths: 128-256 tokens (2× sphere-16 baseline)
Output lengths: 256-512 tokens (same as sphere-16)

Two experiments:
  1. equal_1to1  — All 16 GPUs get equal DP attention weight (1:1)
  2. weighted_2to1 — Full-compute GPUs get 2× DP attention weight (2:1)


How to run
----------

1. Ensure DisagMoE is installed on all 8 nodes (see sphere-16/README.txt).

2. Set up MPS throttling on the last 4 nodes:

     bash experiments/scripts/sphere-16-asym/setup_mps_throttle.sh

3. Start Ray cluster (all 8 nodes):

     conda activate disag12
     ray start --head --node-ip-address=10.0.0.1 --port=6379 \
       --dashboard-port=8265 --min-worker-port=30000 --max-worker-port=39999

     for node in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
       ssh $node "source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12 && ray start --address='10.0.0.1:6379'"
     done

     ray status   # Should show 8 nodes, 16 GPUs

4. Run experiments:

     cd ~/DisagMoE
     bash experiments/scripts/sphere-16-asym/run_experiments.sh

5. Tear down MPS:

     bash experiments/scripts/sphere-16-asym/teardown_mps_throttle.sh

6. (Optional) Tear down Ray:

     for node in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
       ssh $node "source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12 && ray stop"
     done
