Sphere-16 Cluster Setup & Launch Notes
=======================================

Cluster: 8 nodes × 2 L40S GPUs = 16 GPUs total
Head node: sgpu2 (10.0.0.1)
Workers:   sgpu3 (10.0.0.2), sgpu4 (10.0.0.3), sgpu5 (10.0.0.4),
           sgpu6 (10.0.0.5), sgpu7 (10.0.0.6), sgpu8 (10.0.0.7),
           sgpu9 (10.0.0.8)
Network:   RoCE via ens1f1np1 (mlx5_1 HCA)
Conda env: disag12 (Python 3.12)
Note: there is NO NFS on sphere cluster, every node is fully bare-metal.


1. Install DisagMoE (all 8 nodes)
----------------------------------
Run from sgpu2. The script is idempotent (safe to re-run).

  # Copy script to workers
  for node in sgpu3 sgpu4 sgpu5 sgpu6 sgpu7 sgpu8 sgpu9; do
    scp benchmark/scripts/install_disagmoe_ubuntu2404.sh $node:~/
  done

  # Run on all workers in parallel (backgrounded)
  for node in sgpu3 sgpu4 sgpu5 sgpu6 sgpu7 sgpu8 sgpu9; do
    ssh $node "nohup bash ~/install_disagmoe_ubuntu2404.sh > ~/install_disagmoe.log 2>&1 &"
  done

  # Run locally on sgpu2
  bash benchmark/scripts/install_disagmoe_ubuntu2404.sh

  # Monitor progress
  for node in sgpu3 sgpu4 sgpu5 sgpu6 sgpu7 sgpu8 sgpu9; do
    echo "=== $node ===" && ssh $node "tail -3 ~/install_disagmoe.log"
  done

  # Verify on all nodes
  for node in sgpu2 sgpu3 sgpu4 sgpu5 sgpu6 sgpu7 sgpu8 sgpu9; do
    echo -n "$node: "
    ssh $node "source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12 && python -c 'import disagmoe; print(\"OK\")'"
  done

To recompile only (after code changes):

  # On each node, in disag12 env:
  cd ~/DisagMoE && make clean && make pip


2. Start Ray Cluster
--------------------
Must be done before launching the server. Always activate disag12 first.

  # On sgpu2 (head):
  conda activate disag12
  ray start --head --node-ip-address=10.0.0.1 --port=6379 \
    --dashboard-port=8265 --min-worker-port=30000 --max-worker-port=39999

  # On each worker (sgpu3-9):
  for node in sgpu3 sgpu4 sgpu5 sgpu6 sgpu7 sgpu8 sgpu9; do
    ssh $node "source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12 && ray start --address='10.0.0.1:6379'"
  done

  # Verify (should show 8 active nodes, 16 GPUs):
  ray status

  # To tear down:
  for node in sgpu2 sgpu3 sgpu4 sgpu5 sgpu6 sgpu7 sgpu8 sgpu9; do
    ssh $node "source ~/miniconda3/etc/profile.d/conda.sh && conda activate disag12 && ray stop"
  done


3. Launch DisagMoE Server
-------------------------
Run from sgpu2 in ~/DisagMoE with disag12 active and Ray cluster running.

  conda activate disag12
  cd ~/DisagMoE
  bash experiments/scripts/sphere-16/launch_server.sh

Server listens on 0.0.0.0:6699 once "Launching Flask Server" appears.
See launch_server.sh for model, placement, and runtime config.


Notes
-----
- Ray dashboard available at http://10.0.0.1:8265 when cluster is up.
- The server uses ray.init(address="auto"), so Ray must be running first.
- If Ray gets into a bad state, do a full teardown (step 2) and restart.
- gdrdrv kernel module must be loaded on each node (the install script
  handles this, but after a reboot you may need: sudo bash ~/gdrcopy/insmod.sh).
