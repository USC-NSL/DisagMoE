# Bandwidth & Congestion-Aware Network Simulation

## Overview

The simulation models network transfer delays between GPUs using a **dual receive-queue** congestion model. Each GPU maintains two independent FIFO receive queues:

1. **NVLink queue** — for intra-node (same-host) transfers
2. **RDMA queue** — for inter-node (cross-host) transfers

Each queue processes one transfer at a time at the full link bandwidth. The two queues operate independently and can drain in parallel.

## Why Two Queues?

In real GPU clusters, intra-node traffic (NVLink) and inter-node traffic (InfiniBand/RoCE RDMA) use physically separate network fabrics. A GPU receiving data from both a local peer (NVLink) and a remote peer (RDMA) does not contend on a shared link — the two paths are independent. However, two intra-node senders targeting the same GPU do share NVLink bandwidth, and two inter-node senders share the NIC.

## Congestion Model

### Receiver-Side Only

We model congestion only on the **receiver side**. Sender-side contention is not modeled — this is a deliberate simplification. In practice, senders rarely saturate their outgoing links because traffic fans out to many destinations.

### Transfer Time Computation

Given a transfer of `num_tokens` from GPU `src` to GPU `dst`:

```
data_bytes = num_tokens × hidden_dim × bytes_per_element
bandwidth  = NVLink_BW  if same_host(src, dst)  else  RDMA_BW
solo_delay = data_bytes / bandwidth   (converted to simulation ticks)
```

`solo_delay` is the time the transfer takes when it has the full link to itself.

### Queueing Behavior

When multiple transfers arrive at the same GPU on the same link type, they are serialized in FIFO order. Each transfer occupies the channel for its full `solo_delay`. The total receive time on one channel is the sum of all queued transfer times.

Because the NVLink and RDMA queues are independent, the GPU finishes receiving when the slower queue drains:

```
gpu_receive_time = max(sum_of_nvlink_transfers, sum_of_rdma_transfers)
```

### Phase-Level Timing (sync / tbo simulators)

In the synchronous and TBO (two-batch overlap) simulators, all tokens in an iteration (or microbatch) go through each phase together. The phase cannot advance until every GPU finishes its receives. So the overall phase transfer time is:

```
phase_transfer_time = max over all destination GPUs of gpu_receive_time
```

### Async Simulator

In the async (simpy-based) simulator, each GPU's `IngressPort` has two `simpy.Resource(capacity=1)` objects — one for NVLink and one for RDMA. A transfer request acquires the appropriate resource (blocking if another transfer is in progress on that channel), holds it for `solo_delay` ticks, then releases it. This naturally serializes same-channel transfers while allowing NVLink and RDMA to proceed in parallel.

## Topology

GPU-to-host assignment is determined by `n_gpu_per_host`:

```
host_of(gpu) = gpu // n_gpu_per_host
same_host(a, b) = (a // n_gpu_per_host) == (b // n_gpu_per_host)
```

## Parameters

| Parameter | Description |
|---|---|
| `n_gpu_per_host` | Number of GPUs per physical host |
| `hidden_dim` | Model hidden dimension (determines bytes per token) |
| `bytes_per_element` | 2 for FP16, 4 for FP32 |
| `intra_node_bw_gbps` | NVLink bandwidth in GB/s (e.g. 800) |
| `inter_node_bw_gbps` | RDMA bandwidth in GB/s (e.g. 100 or 200) |
| `net_delay_fn(src, dst, num_tokens)` | Returns transfer time in ticks for full bandwidth |

## Fallback

When `net_delay_fn` is `None` (bandwidth-aware mode disabled), fixed per-hop delays (`net_t_attn_to_expert`, `net_t_expert_to_attn`) are used instead, with no congestion modeling.
