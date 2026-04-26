#!/usr/bin/env python3
"""
1:1 cross-node P2P benchmark: NCCL vs NIXL over RoCE.

Both ranks init a 2-rank gloo PG on the RoCE IP interface. Timed window is
bracketed by dist.barrier() on both ranks. Receiver reads mlx5_1
port_rcv_data IB counter for ground-truth RX byte count; compared against
the sender's self-measured throughput, the two must agree within a few
percent AND both must be <= 200 Gbps link rate (25,000 MB/s).
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


IB_DEVICE = "mlx5_1"
IB_PORT = 1
_CTR_DIR = f"/sys/class/infiniband/{IB_DEVICE}/ports/{IB_PORT}/counters"
RCV_COUNTER = f"{_CTR_DIR}/port_rcv_data"
XMIT_COUNTER = f"{_CTR_DIR}/port_xmit_data"
# IBTA spec: port_{rcv,xmit}_data counts data octets divided by 4; *4 -> bytes.
IB_COUNTER_UNIT = 4


def read_rx_bytes():
    with open(RCV_COUNTER) as f:
        return int(f.read().strip()) * IB_COUNTER_UNIT


def read_tx_bytes():
    with open(XMIT_COUNTER) as f:
        return int(f.read().strip()) * IB_COUNTER_UNIT


def _init_gloo(rank, master_addr, master_port, ifname):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["GLOO_SOCKET_IFNAME"] = ifname
    os.environ["TP_SOCKET_IFNAME"] = ifname
    os.environ["NCCL_SOCKET_IFNAME"] = ifname
    os.environ["NCCL_IB_HCA"] = IB_DEVICE
    dist.init_process_group(backend="gloo", rank=rank, world_size=2)


def run_nccl(role, msg_bytes, iters, warmup):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nccl_ext import NcclChannel, get_nccl_unique_id_bytes

    rank = 0 if role == "receiver" else 1
    torch.cuda.set_device(0)

    n_elem = msg_bytes // 2
    buf = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0")

    if rank == 0:
        uid_bytes = get_nccl_unique_id_bytes()
    else:
        uid_bytes = None
    uid_list = [uid_bytes]
    dist.broadcast_object_list(uid_list, src=0)
    uid_bytes = uid_list[0]

    ch = NcclChannel(rank, 1 - rank, uid_bytes)
    ch.initialize()

    for _ in range(warmup):
        if role == "sender":
            ch.send(buf)
        else:
            ch.recv(buf)
    ch.sync()
    dist.barrier()

    dist.barrier()
    if role == "receiver":
        rx_before = read_rx_bytes()
    t_start = time.perf_counter()
    for _ in range(iters):
        if role == "sender":
            ch.send(buf)
        else:
            ch.recv(buf)
    ch.sync()
    t_end = time.perf_counter()
    if role == "receiver":
        rx_after = read_rx_bytes()
    dist.barrier()

    if role == "receiver":
        return {
            "role": "receiver",
            "elapsed_s": t_end - t_start,
            "rx_bytes_hw": rx_after - rx_before,
        }
    return {
        "role": "sender",
        "elapsed_s": t_end - t_start,
    }


def _nixl_setup(role, msg_bytes, local_ip, remote_ip, nixl_port, pipeline_depth):
    from nixl._api import nixl_agent, nixl_agent_config

    os.environ["NIXL_LOG_LEVEL"] = "ERROR"
    os.environ["UCX_NET_DEVICES"] = f"{IB_DEVICE}:1"

    agent_name = "sender" if role == "sender" else "receiver"
    peer_name = "receiver" if role == "sender" else "sender"

    listen_port = nixl_port if role == "receiver" else nixl_port + 1
    cfg = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=listen_port,
        backends=["UCX"],
    )
    agent = nixl_agent(agent_name, cfg)

    torch.cuda.set_device(0)
    n_elem = msg_bytes // 2

    if role == "sender":
        bufs = torch.randn(pipeline_depth, n_elem, dtype=torch.bfloat16, device="cuda:0")
        reg = agent.register_memory(bufs, backends=["UCX"])
        descs_list = []
        for i in range(pipeline_depth):
            descs_list.append(
                agent.get_xfer_descs(
                    [(bufs[i].data_ptr(), msg_bytes, 0)], mem_type="VRAM"
                )
            )
        descs = descs_list[0]
    else:
        buf = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0")
        reg = agent.register_memory(buf, backends=["UCX"])
        descs = agent.get_xfer_descs([(buf.data_ptr(), msg_bytes, 0)], mem_type="VRAM")
        descs_list = None

    peer_listen_port = nixl_port if role == "sender" else nixl_port + 1

    dist.barrier()
    time.sleep(1)

    agent.send_local_metadata(ip_addr=remote_ip, port=peer_listen_port)

    for _ in range(100):
        if agent.check_remote_metadata(peer_name):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"[{agent_name}] Timeout waiting for remote metadata")

    agent.make_connection(peer_name)

    ser_descs = agent.get_serialized_descs(descs)
    agent.send_notif(peer_name, ser_descs)

    remote_descs = None
    for _ in range(100):
        notifs = agent.get_new_notifs()
        if peer_name in notifs and len(notifs[peer_name]) > 0:
            remote_descs = agent.deserialize_descs(notifs[peer_name][0])
            break
        time.sleep(0.1)
    if remote_descs is None:
        raise RuntimeError(f"[{agent_name}] Timeout waiting for remote descriptors")

    return agent, reg, descs_list, remote_descs, peer_name


def run_nixl(role, msg_bytes, iters, warmup, local_ip, remote_ip, nixl_port,
             pipeline_depth):
    agent, reg, descs_list, remote_descs, peer_name = _nixl_setup(
        role, msg_bytes, local_ip, remote_ip, nixl_port, pipeline_depth
    )

    if role == "sender":
        handles = []
        for i in range(pipeline_depth):
            handles.append(
                agent.initialize_xfer("WRITE", descs_list[i], remote_descs, peer_name)
            )
        dist.barrier()

        for _ in range(warmup):
            status = agent.transfer(handles[0])
            if status == "ERR":
                raise RuntimeError("NIXL warmup transfer failed")
            while agent.check_xfer_state(handles[0]) != "DONE":
                pass
        torch.cuda.synchronize()
        dist.barrier()

        in_flight = [False] * pipeline_depth
        dist.barrier()
        t_start = time.perf_counter()
        for i in range(iters):
            slot = i % pipeline_depth
            if in_flight[slot]:
                while agent.check_xfer_state(handles[slot]) != "DONE":
                    pass
            agent.transfer(handles[slot])
            in_flight[slot] = True
        for slot in range(pipeline_depth):
            if in_flight[slot]:
                while agent.check_xfer_state(handles[slot]) != "DONE":
                    pass
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        dist.barrier()

        for h in handles:
            agent.release_xfer_handle(h)
        agent.deregister_memory(reg)
        return {
            "role": "sender",
            "elapsed_s": t_end - t_start,
        }
    else:
        dist.barrier()
        dist.barrier()

        dist.barrier()
        rx_before = read_rx_bytes()
        t_start = time.perf_counter()
        # Receiver is passive: RDMA WRITE is one-sided.
        dist.barrier()
        t_end = time.perf_counter()
        rx_after = read_rx_bytes()

        agent.deregister_memory(reg)
        return {
            "role": "receiver",
            "elapsed_s": t_end - t_start,
            "rx_bytes_hw": rx_after - rx_before,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["sender", "receiver"])
    p.add_argument("--backend", required=True, choices=["nccl", "nixl"])
    p.add_argument("--msg-bytes", type=int, required=True)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--master-addr", default="10.0.0.5")
    p.add_argument("--local-ip", default="10.0.0.5")
    p.add_argument("--remote-ip", default="10.0.0.6")
    p.add_argument("--ifname", default="ens1f1np1")
    p.add_argument("--nixl-port", type=int, default=15000)
    p.add_argument("--pipeline-depth", type=int, default=16)
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rank = 0 if args.role == "receiver" else 1
    _init_gloo(rank, args.master_addr, args.master_port, args.ifname)

    try:
        if args.backend == "nccl":
            result = run_nccl(args.role, args.msg_bytes, args.iters, args.warmup)
        else:
            result = run_nixl(
                args.role,
                args.msg_bytes,
                args.iters,
                args.warmup,
                args.local_ip,
                args.remote_ip,
                args.nixl_port,
                args.pipeline_depth,
            )
    finally:
        dist.destroy_process_group()

    if result["role"] == "receiver":
        elapsed_s = result["elapsed_s"]
        rx_bytes_hw = result["rx_bytes_hw"]
        expected_bytes = args.iters * args.msg_bytes
        rx_tput_hw_mbps = rx_bytes_hw / elapsed_s / 1e6
        rx_tput_expected_mbps = expected_bytes / elapsed_s / 1e6

        print(
            f"[{args.role}] {args.backend} {args.msg_bytes}B: "
            f"elapsed={elapsed_s * 1e3:.1f}ms  "
            f"rx_bytes_hw={rx_bytes_hw / 1e6:.1f}MB  "
            f"expected={expected_bytes / 1e6:.1f}MB  "
            f"rx_tput_hw={rx_tput_hw_mbps:.1f}MB/s  "
            f"rx_tput_expected={rx_tput_expected_mbps:.1f}MB/s"
        )

        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(
                    {
                        "backend": args.backend,
                        "msg_bytes": args.msg_bytes,
                        "role": "receiver",
                        "iters": args.iters,
                        "elapsed_s": elapsed_s,
                        "rx_bytes_hw": rx_bytes_hw,
                        "rx_bytes_expected": expected_bytes,
                        "rx_throughput_mbps_hw": rx_tput_hw_mbps,
                        "rx_throughput_mbps_expected": rx_tput_expected_mbps,
                    },
                    f,
                )
    else:
        elapsed_s = result["elapsed_s"]
        total_bytes = args.iters * args.msg_bytes
        tput_mbps = total_bytes / elapsed_s / 1e6
        avg_us = elapsed_s / args.iters * 1e6
        msg_rate = args.iters / elapsed_s

        print(
            f"[{args.role}] {args.backend} {args.msg_bytes}B: "
            f"elapsed={elapsed_s * 1e3:.1f}ms  avg={avg_us:.1f}us  "
            f"rate={msg_rate:.0f}msg/s  tput={tput_mbps:.1f}MB/s"
        )

        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(
                    {
                        "backend": args.backend,
                        "msg_bytes": args.msg_bytes,
                        "role": "sender",
                        "iters": args.iters,
                        "elapsed_s": elapsed_s,
                        "avg_us": avg_us,
                        "msg_rate": msg_rate,
                        "throughput_mbps": tput_mbps,
                    },
                    f,
                )


if __name__ == "__main__":
    main()
