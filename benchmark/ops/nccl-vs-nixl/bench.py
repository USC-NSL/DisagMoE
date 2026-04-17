#!/usr/bin/env python3
"""
Cross-node open-loop P2P benchmark: NCCL vs NIXL over RoCE.

Matches the dispatcher/pool pipelining pattern:
  NCCL:  sender fires isend() x N, receiver pre-posts irecv() x N, both wait().
  NIXL:  sender creates a handle pool (pipeline_depth), posts WRITE in open loop,
         polls+reposts as handles complete.  Receiver is passive (one-sided RDMA).

Message sizes: 32, 64, 128 bytes.

Usage: see run.sh
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def run_nccl(role, msg_bytes, iters, warmup, master_addr, ifname, master_port=29500):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nccl_ext import NcclChannel, get_nccl_unique_id_bytes

    rank = 0 if role == "receiver" else 1

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_SOCKET_IFNAME"] = ifname
    os.environ["NCCL_IB_HCA"] = "mlx5_1"

    dist.init_process_group(backend="gloo", rank=rank, world_size=2)
    torch.cuda.set_device(0)

    n_elem = msg_bytes // 2  # bf16 = 2 bytes
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

    ch.sync()
    t_start = time.perf_counter()
    for _ in range(iters):
        if role == "sender":
            ch.send(buf)
        else:
            ch.recv(buf)
        ch.sync()
    t_end = time.perf_counter()

    dist.barrier()
    dist.destroy_process_group()
    return t_start, t_end


def _nixl_setup(role, msg_bytes, local_ip, remote_ip, nixl_port):
    from nixl._api import nixl_agent, nixl_agent_config

    os.environ["NIXL_LOG_LEVEL"] = "ERROR"
    os.environ["UCX_NET_DEVICES"] = "mlx5_1:1"

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
    buf = torch.randn(msg_bytes // 2, dtype=torch.bfloat16, device="cuda:0")

    reg = agent.register_memory(buf, backends=["UCX"])
    descs = agent.get_xfer_descs([(buf.data_ptr(), msg_bytes, 0)], mem_type="VRAM")

    peer_listen_port = nixl_port if role == "sender" else nixl_port + 1

    time.sleep(2)
    # send_local_metadata pushes OUR metadata TO the remote agent's listener (not self)
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

    return agent, reg, descs, remote_descs, peer_name


def run_nixl(role, msg_bytes, iters, warmup, local_ip, remote_ip, nixl_port):
    agent, reg, descs, remote_descs, peer_name = _nixl_setup(
        role, msg_bytes, local_ip, remote_ip, nixl_port
    )

    if role == "sender":
        xfer_h = agent.initialize_xfer("WRITE", descs, remote_descs, peer_name)

        for _ in range(warmup):
            status = agent.transfer(xfer_h)
            if status == "ERR":
                raise RuntimeError("NIXL warmup transfer failed")
            while agent.check_xfer_state(xfer_h) != "DONE":
                pass
        torch.cuda.synchronize()

        agent.send_notif(peer_name, b"WARMUP_DONE")
        for _ in range(100):
            notifs = agent.get_new_notifs()
            if peer_name in notifs and any(
                n == b"WARMUP_DONE" for n in notifs[peer_name]
            ):
                break
            time.sleep(0.1)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(iters):
            agent.transfer(xfer_h)
            while agent.check_xfer_state(xfer_h) != "DONE":
                pass
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        agent.send_notif(peer_name, b"BENCH_DONE")
        time.sleep(1)
        agent.release_xfer_handle(xfer_h)
    else:
        for _ in range(300):
            notifs = agent.get_new_notifs()
            if peer_name in notifs and any(
                n == b"BENCH_DONE" for n in notifs[peer_name]
            ):
                break
            time.sleep(0.1)
        t_start, t_end = None, None

    agent.deregister_memory(reg)
    return t_start, t_end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["sender", "receiver"])
    p.add_argument("--backend", required=True, choices=["nccl", "nixl"])
    p.add_argument("--msg-bytes", type=int, required=True)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--master-addr", default="10.0.0.1")
    p.add_argument("--local-ip", default="10.0.0.1")
    p.add_argument("--remote-ip", default="10.0.0.2")
    p.add_argument("--ifname", default="ens1f1np1")
    p.add_argument("--nixl-port", type=int, default=15000)
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.backend == "nccl":
        t_start, t_end = run_nccl(
            args.role,
            args.msg_bytes,
            args.iters,
            args.warmup,
            args.master_addr,
            args.ifname,
            args.master_port,
        )
    else:
        t_start, t_end = run_nixl(
            args.role,
            args.msg_bytes,
            args.iters,
            args.warmup,
            args.local_ip,
            args.remote_ip,
            args.nixl_port,
        )

    if t_start is None:
        print(
            f"[{args.role}] {args.backend} {args.msg_bytes}B: passive (no measurements)"
        )
        return

    elapsed_s = t_end - t_start
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
        with open(args.out, "w") as f:
            json.dump(
                {
                    "backend": args.backend,
                    "msg_bytes": args.msg_bytes,
                    "role": args.role,
                    "avg_us": avg_us,
                    "msg_rate": msg_rate,
                    "throughput_mbps": tput_mbps,
                },
                f,
            )


if __name__ == "__main__":
    main()
