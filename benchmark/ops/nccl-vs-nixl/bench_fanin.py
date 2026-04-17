#!/usr/bin/env python3
"""
Fan-in P2P benchmark: 3 senders → 1 receiver, synchronized start.

4-rank gloo process group for coordination:
  rank 0 = receiver (sgpu6)
  ranks 1,2,3 = senders (sgpu7,sgpu8,sgpu9)

NCCL: receiver creates 3 independent NcclChannels (one per sender).
      Each sender creates 1 NcclChannel to receiver.
      gloo barrier synchronizes all 4 ranks before timed section.
      Receiver posts recv on all 3 channels, senders send concurrently.

NIXL: receiver registers memory, 3 senders each WRITE concurrently.
      Receiver broadcasts "GO" notification, senders start on receipt.
"""

import argparse
import json
import os
import time
import threading

import torch
import torch.distributed as dist


def run_nccl_fanin(
    rank, world_size, msg_bytes, iters, warmup, master_addr, ifname, master_port
):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nccl_ext import NcclChannel, get_nccl_unique_id_bytes

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_SOCKET_IFNAME"] = ifname
    os.environ["NCCL_IB_HCA"] = "mlx5_1"

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)

    n_elem = msg_bytes // 2  # bf16 = 2 bytes
    buf = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0")

    if rank == 0:
        channels = []
        for sender_rank in range(1, world_size):
            uid_bytes = get_nccl_unique_id_bytes()
            dist.send(
                torch.frombuffer(uid_bytes, dtype=torch.uint8).clone(), dst=sender_rank
            )
            ch = NcclChannel(0, sender_rank, uid_bytes)
            ch.initialize()
            channels.append(ch)

        for _ in range(warmup):
            for ch in channels:
                ch.recv(buf)
            for ch in channels:
                ch.sync()
        dist.barrier()

        dist.barrier()
        for _ in range(iters):
            for ch in channels:
                ch.recv(buf)
            for ch in channels:
                ch.sync()

        dist.barrier()
        dist.destroy_process_group()
        return None, None
    else:
        uid_tensor = torch.empty(128, dtype=torch.uint8)
        dist.recv(uid_tensor, src=0)
        uid_bytes = uid_tensor.numpy().tobytes()
        ch = NcclChannel(rank, 0, uid_bytes)
        ch.initialize()

        for _ in range(warmup):
            ch.send(buf)
            ch.sync()
        dist.barrier()

        dist.barrier()
        t_start = time.perf_counter()
        for _ in range(iters):
            ch.send(buf)
            ch.sync()
        t_end = time.perf_counter()

        dist.barrier()
        dist.destroy_process_group()
        return t_start, t_end


def run_nixl_fanin(
    rank, world_size, msg_bytes, iters, warmup, local_ip, all_ips, nixl_port
):
    from nixl._api import nixl_agent, nixl_agent_config

    os.environ["NIXL_LOG_LEVEL"] = "ERROR"
    os.environ["UCX_NET_DEVICES"] = "mlx5_1:1"

    torch.cuda.set_device(0)
    n_elem = msg_bytes // 2
    buf = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0")

    agent_name = f"rank{rank}"
    listen_port = nixl_port + rank
    cfg = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=listen_port,
        backends=["UCX"],
    )
    agent = nixl_agent(agent_name, cfg)

    reg = agent.register_memory(buf, backends=["UCX"])
    descs = agent.get_xfer_descs([(buf.data_ptr(), msg_bytes, 0)], mem_type="VRAM")

    time.sleep(2)

    if rank == 0:
        for sender_rank in range(1, world_size):
            peer_name = f"rank{sender_rank}"
            peer_ip = all_ips[sender_rank]
            peer_port = nixl_port + sender_rank
            # send_local_metadata pushes OUR metadata TO the remote agent's listener (not self)
            agent.send_local_metadata(ip_addr=peer_ip, port=peer_port)

        for sender_rank in range(1, world_size):
            peer_name = f"rank{sender_rank}"
            for _ in range(100):
                if agent.check_remote_metadata(peer_name):
                    break
                time.sleep(0.1)
            agent.make_connection(peer_name)

        ser_descs = agent.get_serialized_descs(descs)
        for sender_rank in range(1, world_size):
            agent.send_notif(f"rank{sender_rank}", ser_descs)

        for sender_rank in range(1, world_size):
            peer_name = f"rank{sender_rank}"
            for _ in range(200):
                notifs = agent.get_new_notifs()
                if peer_name in notifs and any(
                    n == b"SENDER_READY" for n in notifs[peer_name]
                ):
                    break
                time.sleep(0.1)

        time.sleep(0.5)
        for sender_rank in range(1, world_size):
            agent.send_notif(f"rank{sender_rank}", b"GO")

        for _ in range(600):
            done_count = 0
            notifs = agent.get_new_notifs()
            for sender_rank in range(1, world_size):
                peer_name = f"rank{sender_rank}"
                if peer_name in notifs and any(n == b"DONE" for n in notifs[peer_name]):
                    done_count += 1
            if done_count == world_size - 1:
                break
            time.sleep(0.1)

        agent.deregister_memory(reg)
        return None, None
    else:
        receiver_name = "rank0"
        receiver_ip = all_ips[0]
        receiver_port = nixl_port + 0
        agent.send_local_metadata(ip_addr=receiver_ip, port=receiver_port)

        for _ in range(100):
            if agent.check_remote_metadata(receiver_name):
                break
            time.sleep(0.1)
        agent.make_connection(receiver_name)

        remote_descs = None
        for _ in range(200):
            notifs = agent.get_new_notifs()
            if receiver_name in notifs and len(notifs[receiver_name]) > 0:
                for n in notifs[receiver_name]:
                    if n != b"GO":
                        remote_descs = agent.deserialize_descs(n)
                        break
            if remote_descs is not None:
                break
            time.sleep(0.1)
        if remote_descs is None:
            raise RuntimeError(f"[{agent_name}] Timeout waiting for remote descriptors")

        xfer_h = agent.initialize_xfer("WRITE", descs, remote_descs, receiver_name)

        for _ in range(warmup):
            agent.transfer(xfer_h)
            while agent.check_xfer_state(xfer_h) != "DONE":
                pass
        torch.cuda.synchronize()

        agent.send_notif(receiver_name, b"SENDER_READY")

        for _ in range(200):
            notifs = agent.get_new_notifs()
            if receiver_name in notifs and any(
                n == b"GO" for n in notifs[receiver_name]
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

        agent.send_notif(receiver_name, b"DONE")
        time.sleep(1)
        agent.release_xfer_handle(xfer_h)
        agent.deregister_memory(reg)
        return t_start, t_end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--backend", required=True, choices=["nccl", "nixl"])
    p.add_argument("--msg-bytes", type=int, required=True)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--master-addr", default="10.0.0.1")
    p.add_argument("--master-port", type=int, default=31000)
    p.add_argument("--ifname", default="ens1f1np1")
    p.add_argument("--all-ips", default="10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4")
    p.add_argument("--nixl-port", type=int, default=16000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    all_ips = args.all_ips.split(",")

    if args.backend == "nccl":
        t_start, t_end = run_nccl_fanin(
            args.rank,
            args.world_size,
            args.msg_bytes,
            args.iters,
            args.warmup,
            args.master_addr,
            args.ifname,
            args.master_port,
        )
    else:
        t_start, t_end = run_nixl_fanin(
            args.rank,
            args.world_size,
            args.msg_bytes,
            args.iters,
            args.warmup,
            all_ips[args.rank],
            all_ips,
            args.nixl_port,
        )

    if t_start is None:
        print(
            f"[rank{args.rank}] {args.backend} {args.msg_bytes}B: receiver (no measurements)"
        )
        return

    elapsed_s = t_end - t_start
    total_bytes = args.iters * args.msg_bytes
    tput_mbps = total_bytes / elapsed_s / 1e6
    avg_us = elapsed_s / args.iters * 1e6
    msg_rate = args.iters / elapsed_s

    print(
        f"[rank{args.rank}] {args.backend} {args.msg_bytes}B: "
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
                    "rank": args.rank,
                    "elapsed_s": elapsed_s,
                    "avg_us": avg_us,
                    "msg_rate": msg_rate,
                    "throughput_mbps": tput_mbps,
                },
                f,
            )


if __name__ == "__main__":
    main()
