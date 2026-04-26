#!/usr/bin/env python3
"""
Fan-in P2P benchmark: N senders -> 1 receiver, synchronized start.

All ranks init a world-size gloo PG on the RoCE IP interface. Start and end
of the timed section are anchored by dist.barrier() across every rank,
eliminating start-time drift that previously let per-sender windows
under-overlap and report aggregate throughput above the physical link rate.

Cross-verification: the receiver reads mlx5_1 port_rcv_data IB counter
before/after the timed window to get a ground-truth RX byte count. This is
compared to the sum of per-sender self-measured throughputs; the two must
agree within a few percent AND both must be <= 200 Gbps link rate
(25,000 MB/s). Large disagreement or super-link-rate numbers indicate a
measurement bug.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


IB_DEVICES = ["mlx5_0", "mlx5_1"]
IB_PORT = 1
# IBTA spec: port_{rcv,xmit}_data reports total data octets divided by 4
# (i.e. counts 4-byte "lanes"); multiply by 4 to recover bytes.
IB_COUNTER_UNIT = 4


def _ib_counter_path(dev, counter):
    return f"/sys/class/infiniband/{dev}/ports/{IB_PORT}/counters/{counter}"


def read_rx_bytes():
    total = 0
    for dev in IB_DEVICES:
        with open(_ib_counter_path(dev, "port_rcv_data")) as f:
            total += int(f.read().strip()) * IB_COUNTER_UNIT
    return total


def read_tx_bytes():
    total = 0
    for dev in IB_DEVICES:
        with open(_ib_counter_path(dev, "port_xmit_data")) as f:
            total += int(f.read().strip()) * IB_COUNTER_UNIT
    return total


def _init_gloo(rank, world_size, master_addr, master_port, ifname):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["GLOO_SOCKET_IFNAME"] = ifname
    os.environ["TP_SOCKET_IFNAME"] = ifname
    os.environ["NCCL_SOCKET_IFNAME"] = ifname
    os.environ["NCCL_IB_HCA"] = "mlx5_1"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def run_nccl_fanin(rank, world_size, msg_bytes, iters, warmup):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nccl_ext import NcclChannel, get_nccl_unique_id_bytes

    torch.cuda.set_device(0)
    n_elem = msg_bytes // 2
    buf = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0")

    if rank == 0:
        channels = []
        for sender_rank in range(1, world_size):
            uid_bytes = get_nccl_unique_id_bytes()
            dist.send(
                torch.frombuffer(uid_bytes, dtype=torch.uint8).clone(),
                dst=sender_rank,
            )
            ch = NcclChannel(0, 1, uid_bytes)
            ch.initialize()
            channels.append(ch)

        for _ in range(warmup):
            for ch in channels:
                ch.recv(buf)
        for ch in channels:
            ch.sync()
        dist.barrier()

        dist.barrier()
        rx_before = read_rx_bytes()
        t_start = time.perf_counter()
        for _ in range(iters):
            for ch in channels:
                ch.recv(buf)
        for ch in channels:
            ch.sync()
        t_end = time.perf_counter()
        rx_after = read_rx_bytes()
        dist.barrier()

        return {
            "role": "receiver",
            "elapsed_s": t_end - t_start,
            "rx_bytes_hw": rx_after - rx_before,
        }
    else:
        uid_tensor = torch.empty(128, dtype=torch.uint8)
        dist.recv(uid_tensor, src=0)
        uid_bytes = uid_tensor.numpy().tobytes()
        ch = NcclChannel(1, 0, uid_bytes)
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

        return {
            "role": "sender",
            "elapsed_s": t_end - t_start,
        }


def run_nccl_gather_fanin(rank, world_size, msg_bytes, iters, warmup):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nccl_ext import NcclChannel, get_nccl_unique_id_bytes, nccl_group_start, nccl_group_end

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

    ch = NcclChannel(rank, 0, uid_bytes, world_size)
    ch.initialize()

    if rank == 0:
        for _ in range(warmup):
            nccl_group_start()
            for sender in range(1, world_size):
                ch.recv_from(buf, sender)
            nccl_group_end()
        ch.sync()
        dist.barrier()

        dist.barrier()
        rx_before = read_rx_bytes()
        t_start = time.perf_counter()
        for _ in range(iters):
            nccl_group_start()
            for sender in range(1, world_size):
                ch.recv_from(buf, sender)
            nccl_group_end()
        ch.sync()
        t_end = time.perf_counter()
        rx_after = read_rx_bytes()
        dist.barrier()

        return {
            "role": "receiver",
            "elapsed_s": t_end - t_start,
            "rx_bytes_hw": rx_after - rx_before,
        }
    else:
        for _ in range(warmup):
            nccl_group_start()
            ch.send_to(buf, 0)
            nccl_group_end()
        ch.sync()
        dist.barrier()

        dist.barrier()
        t_start = time.perf_counter()
        for _ in range(iters):
            nccl_group_start()
            ch.send_to(buf, 0)
            nccl_group_end()
        ch.sync()
        t_end = time.perf_counter()
        dist.barrier()

        return {
            "role": "sender",
            "elapsed_s": t_end - t_start,
        }


def run_nixl_fanin(
    rank, world_size, msg_bytes, iters, warmup, all_ips, nixl_port, pipeline_depth
):
    from nixl._api import nixl_agent, nixl_agent_config

    os.environ["NIXL_LOG_LEVEL"] = "ERROR"
    os.environ["UCX_NET_DEVICES"] = "mlx5_1:1"

    torch.cuda.set_device(0)
    n_elem = msg_bytes // 2

    agent_name = f"rank{rank}"
    listen_port = nixl_port + rank
    cfg = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=listen_port,
        backends=["UCX"],
    )
    agent = nixl_agent(agent_name, cfg)

    if rank == 0:
        buf = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0")
        reg = agent.register_memory(buf, backends=["UCX"])
        descs = agent.get_xfer_descs(
            [(buf.data_ptr(), msg_bytes, 0)], mem_type="VRAM"
        )
    else:
        bufs = torch.randn(
            pipeline_depth, n_elem, dtype=torch.bfloat16, device="cuda:0"
        )
        reg = agent.register_memory(bufs, backends=["UCX"])
        descs_list = []
        for i in range(pipeline_depth):
            descs_list.append(
                agent.get_xfer_descs(
                    [(bufs[i].data_ptr(), msg_bytes, 0)], mem_type="VRAM"
                )
            )

    dist.barrier()
    time.sleep(1)

    if rank == 0:
        for sender_rank in range(1, world_size):
            peer_ip = all_ips[sender_rank]
            peer_port = nixl_port + sender_rank
            agent.send_local_metadata(ip_addr=peer_ip, port=peer_port)

        for sender_rank in range(1, world_size):
            peer_name = f"rank{sender_rank}"
            for _ in range(100):
                if agent.check_remote_metadata(peer_name):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(
                    f"[{agent_name}] Timeout waiting for remote metadata: {peer_name}"
                )
            agent.make_connection(peer_name)

        ser_descs = agent.get_serialized_descs(descs)
        for sender_rank in range(1, world_size):
            agent.send_notif(f"rank{sender_rank}", ser_descs)

        dist.barrier()
        dist.barrier()

        dist.barrier()
        rx_before = read_rx_bytes()
        t_start = time.perf_counter()
        # Receiver is passive: data arrives via one-sided RDMA WRITE.
        # Block on the closing barrier until every sender finishes posting.
        dist.barrier()
        t_end = time.perf_counter()
        rx_after = read_rx_bytes()

        agent.deregister_memory(reg)
        return {
            "role": "receiver",
            "elapsed_s": t_end - t_start,
            "rx_bytes_hw": rx_after - rx_before,
        }
    else:
        receiver_name = "rank0"
        receiver_ip = all_ips[0]
        receiver_port = nixl_port + 0
        agent.send_local_metadata(ip_addr=receiver_ip, port=receiver_port)

        for _ in range(100):
            if agent.check_remote_metadata(receiver_name):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                f"[{agent_name}] Timeout waiting for remote metadata: {receiver_name}"
            )
        agent.make_connection(receiver_name)

        remote_descs = None
        for _ in range(100):
            notifs = agent.get_new_notifs()
            if receiver_name in notifs and len(notifs[receiver_name]) > 0:
                remote_descs = agent.deserialize_descs(notifs[receiver_name][0])
                break
            time.sleep(0.1)
        if remote_descs is None:
            raise RuntimeError(
                f"[{agent_name}] Timeout waiting for remote descriptors"
            )

        handles = []
        for i in range(pipeline_depth):
            handles.append(
                agent.initialize_xfer(
                    "WRITE", descs_list[i], remote_descs, receiver_name
                )
            )
        dist.barrier()

        for _ in range(warmup):
            agent.transfer(handles[0])
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


def run_uccl_fanin(rank, world_size, msg_bytes, iters, warmup, all_ips):
    from uccl import p2p

    torch.cuda.set_device(0)
    n_elem = msg_bytes // 2
    buf = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0")

    os.environ["UCCL_P2P_LOG_LEVEL"] = "WARNING"
    os.environ["UCCL_P2P_RDMA_GID_INDEX"] = "3"
    ep = p2p.Endpoint(0)
    ok, mr_id = ep.reg(buf.data_ptr(), msg_bytes)
    assert ok, "UCCL memory registration failed"

    local_metadata = ep.get_metadata()
    meta_len = len(local_metadata)

    all_meta = [local_metadata]
    dist.broadcast_object_list(all_meta, src=0)
    receiver_metadata = all_meta[0]

    if rank == 0:
        ep.start_passive_accept()

        sender_conn_ids = []
        for sender_rank in range(1, world_size):
            remote_meta_tensor = torch.zeros(meta_len, dtype=torch.uint8)
            dist.recv(remote_meta_tensor, src=sender_rank)
            remote_metadata = bytes(remote_meta_tensor.tolist())
            ok, conn_id = ep.add_remote_endpoint(remote_metadata)
            assert ok, f"UCCL add_remote_endpoint failed for rank {sender_rank}"
            sender_conn_ids.append(conn_id)

        dist.barrier()

        for _ in range(warmup):
            for cid in sender_conn_ids:
                ep.recv(cid, mr_id, buf.data_ptr(), msg_bytes)
        dist.barrier()

        dist.barrier()
        rx_before = read_rx_bytes()
        t_start = time.perf_counter()
        for _ in range(iters):
            for cid in sender_conn_ids:
                ep.recv(cid, mr_id, buf.data_ptr(), msg_bytes)
        t_end = time.perf_counter()
        rx_after = read_rx_bytes()
        dist.barrier()

        return {
            "role": "receiver",
            "elapsed_s": t_end - t_start,
            "rx_bytes_hw": rx_after - rx_before,
        }
    else:
        r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(receiver_metadata)

        dist.send(torch.ByteTensor(list(local_metadata)), dst=0)

        ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
        assert ok, f"UCCL connect to {r_ip}:{r_port} failed"

        dist.barrier()

        for _ in range(warmup):
            ep.send(conn_id, mr_id, buf.data_ptr(), msg_bytes)
        dist.barrier()

        dist.barrier()
        t_start = time.perf_counter()
        for _ in range(iters):
            ep.send(conn_id, mr_id, buf.data_ptr(), msg_bytes)
        t_end = time.perf_counter()
        dist.barrier()

        return {
            "role": "sender",
            "elapsed_s": t_end - t_start,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--backend", required=True, choices=["nccl", "nixl", "nccl_gather", "uccl"])
    p.add_argument("--msg-bytes", type=int, required=True)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--master-addr", default="10.0.0.5")
    p.add_argument("--master-port", type=int, default=31000)
    p.add_argument("--ifname", default="ens1f1np1")
    p.add_argument("--all-ips", default="10.0.0.5,10.0.0.6,10.0.0.7,10.0.0.8")
    p.add_argument("--nixl-port", type=int, default=16000)
    p.add_argument("--pipeline-depth", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    all_ips = args.all_ips.split(",")

    _init_gloo(
        args.rank,
        args.world_size,
        args.master_addr,
        args.master_port,
        args.ifname,
    )

    try:
        if args.backend == "nccl":
            result = run_nccl_fanin(
                args.rank,
                args.world_size,
                args.msg_bytes,
                args.iters,
                args.warmup,
            )
        elif args.backend == "nccl_gather":
            result = run_nccl_gather_fanin(
                args.rank,
                args.world_size,
                args.msg_bytes,
                args.iters,
                args.warmup,
            )
        elif args.backend == "uccl":
            result = run_uccl_fanin(
                args.rank,
                args.world_size,
                args.msg_bytes,
                args.iters,
                args.warmup,
                all_ips,
            )
        else:
            result = run_nixl_fanin(
                args.rank,
                args.world_size,
                args.msg_bytes,
                args.iters,
                args.warmup,
                all_ips,
                args.nixl_port,
                args.pipeline_depth,
            )
    finally:
        dist.destroy_process_group()

    n_senders = args.world_size - 1

    if result["role"] == "receiver":
        elapsed_s = result["elapsed_s"]
        rx_bytes_hw = result["rx_bytes_hw"]
        expected_bytes = args.iters * args.msg_bytes * n_senders
        rx_tput_hw_mbps = rx_bytes_hw / elapsed_s / 1e6
        rx_tput_expected_mbps = expected_bytes / elapsed_s / 1e6

        print(
            f"[rank{args.rank}] {args.backend} {args.msg_bytes}B receiver: "
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
                        "rank": args.rank,
                        "iters": args.iters,
                        "n_senders": n_senders,
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
            f"[rank{args.rank}] {args.backend} {args.msg_bytes}B sender: "
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
