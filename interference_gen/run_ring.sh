#!/usr/bin/env bash
set -euo pipefail

# Launch trace-driven interference in a ring topology across N nodes.
#
# Each node both sends and receives, creating full-duplex interference
# on every link.  Ring: node[0]→node[1]→...→node[N-1]→node[0].
#
# Usage:
#   ./run_ring.sh --nodes sgpu0:10.0.0.1,sgpu2:10.0.0.2,sgpu3:10.0.0.3,sgpu4:10.0.0.4 \
#       --trace oracle_hpc --link-capacity-gbps 200
#
#   ./run_ring.sh --nodes sgpu0:10.0.0.1,sgpu2:10.0.0.2,sgpu3:10.0.0.3,sgpu4:10.0.0.4 \
#       --mode constant --target-rate-gbps 10 --duration 300
#
#   # Stop all:
#   ./run_ring.sh --nodes sgpu0:10.0.0.1,sgpu2:10.0.0.2,sgpu3:10.0.0.3,sgpu4:10.0.0.4 --stop

# --- Defaults ---
NODES_STR=""
STOP_ONLY=false
PROFILE=false
PROFILE_INTERVAL=0.01
SKIP_CALIBRATION=false
TCP_OVERHEAD=""
CALIBRATION_CSV=""
NUM_STREAMS=8        # Parallel TCP streams for throughput
UCX_TRANSPORT="rc"   # rc (RDMA) or tcp
CAL_PORT=18600
CAL_UCX_BASE_PORT=18700
CAL_IB_DURATION=3
UCX_ENV=""
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
SCP_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_RUNTIME_DIR="interference_gen_runtime"
BASE_PORT=18515

KILL_PATTERN="interfere.py|profile_bw.py|ucx_sender|ucx_receiver|ucx_perftest|ib_write_bw"

LOCAL_HOST=$(hostname -s)

die() { echo "ERROR: $*" >&2; exit 1; }

# --- Parse args ---
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)
            NODES_STR="$2"; shift 2 ;;
        --stop)
            STOP_ONLY=true; shift ;;
        --profile)
            PROFILE=true; shift ;;
        --no-profile)
            PROFILE=false; shift ;;
        --profile-interval)
            PROFILE_INTERVAL="$2"; shift 2 ;;
        --skip-calibration)
            SKIP_CALIBRATION=true; shift ;;
        --tcp-overhead)
            TCP_OVERHEAD="$2"; SKIP_CALIBRATION=true; shift 2 ;;
        --calibration-csv)
            CALIBRATION_CSV="$2"; SKIP_CALIBRATION=true; shift 2 ;;
        --num-streams)
            NUM_STREAMS="$2"; shift 2 ;;
        --transport)
            UCX_TRANSPORT="$2"; shift 2 ;;
        *)
            PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

set_ucx_env() {
    if [ "$UCX_TRANSPORT" = "tcp" ]; then
        UCX_ENV="UCX_TLS=tcp UCX_TCP_SNDBUF=4194304 UCX_TCP_RCVBUF=4194304 UCX_TCP_NODELAY=y"
    else
        UCX_ENV="UCX_TLS=rc UCX_NET_DEVICES=mlx5_1:1"
    fi
}
set_ucx_env

[ -n "$NODES_STR" ] || die "Required: --nodes host1:ip1,host2:ip2,..."

# --- Parse node list ---
IFS=',' read -ra NODE_ENTRIES <<< "$NODES_STR"
NUM_NODES=${#NODE_ENTRIES[@]}
[ "$NUM_NODES" -ge 2 ] || die "Need at least 2 nodes for a ring"

declare -a HOSTS IPS
for entry in "${NODE_ENTRIES[@]}"; do
    IFS=':' read -r host ip <<< "$entry"
    [ -n "$host" ] && [ -n "$ip" ] || die "Invalid node entry: $entry (expected host:ip)"
    HOSTS+=("$host")
    IPS+=("$ip")
done

echo "[ring] Topology ($NUM_NODES nodes, transport=tcp):"
for ((i=0; i<NUM_NODES; i++)); do
    next=$(( (i + 1) % NUM_NODES ))
    echo "  ${HOSTS[$i]} (${IPS[$i]}) → ${HOSTS[$next]} (${IPS[$next]})"
done

# --- Helper: run command on a node (SSH if remote, direct if local) ---
run_on() {
    local host="$1"; shift
    local host_short="${host%%.*}"
    if [ "$host_short" = "$LOCAL_HOST" ]; then
        bash -c "$*"
    else
        ssh $SSH_OPTS "$host" "$*"
    fi
}

# --- Cleanup ---
cleanup() {
    echo "[ring] Cleaning up all nodes..."
    for host in "${HOSTS[@]}"; do
        host_short="${host%%.*}"
        if [ "$host_short" = "$LOCAL_HOST" ]; then
            pkill -9 -f "ucx_ring_node|ucx_sender|ucx_receiver|interfere.py" >/dev/null 2>&1 || true
        else
            ssh $SSH_OPTS "$host" \
                "pkill -9 -f ucx_ring_node; pkill -9 -f ucx_sender; pkill -9 -f ucx_receiver; pkill -9 -f interfere.py; true" \
                2>/dev/null || true
        fi
    done
    echo "[ring] Done."
}

if $STOP_ONLY; then
    cleanup
    exit 0
fi

# --- Build if needed ---
if [ ! -x "$ROOT_DIR/ucx_ring_node" ]; then
    echo "[ring] UCX binaries not found. Building..."
    bash "$ROOT_DIR/build_ucx.sh" || die "Failed to build UCX tools."
fi

# --- Kill existing interference BEFORE deploy (so SCP can overwrite binaries) ---
echo "[ring] Stopping any existing interference..."
for host in "${HOSTS[@]}"; do
    host_short="${host%%.*}"
    if [ "$host_short" = "$LOCAL_HOST" ]; then
        pkill -f "ucx_ring_node|ucx_sender|ucx_receiver|interfere.py" >/dev/null 2>&1 || true
    else
        ssh $SSH_OPTS "$host" \
            "pkill -f ucx_ring_node; pkill -f ucx_sender; pkill -f ucx_receiver; pkill -f interfere.py; true" \
            2>/dev/null || true
    fi
done
sleep 2
for host in "${HOSTS[@]}"; do
    host_short="${host%%.*}"
    if [ "$host_short" = "$LOCAL_HOST" ]; then
        pkill -9 -f "ucx_ring_node|ucx_sender|ucx_receiver|interfere.py" >/dev/null 2>&1 || true
    else
        ssh $SSH_OPTS "$host" \
            "pkill -9 -f ucx_ring_node; pkill -9 -f ucx_sender; pkill -9 -f ucx_receiver; pkill -9 -f interfere.py; true" \
            2>/dev/null || true
    fi
done

# Wait for TCP ports to be free (TIME_WAIT cleanup)
echo "[ring] Waiting for TCP ports to be free..."
for ((retry=0; retry<30; retry++)); do
    port_busy=false
    for ((i=0; i<NUM_NODES; i++)); do
        prev=$(( (i - 1 + NUM_NODES) % NUM_NODES ))
        recv_port=$((BASE_PORT + prev))
        host="${HOSTS[$i]}"
        if run_on "$host" "ss -tlnp 2>/dev/null | grep -q ':${recv_port} '" 2>/dev/null; then
            port_busy=true
            break
        fi
    done
    if ! $port_busy; then
        echo "[ring] All ports free."
        break
    fi
    if [ "$retry" -eq 0 ]; then
        echo "[ring] Ports still in TIME_WAIT, waiting..."
    fi
    sleep 2
done

# --- Verify SSH to all nodes & deploy ---
echo "[ring] Verifying SSH and deploying to all nodes..."
for host in "${HOSTS[@]}"; do
    host_short="${host%%.*}"
    if [ "$host_short" = "$LOCAL_HOST" ]; then
        continue
    fi
    ssh $SSH_OPTS "$host" "true" 2>/dev/null \
        || die "Cannot SSH to $host"
    ssh $SSH_OPTS "$host" "mkdir -p '$REMOTE_RUNTIME_DIR'"
    scp $SCP_OPTS \
        "$ROOT_DIR/interfere.py" \
        "$ROOT_DIR/ucx_ring_node" \
        "$ROOT_DIR/ucx_sender" \
        "$ROOT_DIR/ucx_receiver" \
        "$host:$REMOTE_RUNTIME_DIR/"
done

# --- Extract link-capacity-gbps from passthrough args for calibration ---
LINK_CAP_GBPS=""
for ((i=0; i<${#PASSTHROUGH_ARGS[@]}; i++)); do
    if [ "${PASSTHROUGH_ARGS[$i]}" = "--link-capacity-gbps" ] && [ $((i+1)) -lt ${#PASSTHROUGH_ARGS[@]} ]; then
        LINK_CAP_GBPS="${PASSTHROUGH_ARGS[$((i+1))]}"
        break
    fi
done

# --- Extract net-device from passthrough args (default mlx5_1:1) ---
NET_DEVICE="mlx5_1:1"
for ((i=0; i<${#PASSTHROUGH_ARGS[@]}; i++)); do
    if [ "${PASSTHROUGH_ARGS[$i]}" = "--net-device" ] && [ $((i+1)) -lt ${#PASSTHROUGH_ARGS[@]} ]; then
        NET_DEVICE="${PASSTHROUGH_ARGS[$((i+1))]}"
        break
    fi
done

IB_DEV="${NET_DEVICE%%:*}"
IB_PORT="${NET_DEVICE##*:}"

# --- TCP Calibration Sweep ---
# Sweeps 10%–90% of link capacity between nodes[0] and nodes[1], measuring
# actual interference at each level with ib_write_bw.
if ! $SKIP_CALIBRATION && [ -n "$LINK_CAP_GBPS" ]; then
    CAL_HOST0="${HOSTS[0]}"
    CAL_HOST1="${HOSTS[1]}"
    CAL_IP0="${IPS[0]}"
    CAL_IP1="${IPS[1]}"
    CAL_HOST0_SHORT="${CAL_HOST0%%.*}"
    CAL_HOST1_SHORT="${CAL_HOST1%%.*}"
    CALIBRATION_CSV="/tmp/tcp_calibration_$$.csv"
    CAL_LEVELS="0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
    echo "[calibrate] Starting TCP calibration sweep between $CAL_HOST0 and $CAL_HOST1..."

    # Phase A: Baseline ib_write_bw (no interference)
    echo "[calibrate] Measuring baseline BW with ib_write_bw..."
    run_on "$CAL_HOST1" \
        "ib_write_bw -d $IB_DEV -i $IB_PORT -p $CAL_PORT -D 5 --report_gbits -s 65536" &
    IB_SERVER_PID=$!
    sleep 2
    BASELINE_OUT=$(run_on "$CAL_HOST0" \
        "ib_write_bw -d $IB_DEV -i $IB_PORT -p $CAL_PORT -D 5 --report_gbits -s 65536 $CAL_IP1" 2>/dev/null) || true
    wait $IB_SERVER_PID 2>/dev/null || true

    B_BASELINE=$(echo "$BASELINE_OUT" | grep -E '^\s+[0-9]+' | awk '{print $4}')
    if [ -z "$B_BASELINE" ]; then
        echo "[calibrate] WARNING: Could not parse baseline BW. Skipping calibration."
        CALIBRATION_CSV=""
    else
        echo "[calibrate] Baseline BW: ${B_BASELINE} Gbps"
    fi

    # Phase B: Sweep each level
    if [ -n "$B_BASELINE" ] && [ -n "$CALIBRATION_CSV" ]; then
        echo "tcp_payload_gbps,actual_interference_gbps" > "$CALIBRATION_CSV"
        CAL_LEVEL_IDX=0

        for LEVEL in $CAL_LEVELS; do
            # Each level uses NUM_STREAMS ports
            CAL_PORT_BASE=$((CAL_UCX_BASE_PORT + CAL_LEVEL_IDX * NUM_STREAMS))
            CAL_RATE_GBPS=$(python3 -c "print(float('$LINK_CAP_GBPS') * $LEVEL / 100.0)")
            PER_STREAM_RATE=$(python3 -c "print(float('$LINK_CAP_GBPS') * $LEVEL / 100.0 / $NUM_STREAMS)")

            echo "[calibrate]   ${LEVEL}% → ${CAL_RATE_GBPS} Gbps (${NUM_STREAMS} streams, ports ${CAL_PORT_BASE}–$((CAL_PORT_BASE + NUM_STREAMS - 1)))..."

            # Generate a short constant-rate schedule (per-stream rate)
            CAL_SCHEDULE="/tmp/ucx_cal_sched_${LEVEL}_$$.bin"
            python3 -c "
import sys; sys.path.insert(0, '$ROOT_DIR')
from interfere import write_schedule_v2
rate_bps = $PER_STREAM_RATE * 1e9 / 8.0
window_sec = 0.001
num_windows = 20000  # 20s schedule
rates = [rate_bps] * num_windows
write_schedule_v2(rates, window_sec, 65536, '$CAL_SCHEDULE')
" 2>/dev/null || { echo "[calibrate] WARNING: schedule gen failed for ${LEVEL}%"; CAL_LEVEL_IDX=$((CAL_LEVEL_IDX+1)); continue; }

            # Deploy schedule to calibration nodes if remote
            if [ "$CAL_HOST1_SHORT" != "$LOCAL_HOST" ]; then
                scp $SCP_OPTS "$CAL_SCHEDULE" "$CAL_HOST1:$REMOTE_RUNTIME_DIR/cal_schedule_${LEVEL}.bin" 2>/dev/null
            fi
            if [ "$CAL_HOST0_SHORT" != "$LOCAL_HOST" ]; then
                scp $SCP_OPTS "$CAL_SCHEDULE" "$CAL_HOST0:$REMOTE_RUNTIME_DIR/cal_schedule_${LEVEL}.bin" 2>/dev/null
            fi

            # Start NUM_STREAMS ucx_receiver instances on node1
            CAL_RECV_PIDS=()
            for ((si=0; si<NUM_STREAMS; si++)); do
                cal_p=$((CAL_PORT_BASE + si))
                if [ "$CAL_HOST1_SHORT" = "$LOCAL_HOST" ]; then
                    env $UCX_ENV "$ROOT_DIR/ucx_receiver" "$cal_p" --max-msg 65536 > /dev/null 2>&1 &
                else
                    ssh $SSH_OPTS "$CAL_HOST1" \
                        "cd \$HOME/$REMOTE_RUNTIME_DIR && $UCX_ENV ./ucx_receiver $cal_p --max-msg 65536" > /dev/null 2>&1 &
                fi
                CAL_RECV_PIDS+=($!)
            done
            sleep 1

            # Start NUM_STREAMS ucx_sender instances on node0
            CAL_SEND_PIDS=()
            for ((si=0; si<NUM_STREAMS; si++)); do
                cal_p=$((CAL_PORT_BASE + si))
                if [ "$CAL_HOST0_SHORT" = "$LOCAL_HOST" ]; then
                    env $UCX_ENV "$ROOT_DIR/ucx_sender" "$CAL_IP1" "$cal_p" \
                        --schedule "$CAL_SCHEDULE" --duration 15 > /dev/null 2>&1 &
                else
                    ssh $SSH_OPTS "$CAL_HOST0" \
                        "cd \$HOME/$REMOTE_RUNTIME_DIR && $UCX_ENV ./ucx_sender $CAL_IP1 $cal_p \
                        --schedule cal_schedule_${LEVEL}.bin --duration 15" > /dev/null 2>&1 &
                fi
                CAL_SEND_PIDS+=($!)
            done

            # Wait for TCP to stabilize
            sleep 3

            # Measure with ib_write_bw
            run_on "$CAL_HOST1" \
                "ib_write_bw -d $IB_DEV -i $IB_PORT -p $CAL_PORT -D $CAL_IB_DURATION --report_gbits -s 65536" &
            IB_SERVER_PID=$!
            sleep 1
            LOADED_OUT=$(run_on "$CAL_HOST0" \
                "ib_write_bw -d $IB_DEV -i $IB_PORT -p $CAL_PORT -D $CAL_IB_DURATION --report_gbits -s 65536 $CAL_IP1" 2>/dev/null) || true
            wait $IB_SERVER_PID 2>/dev/null || true

            B_LOADED=$(echo "$LOADED_OUT" | grep -E '^\s+[0-9]+' | awk '{print $4}')

            # Stop calibration interference
            for pid in "${CAL_SEND_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
            for pid in "${CAL_SEND_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
            if [ "$CAL_HOST1_SHORT" = "$LOCAL_HOST" ]; then
                pkill -f "ucx_receiver" >/dev/null 2>&1 || true
            else
                run_on "$CAL_HOST1" "pkill -f 'ucx_receiver' >/dev/null 2>&1 || true" 2>/dev/null || true
            fi
            if [ "$CAL_HOST0_SHORT" != "$LOCAL_HOST" ]; then
                run_on "$CAL_HOST0" "pkill -f 'ucx_sender' >/dev/null 2>&1 || true" 2>/dev/null || true
            fi
            for pid in "${CAL_RECV_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
            rm -f "$CAL_SCHEDULE"

            if [ -n "$B_LOADED" ]; then
                ACTUAL_INTERF=$(python3 -c "print(f'{max(0, float(\"$B_BASELINE\") - float(\"$B_LOADED\")):.4f}')")
                echo "${CAL_RATE_GBPS},${ACTUAL_INTERF}" >> "$CALIBRATION_CSV"
                echo "[calibrate]   ${LEVEL}%: TCP payload=${CAL_RATE_GBPS} Gbps → actual interference=${ACTUAL_INTERF} Gbps"
            else
                echo "[calibrate]   ${LEVEL}%: WARNING — could not parse ib_write_bw output, skipping"
            fi

            CAL_LEVEL_IDX=$((CAL_LEVEL_IDX+1))
        done

        # Validate: need at least 2 data points
        CAL_POINT_COUNT=$(tail -n +2 "$CALIBRATION_CSV" | wc -l)
        if [ "$CAL_POINT_COUNT" -lt 2 ]; then
            echo "[calibrate] WARNING: Only $CAL_POINT_COUNT calibration points. Falling back to no correction."
            CALIBRATION_CSV=""
        else
            echo "[calibrate] Calibration complete: $CAL_POINT_COUNT points saved to $CALIBRATION_CSV"
        fi

        # Wait for all calibration ports to clear
        sleep 2
    fi
elif [ -z "$CALIBRATION_CSV" ] && [ -z "$TCP_OVERHEAD" ]; then
    if ! $SKIP_CALIBRATION && [ -z "$LINK_CAP_GBPS" ]; then
        echo "[calibrate] WARNING: --link-capacity-gbps not set, skipping calibration."
    fi
fi

# Report calibration status
if [ -n "$CALIBRATION_CSV" ] && [ -f "$CALIBRATION_CSV" ]; then
    echo "[ring] Using calibration curve: $CALIBRATION_CSV"
elif [ -n "$TCP_OVERHEAD" ]; then
    echo "[ring] Using uniform TCP overhead factor: $TCP_OVERHEAD"
else
    echo "[ring] No TCP calibration (rates used as-is)"
fi

# --- Generate schedule locally (with calibration applied) ---
echo "[ring] Generating rate schedule..."
SCHEDULE_PATH="/tmp/ucx_ring_schedule_$$.bin"

# Determine calibration arg for Python
CAL_CSV_PATH=""
if [ -n "$CALIBRATION_CSV" ] && [ -f "$CALIBRATION_CSV" ]; then
    CAL_CSV_PATH="$CALIBRATION_CSV"
fi
CAL_TCP_OVERHEAD="${TCP_OVERHEAD:-1.0}"

python3 -c "
import sys; sys.path.insert(0, '$ROOT_DIR')
from interfere import resolve_trace_path, load_trace_rates, write_schedule_v2
from interfere import load_calibration_curve, calibrated_rate
import os

# Parse relevant args
args = '${PASSTHROUGH_ARGS[*]}'.split()
trace_name = None
link_cap = None
duration = 0
msg_size = 65536
window_ms = 1
mode = 'trace'
target_rate = None

i = 0
while i < len(args):
    if args[i] == '--trace' and i+1 < len(args):
        trace_name = args[i+1]; i += 2
    elif args[i] == '--link-capacity-gbps' and i+1 < len(args):
        link_cap = float(args[i+1]); i += 2
    elif args[i] == '--duration' and i+1 < len(args):
        duration = float(args[i+1]); i += 2
    elif args[i] == '--msg-size' and i+1 < len(args):
        msg_size = int(args[i+1]); i += 2
    elif args[i] == '--trace-window-ms' and i+1 < len(args):
        window_ms = float(args[i+1]); i += 2
    elif args[i] == '--mode' and i+1 < len(args):
        mode = args[i+1]; i += 2
    elif args[i] == '--target-rate-gbps' and i+1 < len(args):
        target_rate = float(args[i+1]); i += 2
    else:
        i += 1

# Load calibration
cal_csv = '$CAL_CSV_PATH'
cal_curve = None
tcp_overhead = float('$CAL_TCP_OVERHEAD')
if cal_csv and os.path.isfile(cal_csv):
    cal_curve = load_calibration_curve(cal_csv)
    print(f'[schedule] Applying calibration curve ({len(cal_curve)-1} data points)')
elif tcp_overhead > 1.0:
    print(f'[schedule] Applying uniform TCP overhead: {tcp_overhead:.4f}x')

num_streams = $NUM_STREAMS

if mode == 'trace' and trace_name:
    trace_path = resolve_trace_path(trace_name)
    link_cap_bps = link_cap * 1e9 / 8.0
    rates, window_sec = load_trace_rates(trace_path, link_cap_bps, window_ms=window_ms)
    if cal_curve:
        rates = [calibrated_rate(r, cal_curve) for r in rates]
    elif tcp_overhead > 1.0:
        rates = [r / tcp_overhead for r in rates]
    # Divide across parallel streams
    rates = [r / num_streams for r in rates]
    print(f'[schedule] Dividing rates across {num_streams} parallel streams per node')
    write_schedule_v2(rates, window_sec, msg_size, '$SCHEDULE_PATH')
elif mode == 'constant':
    if target_rate:
        rate_bps = target_rate * 1e9 / 8.0
    elif link_cap:
        rate_bps = link_cap * 1e9 / 8.0
    else:
        rate_bps = 25e9
    if cal_curve:
        rate_bps = calibrated_rate(rate_bps, cal_curve)
    elif tcp_overhead > 1.0:
        rate_bps /= tcp_overhead
    # Divide across parallel streams
    rate_bps /= num_streams
    print(f'[schedule] Dividing rates across {num_streams} parallel streams per node')
    dur = duration if duration > 0 else 3600
    window_sec = 0.001
    num_windows = int(dur / window_sec) + 1
    rates = [rate_bps] * num_windows
    write_schedule_v2(rates, window_sec, msg_size, '$SCHEDULE_PATH')
else:
    print('ERROR: --trace required for trace mode', file=sys.stderr)
    sys.exit(1)
" || die "Failed to generate schedule"

[ -f "$SCHEDULE_PATH" ] || die "Schedule file not generated"

# Extract duration from passthrough args
DURATION=0
for ((i=0; i<${#PASSTHROUGH_ARGS[@]}; i++)); do
    if [ "${PASSTHROUGH_ARGS[$i]}" = "--duration" ] && [ $((i+1)) -lt ${#PASSTHROUGH_ARGS[@]} ]; then
        DURATION="${PASSTHROUGH_ARGS[$((i+1))]}"
        break
    fi
done

# Extract msg-size from passthrough args (default 65536)
MSG_SIZE=65536
for ((i=0; i<${#PASSTHROUGH_ARGS[@]}; i++)); do
    if [ "${PASSTHROUGH_ARGS[$i]}" = "--msg-size" ] && [ $((i+1)) -lt ${#PASSTHROUGH_ARGS[@]} ]; then
        MSG_SIZE="${PASSTHROUGH_ARGS[$((i+1))]}"
        break
    fi
done

# Deploy schedule to all remote nodes
echo "[ring] Deploying schedule to all nodes..."
for host in "${HOSTS[@]}"; do
    host_short="${host%%.*}"
    [ "$host_short" = "$LOCAL_HOST" ] && continue
    scp $SCP_OPTS "$SCHEDULE_PATH" "$host:$REMOTE_RUNTIME_DIR/ring_schedule.bin" 2>/dev/null
done
# Local copy
cp "$SCHEDULE_PATH" "$ROOT_DIR/ring_schedule.bin"

# --- Set up cleanup on exit ---
trap cleanup EXIT INT TERM

# (existing interference already killed before deploy)

# --- Launch ring ---
# Each node runs ONE ucx_ring_node process that both sends and receives.
# Node[i]: sends to node[(i+1)%N], receives from node[(i-1+N)%N]
#   --send-port = BASE_PORT + i        (matches the recv-port on node[i+1])
#   --recv-port = BASE_PORT + ((i-1+N)%N)  (matches the send-port on node[i-1])

declare -a PIDS

echo "[ring] Starting ring nodes ($NUM_STREAMS streams per node)..."
for ((i=0; i<NUM_NODES; i++)); do
    next=$(( (i + 1) % NUM_NODES ))
    prev=$(( (i - 1 + NUM_NODES) % NUM_NODES ))
    host="${HOSTS[$i]}"
    send_ip="${IPS[$next]}"
    host_short="${host%%.*}"

    DURATION_ARG=""
    if [ "$DURATION" != "0" ]; then
        DURATION_ARG="--duration $DURATION"
    fi

    for ((si=0; si<NUM_STREAMS; si++)); do
        send_port=$((BASE_PORT + i * NUM_STREAMS + si))
        recv_port=$((BASE_PORT + prev * NUM_STREAMS + si))

        if [ "$si" -eq 0 ]; then
            echo "  ${host}: ${NUM_STREAMS} streams, send→${HOSTS[$next]} ports $((BASE_PORT + i * NUM_STREAMS))–$((BASE_PORT + i * NUM_STREAMS + NUM_STREAMS - 1))"
        fi

        if [ "$host_short" = "$LOCAL_HOST" ]; then
            env $UCX_ENV \
                "$ROOT_DIR/ucx_ring_node" \
                --send-ip "$send_ip" --send-port "$send_port" \
                --recv-port "$recv_port" \
                --schedule "$ROOT_DIR/ring_schedule.bin" \
                --msg-size "$MSG_SIZE" \
                $DURATION_ARG &
            PIDS+=($!)
        else
            ssh $SSH_OPTS "$host" \
                "cd \$HOME/$REMOTE_RUNTIME_DIR && \
                $UCX_ENV \
                ./ucx_ring_node \
                --send-ip $send_ip --send-port $send_port \
                --recv-port $recv_port \
                --schedule ring_schedule.bin \
                --msg-size $MSG_SIZE \
                $DURATION_ARG" &
            PIDS+=($!)
        fi
    done
done

echo "[ring] All $NUM_NODES ring nodes launched ($NUM_STREAMS processes per node, $((NUM_NODES * NUM_STREAMS)) total)."

# Wait for processes to start and connect
echo "[ring] Waiting 15s for all nodes to connect..."
sleep 15

# Periodic health check: verify all nodes are running + show BW
# TCP: read netdev counters (bytes)
NETDEV=$(run_on "${HOSTS[0]}" "ls /sys/class/infiniband/$IB_DEV/device/net/ 2>/dev/null | head -1" 2>/dev/null)
[ -z "$NETDEV" ] && NETDEV="ens1f1np1"
COUNTER_TX="/sys/class/net/$NETDEV/statistics/tx_bytes"
COUNTER_RX="/sys/class/net/$NETDEV/statistics/rx_bytes"
COUNTER_MULT=1

read_counters() {
    # Read TX and RX counters from a host, print "tx rx" on one line
    local host="$1"
    run_on "$host" "echo \$(cat $COUNTER_TX 2>/dev/null || echo 0) \$(cat $COUNTER_RX 2>/dev/null || echo 0)" 2>/dev/null
}

check_ring_status() {
    local label="$1"
    echo "[$label] Checking node status (measuring BW over 2s)..."

    # First reading
    local -a TX1 RX1
    for ((i=0; i<NUM_NODES; i++)); do
        local vals
        vals=$(read_counters "${HOSTS[$i]}")
        TX1[$i]=$(echo "$vals" | awk '{print $1}')
        RX1[$i]=$(echo "$vals" | awk '{print $2}')
    done

    sleep 2

    # Second reading + report
    local all_ok=true
    for ((i=0; i<NUM_NODES; i++)); do
        local host="${HOSTS[$i]}"
        local count
        count=$(run_on "$host" "pgrep -c -f ucx_ring_node 2>/dev/null || echo 0" 2>/dev/null)
        local vals
        vals=$(read_counters "$host")
        local tx2 rx2
        tx2=$(echo "$vals" | awk '{print $1}')
        rx2=$(echo "$vals" | awk '{print $2}')

        # Compute Gbps over 2s
        local tx_gbps=$(( (tx2 - ${TX1[$i]}) * COUNTER_MULT * 8 / 2 / 1000000000 ))
        local rx_gbps=$(( (rx2 - ${RX1[$i]}) * COUNTER_MULT * 8 / 2 / 1000000000 ))

        if [ "$count" -gt 0 ]; then
            local status="OK"
            if [ "$tx_gbps" -lt 1 ] && [ "$rx_gbps" -lt 1 ]; then
                status="WARN (no traffic)"
            fi
            echo "  ${host}: $status — TX=${tx_gbps} Gbps, RX=${rx_gbps} Gbps"
        else
            echo "  ${host}: FAILED (ucx_ring_node not running)"
            all_ok=false
        fi
    done

    if ! $all_ok; then
        echo "[$label] WARNING: some nodes failed"
    fi
}

# Initial check after startup
check_ring_status "ring"

# --- Periodic ib_write_bw monitoring ---
# Measures actual interference between nodes[0] and nodes[1] using ib_write_bw.
IB_MONITOR_PORT=$((CAL_PORT + 1))  # 18601
IB_MONITOR_DURATION=3
MONITOR_BASELINE="${B_BASELINE:-}"
MONITOR_HOST0="${HOSTS[0]}"
MONITOR_HOST1="${HOSTS[1]}"
MONITOR_IP1="${IPS[1]}"

measure_ring_interference() {
    local label="$1"
    run_on "$MONITOR_HOST1" \
        "ib_write_bw -d $IB_DEV -i $IB_PORT -p $IB_MONITOR_PORT -D $IB_MONITOR_DURATION --report_gbits -s 65536" &
    local ib_srv_pid=$!
    sleep 1
    local ib_out
    ib_out=$(run_on "$MONITOR_HOST0" \
        "ib_write_bw -d $IB_DEV -i $IB_PORT -p $IB_MONITOR_PORT -D $IB_MONITOR_DURATION --report_gbits -s 65536 $MONITOR_IP1" 2>/dev/null) || true
    wait $ib_srv_pid 2>/dev/null || true

    local bw_avg
    bw_avg=$(echo "$ib_out" | grep -E '^\s+[0-9]+' | awk '{print $4}')
    if [ -n "$bw_avg" ]; then
        if [ -n "$MONITOR_BASELINE" ]; then
            local interf
            interf=$(python3 -c "print(f'{max(0, float(\"$MONITOR_BASELINE\") - float(\"$bw_avg\")):.2f}')")
            echo "[$label] ib_write_bw: ${bw_avg} Gbps (baseline=${MONITOR_BASELINE}, interference=${interf} Gbps)"
        else
            echo "[$label] ib_write_bw: ${bw_avg} Gbps"
        fi
    else
        echo "[$label] ib_write_bw: measurement failed"
    fi
}

# Optionally launch BW profiler (local node only)
PROFILE_PID=""
if $PROFILE; then
    echo "[ring] Starting BW profiler (local HCA counters)..."
    python3 "$ROOT_DIR/profile_bw.py" \
        --peer-host "${HOSTS[1]}" --peer-ip "${IPS[1]}" \
        --interval "$PROFILE_INTERVAL" &
    PROFILE_PID=$!
    echo "[ring] Profiler PID=$PROFILE_PID"
fi

echo "[ring] Press Ctrl+C to stop."

# Periodic health check + ib_write_bw every 30s while waiting for processes
STATUS_INTERVAL=30
while true; do
    # Wait up to STATUS_INTERVAL seconds for any process to exit
    for ((s=0; s<STATUS_INTERVAL; s++)); do
        # Check if any PID has exited
        any_exited=false
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                any_exited=true
                break
            fi
        done
        if $any_exited; then break 2; fi
        sleep 1
    done
    # Still running — print status + measure interference
    check_ring_status "ring"
    measure_ring_interference "monitor"
done

# Stop profiler
if [ -n "$PROFILE_PID" ] && kill -0 "$PROFILE_PID" 2>/dev/null; then
    kill "$PROFILE_PID" 2>/dev/null || true
    wait "$PROFILE_PID" 2>/dev/null || true
fi

echo "[ring] A process exited. Cleaning up..."
