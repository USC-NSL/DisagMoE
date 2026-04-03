#!/usr/bin/env bash
set -euo pipefail

# Run trace-driven workload traffic on a node pair (server + client).
# Launches ucx_receiver on the peer (via SSH) and ucx_sender locally,
# both orchestrated through interfere.py.
#
# The sender uses token-bucket rate control for smooth, continuous
# traffic that matches the trace's BW pattern at sub-ms fidelity.
#
# Usage:
#   # Trace replay — Oracle HPC noise pattern:
#   ./run_interfere.sh --peer-host sgpu2 --peer-ip 10.0.0.2 \
#       --trace oracle_hpc --link-capacity-gbps 200
#
#   # Constant rate — steady 10 Gbps:
#   ./run_interfere.sh --peer-host sgpu2 --peer-ip 10.0.0.2 \
#       --mode constant --target-rate-gbps 10 --duration 300
#
#   # BW profiler runs by default (logs to bw_profile_<timestamp>.csv).
#   # Disable with --no-profile, or set output path with --profile-output.
#
#   # Stop all on both nodes:
#   ./run_interfere.sh --peer-host sgpu2 --stop

# --- Defaults ---
PEER_HOST=""
PEER_IP=""
HEAD_IP=""
STOP_ONLY=false
PROFILE=false
PROFILE_OUTPUT=""
SKIP_CALIBRATION=false
TCP_OVERHEAD=""
CALIBRATION_CSV=""
NUM_STREAMS=8
UCX_TRANSPORT="rc"   # rc (RDMA) or tcp
CAL_PORT=18600
CAL_UCX_BASE_PORT=18700
CAL_IB_DURATION=3
UCX_ENV=""           # set in set_ucx_env()
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
SCP_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_RUNTIME_DIR="interference_gen_runtime"

# Process names to kill during cleanup
KILL_PATTERN="interfere.py|profile_bw.py|ucx_sender|ucx_receiver|ucx_perftest|ib_write_bw"

# Detect head IP from mlx5_1 interface
detect_head_ip() {
    local ip
    # Try the netdev corresponding to mlx5_1 (our RDMA device) first
    for dev in ens1f1np1 ibp65s0; do
        ip=$(ip -4 addr show dev "$dev" 2>/dev/null | grep -oP 'inet \K[0-9.]+' || true)
        [ -n "$ip" ] && break
    done
    if [ -z "$ip" ]; then
        ip=$(hostname -I | awk '{print $1}')
    fi
    echo "$ip"
}

die() { echo "ERROR: $*" >&2; exit 1; }

cleanup() {
    echo "[run] Cleaning up..."
    # Kill local processes
    pkill -f "$KILL_PATTERN" >/dev/null 2>&1 || true
    # Kill remote processes
    if [ -n "$PEER_HOST" ]; then
        ssh $SSH_OPTS "$PEER_HOST" \
            "pkill -f '$KILL_PATTERN' >/dev/null 2>&1 || true" 2>/dev/null || true
    fi
    echo "[run] Done."
}

# Parse --peer-host and --stop first, pass everything else through
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --peer-host)
            PEER_HOST="$2"; shift 2 ;;
        --peer-ip)
            PEER_IP="$2"; shift 2 ;;
        --head-ip)
            HEAD_IP="$2"; shift 2 ;;
        --stop)
            STOP_ONLY=true; shift ;;
        --profile)
            PROFILE=true; shift ;;
        --no-profile)
            PROFILE=false; shift ;;
        --profile-output)
            PROFILE=true; PROFILE_OUTPUT="$2"; shift 2 ;;
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

# Validate
[ -n "$PEER_HOST" ] || die "Required: --peer-host <hostname>"

if $STOP_ONLY; then
    cleanup
    exit 0
fi

[ -n "$PEER_IP" ] || die "Required: --peer-ip <ip>"

# Detect head IP if not specified
if [ -z "$HEAD_IP" ]; then
    HEAD_IP=$(detect_head_ip)
fi

echo "[run] Head: $(hostname) ($HEAD_IP)"
echo "[run] Peer: $PEER_HOST ($PEER_IP)"

# Check that UCX binaries exist locally
if [ ! -x "$ROOT_DIR/ucx_sender" ] || [ ! -x "$ROOT_DIR/ucx_receiver" ]; then
    echo "[run] UCX binaries not found. Building..."
    bash "$ROOT_DIR/build_ucx.sh" || die "Failed to build UCX tools. Check UCX installation."
fi

# Verify SSH connectivity
ssh $SSH_OPTS "$PEER_HOST" "true" 2>/dev/null \
    || die "Cannot SSH to $PEER_HOST"

# Prepare remote: copy interfere.py, ucx_receiver, ucx_sender binaries, and any referenced files
echo "[run] Preparing remote runtime on $PEER_HOST..."
ssh $SSH_OPTS "$PEER_HOST" "mkdir -p '$REMOTE_RUNTIME_DIR'"
scp $SCP_OPTS \
    "$ROOT_DIR/interfere.py" \
    "$ROOT_DIR/ucx_receiver" \
    "$ROOT_DIR/ucx_sender" \
    "$PEER_HOST:$REMOTE_RUNTIME_DIR/"

# Copy any file args referenced in passthrough (e.g., calibration CSVs)
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [ -f "$arg" ]; then
        scp $SCP_OPTS "$arg" "$PEER_HOST:$REMOTE_RUNTIME_DIR/" 2>/dev/null || true
    fi
done

# Set up cleanup on exit
trap cleanup EXIT INT TERM

# Kill any existing interference
pkill -f "$KILL_PATTERN" >/dev/null 2>&1 || true
ssh $SSH_OPTS "$PEER_HOST" \
    "pkill -f '$KILL_PATTERN' >/dev/null 2>&1 || true" 2>/dev/null || true
sleep 3  # Allow cleanup

# --- Extract link-capacity-gbps and net-device from passthrough args for calibration ---
LINK_CAP_GBPS=""
NET_DEVICE="mlx5_1:1"
for ((i=0; i<${#PASSTHROUGH_ARGS[@]}; i++)); do
    if [ "${PASSTHROUGH_ARGS[$i]}" = "--link-capacity-gbps" ] && [ $((i+1)) -lt ${#PASSTHROUGH_ARGS[@]} ]; then
        LINK_CAP_GBPS="${PASSTHROUGH_ARGS[$((i+1))]}"
    fi
    if [ "${PASSTHROUGH_ARGS[$i]}" = "--net-device" ] && [ $((i+1)) -lt ${#PASSTHROUGH_ARGS[@]} ]; then
        NET_DEVICE="${PASSTHROUGH_ARGS[$((i+1))]}"
    fi
done

IB_DEV="${NET_DEVICE%%:*}"
IB_PORT="${NET_DEVICE##*:}"

# --- TCP Calibration Sweep ---
# Sweeps 10%–90% of link capacity, measuring actual interference at each level
# with ib_write_bw.  Produces a calibration CSV that maps TCP payload rate to
# actual interference observed.  Each level uses a different UCX port to avoid
# TCP TIME_WAIT collisions.
if ! $SKIP_CALIBRATION && [ -n "$LINK_CAP_GBPS" ]; then
    CALIBRATION_CSV="/tmp/tcp_calibration_$$.csv"
    CAL_LEVELS="0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
    echo "[calibrate] Starting TCP calibration sweep (levels: $CAL_LEVELS)..."

    # Phase A: Baseline ib_write_bw (no interference)
    echo "[calibrate] Measuring baseline BW with ib_write_bw..."
    ssh $SSH_OPTS "$PEER_HOST" \
        "ib_write_bw -d $IB_DEV -i $IB_PORT -p $CAL_PORT -D 5 --report_gbits -s 65536" &
    IB_SERVER_PID=$!
    sleep 2
    BASELINE_OUT=$(ib_write_bw -d "$IB_DEV" -i "$IB_PORT" -p "$CAL_PORT" \
        -D 5 --report_gbits -s 65536 "$PEER_IP" 2>/dev/null) || true
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
        CAL_SWEEP_OK=true

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

            # Start NUM_STREAMS ucx_receiver instances on peer
            CAL_RECV_PIDS=()
            for ((si=0; si<NUM_STREAMS; si++)); do
                cal_p=$((CAL_PORT_BASE + si))
                ssh $SSH_OPTS "$PEER_HOST" \
                    "cd \$HOME/$REMOTE_RUNTIME_DIR && $UCX_ENV ./ucx_receiver $cal_p --max-msg 65536" > /dev/null 2>&1 &
                CAL_RECV_PIDS+=($!)
            done
            sleep 1

            # Start NUM_STREAMS ucx_sender instances locally
            CAL_SEND_PIDS=()
            for ((si=0; si<NUM_STREAMS; si++)); do
                cal_p=$((CAL_PORT_BASE + si))
                env $UCX_ENV "$ROOT_DIR/ucx_sender" "$PEER_IP" "$cal_p" \
                    --schedule "$CAL_SCHEDULE" --duration 15 > /dev/null 2>&1 &
                CAL_SEND_PIDS+=($!)
            done

            # Wait for TCP to stabilize
            sleep 3

            # Measure with ib_write_bw
            ssh $SSH_OPTS "$PEER_HOST" \
                "ib_write_bw -d $IB_DEV -i $IB_PORT -p $CAL_PORT -D $CAL_IB_DURATION --report_gbits -s 65536" &
            IB_SERVER_PID=$!
            sleep 1
            LOADED_OUT=$(ib_write_bw -d "$IB_DEV" -i "$IB_PORT" -p "$CAL_PORT" \
                -D "$CAL_IB_DURATION" --report_gbits -s 65536 "$PEER_IP" 2>/dev/null) || true
            wait $IB_SERVER_PID 2>/dev/null || true

            B_LOADED=$(echo "$LOADED_OUT" | grep -E '^\s+[0-9]+' | awk '{print $4}')

            # Stop calibration interference
            for pid in "${CAL_SEND_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
            for pid in "${CAL_SEND_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
            ssh $SSH_OPTS "$PEER_HOST" \
                "pkill -f 'ucx_receiver' >/dev/null 2>&1 || true" 2>/dev/null || true
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
    echo "[run] Using calibration curve: $CALIBRATION_CSV"
elif [ -n "$TCP_OVERHEAD" ]; then
    echo "[run] Using uniform TCP overhead factor: $TCP_OVERHEAD"
else
    echo "[run] No TCP calibration (rates used as-is)"
fi

# Rewrite passthrough args: replace local file paths with remote basenames
REMOTE_ARGS=()
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [ -f "$arg" ]; then
        REMOTE_ARGS+=("$(basename "$arg")")
    else
        REMOTE_ARGS+=("$arg")
    fi
done

# Launch server (ucx_receiver) on PEER via SSH (background)
REMOTE_CSV="recv_bw.csv"
echo "[run] Starting server (ucx_receiver) on $PEER_HOST..."
ssh $SSH_OPTS "$PEER_HOST" \
    "cd \$HOME/$REMOTE_RUNTIME_DIR && python3 interfere.py --role server --peer-ip $HEAD_IP \
    --ucx-transport $UCX_TRANSPORT \
    --num-streams $NUM_STREAMS --csv $REMOTE_CSV ${REMOTE_ARGS[*]}" &
SERVER_PID=$!

# Give server time to bind ports
sleep 3

# Launch client (ucx_sender + schedule generation) LOCALLY (background)
echo "[run] Starting client (ucx_sender) locally..."
CAL_ARGS=(--num-streams "$NUM_STREAMS")
if [ -n "$CALIBRATION_CSV" ] && [ -f "$CALIBRATION_CSV" ]; then
    CAL_ARGS+=(--calibration-csv "$CALIBRATION_CSV")
elif [ -n "$TCP_OVERHEAD" ]; then
    CAL_ARGS+=(--tcp-overhead "$TCP_OVERHEAD")
fi
python3 "$ROOT_DIR/interfere.py" --role client --peer-ip "$PEER_IP" \
    --ucx-transport "$UCX_TRANSPORT" \
    "${CAL_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}" &
CLIENT_PID=$!

echo "[run] Server SSH PID=$SERVER_PID, Client PID=$CLIENT_PID"

# Optionally launch BW profiler
PROFILE_PID=""
if $PROFILE; then
    PROFILE_ARGS=(--peer-host "$PEER_HOST" --peer-ip "$PEER_IP" --head-ip "$HEAD_IP")
    if [ -n "$PROFILE_OUTPUT" ]; then
        PROFILE_ARGS+=(--output "$PROFILE_OUTPUT")
    fi
    echo "[run] Starting BW profiler..."
    python3 "$ROOT_DIR/profile_bw.py" "${PROFILE_ARGS[@]}" &
    PROFILE_PID=$!
    echo "[run] Profiler PID=$PROFILE_PID"
fi

echo "[run] Press Ctrl+C to stop."

# --- Periodic ib_write_bw monitoring every 30s ---
# Measures actual interference by running ib_write_bw alongside the
# interference traffic and comparing to the calibration baseline.
IB_MONITOR_INTERVAL=30
IB_MONITOR_DURATION=3
IB_MONITOR_PORT=$((CAL_PORT + 1))  # 18601, separate from calibration port

# Retrieve baseline from calibration (if available)
MONITOR_BASELINE="${B_BASELINE:-}"

measure_interference() {
    local label="$1"
    # Run ib_write_bw to measure available BW
    ssh $SSH_OPTS "$PEER_HOST" \
        "ib_write_bw -d $IB_DEV -i $IB_PORT -p $IB_MONITOR_PORT -D $IB_MONITOR_DURATION --report_gbits -s 65536" &
    local ib_srv_pid=$!
    sleep 1
    local ib_out
    ib_out=$(ib_write_bw -d "$IB_DEV" -i "$IB_PORT" -p "$IB_MONITOR_PORT" \
        -D "$IB_MONITOR_DURATION" --report_gbits -s 65536 "$PEER_IP" 2>/dev/null) || true
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

# If we don't have a baseline yet (calibration was skipped), measure one now
if [ -z "$MONITOR_BASELINE" ]; then
    # Quick baseline (both sender/receiver are running but may still be starting)
    echo "[monitor] No calibration baseline available. Measuring baseline in 5s..."
    sleep 5
fi

# Periodic monitoring loop — also checks if processes are still alive
while true; do
    for ((s=0; s<IB_MONITOR_INTERVAL; s++)); do
        # Check if either process has exited
        if ! kill -0 "$SERVER_PID" 2>/dev/null || ! kill -0 "$CLIENT_PID" 2>/dev/null; then
            break 2
        fi
        sleep 1
    done
    # Both still running — measure interference
    if kill -0 "$SERVER_PID" 2>/dev/null && kill -0 "$CLIENT_PID" 2>/dev/null; then
        measure_interference "monitor"
    else
        break
    fi
done

# If profiler is running, stop it
if [ -n "$PROFILE_PID" ] && kill -0 "$PROFILE_PID" 2>/dev/null; then
    kill "$PROFILE_PID" 2>/dev/null || true
    wait "$PROFILE_PID" 2>/dev/null || true
fi

# Retrieve receiver CSV from peer
echo "[run] Retrieving receiver CSV from $PEER_HOST..."
LOCAL_CSV="$ROOT_DIR/recv_bw_$(date +%Y%m%d_%H%M%S).csv"
scp $SCP_OPTS "$PEER_HOST:$REMOTE_RUNTIME_DIR/$REMOTE_CSV" "$LOCAL_CSV" 2>/dev/null && \
    echo "[run] Receiver CSV: $LOCAL_CSV" || \
    echo "[run] WARNING: could not retrieve receiver CSV"
