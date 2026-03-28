#!/usr/bin/bash
# lib/benchmark.sh — Benchmark request helper
# Source this file; do not execute directly.
#
# Requires (from config.sh):
#   SERVER_PORT, BENCH_CURL_TIMEOUT
#   BENCH_RATE, BENCH_TIME, BENCH_GENERATOR, BENCH_DATASET_PATH, BENCH_MAX_CONTEXT_LEN
#   BENCH_MIN_IN, BENCH_MAX_IN, BENCH_MIN_OUT, BENCH_MAX_OUT

log_bench() { echo "$(date '+%Y-%m-%d %H:%M:%S') [bench] $*"; }

# run_benchmark <result_json_path> <cmd_file_path>
run_benchmark() {
    local result_file="$1"
    local cmd_file="$2"

    log_bench "Sending benchmark:" \
        "rate=${BENCH_RATE} rps, time=${BENCH_TIME}s," \
        "generator=${BENCH_GENERATOR}, dataset=${BENCH_DATASET_PATH:-none}," \
        "max_seq_len=${BENCH_MAX_CONTEXT_LEN:-none}," \
        "in=${BENCH_MIN_IN}-${BENCH_MAX_IN}, out=${BENCH_MIN_OUT}-${BENCH_MAX_OUT}"

    local payload
    payload=$(printf '{
    "rate": %d,
    "time": %d,
    "distribution": "%s",
    "dataset_path": "%s",
    "dataset_max_context_len": %s,
    "min_input_len": %d,
    "max_input_len": %d,
    "min_output_len": %d,
    "max_output_len": %d
}' "$BENCH_RATE" "$BENCH_TIME" \
   "$BENCH_GENERATOR" "$BENCH_DATASET_PATH" "${BENCH_MAX_CONTEXT_LEN:-null}" \
   "$BENCH_MIN_IN" "$BENCH_MAX_IN" \
   "$BENCH_MIN_OUT" "$BENCH_MAX_OUT")

    {
        printf '# Benchmark command\n'
        printf '# Generated: %s\n\n' "$(date)"
        printf 'curl -s \\\n'
        printf '    -o %q \\\n' "$result_file"
        printf '    -w "%%{http_code}" \\\n'
        printf '    -X POST "http://localhost:%s/run_once" \\\n' "$SERVER_PORT"
        printf '    -H "Content-Type: application/json" \\\n'
        printf "    -d '%s' \\\n" "$payload"
        printf '    --max-time %s\n' "$BENCH_CURL_TIMEOUT"
    } > "$cmd_file"

    local http_code
    http_code=$(curl -s \
        -o "$result_file" \
        -w "%{http_code}" \
        -X POST "http://localhost:${SERVER_PORT}/run_once" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time "$BENCH_CURL_TIMEOUT")

    if [ "$http_code" = "200" ]; then
        log_bench "Benchmark complete (HTTP 200). Result: $result_file"
        return 0
    else
        log_bench "ERROR: HTTP $http_code (000 = curl timeout after ${BENCH_CURL_TIMEOUT}s)."
        printf '{"error":"http_%s"}\n' "$http_code" > "$result_file"
        return 1
    fi
}
