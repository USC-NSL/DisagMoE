#!/usr/bin/bash
# lib/benchmark.sh — Benchmark request helper
# Source this file; do not execute directly.
#
# Requires (from config.sh):
#   SERVER_PORT, BENCH_CURL_TIMEOUT_SHAREGPT, BENCH_CURL_TIMEOUT_GSM8K
#   BENCH_RATE, BENCH_TIME, BENCH_GENERATOR, BENCH_DATASET_PATH, BENCH_MAX_CONTEXT_LEN
#   BENCH_MIN_IN, BENCH_MAX_IN, BENCH_MIN_OUT, BENCH_MAX_OUT

log_bench() { echo "$(date '+%Y-%m-%d %H:%M:%S') [bench] $*"; }

# run_benchmark <result_json_path> <cmd_file_path>
#   Builds the curl request, saves it to <cmd_file_path>, then POSTs to
#   /run_once and writes the response JSON to <result_json_path>.
#   Returns 0 on HTTP 200, 1 otherwise.
run_benchmark() {
    local result_file="$1"
    local cmd_file="$2"

    local curl_timeout
    if [[ "${BENCH_DATASET_PATH:-}" == *gsm8k* ]]; then
        curl_timeout="${BENCH_CURL_TIMEOUT_GSM8K:-300}"
    else
        curl_timeout="${BENCH_CURL_TIMEOUT_SHAREGPT:-600}"
    fi

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

    # Save the exact curl command for reproducibility / manual replay
    {
        printf '# Benchmark command\n'
        printf '# Generated: %s\n\n' "$(date)"
        printf 'curl -s \\\n'
        printf '    -o %q \\\n' "$result_file"
        printf '    -w "%%{http_code}" \\\n'
        printf '    -X POST "http://localhost:%s/run_once" \\\n' "$SERVER_PORT"
        printf '    -H "Content-Type: application/json" \\\n'
        printf "    -d '%s' \\\n" "$payload"
        printf '    --max-time %s\n' "$curl_timeout"
    } > "$cmd_file"

    local http_code
    # Unset LD_LIBRARY_PATH for curl: env.sh's conda paths conflict with
    # the system libldap (OpenSSL version mismatch), silently breaking curl.
    http_code=$(env -i HOME="$HOME" PATH="$PATH" curl -s \
        -o "$result_file" \
        -w "%{http_code}" \
        -X POST "http://localhost:${SERVER_PORT}/run_once" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time "$curl_timeout")

    if [ "$http_code" = "200" ]; then
        log_bench "Benchmark complete (HTTP 200). Result: $result_file"
        return 0
    else
        log_bench "ERROR: HTTP $http_code (000 = curl timeout after ${curl_timeout}s)."
        printf '{"error":"http_%s"}\n' "$http_code" > "$result_file"
        return 1
    fi
}
