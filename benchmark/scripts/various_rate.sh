#!/usr/bin/bash

MIN_INPUT_LEN=10
MAX_INPUT_LEN=11
MIN_OUTPUT_LEN=50
MAX_OUTPUT_LEN=51

for rate in 1000 2000 3000 4000 5000; do

curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 1,
            "time": 1,
            "distribution": "poisson",
            "min_input_len": '$MIN_INPUT_LEN',
            "max_input_len": '$MAX_INPUT_LEN',
            "min_output_len": '$MIN_OUTPUT_LEN',
            "max_output_len": '$MAX_OUTPUT_LEN'
        }'

curl -X POST http://localhost:6699/run_once \
    -H "Content-Type: application/json" \
    -d "{
        \"rate\": $rate,
        \"time\": 120,
        \"distribution\": \"poisson\",
        \"min_input_len\": $MIN_INPUT_LEN,
        \"max_input_len\": $MAX_INPUT_LEN,
        \"min_output_len\": $MIN_OUTPUT_LEN,
        \"max_output_len\": $MAX_OUTPUT_LEN
    }"

done


