#!/usr/bin/bash

curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 250,
            "time": 20,
            "distribution": "poisson",
            "min_input_len": 200,
            "max_input_len": 500,
            "min_output_len": 100,
            "max_output_len": 300
        }'

# curl -X POST http://localhost:6699/run_once \
#         -H "Content-Type: application/json" \
#         -d '{
#             "rate": 100,
#             "time": 300,
#             "distribution": "incremental_poisson"
#         }'