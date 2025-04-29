#!/usr/bin/bash

# sanity check
curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 1,
            "time": 1,
            "distribution": "poisson"
        }'


# curl -X POST http://localhost:6699/run_once \
#         -H "Content-Type: application/json" \
#         -d '{
#             "rate": 100,
#             "time": 300,
#             "distribution": "incremental_poisson"
#         }'




