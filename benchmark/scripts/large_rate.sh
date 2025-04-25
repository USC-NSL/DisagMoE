# curl -X POST http://localhost:6699/run_once \
#         -H "Content-Type: application/json" \
#         -d '{
#             "rate": 500,
#             "time": 120,
#             "distribution": "poisson"
#         }'

curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 1000,
            "time": 120,
            "distribution": "poisson"
        }'

# curl -X POST http://localhost:6699/run_once \
#         -H "Content-Type: application/json" \
#         -d '{
#             "rate": 1500,
#             "time": 120,
#             "distribution": "poisson"
#         }'

curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 2000,
            "time": 120,
            "distribution": "poisson"
        }'

curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 2500,
            "time": 120,
            "distribution": "poisson"
        }'

# curl -X POST http://localhost:6699/run_once \
#         -H "Content-Type: application/json" \
#         -d '{
#             "rate": 3000,
#             "time": 120,
#             "distribution": "poisson"
#         }'
