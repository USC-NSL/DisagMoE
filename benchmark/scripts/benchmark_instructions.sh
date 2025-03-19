curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 10,
            "time": 10,
            "distribution": "poisson"
        }'



