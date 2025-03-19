curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 20,
            "time": 120,
            "distribution": "poisson"
        }'



