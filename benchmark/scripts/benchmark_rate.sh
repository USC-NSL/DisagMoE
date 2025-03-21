curl -X POST http://localhost:6699/set_schedule \
        -H "Content-Type: application/json" \
        -d '{
            "policy": "mbfs",
            "step": 1
        }'

for i in {0..10}; do
    rate=$((1 + i * 3))

    curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": '$rate',
            "time": 300,
            "distribution": "poisson"
        }'
done