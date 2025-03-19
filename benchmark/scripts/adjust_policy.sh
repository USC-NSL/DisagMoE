curl -X POST http://localhost:6699/set_schedule \
        -H "Content-Type: application/json" \
        -d '{
            "policy": "flfs",
            "step": 32
        }'