curl -X POST http://localhost:6699/set_schedule \
        -H "Content-Type: application/json" \
        -d '{
            "policy": "mbfs",
            "step": 1
        }'

curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 20,
            "time": 60,
            "distribution": "poisson"
        }'


for i in {1..6}; do
    step=$((2 ** (i - 1)))
    curl -X POST http://localhost:6699/set_schedule \
        -H "Content-Type: application/json" \
        -d "{
            \"policy\": \"mbflfs\",
            \"step\": $step
        }"

    curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 20,
            "time": 60,
            "distribution": "poisson"
        }'
done


curl -X POST http://localhost:6699/set_schedule \
        -H "Content-Type: application/json" \
        -d '{
            "policy": "flfs",
            "step": 1
        }'

curl -X POST http://localhost:6699/run_once \
        -H "Content-Type: application/json" \
        -d '{
            "rate": 20,
            "time": 60,
            "distribution": "poisson"
        }'