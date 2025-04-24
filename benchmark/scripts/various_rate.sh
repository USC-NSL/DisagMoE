for rate in 100 200 400 600 800 1200 1600; do

curl -X POST http://localhost:6699/run_once \
    -H "Content-Type: application/json" \
    -d "{
        \"rate\": $rate,
        \"time\": 120,
        \"distribution\": \"poisson\"
    }"

done


