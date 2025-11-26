#!/usr/bin/bash

PROFILE_START_MIN_BATCH_SIZE=200
PROFILE_NUM_STEPS=20
PROFILE_DIR="torch_profile"

curl -X POST http://localhost:6699/init_profile \
  -H 'Content-Type: application/json' \
  -d "{
        \"profile_start_min_batch_size\": $PROFILE_START_MIN_BATCH_SIZE,
        \"profile_num_steps\": $PROFILE_NUM_STEPS,
        \"profile_dir\": \"$PROFILE_DIR\"
      }"