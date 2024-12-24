#!/bin/bash

# update this list each time the instance group is updated
declare -a clients=("02bb" "78b7" "jglc" "jk81")

for client in "${clients[@]}"
do
        gcloud compute ssh "disag-group-$client" --zone=asia-southeast1-c -- $@ &
done

wait