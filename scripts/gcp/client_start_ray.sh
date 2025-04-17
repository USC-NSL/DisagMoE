#!/bin/bash

# update the MASTER address each time the master is rebuilt
export MASTER=10.148.0.11

bash client.sh /opt/conda/bin/conda run -n dmoe ray start --address="$MASTER:6379" --num-cpus=7