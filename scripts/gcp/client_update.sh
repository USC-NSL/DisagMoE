#!/bin/bash

export WORK_DIR=/home/$(whoami)/DisagMoE

bash client.sh "cd $WORK_DIR; /opt/conda/bin/conda run -n dmoe make pip"