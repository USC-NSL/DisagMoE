#!/usr/bin/bash

prefix=mqa_top1_

mkdir -p "{$prefix}_short"
python benchmark/scripts/benchsuite.py  "short"
mv $(ls -d mqa_top1/*) "{$prefix}_short"

mkdir -p "{$prefix}_reasonable"
python benchmark/scripts/benchsuite.py "reasonable"
mv $(ls -d mqa_top1/*) "{$prefix}_reasonable"

mkdir -p "{$prefix}_reasonable_v2"
python benchmark/scripts/benchsuite.py "reasonable_v2"
mv $(ls -d mqa_top1/*) "{$prefix}_reasonable_v2"

