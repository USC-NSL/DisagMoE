#!/usr/bin/bash

prefix=mqa_top1

mkdir -p "$prefix"_short
python benchmark/scripts/benchsuite.py  "short"
mv $(ls -d mqa_top1/*) "$prefix"_short

mkdir -p "$prefix"reasonable
python benchmark/scripts/benchsuite.py "reasonable"
mv $(ls -d mqa_top1/*) "$prefix"reasonable

mkdir -p "$prefix"reasonable_v2
python benchmark/scripts/benchsuite.py "reasonable_v2"
mv $(ls -d mqa_top1/*) "$prefix"reasonable_v2

