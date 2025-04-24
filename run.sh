#!/usr/bin/bash

prefix=gqa_top2

mkdir -p "$prefix"_short
python benchmark/scripts/benchsuite.py  "short"
mv $prefix/* "$prefix"_short

mkdir -p "$prefix"_reasonable
python benchmark/scripts/benchsuite.py "reasonable"
mv $prefix/* "$prefix"_reasonable

mkdir -p "$prefix"_reasonable_v2
python benchmark/scripts/benchsuite.py "reasonable_v2"
mv $prefix/* "$prefix"_reasonable_v2