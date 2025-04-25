#!/usr/bin/bash

prefix=gqa_top2

mkdir -p "$prefix"_short
python benchmark/scripts/benchsuite.py  "short"
mv $prefix/* "$prefix"_short

mkdir -p "$prefix"_medium
python benchmark/scripts/benchsuite.py "medium"
mv $prefix/* "$prefix"_medium

mkdir -p "$prefix"_reasonable
python benchmark/scripts/benchsuite.py "reasonable"
mv $prefix/* "$prefix"_reasonable