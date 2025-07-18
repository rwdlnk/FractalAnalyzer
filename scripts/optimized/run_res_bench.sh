#!/bin/bash

# Run the benchmark with your available data
python3 resolution_benchmark_optimized.py \
    --base-dirs /media/rod/5B83-F7CA/Data/svofRuns/Dalziel/1600x1600 \
    --target-time 0.0 \
    --output ./benchmark_results \
    --mixing-method dalziel \
    --h0 0.5
