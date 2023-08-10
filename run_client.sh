#!/bin/sh

python benchmarks/benchmark_serving.py \
    --backend vllm \
    --tokenizer facebook/opt-125m --dataset data/ShareGPT_V3_unfiltered_cleaned_split.json
