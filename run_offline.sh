#!/bin/sh

TOKENIZERS_PARALLELISM=true python benchmarks/benchmark_throughput.py \
    --backend vllm \
    --tokenizer facebook/opt-125m --dataset data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --tensor-parallel-size 1 \
    --n 1 \
    --num-prompts 10 \
    --trust-remote-code
