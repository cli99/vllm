#!/bin/sh

MODEL_NAME=~/llama2-7b-hf

python benchmarks/benchmark_serving.py \
    --backend vllm \
    --tokenizer $MODEL_NAME \
    --dataset data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --seed 0
