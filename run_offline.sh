#!/bin/sh

# MODEL_NAME=facebook/opt-6.7B
# MODEL_NAME=mosaicml/mpt-7b
# MODEL_NAME=openlm-research/open_llama_7b_v2
# MODEL_NAME=openlm-research/open_llama_3b_v2
MODEL_NAME=mosaicml/mpt-7b-chat
TOKENIZER=EleutherAI/gpt-neox-20b
# MODEL_NAME=meta-llama/Llama-2-7b

# TOKENIZERS_PARALLELISM=true python benchmarks/benchmark_throughput.py \
#     --backend vllm \
#     --tokenizer $MODEL_NAME --dataset data/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --tensor-parallel-size 1 \
#     --n 1 \
#     --num-prompts 10 \
#     --trust-remote-code --seed 0

TOKENIZERS_PARALLELISM=true python benchmarks/benchmark_latency.py \
    --model $MODEL_NAME \
    --tokenizer $TOKENIZER \
    --tensor-parallel-size 1 \
    --input-len 32 \
    --output-len 128 \
    --batch-size 1 \
    --n 1 \
    --num-iters 20 \
    --use-dummy-weights \
    --trust-remote-code
