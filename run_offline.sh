#!/bin/sh

# MODEL_NAME=facebook/opt-6.7B
# MODEL_NAME=mosaicml/mpt-7b
# MODEL_NAME=NousResearch/Nous-Hermes-Llama2-13b

# MODEL_NAME=NousResearch/Llama-2-7b-hf
MODEL_NAME=NousResearch/Llama-2-13b-hf
# MODEL_NAME=NousResearch/Llama-2-70b-hf
TOKENIZER=${MODEL_NAME}
# TOKENIZER="hf-internal-testing/llama-tokenizer"
# TOKENIZER="meta-llama/Llama-2-7b-chat-hf"
# MODEL_NAME=meta-llama/Llama-2-7b
OUTPUT_FILE="latency_llama-2-13b.csv"

# TOKENIZERS_PARALLELISM=true python benchmarks/benchmark_throughput.py \
#     --backend vllm \
#     --tokenizer $MODEL_NAME --dataset data/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --tensor-parallel-size 1 \
#     --n 1 \
#     --num-prompts 10 \
#     --trust-remote-code --seed 0

# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# for i in 1 2 4 8 16 32 64; do
for i in 1; do
    echo "Running latency benchmark for batch size ${i}"
    TOKENIZERS_PARALLELISM=true python3 benchmarks/benchmark_latency.py \
        --model $MODEL_NAME \
        --tokenizer $TOKENIZER \
        --tensor-parallel-size 2 \
        --n 1 \
        --input-len 128 \
        --output-len 128 \
        --batch-size ${i} \
        --num-iters 1 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --output-file ${OUTPUT_FILE}
done
        # --use-dummy-weights \
