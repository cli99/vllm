#!/bin/sh

python -m vllm.entrypoints.api_server \
    --model facebook/opt-125m --swap-space 8 \
    --use-dummy-weights \
    --disable-log-requests
