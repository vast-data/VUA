#!/bin/bash

set -eu

api_key=fke9dfkjw9rjqw94rtj29
model=${SERVE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}

export PYTHONPATH=$(pwd)/example

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python example/vllm-wrap.py serve ${model} --cpu-offload-gb 15 --api-key ${api_key} \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"RemoteStorageConnector","kv_role":"kv_both","kv_rank":1,"kv_parallel_size":1,"kv_buffer_size":5e9}'
