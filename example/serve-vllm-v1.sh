#!/bin/bash

set -eu

api_key=fke9dfkjw9rjqw94rtj29

SCRIPTPATH=$(dirname $(realpath ${BASH_SOURCE}))

kv_transfer_config=(
    --kv-transfer-config '{"kv_connector":"VUAStorageConnector_V1","kv_role":"kv_both","kv_connector_extra_config": {"shared_storage_path": "local_storage"}}'
)

VLLM_USE_V1=1 python "${SCRIPTPATH}/../bin/vua-vllm" serve meta-llama/Llama-3.2-3B-Instruct \
    --gpu-memory-utilization 0.85 --port 6677 \
    --tensor-parallel-size 2 \
    --api-key ${api_key} \
    "${kv_transfer_config[@]}" \
    --no-enable-prefix-caching \
    "$@"
