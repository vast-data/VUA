#!/bin/bash

set -eu

api_key=fke9dfkjw9rjqw94rtj29

vllm serve meta-llama/Llama-3.2-3B-Instruct --gpu-memory-utilization 0.85 --port 6677  --tensor-parallel-size 2 --api-key ${api_key} --kv-transfer-config '{"kv_connector":"SharedStorageConnector","kv_role":"kv_both","kv_connector_extra_config": {"shared_storage_path": "local_storage"}}' --no-enable-prefix-caching --max-num-batched-tokens 131072
