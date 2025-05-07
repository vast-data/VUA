## Testing KV Connector V1

First, let's make sure we can execute a standalone vLLM that uses the VUA KV Connector V1:

```
bin/vua-vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --api-key fke9dfkjw9rjqw94rtj29 \
    --kv-transfer-config '{"kv_connector":"VUAStorageConnector_V1","kv_role":"kv_both","kv_connector_extra_config": {"shared_storage_path": "/mnt/nfsrdma"}}'
    --no-enable-prefix-caching \
    --max-num-batched-tokens 30720 \
```

Notes:

- This configuration disables prefix caching, so we can see the time different between external KV cache fetch and prefill stages.
- I used two GPUs. You may try a different model with a single GPU too.
- Storage path configured here is NFS with [VAST NFS](https://vastnfs.vastdata.com/docs/4.0/index.html), but you can use any file system that supports GDS, or a local file system that does not support GDS.
- Here we pass `--max-num-batched-tokens 30720` so that the KV cache for the query we are doing is split to 4 parts token-wise.
- We will test with a 126,381 token query.


First query will take time to write KV cache to the storage path because the cache is empty (and write path is not yet optimized):

```
$ ./example/test-latency.py 127.0.0.1:8080
...
Response time: 14.83 secs

$ ls -l /mnt/nfsrdma/
total 0
drwxr-xr-x 2 user user 4096 May  6 17:09 0%4f440dd10e777d82a6fc3bf9635748e3140a0b2f
drwxr-xr-x 2 user user 4096 May  6 17:09 0%6d88220f2b3e743b6328f93a84c75f98146ca139
drwxr-xr-x 2 user user 4096 May  6 17:09 0%8f02a7a5c925152aea58542866db6580982538d0
drwxr-xr-x 2 user user 4096 May  6 17:09 0%a2d3912ffbdd127f39f2eb20afcb97f9ac0f9928
drwxr-xr-x 2 user user 4096 May  6 17:09 1%4f440dd10e777d82a6fc3bf9635748e3140a0b2f
drwxr-xr-x 2 user user 4096 May  6 17:09 1%6d88220f2b3e743b6328f93a84c75f98146ca139
drwxr-xr-x 2 user user 4096 May  6 17:09 1%8f02a7a5c925152aea58542866db6580982538d0
drwxr-xr-x 2 user user 4096 May  6 17:09 1%a2d3912ffbdd127f39f2eb20afcb97f9ac0f9928

```

Further queries will utilize KV cache from remote storage:

```
$ ./example/test-latency.py 127.0.0.1:8080
...
Response time: 3.648 secs
```

You can test prefill time by temporarily disabling the use of external KV cache:

```
$ touch /mnt/nfsrdma/disable-get
$ ./example/test-latency.py 127.0.0.1:8080
...
Response time: 12.062 secs
```
