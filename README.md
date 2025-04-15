# VUA

VUA is a system for storing and retrieving key-value caches of deep learning models, utilizing a directory structure derived from token values.

## Developer Quick Start

1. **Understanding VUA**

   VUA splits tokens into groups defined by a fixed split factor (see `VUAConfig.split_factor`). Tokens are converted to directory names where cache data is stored. The `put()` method splits key-value caches and stores fragmented tensor data into these directories, while `get_closest()` fetches the closest matching cache fragment based on supplied token prefixes.

2. **Exploring the Codebase**

   - **`src/vua/core.py`**: Contains the primary VUA implementation with `put()` and `get_closest()`, plus token-to-path conversion logic.
   - **`src/vua/serdes.py`**: Provides tensor serialization/deserialization functions (`tensor_to_bytes` and `bytes_to_tensor`) for efficiently handling PyTorch tensors. Compatible to safetensors.
   - **`tests/test_vua.py`**: Offers tests to validate token processing, cache storage and retrieval, and tensor serialization.

3. **Running the Tests**

   Run the tests by executing:
   ```bash
   uv run python -m unittest discover -s tests
   ```
4. **Using VUA in Your Project**

   - Create a VUA instance by providing a configuration (e.g. `VUAConfig`) and a directory path for cache storage.
   - Utilize `put()` to store computed key-value caches and `get_closest()` to retrieve cached data based on token queries.
   - **Batched Operations:** VUA supports batched put and get operations. If you provide a 2D tensor of tokens to `put()`, it processes each sequence in parallel. Similarly, calling `get_closest()` with a list of token tensors returns a list of corresponding `ClosestKV` results.

   **Example:**

   ```python
   import os
   import torch
   from vua.core import VUA, VUAConfig

   # Set up cache storage directory
   cache_dir = "./vua_cache"
   os.makedirs(cache_dir, exist_ok=True)

   # Create a VUA instance
   vua = VUA(VUAConfig, cache_dir)

   # Generate sample tokens ensuring the length is divisible by split_factor
   tokens = torch.randint(0, 0xFFFF, (1, 512), dtype=torch.uint16)
   trimmed_tokens = VUAConfig.trim_to_split_factor(tokens)

   # Create a sample key-value cache (for demonstration purposes)
   # This is a list with one layer and two tensor pairs (keys and values)
   kvcache = [[torch.randn(1, 2, trimmed_tokens.size(1), 64), torch.randn(1, 2, trimmed_tokens.size(1), 64)]]

   # Store the cache using put()
   vua.put(trimmed_tokens, kvcache)

   # Retrieve the cache using get_closest()
   result = vua.get_closest(trimmed_tokens, device="cpu")
   print("Retrieved tokens:", result.tokens)
   print("Retrieved data:", result.data)
   ```

5. **Examples directory**

   The examples directory contains two main use cases:

   - Usage with 'transformers' library

   ```
   uv run pyth./example/on-transformers.py
   ```

   - Usage to experimental vLLM connector for offline kvcache storage

   ```
   uv run ./serve-vllm.sh
   ```


6. **Debugging and Logging**

   VUA leverages Python’s `logging` module for detailed debug output. Configure custom log handlers during development to monitor directory navigation and cache operations effectively.
