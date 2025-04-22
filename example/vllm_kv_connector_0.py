"""A template connector for remote KV storage that does nothing.

This connector always returns query misses and stores no KV cache data.

Based on some code from LMCache.
"""

from typing import TYPE_CHECKING, List, Tuple, Union
import torch
import logging
from copy import deepcopy
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.sequence import IntermediateTensors
from torch.nn.utils.rnn import pad_sequence
from vllm.config import CacheConfig
from vllm import _custom_ops as ops
import os

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = logging.getLogger(__name__)

vuacache = None

read_only_cache = os.getenv("VUA_READ_ONLY_CACHE", "") == "1"
skip_cache = os.getenv("VUA_SKIP_CACHE", "") == "1"

class RemoteStorageConnector(KVConnectorBase):
    """
    A template connector for remote KV storage.

    This connector does not actually store any KV cache data and always returns a miss.
    """

    def __init__(self, rank: int, local_rank: int, config):
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.cache_config = config.cache_config

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input,
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = getattr(model_config, "head_dim",
                            int(hidden_size // num_attention_heads))

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You have some decode requests while using "
                               "SimpleConnector. Their KVCache won't be sent.")
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            kvcache_layers = []
            for layer_idx, layer_id in enumerate(range(start_layer, end_layer)):
                kv_cache = kv_caches[layer_id - start_layer]
                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                # Index directly into pre-allocated memory (no extra allocations):
                k = key_cache[current_slot_mapping].permute(1, 0, 2).unsqueeze(0)
                v = value_cache[current_slot_mapping].permute(1, 0, 2).unsqueeze(0)
                kvcache_layers.append((k, v))
                del k
                del v

            prefix_trimmed = vuacache.config().trim_to_split_factor(current_tokens)
            if not read_only_cache:
                vuacache.put(prefix_trimmed, kvcache_layers)

            #
            # Currently ignoring:
            #
            # hidden_or_intermediate_states[start_pos:end_pos].size())
            logger.warning(f"HIDDEN {hidden_or_intermediate_states.size()}")
        logger.warning("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        model_input, bypass_model_exec, hidden_or_intermediate_states = retrieve_kv(
            model_executable, model_input, self.cache_config, kv_caches, self.tp_size)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self) -> None:
        # Nothing to close for remote storage.
        pass


def retrieve_kv(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    cache_config: CacheConfig,
    kv_caches: List[torch.Tensor],
    tp_size: int,
) -> Tuple["ModelInputForGPUWithSamplingMetadata", bool, Union[
        torch.Tensor, IntermediateTensors]]:
    """Retrieve the KV caches from VUA for the current model_input. And
    rebuild the model_input to reflect the changes in KV if necessary.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to put KV to
    :type kv_caches: List[torch.Tensor]

    :return: The rebuilt model_input to reflect the changes in KV.
    :return: The boolean value to indicate whether the
             entire execute_model should be skipped
    """

    from vllm.attention.backends.flash_attn import FlashAttentionMetadata
    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    query_start_loc = model_input.attn_metadata.query_start_loc
    assert query_start_loc is not None
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    assert slot_mapping is not None
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    # The following metadata are needed to rebuilt the model input
    full_tokens_list = []
    num_computed_tokens_list = []
    lmc_num_computed_tokens_list = []

    start_pos_list = []
    is_prefill_list = []

    do_sample_list = []

    next_start_pos = 0
    num_request_not_found = 0

    # idx is on a sequence, not a sequence group.
    idx = 0

    assert model_input.sampling_metadata is not None
    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    start_layer = model_executable.model.start_layer
    end_layer = model_executable.model.end_layer

    model_config = model_executable.model.config
    hidden_size = model_config.hidden_size
    num_attention_heads = model_config.num_attention_heads
    num_heads = int(model_config.num_key_value_heads / tp_size)
    head_size = getattr(model_config, "head_dim",
                        int(hidden_size // num_attention_heads))

    chunk_prefill_full_hit = True
    for seq_group in seq_group_list:
        seq_ids = seq_group.seq_ids
        for seq_id in seq_ids:
            seq_data = seq_group.seq_data[seq_id]
            is_prefill_list.append(seq_group.is_prompt)
            total_seq_len = seq_lens[idx]
            do_sample_list.append(True)

            full_token_tensor = torch.tensor(
                seq_data.get_token_ids()[:total_seq_len], device="cpu")
            full_tokens_list.append(full_token_tensor)

            vllm_num_required_tokens = (query_start_loc[idx + 1] -
                                        query_start_loc[idx]).item()
            assert isinstance(vllm_num_required_tokens, int)

            start_pos = next_start_pos
            end_pos = start_pos + vllm_num_required_tokens
            next_start_pos = end_pos
            start_pos_list.append(start_pos)

            # number of tokens already computed by vllm
            # (e.g., chunk prefill, prefix caching)
            vllm_num_computed_tokens = total_seq_len - vllm_num_required_tokens

            # construct token mesk to indicate what tokens should be retrieved
            # from lmc. Tokens computed in vllm already should be skipped
            token_mask = torch.ones_like(full_token_tensor, dtype=torch.bool)
            chunk_size = vuacache.config().split_factor
            vllm_num_computed_tokens_align = vllm_num_computed_tokens \
                // chunk_size * chunk_size
            token_mask[:vllm_num_computed_tokens_align] = False

            # TODO(Jiayi): Please get rid of this in the future
            # Please only pass the required slot_mapping to the engine
            if vllm_num_computed_tokens > 0:
                slot_mapping_req_full = torch.full((total_seq_len, ),
                                                   -1,
                                                   device=slot_mapping.device,
                                                   dtype=slot_mapping.dtype)
                slot_mapping_req_full[vllm_num_computed_tokens:] = \
                    slot_mapping[start_pos:end_pos]
            else:
                slot_mapping_req_full = slot_mapping[start_pos:end_pos]

            prefix_trimmed = vuacache.config().trim_to_split_factor(full_token_tensor)
            logger.warning(f"GET {prefix_trimmed} vllm_num_computed_tokens={vllm_num_computed_tokens}")

            if not skip_cache:
                res = vuacache.get_closest(prefix_trimmed, model_input.input_tokens.device)
            else:
                res = None

            if res:
                # Reverse the trimming: assume res is a tensor mask for the trimmed tokens,
                # and align it with the full token tensor. We'll pad the beginning with False.
                kvcache_layers = res.data
                nr_fetched_token_caches = len(res.tokens)
                logger.warning(f"GET hit nr_fetched_token_caches={nr_fetched_token_caches}")

                # Probably a slower (and wrong?) way:
                for layer_idx, layer_id in enumerate(range(start_layer, end_layer)):
                    kv_cache = kv_caches[layer_id - start_layer]
                    key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                    value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                    (k, v) = kvcache_layers[layer_idx]
                    k = k.squeeze(0).permute(1, 0, 2)
                    v = v.squeeze(0).permute(1, 0, 2)
                    slot_slice = slot_mapping_req_full[:nr_fetched_token_caches]
                    key_cache[slot_slice] = k
                    value_cache[slot_slice] = v

                # Probably a faster (and wrong?) way that needs fixing:
                #
                # slot_slice = slot_mapping_req_full[:nr_fetched_token_caches]
                # for layer_idx, layer_id in enumerate(range(start_layer, end_layer)):
                #     layer = model_executable.model.layers[layer_id]
                #     kv_cache = kv_caches[layer_id - start_layer]
                #     key_cache, value_cache = kv_cache[0], kv_cache[1]
                #     (k, v) = kvcache_layers[layer_idx]
                #     ops.reshape_and_cache_flash(
                #         k.squeeze(0).permute(1, 0, 2).to(key_cache.device),
                #         v.squeeze(0).permute(1, 0, 2).to(value_cache.device),
                #         key_cache,
                #         value_cache,
                #         slot_slice,
                #         layer.self_attn.attn.kv_cache_dtype,
                #         layer.self_attn.attn._k_scale,
                #         layer.self_attn.attn._v_scale,
                #     )

                ret_token_mask = torch.cat([
                    torch.full((nr_fetched_token_caches, ),
                                dtype=torch.bool, fill_value=True),
                    torch.full((len(full_token_tensor) - nr_fetched_token_caches, ),
                                dtype=torch.bool, fill_value=False)])
            else:
                ret_token_mask = torch.zeros_like(full_token_tensor, dtype=torch.bool)

            lmc_num_computed_tokens = max(
                    torch.sum(ret_token_mask).item() -
                    (vllm_num_computed_tokens -
                     vllm_num_computed_tokens_align),
                    0
                )

            assert isinstance(lmc_num_computed_tokens, int)

            # total number of computed tokens (vllm + lmc)
            num_computed_tokens = vllm_num_computed_tokens + \
                lmc_num_computed_tokens

            # TODO(Jiayi): currently we do not skip anything if chunked prefill
            # is batched with any decode or other chunked prefills.
            if num_computed_tokens != total_seq_len:
                chunk_prefill_full_hit = False
            else:
                lmc_num_computed_tokens -= 1
                num_computed_tokens -= 1

            num_computed_tokens_list.append(num_computed_tokens)
            lmc_num_computed_tokens_list.append(lmc_num_computed_tokens)

            # No cache found, move on
            if lmc_num_computed_tokens == 0:
                num_request_not_found += 1

            # Inject the lmc retrieved kv cache
            logger.warning(f"Injected token number: {lmc_num_computed_tokens}")

            idx += 1

    seq_cnt = len(query_start_loc) - 1
    assert idx == seq_cnt
    assert len(lmc_num_computed_tokens_list) == seq_cnt
    assert len(num_computed_tokens_list) == seq_cnt

    if chunk_prefill_full_hit:
        num_tok = len(model_input.input_tokens)
        num_dim = model_executable.model.embed_tokens.embedding_dim
        dtype = model_executable.model.embed_tokens.weight.dtype
        device = model_input.input_tokens.device
        hidden_or_intermediate_states = torch.zeros(num_tok,
                                                    num_dim,
                                                    device=device,
                                                    dtype=dtype)
        logger.warning("Skip the entire model forward!")
        return model_input, True, hidden_or_intermediate_states

    if num_request_not_found < seq_cnt:
        rebuilt_model_input = build_partial_prefill_input(
            model_input,
            full_tokens_list,
            num_computed_tokens_list,
            start_pos_list,
            slot_mapping,
            lmc_num_computed_tokens_list,
            is_prefill_list,
            do_sample_list,
            kv_caches[0][0].device,
            cache_config,
        )
        logger.warning("Rebuilt the input!")
        return rebuilt_model_input, False, None

    logger.warning("Returning the original input!")
    return model_input, False, None


def build_partial_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    full_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping_flat: torch.Tensor,
    lmc_num_computed_tokens_list: List[int],
    is_prefill_list: List[bool],
    do_sample_list: List[bool],
    device: torch.device,
    cache_config: CacheConfig,
) -> "ModelInputForGPUWithSamplingMetadata":
    """Helper function to rebuild the model input for the current request.
    """
    assert model_input.attn_metadata is not None
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata
    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."
    assert model_input.attn_metadata.context_lens_tensor is not None
    assert model_input.attn_metadata.block_tables is not None
    assert model_input.attn_metadata.query_start_loc is not None
    assert model_input.input_positions is not None

    rebuilt_input_tokens = []
    rebuilt_input_positions = []
    rebuilt_query_lens = []
    rebuilt_num_prefills = 0
    rebuilt_num_prefill_tokens = 0
    rebuilt_slot_mapping = []
    rebuilt_max_query_len = 0

    rebuilt_block_tables = []

    rebuilt_query_start_loc = [0]
    rebuilt_context_lens_tensor = []
    rebuilt_selected_token_indices = []

    last_query_start_loc = 0

    # recounting query and context lengths
    for idx in range(len(full_tokens_list)):
        token_tensor = full_tokens_list[idx]
        num_token = len(token_tensor)
        num_computed_token = num_computed_tokens_list[idx]
        start_pos = start_pos_list[idx]
        is_prefill = is_prefill_list[idx]
        lmc_num_computed_tokens = lmc_num_computed_tokens_list[idx]
        rebuilt_input_tokens.append(token_tensor[num_computed_token:])
        q_len = num_token - num_computed_token
        assert q_len > 0
        rebuilt_query_lens.append(q_len)
        start_input_pos_idx = start_pos + lmc_num_computed_tokens
        end_input_pos_idx = start_input_pos_idx + q_len
        rebuilt_input_positions.append(
            model_input.input_positions[start_input_pos_idx:end_input_pos_idx])
        # Attn metadata-related
        if is_prefill:
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
        else:
            assert q_len == 1

        start_slot_idx = start_pos + lmc_num_computed_tokens
        end_slot_idx = start_slot_idx + q_len
        new_slot_mapping = slot_mapping_flat[start_slot_idx:end_slot_idx]
        rebuilt_slot_mapping.append(new_slot_mapping)
        rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)

        last_query_start_loc += q_len
        rebuilt_query_start_loc.append(last_query_start_loc)  # start with 0
        rebuilt_context_lens_tensor.append(num_computed_token)

        # recover `block_table`
        if len(model_input.attn_metadata.block_tables[idx]) > 0:
            rebuilt_block_tables.append(
                model_input.attn_metadata.block_tables[idx])
        else:
            slot_mapping_req = slot_mapping_flat[start_pos:end_slot_idx]
            vllm_block_size = cache_config.block_size
            rebuilt_block_table = slot_mapping_req[::16].to(torch.int32) \
                // vllm_block_size
            rebuilt_block_tables.append(rebuilt_block_table)

        # Sampling metadata related
        # seq_groups (use rebuilt query lens)
        if do_sample_list[idx]:
            rebuilt_selected_token_indices.append(last_query_start_loc - 1)

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
    rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(
        device)
    rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

    rebuilt_attn_metadata.block_tables = pad_sequence(
        rebuilt_block_tables, batch_first=True).to(device)

    rebuilt_attn_metadata.query_start_loc = torch.tensor(
        rebuilt_query_start_loc,
        dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
    rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
        rebuilt_context_lens_tensor,
        dtype=model_input.attn_metadata.context_lens_tensor.dtype,
    ).to(device)

    rebuilt_attn_metadata._cached_prefill_metadata = None
    rebuilt_sampling_metadata = None
    # rebuilt sampling_metadata
    if model_input.sampling_metadata is not None:
        rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
        for idx, q_len in enumerate(rebuilt_query_lens):
            if rebuilt_sampling_metadata.seq_groups is not None:
                rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

        rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
            rebuilt_selected_token_indices,
            dtype=model_input.sampling_metadata.selected_token_indices.dtype,
        ).to(device)

    # import here to avoid circular import.
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=torch.cat(rebuilt_input_tokens).to(device),
        input_positions=torch.cat(rebuilt_input_positions).to(device),
        seq_lens=model_input.seq_lens,
        query_lens=rebuilt_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=rebuilt_attn_metadata,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        sampling_metadata=rebuilt_sampling_metadata,
        is_prompt=model_input.is_prompt,
        async_callback=model_input.async_callback,
    )

    return rebuilt_model_input
