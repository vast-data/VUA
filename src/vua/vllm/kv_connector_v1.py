# SPDX-License-Identifier: Apache-2.0
#
# The implementation of this KV connector for VLLM is roughly based on the following:
#
# - SharedStorageConnector from vllm (ed7a29d9f8b48)
#      https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py
#
# - LMCacheConnectorV1Impl from LMCache (ccb5204517edbb6)
#      https://github.com/LMCache/LMCache/blob/dev/lmcache/integration/vllm/vllm_v1_adapter.py
#
import hashlib
import os
import time
import safetensors
import torch
import itertools

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.utils import PlaceholderModule

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData

logger = init_logger(__name__)

try:
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
except ImportError:
    fastsafetensors = PlaceholderModule("fastsafetensors")
    SafeTensorsFileLoader = fastsafetensors.placeholder_attr(
        "SafeTensorsFileLoader")
    SingleGroup = fastsafetensors.placeholder_attr("SingleGroup")


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in external
    external_cached_tokens: int


@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool
    # Prefix that we don't need to load/save
    existing_token_ids: int
    # If transition to store mode, these will be the new token_ids and slot_mapping
    store_token_ids: Optional[torch.Tensor]
    store_slot_mapping: Optional[torch.Tensor]

    @staticmethod
    def make_meta(token_ids: list[int], block_ids: list[int], block_size: int,
                  chunk_size: int, is_store: bool, existing_token_ids: int,
                  load_spec: Optional[LoadSpec]) -> "ReqMeta":
        valid_num_tokens = align_to_size(len(token_ids), chunk_size)
        nr_trimmed_tokens = valid_num_tokens
        trimmed_nr_block_ids = len(block_ids)
        store_token_ids = None
        store_slot_mapping = None

        if load_spec and valid_num_tokens > load_spec.external_cached_tokens:
            store_token_ids = torch.tensor(token_ids)[:nr_trimmed_tokens]
            block_ids_tensor = torch.tensor(block_ids)[:trimmed_nr_block_ids]
            store_slot_mapping = ReqMeta._compute_slot_mapping(
                block_ids_tensor, block_size, nr_trimmed_tokens)
            nr_trimmed_tokens = load_spec.external_cached_tokens
            trimmed_nr_block_ids = nr_trimmed_tokens // block_size

        token_ids_tensor = torch.tensor(token_ids)[:nr_trimmed_tokens]
        block_ids_tensor = torch.tensor(block_ids)[:trimmed_nr_block_ids]
        slot_mapping = ReqMeta._compute_slot_mapping(block_ids_tensor, block_size, nr_trimmed_tokens)

        logger.info(f"make_meta: len(token_ids)={len(token_ids)} len(block_ids)={len(block_ids)} existing_token_ids={existing_token_ids} chunk_size={chunk_size} valid_num_tokens={valid_num_tokens} nr_trimmed_tokens={nr_trimmed_tokens} slot_mapping.size()={slot_mapping.size()} ")
        return ReqMeta(
            token_ids=token_ids_tensor,
            existing_token_ids=existing_token_ids,
            slot_mapping=slot_mapping,
            is_store=is_store,
            store_token_ids=store_token_ids,
            store_slot_mapping=store_slot_mapping,
        )

    @staticmethod
    def _compute_slot_mapping(block_ids_tensor: torch.Tensor,
                              block_size: int,
                              num_tokens: int) -> torch.Tensor:
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
            block_ids_tensor.reshape((num_blocks, 1)) * block_size
        return slot_mapping.flatten()[:num_tokens]

    def transition_to_store(self):
        if self.store_token_ids is None:
            return

        logger.info(f"transition to store: {self.existing_token_ids} -> {len(self.token_ids)}, now {len(self.store_token_ids)}")
        self.existing_token_ids = len(self.token_ids)
        self.token_ids = self.store_token_ids
        self.slot_mapping = self.store_slot_mapping
        self.is_store = True
        self.store_token_ids = None
        self.store_slot_mapping = None


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute]
            .copy(),
            allocated_block_ids=new_request.block_ids.copy(),
        )

    def update(
        self,
        cached_request: "CachedRequestData",
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """
        self.token_ids.extend(cached_request.new_token_ids)
        self.allocated_block_ids.extend(cached_request.new_block_ids)


@dataclass
class SharedStorageConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        chunk_size: int,
        is_store: bool,
        existing_token_ids: int,
        load_spec: Optional[LoadSpec],
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, chunk_size, is_store, existing_token_ids, load_spec))


class VUAStorageConnector_V1(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._request_trackers: dict[str, RequestTracker] = {}
        self._requests_need_load: dict[str, Request] = {}
        self.load_specs: dict[str, LoadSpec] = {}
        transfer_config = vllm_config.kv_transfer_config
        self._max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._storage_path = transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp")
        self.kv_caches: dict[str, torch.Tensor] = {}
        logger.info(vllm_config.kv_transfer_config)
        self._rank_prefix = str(vllm_config.parallel_config.rank) + "%"
        logger.info("Shared storage path is %s, rank %d", self._storage_path,
                    vllm_config.parallel_config.rank)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        attn_metadata = forward_context.attn_metadata

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = \
            self._get_connector_metadata()
        assert isinstance(metadata, SharedStorageConnectorMetadata)

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        if torch.distributed.is_initialized():
            pg = torch.distributed.group.WORLD
        else:
            pg = SingleGroup()

        device = torch.device(f'cuda:{pg.rank()}')
        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue

            load_time = time.time()
            partial_slot_mapping = request.slot_mapping[request.existing_token_ids:]
            logger.info("Inject KV cache of %d tokens to the paged memory at offset %d",
                        len(partial_slot_mapping), request.existing_token_ids)

            assert(request.existing_token_ids % self._max_num_batched_tokens == 0)
            assert(len(request.token_ids) % self._max_num_batched_tokens == 0)
            start = request.existing_token_ids // self._max_num_batched_tokens
            end = len(request.token_ids) // self._max_num_batched_tokens

            logger.info(f"Prefix range: {start}..{end}")
            for chunk_idx in range(start, end):
                tokens_ids = request.token_ids[:(chunk_idx + 1) * self._max_num_batched_tokens]
                hash_layer_name = self._generate_hash_layer_name("layer", tokens_ids)
                logger.info(f"  {chunk_idx}: {hash_layer_name}")

            batch_size = 8 # Larger batches mean more RAM used temporarily in the loop
            for layers_batch in itertools.batched(forward_context.no_compile_layers, batch_size):
                # Load the KV for each request each layer
                loader = SafeTensorsFileLoader(pg, device)
                filenames = []
                for layer_name in layers_batch:
                    for chunk_idx in range(start, end):
                        (_, filename) = self._generate_filename_debug(
                            layer_name, request.token_ids[:(chunk_idx + 1) * self._max_num_batched_tokens])
                        filenames.append(filename)

                loader.add_filenames({0: filenames})
                try:
                    fb = loader.copy_files_to_device()
                    inject_start_time = time.time()
                    for layer_name in layers_batch:
                        attn_layer = forward_context.no_compile_layers[layer_name]
                        kv_cache_layer = attn_layer.kv_cache[\
                                forward_context.virtual_engine]
                        for chunk_idx in range(start, end):
                            tokens_ids = request.token_ids[:(chunk_idx + 1) * self._max_num_batched_tokens]
                            hash_layer_name = self._generate_hash_layer_name(layer_name, tokens_ids)
                            partial_slot_mapping = request.slot_mapping[
                                (chunk_idx * self._max_num_batched_tokens):
                                ((chunk_idx + 1) * self._max_num_batched_tokens)]
                            kv_cache = fb.get_tensor(hash_layer_name)
                            inject_kv_into_layer(kv_cache_layer, kv_cache, partial_slot_mapping)
                        inject_end_time = time.time()
                        inject_total_time = inject_end_time - inject_start_time
                    try:
                        pass
                    finally:
                        fb.close()
                finally:
                    loader.close()

            load_end = time.time()
            logger.info("Load + injection done (%.4f seconds, injection %.4f)", load_end - load_time, inject_total_time)
            request.transition_to_store()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        if self._check_disable_put():
            return False

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping,
                                                                ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping,
                                                               ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, SharedStorageConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                (hash_layer_name, filename) = self._generate_filename_debug(
                    layer_name, request.token_ids[:len(request.slot_mapping)],
                    create_folder=True)
                if layer_name in ["model.layers.0.self_attn.attn"]:
                    logger.info(f"SAVE KV LAYER request.slot_mapping.size()={request.slot_mapping.size()} filename={filename}")
                if os.path.exists(filename):
                    continue

                partial_slot_mapping = request.slot_mapping[request.existing_token_ids:]
                kv_cache = extract_kv_from_layer(kv_layer, partial_slot_mapping)
                tensors = {hash_layer_name: kv_cache.detach().cpu()}
                self.kv_caches[layer_name] = (filename, tensors)

    def wait_for_save(self):
        logger.info(f"wait_for_save: {len(self.kv_caches)} files")
        # TODO: save in parallel and use async IO
        for (layer_name, (filename, tensors)) in list(self.kv_caches.items()):
            safetensors.torch.save_file(tensors, filename)
        self.kv_caches = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self._check_disable_get():
            return 0

        # NOTE: in this implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        num_tokens_to_check = self._search_matching_prefix(request.prompt_token_ids, 1)
        if num_tokens_to_check == 0:
            return 0

        if num_tokens_to_check == request.num_tokens:
            num_tokens_to_check -= 1

        logger.info(f"External Cache Hit: {num_tokens_to_check} tokens (computed: {num_computed_tokens})!")
        if num_computed_tokens > num_tokens_to_check:
            return 0

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            external_cached_tokens=num_tokens_to_check)

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV
        return num_tokens_to_check - num_computed_tokens

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        if num_external_tokens == 0:
            # No need to load anything
            return

        self._requests_need_load[request.request_id] = request
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        assert num_external_tokens > 0 and num_external_tokens == \
            self.load_specs[request.request_id].external_cached_tokens - \
            self.load_specs[request.request_id].vllm_cached_tokens, \
            f"Mismatch in number of tokens: {num_external_tokens} vs " \
            f"{self.load_specs[request.request_id].external_cached_tokens} - " \
            f"{self.load_specs[request.request_id].vllm_cached_tokens}" \
            f" for request {request.request_id}"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = SharedStorageConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            logger.info(f"build_connector_meta: finished {finished_req_id}")
            self._request_trackers.pop(finished_req_id, None)

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            load_spec = self.load_specs.pop(new_req.req_id, None)
            sched_tokens = scheduler_output.num_scheduled_tokens[new_req.req_id]
            num_tokens_to_compute = new_req.num_computed_tokens + sched_tokens
            in_need_load = new_req.req_id in self._requests_need_load
            logger.info(f"build_connector_meta: new req {new_req.req_id}: {load_spec}, in_need_load={in_need_load} {new_req.num_computed_tokens} + {sched_tokens} -> {num_tokens_to_compute}")
            request_tracker = RequestTracker.from_new_request(
                new_req, num_tokens_to_compute)
            self._request_trackers[new_req.req_id] = request_tracker

            if in_need_load:
                meta.add_request(token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids,
                                 block_size=self._block_size,
                                 chunk_size=self._max_num_batched_tokens,
                                 is_store=False,
                                 existing_token_ids=0,
                                 load_spec=load_spec)
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                if not self._found_match_for_request(new_req.prompt_token_ids, 1):
                    meta.add_request(token_ids=new_req.prompt_token_ids,
                                     block_ids=new_req.block_ids,
                                     block_size=self._block_size,
                                     chunk_size=self._max_num_batched_tokens,
                                     is_store=True,
                                     existing_token_ids=0,
                                     load_spec=None)

        for cached_req in scheduler_output.scheduled_cached_reqs:
            request_tracker = self._request_trackers[cached_req.req_id]
            prev_token_id_len = len(request_tracker.token_ids)
            request_tracker.update(cached_req)
            logger.info(f"build_connector_meta: cached {cached_req.req_id}, {cached_req.resumed_from_preemption} prev_token_id_len={prev_token_id_len}, now {len(request_tracker.token_ids)}")

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if cached_req.req_id in self._requests_need_load:
                if not cached_req.resumed_from_preemption:
                    break
                # NOTE(rob): cached_req_data does not have the full
                # list of token ids (only new tokens). So we look it
                # up in the actual request object.
                request = self._requests_need_load[cached_req.req_id]
                total_tokens = (len(cached_req.new_token_ids) +
                                cached_req.num_computed_tokens)
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = cached_req.new_block_ids

                meta.add_request(token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size,
                                 chunk_size=self._max_num_batched_tokens,
                                 is_store=False,
                                 existing_token_ids=prev_token_id_len,
                                 load_spec=None)
                total_need_load += 1
            else:
                if not self._found_match_for_request(request_tracker.token_ids, 0):
                    meta.add_request(token_ids=request_tracker.token_ids,
                                     block_ids=request_tracker.allocated_block_ids,
                                     block_size=self._block_size,
                                     chunk_size=self._max_num_batched_tokens,
                                     is_store=True,
                                     existing_token_ids=prev_token_id_len,
                                     load_spec=None)
                else:
                    logger.info(f"build_connector_meta: found match, not issuing store")

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_request(
        self,
        token_ids: list[int],
        diff: int,
    ) -> bool:
        """Check if the cache is hit for the request.
        """
        num_tokens_to_check = align_to_size(
            len(token_ids) - diff, self._max_num_batched_tokens)
        foldername = self._generate_foldername_debug(torch.tensor(
            token_ids)[:num_tokens_to_check], create_folder=False)
        return os.path.exists(foldername)

    def _search_matching_prefix(
        self,
        token_ids: list[int],
        diff: int,
    ) -> int:
        """Find a matching prefix
        """
        token_groups_to_check = align_to_size(
            len(token_ids) - diff, self._max_num_batched_tokens) \
            // self._max_num_batched_tokens

        matches = 0
        for i in range(0, token_groups_to_check):
            end_token = self._max_num_batched_tokens * (i + 1)
            foldername = self._generate_foldername_debug(torch.tensor(
                token_ids)[:end_token], create_folder=False)
            if not os.path.exists(foldername):
                break
            matches = end_token
        return matches

    def _generate_foldername_debug(
        self,
        input_ids: torch.Tensor,
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input
        ids.
        """
        input_ids_bytes = input_ids.numpy().tobytes()
        input_ids_hash = self._rank_prefix + hashlib.sha1(input_ids_bytes).hexdigest()
        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(
        self,
        layer_name: str,
        input_ids: torch.Tensor,
        create_folder=False,
    ) -> (str, str):
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        foldername = self._generate_foldername_debug(input_ids,
                                                     create_folder=create_folder)
        hash_layer_name = os.path.basename(foldername) + "." + layer_name
        return (hash_layer_name, os.path.join(foldername, f"{layer_name}.safetensors"))

    def _generate_hash_layer_name(
        self,
        layer_name: str,
        input_ids: torch.Tensor,
    ) -> (str, str):
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        foldername = self._generate_foldername_debug(input_ids, create_folder=False)
        return os.path.basename(foldername) + "." + layer_name

    def _check_disable_put(
        self,
    ) -> bool:
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        return os.path.exists(os.path.join(self._storage_path, "disable-put"))


    def _check_disable_get(
        self,
    ) -> bool:
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        return os.path.exists(os.path.join(self._storage_path, "disable-get"))


def align_to_size(num_tokens: int, size) -> int:
    """Align the number of tokens to the block size.
    """
    if num_tokens % size == 0:
        return num_tokens
    return (num_tokens - 1) // size * size


KVConnectorFactory.register_connector("VUAStorageConnector_V1",
                                      "vua.vllm.kv_connector_v1",
                                      "VUAStorageConnector_V1")
