import threading
import os
import logging
import torch
import hashlib
from typing import NamedTuple, Tuple, List


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


from . import serdes


class SplitFactorError(Exception):
    pass


class VUAConfig:
    split_factor = 600

    @classmethod
    def tokens_to_paths(cls, tokens):
        """
        Convert a tensor of tokens into a list of hash directory names, where each hash
        includes all tokens from the beginning up to the current group.

        Parameters:
            tokens (Tensor): A PyTorch or numpy-compatible tensor of tokens
            with shape [n] or [1, n].

        Returns:
            List[str]: A list of directory names, each representing a hash of all tokens
            up to that group.

        Raises:
            SplitFactorError: If the total number of tokens is not divisible by
            the split_factor.
        """

        tokens = tokens.squeeze()
        token_list = list(tokens)
        if len(token_list) % cls.split_factor != 0:
            raise SplitFactorError(len(token_list), cls.split_factor)
        num_groups = len(token_list) // cls.split_factor
        path_components = []
        prev = ""
        for i in range(num_groups):
            # Each hash includes all tokens from the beginning up to this group
            group_tokens = token_list[: (i + 1) * cls.split_factor]
            hex_tokens = [format(token, 'x') for token in group_tokens]
            component = ",".join(hex_tokens)
            component = hashlib.sha1((prev + component).encode('utf-8')).hexdigest()
            prev = "hash:" + component + ":"
            path_components.append(component)
        return path_components  # List of strings

    @classmethod
    def trim_to_split_factor(cls, tokens):
        """Trim a tokens tensor so that its length is divisible by split_factor.

        Parameters:
            tokens (Tensor): A PyTorch or numpy-compatible tensor of tokens
            with shape [n] or [1, n].

        Returns:
            Tensor: The trimmed tensor.
        """
        tokens = tokens.squeeze()
        return tokens[:len(tokens) - (len(tokens) % cls.split_factor)]


class ClosestKV(NamedTuple):
    """
    tokens (Tensor): A PyTorch tensor of shape:
        tensor.Size([1, seq_len])

    data (List[Tuple[Tensor, Tensor]]): a KVCache in the struct of
        Transformers library. The list of layers, each layer has a 2-tuple
        for keys and values, and each keys or values tensor is of the
        following shape:

        tensor.Size([1, num_heads, seq_len, head_dim])
    """

    data: torch.Tensor
    tokens: List[Tuple[torch.Tensor, torch.Tensor]]


class VUA:
    def __init__(self, config, root_path):
        self._config = config
        self._root_path = root_path

    def config(self) -> VUAConfig:
        """
        Return configruration for this VUA instance.
        """
        return self._config

    def put(self, tokens, data):
        """
        Save split kvcache data and tokens into a nested directory structure
        derived from the token values.

        The tokens are processed to generate directory path components.
        Directories are created as needed, ensuring that the generated path
        does not exceed system limitations. The data and tokens are stored in
        files named '_data' and '_tokens' respectively within each node.

        Parameters:
            tokens (Tensor): A PyTorch tensor of shape:
                tensor.Size([batch_head, seq_len])
                if provided `tensor.Size([seq_len])`, a 1-batch is assumed.

            data (List[Tuple[Tensor, Tensor]]): a KVCache in the struct of
                Transformers library. The list of layers, each layer has a 2-tuple
                for keys and values, and each keys or values tensor is of the
                following shape:

                tensor.Size([batch_head, num_heads, seq_len, head_dim])

            seq_len must be a multiple of config().split_factor

        Returns:
            None if the root path does not exist, otherwise None.
        """

        if tokens.dim() > 2:
            raise Exception(f"input token tensor dimension too big {tokens.dim()}")

        if tokens.dim() == 2:
            # TODO: we can optimize for common prefixes here. For now it will just
            # be parallel.
            threads = []
            assert data[0][0].size(0) == tokens.size(0), "number token sequences should match batch_head"
            for i in range(tokens.size(0)):
                split_kvcache = [[kv[i].unsqueeze(0) for kv in layer] for layer in data]
                t = threading.Thread(target=self.put, args=(tokens[i], split_kvcache))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            return

        logger.debug(f"put with {len(tokens)} tokens")
        tokens = tokens.squeeze()
        path_components = self._config.tokens_to_paths(tokens)

        # Start with the root path
        def save_group(group_hash, group_idx):
            group_dir = os.path.join(self._root_path, group_hash)
            if os.path.exists(group_dir):
                return

            group_dir_tmp = group_dir + ".tmp"
            try:
                os.mkdir(group_dir_tmp)
                logger.debug(f"group #{group_idx}: dir created {group_dir_tmp}")
            except OSError:
                # check for FileExists
                pass

            # Create symlink to parent if not the first group
            if group_idx > 0:
                parent_hash = path_components[group_idx - 1]
                parent_link = os.path.join(group_dir_tmp, "parent")
                try:
                    os.symlink(os.path.join("..", parent_hash), parent_link)
                except FileExistsError:
                    pass

            # Prepare data and tokens for this group
            sliced_group = []
            for layer in data:
                x = []
                for t in layer:
                    t2 = t[:, :, group_idx*self._config.split_factor:(group_idx+1)*self._config.split_factor, :]
                    x.append(t2.clone())
                sliced_group.append(torch.stack(x))

            sliced_group = serdes.tensor_to_bytes(torch.stack(sliced_group), group_hash + ".data")
            sliced_tokens = serdes.tensor_to_bytes(tokens[group_idx*self._config.split_factor:
                                   (group_idx+1)*self._config.split_factor].clone(), group_hash + ".tokens")

            # Write data and tokens files
            fd = os.open(os.path.join(group_dir_tmp, "_data.safetensors"), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            with os.fdopen(fd, 'wb') as file:
                file.write(sliced_group)
                os.fsync(fd)

            fd = os.open(os.path.join(group_dir_tmp, "_tokens.safetensors"), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            with os.fdopen(fd, 'wb') as file:
                file.write(sliced_tokens)
                os.fsync(fd)

            os.rename(group_dir_tmp, group_dir)

        threads = []
        for group_idx, group_hash in enumerate(path_components):
            t = threading.Thread(target=save_group, args=(group_hash, group_idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def get_closest(self, tokens, device):
        """
        Reconstruct KVCaches from stored fragments based on a token path lookup.

        It supports both a single token tensor and a list of token tensors.
        When provided a list, it returns a list where each element is the
        result corresponding to the best match for the input token tensor; when
        provided a single tensor, it returns a ClosestKV instance or None if no
        match is found.

        Parameters:
            tokens (Tensor or List[Tensor]): A PyTorch tensor of tokens
            (1-dimensional) or a list thereof. device (Device): The device to
            load the stored tensors onto (e.g., 'cpu' or 'cuda:<num>').

        Returns:
            Union[ClosestKV, None] or List[Union[ClosestKV, None]]: For a
            single token tensor input, returns a ClosestKV instance or None;
            for a list of token tensors, returns a list with an entry for each
            tensor.
        """

        if isinstance(tokens, torch.Tensor):
            if tokens.dim() >= 2:
                raise Exception(f"input token tensor dimension too big {tokens.dim()}")
            logger.debug(f"get with tokens.size()={tokens.size()} tokens")
        elif isinstance(tokens, list) and \
                all(isinstance(t, torch.Tensor) and t.dim() == 1 for t in tokens):
            # TODO: we can optimize for common prefixes here. For now it will just
            # be parallel.
            tokens_groups = tokens
            threads = []
            results = [None] * len(tokens_groups)

            for i, token_list in enumerate(tokens_groups):
                def worker(token_list, device, i=i):
                    results[i] = self.get_closest(token_list, device)
                t = threading.Thread(target=worker, args=(token_list, device))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            return results

        results = []
        tokens = tokens.squeeze()
        tokens = self._config.trim_to_split_factor(tokens)
        logger.debug(f"tokens shape: {tokens.size()}")
        path_components = self._config.tokens_to_paths(tokens)

        # Each group is now a flat directory under root, with symlinks to parent
        threads = []

        def load_group(group_hash, group_idx):
            group_dir = os.path.join(self._root_path, group_hash)
            try:
                tokens = None
                data = None
                logger.debug(f"loading group idx {group_idx}")
                with open(os.path.join(group_dir, "_tokens.safetensors"), "rb") as file:
                    tokens = serdes.bytes_to_tensor(file.read(), group_hash + ".tokens")
                with open(os.path.join(group_dir, "_data.safetensors"), "rb") as file:
                    data = serdes.bytes_to_tensor(file.read(), group_hash + ".data")
                logger.debug(f"done loading group idx {group_idx}")
                results.append((group_idx, group_hash, (tokens, data)))
            except FileNotFoundError:
                logger.debug(f"group {group_hash} not found")

        for group_idx, group_hash in enumerate(path_components):
            t = threading.Thread(target=load_group, args=(group_hash, group_idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if not results:
            return None

        results.sort()
        cache_groups = []
        tokens_groups = []

        for (group_idx, _, (tokens, data)) in results:
            tokens_groups.append(tokens)
            cache_groups.append(data.pin_memory().to(device=device, non_blocking=True))

        data = []
        num_layers = len(cache_groups[0])
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for group in cache_groups:
                keys.append(group[layer_idx][0])
                values.append(group[layer_idx][1])
            combined_key = torch.cat(keys, dim=2)
            combined_value = torch.cat(values, dim=2)
            data.append((combined_key, combined_value))

        logger.debug("data combination ended")

        tokens = torch.cat(tokens_groups, dim=0).to(device=device)
        return ClosestKV(tokens=tokens, data=data)
