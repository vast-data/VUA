import threading
import os
import logging
import torch
import pickle
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
    def tokens_to_path(cls, tokens):
        """
        Convert a tensor of tokens into a list of directory path components.

        Parameters:
            tokens (Tensor): A PyTorch or numpy-compatible tensor of tokens
            with shape [n] or [1, n].

        Returns:
            List[str]: A list of directory names, each representing a group of
            tokens converted to hexadecimal.

        Raises:
            SplitFactorError: If the total number of tokens is not divisible by
            the split_factor.
        """

        tokens = tokens.squeeze()
        # Convert tokens to a list in case they're in a tensor-like format
        token_list = list(tokens)
        if len(token_list) % cls.split_factor != 0:
            raise SplitFactorError(len(token_list), cls.split_factor)
        num_groups = len(token_list) // cls.split_factor  # Discard remainder
        path_components = []
        for i in range(num_groups):
            group = token_list[i * cls.split_factor : (i + 1)
                               * cls.split_factor]
            # Convert each token to hex (without prefix) and join with commas
            hex_tokens = [format(token, 'x') for token in group]
            component = ",".join(hex_tokens)
            component = hashlib.sha1(component.encode('utf-8')).hexdigest()
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
        path_components = self._config.tokens_to_path(tokens)

        # Start with the root path
        try:
            current_fd = os.open(self._root_path, os.O_RDONLY | os.O_DIRECTORY)
        except FileNotFoundError:
            return None

        for (group_idx, group) in enumerate(path_components):
            group = path_components[group_idx]
            sliced_group = []
            for layer in data:
                x = []
                for t in layer:
                    t2 = t[:, :, group_idx*self._config.split_factor:(group_idx+1)*self._config.split_factor, :]
                    x.append(t2.clone())
                sliced_group.append(torch.stack(x))

            # Have one big 6-dimention matrix so it would be faster to load it into numpy
            sliced_group = serdes.tensor_to_bytes(torch.stack(sliced_group))
            sliced_tokens = serdes.tensor_to_bytes(tokens[group_idx*self._config.split_factor:
                                   (group_idx+1)*self._config.split_factor].clone())

            logger.debug(f"group #{group_idx}")

            try:
                os.mkdir(group, dir_fd=current_fd)
                logger.debug(f"group #{group_idx}: sub created")
            except OSError:
                # check for FileExists
                pass

            next_fd = os.open(group, os.O_RDONLY | os.O_DIRECTORY,
                              dir_fd=current_fd)
            logger.debug(f"group #{group_idx}: next opened")
            os.close(current_fd)
            current_fd = next_fd

            fd = os.open("_data.safetensors", os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                         dir_fd=current_fd)
            with os.fdopen(fd, 'wb') as file:
                pickle.dump(sliced_group, file)

            fd = os.open("_tokens.safetensors", os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                         dir_fd=current_fd)
            with os.fdopen(fd, 'wb') as file:
                pickle.dump(sliced_tokens, file)
        os.close(current_fd)

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

        def background_loader(cumulative_path, dir_fd, group_idx):
            try:
                try:
                    tokens = None
                    data = None
                    logger.debug(f"loading group idx {group_idx}")
                    fd = os.open("_tokens.safetensors", os.O_RDONLY, dir_fd=dir_fd)
                    with os.fdopen(fd, 'rb') as file:
                        tokens = serdes.bytes_to_tensor(pickle.load(file))

                    fd = os.open("_data.safetensors", os.O_RDONLY, dir_fd=dir_fd)
                    with os.fdopen(fd, 'rb') as file:
                        data = serdes.bytes_to_tensor(pickle.load(file))

                    logger.debug(f"done loading group idx {group_idx}")
                    results.append((group_idx, cumulative_path, (tokens, data)))
                except FileNotFoundError:
                    print("not found")
            finally:
                os.close(dir_fd)

        tokens = tokens.squeeze()
        tokens = self._config.trim_to_split_factor(tokens)
        logger.debug(f"tokens shape: {tokens.size()}")
        path_components = self._config.tokens_to_path(tokens)

        # Access the root path
        try:
            current_fd = os.open(self._root_path, os.O_RDONLY | os.O_DIRECTORY)
        except FileNotFoundError:
            return None

        i = 0
        group_idx = 0
        group = []
        nr_previous_components = 0

        threads = []
        while i < len(path_components) or group:
            if i < len(path_components):
                group.append(path_components[i])
                i += 1
                candidate = "/".join(group)
                full = len(candidate) > 3500
            else:
                full = True

            if not full:
                continue

            group_idx += 1

            # Give the kernel the chance to cache all the dentries
            # of this group
            subpath = "/".join(group)
            logger.debug(f"group #{group_idx}: size {len(group)}, "
                         f"subpath len {len(subpath)} '{subpath[:20]}...'")
            next_fd = None
            try:
                next_fd = os.open(subpath, os.O_RDONLY | os.O_DIRECTORY,
                                  dir_fd=current_fd)
            except FileNotFoundError:
                logger.debug(f"group #{group_idx}: not there")

            # Good, we can try parallel lookup for `_data.safetensors`
            logger.debug(f"lookup iter {i}")
            cumulative_path = ""
            missing = False
            for thread_idx, comp in enumerate(group, 0):
                cumulative_path = os.path.join(cumulative_path, comp)
                try:
                    new_dir_fd = os.open(cumulative_path, os.O_RDONLY | os.O_DIRECTORY, dir_fd=current_fd)
                except OSError:
                    missing = True
                    break
                t = threading.Thread(
                        target=background_loader,
                        args=(cumulative_path, new_dir_fd,
                              nr_previous_components + thread_idx))

                t.start()
                threads.append(t)

            # Go to the next path group
            os.close(current_fd)
            current_fd = None
            if next_fd:
                current_fd = next_fd
            else:
                break
            nr_previous_components += len(group)
            group = []
            if missing:
                break

        # Aggergate the results
        logger.debug("parallel lookup wait")
        for t in threads:
            t.join()

        logger.debug("parallel lookup end")

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
