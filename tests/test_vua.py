import unittest
import torch
import logging

from vua.core import VUA, VUAConfig
from vua.serdes import tensor_to_bytes, bytes_to_tensor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

def generate_rand_kvcache(n_layers, seq_len, batch_size, num_heads, head_size):
    layers = []
    for i in range(0, n_layers):
        s = []
        for kv in [0, 1]:
            size = (batch_size, num_heads, seq_len, head_size)
            t = torch.randn(size, dtype=torch.float16)
            s.append(t)
        layers.append(s)
    return layers


class TestVUAConfig(unittest.TestCase):
    def test_tokens_to_path(self):
        # Create a tensor of tokens that is divisible by split_factor
        tokens = torch.arange(VUAConfig.split_factor * 2)
        paths = VUAConfig.tokens_to_path(tokens)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(isinstance(p, str) for p in paths),
            "Each token group should be converted to a string path component")

    def test_trim_to_split_factor(self):
        # Create a tensor of tokens that isn't divisible by split_factor
        tokens = torch.arange(100)
        trimmed = VUAConfig.trim_to_split_factor(tokens)
        self.assertEqual(len(trimmed) % VUAConfig.split_factor, 0,
            "Trimmed tensor length should be divisible by split_factor")


    def test_put_get(self):
        import tempfile
        import os
        import logging

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            vua_path = os.path.join(temp_dir, 'vua')
            os.mkdir(vua_path)

            # Create a tensor of tokens that isn't divisible by split_factor
            logger.info(f"VUAPath: {vua_path}")
            cache = VUA(VUAConfig, vua_path)

            nr_tokens = 3000
            tokens = torch.randint(low=0, high=0xffff, size=(1, nr_tokens), dtype=torch.uint16)
            half_tokens = tokens[:, :nr_tokens//2]
            quarter_tokens = tokens[:, :nr_tokens//4]
            trimmed = VUAConfig.trim_to_split_factor(tokens)
            trimmed_half = VUAConfig.trim_to_split_factor(half_tokens)
            trimmed_quarter = VUAConfig.trim_to_split_factor(quarter_tokens)
            trimmed_quarter_plus_other = VUAConfig.trim_to_split_factor(torch.cat([quarter_tokens, tokens[:, 3*nr_tokens//4:nr_tokens]], dim=1))
            self.assertEqual(len(trimmed_half) % VUAConfig.split_factor, 0,
                "Trimmed tensor length should be divisible by split_factor")

            kvcache = generate_rand_kvcache(32, half_tokens.size(1), 1, 32, 16)
            cache.put(trimmed_half, kvcache)

            logger.info("---- Doing a get with a double length query")
            # The prefix with which we originally did 'put', yields half that prefix.
            res = cache.get_closest(trimmed, device="cuda:0")
            self.assertEqual(torch.equal(res.tokens.to("cpu"), trimmed_half), True)

            logger.info("---- :: Doing a get with half of it, only gets us the half")
            res = cache.get_closest(trimmed_quarter, device="cuda:0")
            self.assertEqual(torch.equal(res.tokens.to("cpu"), trimmed_quarter), True)

            logger.info("---- :: Doing a get with half of it plus other tokens, only gets us the half")
            res = cache.get_closest(trimmed_quarter_plus_other, device="cuda:0")
            self.assertEqual(torch.equal(res.tokens.to("cpu"), trimmed_quarter), True)

            logger.info("---- :: Batched get")
            res = cache.get_closest([trimmed_quarter,
                                     trimmed_quarter_plus_other], device="cuda:0")
            self.assertNotEqual(res[0], None)
            self.assertNotEqual(res[1], None)

            logger.info("---- :: Batched put")

            batched_seqs = torch.randint(low=0, high=0xffff,
                                         size=(3, cache.config().split_factor * 10), dtype=torch.uint16)
            batched_kvcache = generate_rand_kvcache(24, batched_seqs.size(1), batched_seqs.size(0), 8, 16)
            cache.put(batched_seqs, batched_kvcache)


class TestSerdes(unittest.TestCase):
    def test_tensor_serialization(self):
        # Create a random tensor and test that serializing and deserializing
        # returns a similar tensor
        x = torch.randn(100, 5, 10000)
        b = tensor_to_bytes(x)
        x_rec = bytes_to_tensor(b)
        self.assertTrue(torch.allclose(x, x_rec.float(), atol=1e-6),
                        "Deserialized tensor does not match the original")


if __name__ == '__main__':
    unittest.main()
