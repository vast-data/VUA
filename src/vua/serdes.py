import torch
import struct
import numpy as np
import json
import mmap


BLOCK_SIZE = 4096

def align_up(x, block_size=BLOCK_SIZE):
    return (x + block_size - 1) & ~(block_size - 1)


def alloc_aligned_buffer(size, alignment=BLOCK_SIZE):
    buf = mmap.mmap(-1, size + alignment)
    return memoryview(buf)


def tensor_to_bytes(t, name):
    dtype = t.dtype
    if dtype is np.float16 or dtype is torch.float16:
        dtype = "F16"
    elif dtype is np.float32 or dtype is torch.float32:
        dtype = "F32"
    elif dtype is torch.bfloat16:
        dtype = "BF16"
    elif dtype is np.int64 or dtype is torch.int64:
        dtype = "I64"
    elif dtype is np.int32 or dtype is torch.int32:
        dtype = "I32"
    elif dtype is np.int16 or dtype is torch.int16:
        dtype = "I16"
    elif dtype is np.uint64 or dtype is torch.uint64:
        dtype = "U64"
    elif dtype is np.uint32 or dtype is torch.uint32:
        dtype = "U32"
    elif dtype is np.uint16 or dtype is torch.uint16:
        dtype = "U16"
    else:
        raise Exception(f"unhandled dtype {dtype}")

    tensor = {}
    tensor['dtype'] = dtype
    tensor['shape'] = [int(x) for x in t.size()]
    data = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
    tensor['data_offsets'] = [0, len(data)]
    meta = dict()
    meta[name] = tensor
    str_meta = bytes(json.dumps(meta), 'utf-8')
    n = len(str_meta)
    str_meta += b" " * (((n + 7) & ~7) - n)
    n = len(str_meta)
    return struct.pack('L', n) + str_meta + t.contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_to_bytes_aligned(t: torch.Tensor, name: str) -> memoryview:
    dtype = t.dtype
    if dtype is np.float16 or dtype is torch.float16:
        dtype = "F16"
        np_dtype = np.float16
    elif dtype is np.float32 or dtype is torch.float32:
        dtype = "F32"
        np_dtype = np.float32
    elif dtype is torch.bfloat16:
        dtype = "BF16"
        np_dtype = np.uint16  # no native bfloat16 in NumPy, fake it
        t = t.view(torch.uint16)
    elif dtype is np.int64 or dtype is torch.int64:
        dtype = "I64"
        np_dtype = np.int64
    elif dtype is np.int32 or dtype is torch.int32:
        dtype = "I32"
        np_dtype = np.int32
    elif dtype is np.int16 or dtype is torch.int16:
        dtype = "I16"
        np_dtype = np.int16
    elif dtype is np.uint64 or dtype is torch.uint64:
        dtype = "U64"
        np_dtype = np.uint64
    elif dtype is np.uint32 or dtype is torch.uint32:
        dtype = "U32"
        np_dtype = np.uint32
    elif dtype is np.uint16 or dtype is torch.uint16:
        dtype = "U16"
        np_dtype = np.uint16
    else:
        raise Exception(f"unhandled dtype {dtype}")

    # Metadata
    tensor_meta = {
        'dtype': dtype,
        'shape': list(t.size()),
        'data_offsets': [0, t.numel() * t.element_size()],
    }
    meta = {name: tensor_meta}
    str_meta = json.dumps(meta).encode('utf-8')
    meta_len = len(str_meta)
    str_meta += b' ' * ((8 - (meta_len % 8)) % 8)
    meta_len = len(str_meta)
    header = struct.pack('L', meta_len)

    # Total length calculation
    data_len = t.numel() * t.element_size()
    total_len = len(header) + meta_len + data_len
    total_len_aligned = align_up(total_len)

    # Allocate aligned buffer
    aligned = alloc_aligned_buffer(total_len_aligned)

    # Copy header + metadata
    aligned[:len(header)] = header
    aligned[len(header):len(header)+meta_len] = str_meta

    # Prepare numpy view into the aligned memory for data section
    data_offset = len(header) + meta_len
    aligned_data = np.ndarray(
        shape=(t.numel(),),
        dtype=np_dtype,
        buffer=aligned[data_offset:data_offset + data_len]
    )

    # Copy tensor data directly
    np.copyto(aligned_data, t.flatten())
    eof = len(header) + meta_len + t.numel() * t.element_size()
    return aligned[:eof]


def bytes_to_tensor(b, name):
    size, = struct.unpack('L', b[:8])
    meta = json.loads(b[8:8 + size])
    payload_offset = size + 8
    tensor = meta[name]
    dtype = tensor['dtype']
    data_offsets = tensor['data_offsets']
    shape = tensor['shape']
    data_bytes = b[payload_offset + data_offsets[0]:payload_offset + data_offsets[1]]
    if dtype == 'BF16':
        dtype = torch.bfloat16
        return torch.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    elif dtype == 'F16':
        dtype = np.float16
    elif dtype == 'F32':
        dtype = np.float32
    elif dtype == 'I64':
        dtype = np.int64
    elif dtype == 'I32':
        dtype = np.int32
    elif dtype == 'I16':
        dtype = np.int16
    elif dtype == 'U64':
        dtype = np.uint64
    elif dtype == 'U32':
        dtype = np.uint32
    elif dtype == 'U16':
        dtype = np.uint16
    else:
        raise Exception(f"unhandled dtype {dtype}")

    array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return torch.from_numpy(np.array(array))

def tensor_header_bytes(t: torch.Tensor, name: str) -> bytes:
    dtype = t.dtype
    if dtype is np.float16 or dtype is torch.float16:
        dtype = "F16"
    elif dtype is np.float32 or dtype is torch.float32:
        dtype = "F32"
    elif dtype is torch.bfloat16:
        dtype = "BF16"
    elif dtype is np.int64 or dtype is torch.int64:
        dtype = "I64"
    elif dtype is np.int32 or dtype is torch.int32:
        dtype = "I32"
    elif dtype is np.int16 or dtype is torch.int16:
        dtype = "I16"
    elif dtype is np.uint64 or dtype is torch.uint64:
        dtype = "U64"
    elif dtype is np.uint32 or dtype is torch.uint32:
        dtype = "U32"
    elif dtype is np.uint16 or dtype is torch.uint16:
        dtype = "U16"
    else:
        raise Exception(f"unhandled dtype {dtype}")

    # Metadata
    data_size = t.numel() * t.element_size()
    tensor_meta = {
        'dtype': dtype,
        'shape': list(t.size()),
        'data_offsets': [0, data_size],
    }
    meta = {name: tensor_meta}
    str_meta = json.dumps(meta).encode('utf-8')
    meta_len = len(str_meta)
    str_meta += b' ' * ((8 - (meta_len % 8)) % 8)
    meta_len = len(str_meta)
    header = struct.pack('L', meta_len)
    return (header + str_meta, data_size)
