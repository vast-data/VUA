import torch
import struct
import numpy as np
import json


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
