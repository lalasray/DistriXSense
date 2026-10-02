"""Versioned binary packets used by the inference path and byte accounting.

Indices are bit-packed; timestamps are float32 seconds. Headers carry modality,
shape, representation type and bit width. Network overhead is external to this
application format and must be explicitly configured for a measured link.
"""
from dataclasses import dataclass
import math
import struct
import numpy as np
import torch

HEADER = struct.Struct("<4sBHIIB")
MAGIC = b"DXS1"


@dataclass
class Message:
    modality: str
    values: torch.Tensor  # (B, K, D) floats OR (B, K) indices
    times: torch.Tensor  # (B, K)
    valid: torch.Tensor  # (B, K)
    vocabulary: int = 0


def serialize(message, example, modality_id):
    valid = message.valid[example]
    values = message.values[example][valid].detach().cpu().numpy()
    times = message.times[example][valid].detach().cpu().numpy().astype("<f4")
    if len(times) == 0:
        return b""  # no packet for an absent modality
    if values.ndim == 1:
        width = max(1, math.ceil(math.log2(message.vocabulary)))
        if width > 32 or (values < 0).any() or (values >= message.vocabulary).any():
            raise ValueError("Indices outside vocabulary or unsupported width")
        bits = ((values.astype(np.uint64)[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)
        body = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
        kind, channels = 1, 1
    else:
        kind, width, channels = 0, 32, values.shape[-1]
        body = values.astype("<f4").tobytes()
    return HEADER.pack(MAGIC, kind, modality_id, len(times), channels, width) + times.tobytes() + body


def deserialize(packet, modalities, vocabulary=0, device="cpu"):
    if len(packet) < HEADER.size:
        raise ValueError("Truncated packet")
    magic, kind, mid, length, channels, width = HEADER.unpack_from(packet)
    if magic != MAGIC or mid >= len(modalities) or not length or kind not in (0, 1):
        raise ValueError("Invalid packet header")
    data_size = math.ceil(length*width/8) if kind else length*channels*4
    expected = HEADER.size + length*4 + data_size
    if len(packet) != expected or (kind and (channels != 1 or not 1 <= width <= 32)) or (not kind and (width != 32 or channels < 1)):
        raise ValueError("Invalid packet size or encoding")
    times = np.frombuffer(packet, dtype="<f4", count=length, offset=HEADER.size).copy()
    body = packet[HEADER.size+length*4:]
    if kind:
        bits = np.unpackbits(np.frombuffer(body, dtype=np.uint8), bitorder="little")[:length*width].reshape(length, width)
        values = (bits.astype(np.uint64) << np.arange(width, dtype=np.uint64)).sum(1).astype(np.int64)
        if vocabulary < 1 or (values >= vocabulary).any():
            raise ValueError("Received index outside bank")
    else:
        values = np.frombuffer(body, dtype="<f4").copy().reshape(length, channels)
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("Nonfinite packet data")
    return Message(modalities[mid], torch.from_numpy(values).unsqueeze(0).to(device),
                   torch.from_numpy(times).unsqueeze(0).to(device),
                   torch.ones(1, length, dtype=torch.bool, device=device), vocabulary)


def roundtrip(messages, names, device):
    """Batch-size-one deployment inference, through the actual wire format."""
    received, packets = {}, []
    for n, message in messages.items():
        if message.values.shape[0] != 1:
            raise ValueError("Packet roundtrip requires batch size one")
        packet = serialize(message, 0, names.index(n))
        if packet:
            received[n] = deserialize(packet, names, message.vocabulary, device)
            packets.append(packet)
    return received, packets


def payload_bytes(messages, names, overhead=0):
    batch = next(iter(messages.values())).values.shape[0]
    totals = []
    for i in range(batch):
        total = 0
        for n, m in messages.items():
            packet = serialize(m, i, names.index(n))
            total += len(packet) + (overhead if packet else 0)
        totals.append(total)
    return totals
