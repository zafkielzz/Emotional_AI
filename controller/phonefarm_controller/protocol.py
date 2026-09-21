"""Binary frame codec shared by controller tests and Android protocol fixtures."""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

MAGIC = b"PFAR"
VERSION = 1
HEADER = struct.Struct("!4sBBIQI")
MAX_METADATA_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024


class MessageType(IntEnum):
    HELLO = 1
    ASSIGN = 2
    TENSOR = 3
    RESULT = 4
    ERROR = 5
    RELEASE = 6
    PING = 7
    PONG = 8


class ProtocolError(ValueError):
    """Raised when a frame violates protocol v1."""


@dataclass(frozen=True)
class Frame:
    message_type: MessageType
    metadata: dict[str, Any]
    payload: bytes = b""


def encode_frame(frame: Frame) -> bytes:
    metadata = json.dumps(frame.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(metadata) > MAX_METADATA_BYTES:
        raise ProtocolError("metadata exceeds limit")
    if len(frame.payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("payload exceeds limit")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        int(frame.message_type),
        len(metadata),
        len(frame.payload),
        zlib.crc32(frame.payload) & 0xFFFFFFFF,
    )
    return header + metadata + frame.payload


def decode_frame(data: bytes) -> Frame:
    if len(data) < HEADER.size:
        raise ProtocolError("truncated header")
    magic, version, message_type, metadata_len, payload_len, crc32 = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ProtocolError("unsupported protocol")
    if metadata_len > MAX_METADATA_BYTES or payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError("declared frame exceeds limit")
    expected = HEADER.size + metadata_len + payload_len
    if len(data) != expected:
        raise ProtocolError("truncated or trailing frame data")
    try:
        metadata = json.loads(data[HEADER.size : HEADER.size + metadata_len])
        parsed_type = MessageType(message_type)
    except (ValueError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid frame metadata or type") from error
    payload = data[HEADER.size + metadata_len :]
    if zlib.crc32(payload) & 0xFFFFFFFF != crc32:
        raise ProtocolError("payload checksum mismatch")
    if not isinstance(metadata, dict):
        raise ProtocolError("metadata must be an object")
    return Frame(parsed_type, metadata, payload)

