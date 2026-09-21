# PhoneFarm binary protocol v1

All multi-byte fields use network byte order. The frame header is followed by a UTF-8 JSON metadata object and raw contiguous tensor bytes.

| Field | Type |
| --- | --- |
| magic | 4 bytes: `PFAR` |
| version | uint8: `1` |
| message type | uint8 |
| metadata length | uint32 |
| payload length | uint64 |
| payload CRC32 | uint32 |

Message types: `HELLO=1`, `ASSIGN=2`, `TENSOR=3`, `RESULT=4`, `ERROR=5`, `RELEASE=6`, `PING=7`, `PONG=8`.

Tensor metadata contains `request_id`, `segment_id`, `dtype`, `shape`, and optional `route`. Reject unsupported versions, invalid shape/dtype metadata, and checksum failures.
