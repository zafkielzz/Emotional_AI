# Segment contract

Segment boundaries carry hidden states, position IDs, request identity, and owned past/present KV tensors during decode. Do not transmit the whole KV cache between workers. A manifest identifies exact layer range, dtype, quantization, file hashes, and tensor schemas.
