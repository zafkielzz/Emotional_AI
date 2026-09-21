from controller.phonefarm_controller.protocol import Frame, MessageType, ProtocolError, decode_frame, encode_frame


def test_tensor_frame_round_trip() -> None:
    frame = Frame(MessageType.TENSOR, {"request_id": "r1", "shape": [1, 4], "dtype": "float32"}, b"abcd")
    assert decode_frame(encode_frame(frame)) == frame


def test_checksum_failure_is_rejected() -> None:
    data = bytearray(encode_frame(Frame(MessageType.TENSOR, {}, b"abcd")))
    data[-1] ^= 1
    try:
        decode_frame(bytes(data))
    except ProtocolError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("checksum failure must be rejected")


def test_golden_hex_frame() -> None:
    golden_hex = "504641520103000000330000000000000004ed82cd117b226474797065223a22666c6f61743332222c22726571756573745f6964223a227231222c227368617065223a5b312c345d7d61626364"
    frame = decode_frame(bytes.fromhex(golden_hex))
    assert frame.message_type == MessageType.TENSOR
    assert frame.metadata == {"dtype": "float32", "request_id": "r1", "shape": [1, 4]}
    assert frame.payload == b"abcd"
    assert encode_frame(frame).hex() == golden_hex


