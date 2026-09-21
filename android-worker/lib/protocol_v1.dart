import 'dart:convert';
import 'dart:typed_data';

/// Magic constant for Protocol v1 frames ('PFAR').
final Uint8List protocolMagic = Uint8List.fromList([0x50, 0x46, 0x41, 0x52]);
const int protocolVersion = 1;
const int headerSize = 22;
const int maxMetadataBytes = 64 * 1024;
const int maxPayloadBytes = 512 * 1024 * 1024;

/// Protocol v1 message types.
enum MessageType {
  hello(1),
  assign(2),
  tensor(3),
  result(4),
  error(5),
  release(6),
  ping(7),
  pong(8);

  const MessageType(this.value);
  final int value;

  static MessageType fromValue(int val) {
    for (final type in MessageType.values) {
      if (type.value == val) return type;
    }
    throw ProtocolException('Unknown message type: $val');
  }
}

/// Thrown when a frame violates Protocol v1 specification.
class ProtocolException implements Exception {
  final String message;
  const ProtocolException(this.message);

  @override
  String toString() => 'ProtocolException: $message';
}

/// IEEE 802.3 32-bit Cyclic Redundancy Check.
class Crc32 {
  static final List<int> _table = _initTable();

  static List<int> _initTable() {
    final table = List<int>.filled(256, 0);
    for (int i = 0; i < 256; i++) {
      int c = i;
      for (int k = 0; k < 8; k++) {
        if ((c & 1) != 0) {
          c = 0xEDB88320 ^ (c >>> 1);
        } else {
          c = c >>> 1;
        }
      }
      table[i] = c;
    }
    return table;
  }

  static int compute(List<int> bytes) {
    int crc = 0xFFFFFFFF;
    for (final byte in bytes) {
      final index = (crc ^ byte) & 0xFF;
      crc = _table[index] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF;
  }
}

/// Binary frame in Protocol v1.
class Frame {
  final MessageType messageType;
  final Map<String, dynamic> metadata;
  final Uint8List payload;

  Frame(this.messageType, this.metadata, [Uint8List? payload])
      : payload = payload ?? Uint8List(0);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! Frame) return false;
    if (messageType != other.messageType) return false;
    if (jsonEncode(metadata) != jsonEncode(other.metadata)) return false;
    if (payload.length != other.payload.length) return false;
    for (int i = 0; i < payload.length; i++) {
      if (payload[i] != other.payload[i]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hash(
        messageType,
        jsonEncode(metadata),
        Object.hashAll(payload),
      );

  @override
  String toString() =>
      'Frame(type: $messageType, metadata: $metadata, payloadBytes: ${payload.length})';
}

/// Encodes a Frame into a Protocol v1 binary byte array.
Uint8List encodeFrame(Frame frame) {
  final metaBytes = utf8.encode(jsonEncode(frame.metadata));
  if (metaBytes.length > maxMetadataBytes) {
    throw const ProtocolException('metadata exceeds limit');
  }
  if (frame.payload.length > maxPayloadBytes) {
    throw const ProtocolException('payload exceeds limit');
  }

  final crc = Crc32.compute(frame.payload);
  final totalLength = headerSize + metaBytes.length + frame.payload.length;
  final buffer = Uint8List(totalLength);
  final byteData = ByteData.sublistView(buffer);

  // Magic: PFAR
  buffer[0] = 0x50;
  buffer[1] = 0x46;
  buffer[2] = 0x41;
  buffer[3] = 0x52;

  // Version & Type
  buffer[4] = protocolVersion;
  buffer[5] = frame.messageType.value;

  // Metadata Length (uint32)
  byteData.setUint32(6, metaBytes.length, Endian.big);

  // Payload Length (uint64)
  byteData.setUint64(10, frame.payload.length, Endian.big);

  // Payload CRC32 (uint32)
  byteData.setUint32(18, crc, Endian.big);

  // Metadata bytes
  buffer.setRange(headerSize, headerSize + metaBytes.length, metaBytes);

  // Payload bytes
  if (frame.payload.isNotEmpty) {
    buffer.setRange(
      headerSize + metaBytes.length,
      totalLength,
      frame.payload,
    );
  }

  return buffer;
}

/// Decodes a standalone Protocol v1 binary byte array into a Frame.
Frame decodeFrame(Uint8List data) {
  if (data.length < headerSize) {
    throw const ProtocolException('truncated header');
  }

  // Check magic
  if (data[0] != 0x50 ||
      data[1] != 0x46 ||
      data[2] != 0x41 ||
      data[3] != 0x52) {
    throw const ProtocolException('unsupported protocol');
  }

  final version = data[4];
  if (version != protocolVersion) {
    throw const ProtocolException('unsupported protocol');
  }

  final messageTypeValue = data[5];
  final byteData = ByteData.sublistView(data);
  final metadataLen = byteData.getUint32(6, Endian.big);
  final payloadLen = byteData.getUint64(10, Endian.big);
  final expectedCrc = byteData.getUint32(18, Endian.big);

  if (metadataLen > maxMetadataBytes || payloadLen > maxPayloadBytes) {
    throw const ProtocolException('declared frame exceeds limit');
  }

  final expectedTotal = headerSize + metadataLen + payloadLen;
  if (data.length != expectedTotal) {
    throw const ProtocolException('truncated or trailing frame data');
  }

  final MessageType messageType;
  try {
    messageType = MessageType.fromValue(messageTypeValue);
  } catch (e) {
    throw ProtocolException('invalid frame metadata or type: $e');
  }

  final metaSlice = data.sublist(headerSize, headerSize + metadataLen);
  final Map<String, dynamic> metadata;
  try {
    final decodedJson = jsonDecode(utf8.decode(metaSlice));
    if (decodedJson is! Map<String, dynamic>) {
      throw const ProtocolException('metadata must be an object');
    }
    metadata = decodedJson;
  } catch (e) {
    throw ProtocolException('invalid frame metadata or type: $e');
  }

  final payload = data.sublist(headerSize + metadataLen, expectedTotal);
  final actualCrc = Crc32.compute(payload);
  if (actualCrc != expectedCrc) {
    throw const ProtocolException('payload checksum mismatch');
  }

  return Frame(messageType, metadata, payload);
}

/// Stream frame accumulator for handling chunked TCP or WebSocket frames.
class FrameAccumulator {
  final List<int> _buffer = [];

  void add(List<int> chunk) {
    _buffer.addAll(chunk);
  }

  /// Extracts and returns the next complete Frame, or null if more bytes are required.
  Frame? tryReadFrame() {
    if (_buffer.length < headerSize) return null;

    final headerData = Uint8List.fromList(_buffer.sublist(0, headerSize));
    // Check magic
    if (headerData[0] != 0x50 ||
        headerData[1] != 0x46 ||
        headerData[2] != 0x41 ||
        headerData[3] != 0x52) {
      throw const ProtocolException('unsupported protocol');
    }
    if (headerData[4] != protocolVersion) {
      throw const ProtocolException('unsupported protocol');
    }

    final byteData = ByteData.sublistView(headerData);
    final metadataLen = byteData.getUint32(6, Endian.big);
    final payloadLen = byteData.getUint64(10, Endian.big);

    if (metadataLen > maxMetadataBytes || payloadLen > maxPayloadBytes) {
      throw const ProtocolException('declared frame exceeds limit');
    }

    final frameTotalSize = headerSize + metadataLen + payloadLen;
    if (_buffer.length < frameTotalSize) {
      return null; // Need more data
    }

    final frameBytes = Uint8List.fromList(_buffer.sublist(0, frameTotalSize));
    _buffer.removeRange(0, frameTotalSize);
    return decodeFrame(frameBytes);
  }

  void clear() {
    _buffer.clear();
  }
}
