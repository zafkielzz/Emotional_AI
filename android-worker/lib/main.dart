import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:phonefarm_worker/protocol_v1.dart';

void main() => runApp(const PhoneFarmApp());

class PhoneFarmApp extends StatelessWidget {
  const PhoneFarmApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'PhoneFarm Worker',
    theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
    home: const WorkerPage(),
  );
}

class WorkerPage extends StatefulWidget {
  const WorkerPage({super.key});
  @override
  State<WorkerPage> createState() => _WorkerPageState();
}

class _WorkerPageState extends State<WorkerPage> {
  final _workerId = TextEditingController(text: 'n9_01');
  final _controllerUrl = TextEditingController();
  final _controllerToken = TextEditingController();
  WorkerClient? _client;
  String _status = 'Stopped';
  String _log = 'Enter a Controller URL, then test it.';
  bool _busy = false;
  bool _downloadingModel = false;
  String _downloadProgress = '';

  Future<void> _test() async {
    setState(() {
      _busy = true;
      _log = 'Testing /healthz…';
    });
    try {
      final result = await WorkerClient(
        _controllerUrl.text,
        _workerId.text,
        controllerToken: _controllerToken.text,
      ).health();
      setState(
        () => _log = result
            ? 'Controller is reachable.'
            : 'Controller returned an unexpected response.',
      );
    } catch (error) {
      setState(() => _log = 'Cannot reach controller: $error');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _start() async {
    final client = WorkerClient(
      _controllerUrl.text,
      _workerId.text,
      controllerToken: _controllerToken.text,
      onLog: (line) {
        if (mounted) setState(() => _log = line);
      },
    );
    setState(() {
      _busy = true;
      _log = 'Registering worker…';
    });
    try {
      await client.start();
      _client = client;
      setState(() {
        _status = 'READY';
        _log = 'Registered; HTTP heartbeat and WebSocket relay are active.';
      });
    } catch (error) {
      await client.stop();
      setState(() {
        _status = 'ERROR';
        _log = 'Start failed: $error';
      });
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _stop() async {
    await _client?.stop();
    setState(() {
      _client = null;
      _status = 'Stopped';
      _log = 'Worker stopped.';
    });
  }

  Future<void> _downloadModel() async {
    if (_controllerUrl.text.trim().isEmpty) {
      setState(() => _log = 'Please enter Controller URL first.');
      return;
    }
    setState(() {
      _downloadingModel = true;
      _downloadProgress = 'Connecting...';
      _log = 'Connecting to download SLM model (469 MB)...';
    });

    try {
      final baseUrl = _controllerUrl.text.trim().replaceFirst(RegExp(r'/$'), '');
      final downloadUrl = Uri.parse(
        '$baseUrl/v1/models/download/qwen0.5b?token=${_controllerToken.text.trim()}',
      );

      final client = HttpClient();
      final request = await client.getUrl(downloadUrl);
      final response = await request.close();

      if (response.statusCode != 200) {
        throw HttpException('Download failed: HTTP ${response.statusCode}');
      }

      final totalBytes = response.contentLength;
      int receivedBytes = 0;

      Directory downloadDir = Directory('/sdcard/Download');
      try {
        if (!await downloadDir.exists()) {
          await downloadDir.create(recursive: true);
        }
      } catch (_) {
        downloadDir = Directory('/sdcard/Android/data/com.example.phonefarm_worker/files');
        if (!await downloadDir.exists()) {
          await downloadDir.create(recursive: true);
        }
      }
      final file = File('${downloadDir.path}/qwen2.5-0.5b-instruct-q4_k_m.gguf');
      final sink = file.openWrite();

      await for (final chunk in response) {
        sink.add(chunk);
        receivedBytes += chunk.length;
        if (totalBytes > 0 && mounted) {
          final pct = (receivedBytes * 100 / totalBytes).toStringAsFixed(1);
          final mbReceived = (receivedBytes / (1024 * 1024)).toStringAsFixed(1);
          final mbTotal = (totalBytes / (1024 * 1024)).toStringAsFixed(1);
          setState(() {
            _downloadProgress = '$pct% ($mbReceived/$mbTotal MB)';
          });
        }
      }
      await sink.close();
      client.close();

      if (mounted) {
        setState(() {
          _downloadingModel = false;
          _downloadProgress = 'Model Ready!';
          _log = 'Model saved to ${file.path} (${(receivedBytes / (1024 * 1024)).toStringAsFixed(1)} MB)!';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _downloadingModel = false;
          _downloadProgress = 'Download Failed';
          _log = 'Download error: $e';
        });
      }
    }
  }

  bool _testingSlm = false;

  Future<void> _testSlm() async {
    setState(() {
      _testingSlm = true;
      _log = 'Running local on-device SLM inference test...';
    });
    try {
      const channel = MethodChannel('com.phonefarm/inference');
      final res = await channel.invokeMapMethod<String, dynamic>('runInference', {
        'prompt': '<|im_start|>user\nXin chào! Hãy giới thiệu ngắn gọn trong 1 câu.<|im_end|>\n<|im_start|>assistant\n',
        'max_tokens': 32,
      });
      if (res != null) {
        if (res['status'] == 'success') {
          final txt = (res['response_text'] as String?)?.trim() ?? '';
          final lat = (res['latency_ms'] as num?)?.toDouble() ?? 0.0;
          final toks = (res['tokens_generated'] as num?)?.toInt() ?? 0;
          final tps = (res['tokens_per_second'] as num?)?.toDouble() ?? 0.0;
          setState(() {
            _log = 'SUCCESS!\nLat: ${lat.toStringAsFixed(0)}ms | Toks: $toks | Speed: ${tps.toStringAsFixed(1)} tok/s\nOutput: $txt';
          });
        } else {
          setState(() {
            _log = 'SLM FAILED:\n${res['error']}';
          });
        }
      } else {
        setState(() {
          _log = 'SLM returned null response.';
        });
      }
    } catch (e) {
      setState(() {
        _log = 'SLM test error: $e';
      });
    } finally {
      if (mounted) setState(() => _testingSlm = false);
    }
  }

  @override
  void dispose() {
    _client?.stop();
    _workerId.dispose();
    _controllerUrl.dispose();
    _controllerToken.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('PhoneFarm Worker')),
    body: Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: _workerId,
            enabled: _client == null,
            decoration: const InputDecoration(labelText: 'Worker ID'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _controllerUrl,
            enabled: _client == null,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              labelText: 'Controller URL',
              hintText: 'https://public-controller.example',
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _controllerToken,
            enabled: _client == null,
            obscureText: true,
            decoration: const InputDecoration(labelText: "Controller token"),
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              OutlinedButton(
                onPressed: _busy || _client != null ? null : _test,
                child: const Text('Test connection'),
              ),
              FilledButton(
                onPressed: _busy || _client != null ? null : _start,
                child: const Text('Start worker'),
              ),
              TextButton(
                onPressed: _client == null ? null : _stop,
                child: const Text('Stop'),
              ),
              ElevatedButton.icon(
                onPressed: _downloadingModel ? null : _downloadModel,
                icon: _downloadingModel
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.download),
                label: Text(
                  _downloadingModel
                      ? 'Downloading ($_downloadProgress)'
                      : 'Download SLM Model',
                ),
              ),
              ElevatedButton.icon(
                onPressed: _testingSlm ? null : _testSlm,
                icon: _testingSlm
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.psychology),
                label: Text(_testingSlm ? 'Testing SLM...' : 'Test SLM'),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Text(
            'Status: $_status',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          SelectableText(_log),
          const Spacer(),
          const Text(
            'MVP telemetry: network status is active. RAM, temperature and ONNX execution are added through the Flutter Android bridge in the next worker step.',
          ),
        ],
      ),
    ),
  );
}

class WorkerClient {
  WorkerClient(
    String controllerUrl,
    this.workerId, {
    this.controllerToken = "",
    this.onLog,
  }) : base = Uri.parse(controllerUrl.trim().replaceFirst(RegExp(r'/$'), ''));
  final Uri base;
  final String workerId;
  final String controllerToken;
  final void Function(String line)? onLog;
  Timer? _heartbeat;
  WebSocket? _relay;
  final FrameAccumulator _accumulator = FrameAccumulator();

  Future<bool> health() async {
    final response = await _request('GET', base.replace(path: '/healthz'));
    return response.statusCode == 200 &&
        jsonDecode(await response.transform(utf8.decoder).join())['status'] ==
            'ok';
  }

  Future<void> start() async {
    if (controllerToken.trim().isEmpty) {
      throw StateError("Controller token is required");
    }
    if (!await health()) throw StateError('health check failed');
    await _post('/v1/workers/register', {
      'worker_id': workerId,
      'capabilities': {'platform': 'flutter-android', 'protocol_version': 1},
    });
    await _heartbeatOnce();
    _heartbeat = Timer.periodic(
      const Duration(seconds: 5),
      (_) => _heartbeatOnce(),
    );
    await _openRelay();
  }

  Future<void> stop() async {
    _heartbeat?.cancel();
    _heartbeat = null;
    await _relay?.close();
    _relay = null;
    _accumulator.clear();
  }

  static const _telemetryChannel = MethodChannel('com.phonefarm/telemetry');
  static const _inferenceChannel = MethodChannel('com.phonefarm/inference');

  Future<void> _heartbeatOnce() async {
    try {
      Map<String, dynamic> metrics = {
        'status': 'ready',
        'telemetry_source': 'flutter-worker',
      };
      if (Platform.isAndroid) {
        try {
          final nativeMetrics = await _telemetryChannel
              .invokeMapMethod<String, dynamic>('getMetrics');
          if (nativeMetrics != null) {
            metrics.addAll(nativeMetrics);
          }
        } catch (_) {
          // Fallback to basic metrics if channel is unavailable in tests/mocks
        }
      }
      await _put('/v1/workers/$workerId/heartbeat', {'metrics': metrics});
    } catch (error) {
      onLog?.call('Heartbeat failed: $error');
    }
  }

  Future<void> _openRelay() async {
    if (_heartbeat == null) return;
    try {
      final relayScheme = switch (base.scheme) {
        "https" => "wss",
        "http" => "ws",
        _ => throw ArgumentError("Controller URL must use http or https"),
      };
      final relayUri = base.replace(
        scheme: relayScheme,
        path: "/v1/relay",
        query: null,
        fragment: null,
      );
      _relay = await WebSocket.connect(
        relayUri.toString(),
        headers: {"X-PhoneFarm-Token": controllerToken},
      ).timeout(const Duration(seconds: 8));

      // Send HELLO frame
      final helloFrame = Frame(MessageType.hello, {"worker_id": workerId});
      _relay!.add(encodeFrame(helloFrame));

      _relay!.listen(
        (message) {
          if (message is List<int>) {
            try {
              _accumulator.add(message);
              while (true) {
                final frame = _accumulator.tryReadFrame();
                if (frame == null) break;
                _handleIncomingFrame(frame);
              }
            } catch (e) {
              onLog?.call("Frame parse error: $e");
            }
          }
        },
        onDone: () {
          onLog?.call("Relay disconnected. Retrying in 3s...");
          _relay = null;
          if (_heartbeat != null) {
            Future.delayed(const Duration(seconds: 3), () {
              if (_heartbeat != null && _relay == null) {
                _openRelay();
              }
            });
          }
        },
        onError: (error) {
          onLog?.call("Relay error: $error");
          _relay = null;
          if (_heartbeat != null) {
            Future.delayed(const Duration(seconds: 3), () {
              if (_heartbeat != null && _relay == null) {
                _openRelay();
              }
            });
          }
        },
      );
    } catch (e) {
      onLog?.call("Failed to connect relay: $e. Retrying in 5s...");
      if (_heartbeat != null) {
        Future.delayed(const Duration(seconds: 5), () {
          if (_heartbeat != null && _relay == null) {
            _openRelay();
          }
        });
      }
    }
  }

  void _handleIncomingFrame(Frame frame) {
    switch (frame.messageType) {
      case MessageType.hello:
        onLog?.call("Received HELLO ack from relay.");
        break;
      case MessageType.assign:
        final reqId = frame.metadata['request_id']?.toString() ?? 'req';
        final sourceOrch = frame.metadata['source_worker_id']?.toString() ?? 'orchestrator';
        final role = frame.metadata['role']?.toString() ?? 'Specialist';
        final query = frame.metadata['query']?.toString() ??
            utf8.decode(frame.payload, allowMalformed: true);

        onLog?.call("ASSIGN [$role]: $query");
        _processAssignTask(reqId, sourceOrch, role, query);
        break;
      case MessageType.tensor:
        final reqId = frame.metadata['request_id'];
        final shape = frame.metadata['shape'];
        final dtype = frame.metadata['dtype'];
        onLog?.call(
          "Received TENSOR: req=$reqId, shape=$shape, dtype=$dtype (${frame.payload.length} bytes).",
        );
        break;
      case MessageType.result:
        onLog?.call(
          "Received RESULT: req=${frame.metadata['request_id']}.",
        );
        break;
      case MessageType.error:
        onLog?.call(
          "Received ERROR: ${frame.metadata['error'] ?? frame.metadata}.",
        );
        break;
      case MessageType.ping:
        final pong = Frame(MessageType.pong, {"worker_id": workerId});
        _relay?.add(encodeFrame(pong));
        break;
      case MessageType.pong:
        break;
      case MessageType.release:
        onLog?.call(
          "Received RELEASE for req=${frame.metadata['request_id']}.",
        );
        break;
    }
  }

  Future<void> _processAssignTask(
    String reqId,
    String destination,
    String role,
    String query,
  ) async {
    final sw = Stopwatch()..start();
    String answer = '';
    double elapsedMs = 0.0;
    int tokensGenerated = 0;
    double tokensPerSec = 0.0;
    String engineUsed = 'template_generator';
    String? engineError;

    if (Platform.isAndroid) {
      try {
        final prompt =
            "<|im_start|>system\nBạn là $role trong hệ thống PhoneFarm Edge Multi-Agent. "
            "Hãy phân tích ngắn gọn, súc tích (dưới 80 từ) câu hỏi sau theo đúng chuyên môn của bạn.<|im_end|>\n"
            "<|im_start|>user\n$query<|im_end|>\n<|im_start|>assistant\n";

        final nativeRes = await _inferenceChannel
            .invokeMapMethod<String, dynamic>(
          'runInference',
          {
            'model_path':
                '/sdcard/Download/qwen2.5-0.5b-instruct-q4_k_m.gguf',
            'prompt': prompt,
            'max_tokens': 96,
          },
        );

        if (nativeRes != null) {
          if (nativeRes['status'] == 'success') {
            final text = (nativeRes['response_text'] as String?)?.trim() ?? '';
            if (text.isNotEmpty) {
              answer = text;
              elapsedMs = (nativeRes['latency_ms'] as num?)?.toDouble() ?? 0.0;
              tokensGenerated =
                  (nativeRes['tokens_generated'] as num?)?.toInt() ?? 0;
              tokensPerSec =
                  (nativeRes['tokens_per_second'] as num?)?.toDouble() ?? 0.0;
              engineUsed = 'llama_cpp_on_device';
            }
          } else {
            engineError = nativeRes['error']?.toString();
            onLog?.call("Engine error: $engineError");
          }
        }
      } catch (e) {
        engineError = e.toString();
        onLog?.call("On-device inference fallback: $e");
      }
    }

    if (answer.isEmpty) {
      answer = _generatePersonaResponse(role, query);
      sw.stop();
      elapsedMs = sw.elapsedMilliseconds.toDouble();
      tokensGenerated = answer.split(RegExp(r'\s+')).length;
    }

    final payloadBytes = utf8.encode(answer);
    final resultFrame = Frame(
      MessageType.result,
      {
        'request_id': reqId,
        'source_worker_id': workerId,
        'destination_worker_id': destination,
        'role': role,
        'response_text': answer,
        'latency_ms': elapsedMs,
        'tokens_generated': tokensGenerated,
        'metrics': {
          'execution_engine': engineUsed,
          'model': 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
          'tokens_per_second': tokensPerSec,
          if (engineError != null) 'diagnostic_error': engineError,
        },
      },
      Uint8List.fromList(payloadBytes),
    );

    _relay?.add(encodeFrame(resultFrame));
    if (engineUsed == 'llama_cpp_on_device') {
      onLog?.call(
        "Sent RESULT [$role] via LLM in ${elapsedMs.toStringAsFixed(1)}ms (${tokensPerSec.toStringAsFixed(1)} tok/s)",
      );
    } else {
      onLog?.call(
        "Sent RESULT [$role] via template (Error: ${engineError ?? 'none'})",
      );
    }
  }

  String _generatePersonaResponse(String role, String query) {
    final lowerRole = role.toLowerCase();
    if (lowerRole.contains("strategist") || lowerRole.contains("planner")) {
      return "[$role] Về mặt chiến lược: Bài toán '$query' cần được bóc tách theo 3 giai đoạn: "
          "1) Thiết lập môi trường và tiền xử lý; 2) Triển khai song song theo mô hình phân tán; 3) Tổng hợp và kiểm định chất lượng.";
    } else if (lowerRole.contains("knowledge") || lowerRole.contains("fact")) {
      return "[$role] Về mặt tri thức: Đối với '$query', các nguyên lý quan trọng gồm có: "
          "định luật Amdahl về giới hạn song song hóa, cơ chế giảm tần số (DVFS) khi quá nhiệt, và lượng tử hóa INT4/INT8 để tối ưu RAM.";
    } else if (lowerRole.contains("critic") || lowerRole.contains("adversarial")) {
      return "[$role] Phản biện & Cảnh báo: Cần đặc biệt lưu ý rủi ro nghẽn cổ chai mạng (network bottleneck), "
          "sự trôi lệch pin/nhiệt độ giữa các thiết bị không đồng nhất, và hiện tượng trễ không đồng bộ (straggler effect).";
    } else if (lowerRole.contains("creative") || lowerRole.contains("alternative")) {
      return "[$role] Góc nhìn đột phá: Thay vì chỉ suy luận thuần túy, có thể ứng dụng cơ chế speculative drafting lai ghép "
          "hoặc caching ngữ cảnh thông minh (KV-cache reuse) giữa các phiên truy vấn tương tự.";
    } else if (lowerRole.contains("pragmatic") || lowerRole.contains("feasibility")) {
      return "[$role] Đánh giá khả thi: Với 5 máy Galaxy Note 9 (RAM 6GB-8GB), mô hình SLM 0.5B-1.5B chạy mượt mà mà không lo thiếu RAM. "
          "Khuyến nghị đặt nhiệt độ trần ở 42°C để bảo vệ thiết bị.";
    } else {
      return "[$role] Hoàn tất phân tích chuyên sâu cho yêu cầu: '$query'.";
    }
  }

  Future<HttpClientResponse> _request(String method, Uri uri) async {
    final request = await HttpClient()
        .openUrl(method, uri)
        .timeout(const Duration(seconds: 5));
    return request.close();
  }

  Future<void> _post(String path, Map<String, dynamic> body) =>
      _send('POST', path, body);
  Future<void> _put(String path, Map<String, dynamic> body) =>
      _send('PUT', path, body);
  Future<void> _send(
    String method,
    String path,
    Map<String, dynamic> body,
  ) async {
    final request = await HttpClient().openUrl(
      method,
      base.replace(path: path),
    );
    request.headers.contentType = ContentType.json;
    request.headers.add("X-PhoneFarm-Token", controllerToken);
    request.write(jsonEncode(body));
    final response = await request.close();
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException('$method $path returned ${response.statusCode}');
    }
  }
}
