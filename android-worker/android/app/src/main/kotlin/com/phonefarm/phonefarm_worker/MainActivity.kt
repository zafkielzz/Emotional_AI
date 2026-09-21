package com.phonefarm.phonefarm_worker

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.phonefarm/telemetry"
    private val INFERENCE_CHANNEL = "com.phonefarm/inference"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Telemetry channel
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "getMetrics" -> {
                    try {
                        val metrics = getSystemMetrics()
                        result.success(metrics)
                    } catch (e: Exception) {
                        result.error("METRICS_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }

        // On-Device SLM Inference channel
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, INFERENCE_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "checkModel" -> {
                    val path = call.argument<String>("model_path") ?: "/sdcard/Download/qwen2.5-0.5b-instruct-q4_k_m.gguf"
                    val file = File(path)
                    result.success(mapOf(
                        "exists" to file.exists(),
                        "size_bytes" to if (file.exists()) file.length() else 0L,
                        "path" to file.absolutePath
                    ))
                }
                "checkEngine" -> {
                    val cliFile = File(filesDir, "llama-cli")
                    result.success(mapOf(
                        "cli_ready" to (cliFile.exists() && cliFile.canExecute()),
                        "native_dir" to applicationInfo.nativeLibraryDir
                    ))
                }
                "runInference" -> {
                    val modelPath = call.argument<String>("model_path") ?: "/sdcard/Download/qwen2.5-0.5b-instruct-q4_k_m.gguf"
                    val prompt = call.argument<String>("prompt") ?: ""
                    val maxTokens = call.argument<Int>("max_tokens") ?: 96

                    thread(name = "llama-inference") {
                        try {
                            val resp = executeLlamaCli(modelPath, prompt, maxTokens)
                            runOnUiThread { result.success(resp) }
                        } catch (e: Exception) {
                            runOnUiThread {
                                result.success(mapOf(
                                    "status" to "error",
                                    "error" to (e.message ?: "Unknown error"),
                                    "response_text" to ""
                                ))
                            }
                        }
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun extractAsset(assetPath: String, destFile: File) {
        val parent = destFile.parentFile
        if (parent != null && !parent.exists()) {
            parent.mkdirs()
        }
        assets.open(assetPath).use { input ->
            FileOutputStream(destFile).use { output ->
                input.copyTo(output)
            }
        }
        destFile.setExecutable(true, false)
    }

    private fun executeLlamaCli(modelPath: String, prompt: String, maxTokens: Int): Map<String, Any> {
        val candidates = listOf(
            modelPath,
            "/storage/emulated/0/Download/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "/sdcard/Download/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            File(getExternalFilesDir(null), "qwen2.5-0.5b-instruct-q4_k_m.gguf").absolutePath,
            File(filesDir, "qwen2.5-0.5b-instruct-q4_k_m.gguf").absolutePath,
            "/sdcard/Android/data/com.phonefarm.phonefarm_worker/files/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )
        val modelFile = candidates.map { File(it) }.firstOrNull { it.exists() && it.length() > 1000000L }
        if (modelFile == null) {
            val checked = candidates.joinToString(" | ") { "$it (exists=${File(it).exists()}, len=${if (File(it).exists()) File(it).length() else 0})" }
            return mapOf(
                "status" to "error",
                "error" to "Model not found. Checked: $checked",
                "response_text" to ""
            )
        }

        val nativeCli = File(applicationInfo.nativeLibraryDir, "libllama-cli.so")
        if (nativeCli.exists()) {
            nativeCli.setExecutable(true, false)
        }
        val cliFile = if (nativeCli.exists()) {
            nativeCli
        } else {
            val localCli = File(filesDir, "llama-cli")
            if (!localCli.exists() || !localCli.canExecute()) {
                try {
                    extractAsset("bin/arm64/llama-cli", localCli)
                    localCli.setExecutable(true, false)
                } catch (e: Exception) {
                    // fallback
                }
            }
            if (localCli.exists() && localCli.canExecute()) localCli else nativeCli
        }

        if (!cliFile.exists()) {
            val nativeDirFiles = File(applicationInfo.nativeLibraryDir).list()?.joinToString() ?: "empty/null"
            return mapOf(
                "status" to "error",
                "error" to "Executable not found at ${cliFile.absolutePath}. Native dir contains: [$nativeDirFiles]",
                "response_text" to ""
            )
        }

        val startTime = System.currentTimeMillis()
        val pb = ProcessBuilder(
            cliFile.absolutePath,
            "-m", modelFile.absolutePath,
            "-p", prompt,
            "-n", maxTokens.toString(),
            "-c", "512",
            "-t", "4",
            "--temp", "0.7",
            "--no-warmup",
            "--single-turn",
            "--color", "off",
            "--log-colors", "off",
            "--show-timings"
        )
        val currentLd = System.getenv("LD_LIBRARY_PATH") ?: "/system/lib64"
        pb.environment()["LD_LIBRARY_PATH"] = "${applicationInfo.nativeLibraryDir}:$currentLd"
        pb.redirectErrorStream(true)

        try {
            val process = pb.start()
            // Close stdin immediately so the CLI knows no interactive input is coming
            try { process.outputStream.close() } catch (_: Exception) {}

            var output = ""
            val readerThread = thread(name = "cli-stream-reader") {
                try {
                    output = process.inputStream.bufferedReader().use { it.readText() }
                } catch (e: Exception) {
                    output = "Error reading output: ${e.message}"
                }
            }

            val finished = process.waitFor(45, java.util.concurrent.TimeUnit.SECONDS)
            if (!finished) {
                process.destroyForcibly()
                val elapsedMs = (System.currentTimeMillis() - startTime).toDouble()
                return mapOf(
                    "status" to "error",
                    "error" to "Inference timed out after 45s. Output so far: ${output.takeLast(400)}",
                    "response_text" to "",
                    "latency_ms" to elapsedMs
                )
            }

            readerThread.join(2000)
            val exitCode = process.exitValue()
            val elapsedMs = (System.currentTimeMillis() - startTime).toDouble()

            if (exitCode != 0) {
                return mapOf(
                    "status" to "error",
                    "exit_code" to exitCode,
                    "error" to "Exit $exitCode: ${output.takeLast(600)}",
                    "response_text" to "",
                    "latency_ms" to elapsedMs
                )
            }

            val cleanOutput = output.replace(Regex("\u001B\\[[;\\d]*m"), "")
            val generatedText = if (cleanOutput.contains("Assistant:\n")) {
                cleanOutput.substringAfter("Assistant:\n").substringBefore("\n[ Prompt:").substringBefore("\nUser:").trim()
            } else if (cleanOutput.contains(prompt)) {
                cleanOutput.substringAfter(prompt).substringBefore("\n[ Prompt:").trim()
            } else {
                cleanOutput.substringBefore("\n[ Prompt:").trim()
            }

            val tokenMatch = Regex("Generation:\\s+([\\d\\.]+)\\s+t/s").find(cleanOutput)
            val measuredTps = tokenMatch?.groupValues?.get(1)?.toDoubleOrNull()

            val tokenCount = generatedText.split(Regex("\\s+")).filter { it.isNotBlank() }.size
            val tokensPerSec = measuredTps ?: if (elapsedMs > 0) (tokenCount / (elapsedMs / 1000.0)) else 0.0

            return mapOf(
                "status" to "success",
                "response_text" to generatedText,
                "latency_ms" to elapsedMs,
                "tokens_generated" to tokenCount,
                "tokens_per_second" to tokensPerSec
            )
        } catch (e: Exception) {
            val elapsedMs = (System.currentTimeMillis() - startTime).toDouble()
            return mapOf(
                "status" to "error",
                "error" to "Exception running cli (${cliFile.absolutePath}): ${e.message}",
                "response_text" to "",
                "latency_ms" to elapsedMs
            )
        }
    }

    private fun getSystemMetrics(): Map<String, Any?> {
        val memoryInfo = ActivityManager.MemoryInfo()
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        activityManager?.getMemoryInfo(memoryInfo)

        val batteryIntent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val rawTemp = batteryIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val tempCelsius = if (rawTemp != Int.MIN_VALUE) rawTemp / 10.0 else null

        return mapOf(
            "ram_free_bytes" to (memoryInfo.availMem),
            "ram_total_bytes" to (memoryInfo.totalMem),
            "battery_temperature_celsius" to tempCelsius,
            "status" to "ready"
        )
    }
}

