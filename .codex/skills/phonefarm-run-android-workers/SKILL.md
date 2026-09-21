---
name: phonefarm-run-android-workers
description: Build, deploy, operate, and diagnose PhoneFarm Android worker APKs. Use for Kotlin worker code, ONNX Runtime Mobile, dashboard/ADB installation, controller registration, worker health, TCP tensor transport, and phone-farm connectivity probes.
---

# PhoneFarm Android workers

Build one worker APK that can be installed through the phone-farm dashboard. Prefer outbound connections to the controller relay so workers operate behind NAT or web-managed networks.

## Workflow

1. Build and install the same APK on two phones through dashboard upload or ADB.
2. Configure a worker ID and controller URL in the app.
3. Verify HTTP registration and persistent outbound TCP connectivity before loading a model.
4. Exchange golden synthetic tensors, then run a validated segment.
5. Keep a foreground service running and emit heartbeat/telemetry every five seconds.

Read `references/deployment.md` before real-device deployment. Never place device URLs or credentials in Git.
