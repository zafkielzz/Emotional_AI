---
name: phonefarm-build-model-segments
description: Split, export, validate, and quantize executable PhoneFarm Transformer segments. Use when working on Tiny GPT or Qwen layer ranges, ONNX export, segment manifests, numerical equivalence, model checksums, INT8, or INT4 artifacts.
---

# PhoneFarm model segments

Create executable contiguous layer-range segments, not checkpoint shards. Preserve a PC reference and make every generated artifact traceable through its manifest.

## Workflow

1. Inspect the source model and keep it read-only.
2. Define contiguous layer ranges and their input/output/KV-cache contract.
3. Validate composed PyTorch output before ONNX export.
4. Export one segment at a time; create a manifest with SHA-256 and tensor schema.
5. Validate PC ONNX output against its declared reference.
6. Quantize INT8 before attempting INT4; record operator fallbacks and accuracy changes.

Read `references/segment-contract.md` before changing a segment boundary. Use repository validation scripts rather than inventing an untracked comparison.
