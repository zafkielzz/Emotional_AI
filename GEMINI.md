# PhoneFarm project rules

- Treat `PHONE_FARM_DISTRIBUTED_AI_IMPLEMENTATION_BRIEF.md` as research scope and `IMPLEMENTATION_PLAN.md` as stage status and acceptance evidence.
- Work through stages in order. Do not begin Qwen3 deployment before the Tiny GPT two-phone prefill and decode tests pass.
- Use Conda environment `capstone`; preserve its PyTorch/CUDA installation. Add only missing project dependencies and record resolved versions.
- Treat `/media/zafkiel/WORK_SPACE2/models/Qwen3-8B` as read-only. Store generated artifacts only under `/media/zafkiel/WORK_SPACE2/phonefarm_artifacts` with external-data chunks at most 2 GiB.
- Keep weights, APKs, build output, metrics, device IPs, URLs, and credentials out of Git.
- Maintain a numerical reference for every model, segment manifest, and quantization form. CPU/FP32 is the first correctness baseline; INT8 precedes INT4; accelerators are benchmarks only.
- Use protocol v1 frames for every activation path. KV cache stays with its owning worker and active requests are never migrated.
- Record receive, deserialize, inference, serialize, send, RAM, temperature, and error metrics for every real worker request.
- Keep Android deployment compatible with web dashboard installation and outbound connections. Controller relay is mandatory; direct phone-to-phone routing is optional after probing.
- Run focused tests for changed code. Do not claim a performance benefit without an experiment manifest and raw JSONL evidence.

# Safety & File Protection Rules

- **STRICT PROHIBITION ON FILE/FOLDER REMOVAL**: NEVER run destructive removal commands (including `rm`, `rm -rf`, `rmdir`, `unlink`, `git clean`, or Python `os.remove`/`shutil.rmtree`) on any files, folders, code, models, weights, datasets, or documentation.
- **NO AUTOMATIC DELETION**: Do NOT delete any user files or project directories. If file cleanup is ever deemed necessary, always STOP and ask the user for explicit permission with the full list of files before proceeding.
- **NON-DESTRUCTIVE OPERATIONS ONLY**: Always prefer safe in-place modifications, non-destructive refactoring, renaming, archiving, or creating backups over deleting files.
