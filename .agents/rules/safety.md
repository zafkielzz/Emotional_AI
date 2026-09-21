# Safety & File Protection Rules

- **Strict prohibition on file/folder removal**: NEVER execute destructive removal commands (such as `rm`, `rm -rf`, `rmdir`, `unlink`, `git clean`, or programmatic deletions like `os.remove`/`shutil.rmtree`) on any files, folders, code, models, weights, datasets, or documentation.
- **No silent/automatic deletion**: Never delete any project files or directories. If file cleanup is ever deemed necessary, always pause and ask the user for explicit confirmation with the list of affected files before proceeding.
- **Non-destructive operations only**: Always prioritize safe in-place modifications, non-destructive refactoring, renaming, creating backups, or archiving rather than deleting files.
