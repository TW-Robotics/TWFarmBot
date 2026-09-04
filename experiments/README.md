# Experiments

Reproducible evaluations with their own configs and outputs. Each experiment
gets its own directory only while it is active — do not keep empty
placeholder directories.

Rules (see `docs/architecture.md`):

- Config-driven runs; results stay in the experiment's own `outputs/` folder.
- Delete or archive the directory when the evaluation is done.
