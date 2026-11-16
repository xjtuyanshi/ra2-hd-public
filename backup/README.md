# Backup tools and records

Restore and packaging tools for the historical Build 12 backup. For the current Build 13
pack and public download locations, start with [START_HERE.md](../START_HERE.md).
The incremental restorer is supplied by the Build 13 release; the older tool below
remains a Build 12 reference. [AI_SETUP.md](../AI_SETUP.md) preserves the Mac workflow.

- `env-build12.manifest.json`: every file of the environment release
  `env-build12-20261102` with size, sha256, mode and chunk segments (same as the release
  asset of that name).
- `python/`: pinned package lists for recreating the Python environments.
- `tools/`: farm tools (`build_manifest.py`, `pack_upload.py`, `run-upload.sh`,
  `restore_farm.py`), environment tools (`build_env_manifest.py`, `pack_env_upload.py`,
  `run-env-upload.sh`, `restore_env.py`, `check_models.py`) and
  `launcher-build12.command`. On the original machine they run from
  `gameres/s2x/backup/` (environment tools from `gameres/s2x/backup/env/tools/`).
- `LICENSES/`: third-party licence texts and model cards; see `LICENSES/README.md`.
