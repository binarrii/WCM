"""Root-level pytest hooks.

Keeps the DeepFace-era ``test_tensorflow_init.py`` probe on disk (so it
survives as archaeology of how the migration started) but excludes it
from pytest discovery — otherwise ``uv run pytest`` at the project root
errors out at collection time on the missing ``tensorflow`` module.
"""

collect_ignore_glob = ["test_tensorflow_init.py"]
