import os
from pathlib import Path


EXPERIMENT_ROOT_ENV = "ARON_EXPERIMENT_ROOT"
DEFAULT_EXPERIMENT_ROOT = (
    Path.home()
    / ".gemini"
    / "antigravity"
    / "experiments"
    / "ARON"
    / "reversegnn-compactness"
)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_root():
    root = os.environ.get(EXPERIMENT_ROOT_ENV)
    if root:
        return ensure_dir(Path(root).expanduser())
    return ensure_dir(DEFAULT_EXPERIMENT_ROOT)


def sweep_log_dir():
    return ensure_dir(experiment_root() / "sweep_logs")


def artifact_path(*parts):
    path = experiment_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
