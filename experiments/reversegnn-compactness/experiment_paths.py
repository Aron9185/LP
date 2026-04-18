import os
from pathlib import Path


EXPERIMENT_ROOT_ENV = "ARON_EXPERIMENT_ROOT"
DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIRNAME = "results"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_root():
    root = os.environ.get(EXPERIMENT_ROOT_ENV)
    if root:
        root_path = Path(root).expanduser()
        if root_path.name == DEFAULT_RESULTS_DIRNAME:
            return ensure_dir(root_path.parent)
        return ensure_dir(root_path)
    return ensure_dir(DEFAULT_EXPERIMENT_DIR)


def results_root():
    return ensure_dir(experiment_root() / DEFAULT_RESULTS_DIRNAME)


def sweep_log_dir():
    return ensure_dir(results_root() / "sweep_logs")


def artifact_path(*parts):
    path = results_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def repo_root():
    return Path(__file__).resolve().parents[2]
