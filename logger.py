import logging
import os

_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)

logger = logging.getLogger("Pc_TCN")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _console = logging.StreamHandler()
    _console.setLevel(logging.INFO)
    _console.setFormatter(_fmt)
    logger.addHandler(_console)


def setup_file_handler(run_id: str):
    _file = logging.FileHandler(
        os.path.join(_LOG_DIR, f"Pc_TCN_{run_id}.log"), encoding="utf-8"
    )
    _file.setLevel(logging.DEBUG)
    _fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _file.setFormatter(_fmt)
    logger.addHandler(_file)
    return run_id
