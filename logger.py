import logging
import os

_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)

logger = logging.getLogger("ameath_agent")
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

    _file = logging.FileHandler(
        os.path.join(_LOG_DIR, "agent.log"), encoding="utf-8"
    )
    _file.setLevel(logging.DEBUG)
    _file.setFormatter(_fmt)
    logger.addHandler(_file)
