from __future__ import annotations

import logging
import sys


_LOGGER_NAMESPACE = "miru.edge"


def setup_edge_logging() -> None:
    """Configure timestamped stdout logging for edge runtime once."""
    logger = logging.getLogger(_LOGGER_NAMESPACE)
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the edge logging namespace."""
    prefix = "pipeline_jetson."
    suffix = name[len(prefix) :] if name.startswith(prefix) else name
    suffix = suffix.replace(".", "_")
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{suffix}")
