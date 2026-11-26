"""Simple logging utilities for microbiome ML."""

import contextlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Union


def setup_logging(level: str = "INFO") -> None:
    """Setup basic logging configuration.

    Not called on import: call it from the entry point of an application.
    Does nothing if the root logger already has handlers (``basicConfig``).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (defaults to package name)

    Returns:
        Logger instance
    """
    if name is None:
        name = __name__.split(".")[0]

    logger = logging.getLogger(name)

    # If no handlers, setup basic logging
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


@contextlib.contextmanager
def tee_log(
    path: Union[str, Path],
    attempt: int = 1,
    label: Optional[str] = None,
) -> Iterator[Path]:
    """Send stdout and stderr (file descriptors 1 and 2) to a log file.

    Everything that is printed, logged to a stream handler, written by a
    subprocess or shown in a traceback ends up in the file. This is the
    per-job log convention of the workflows: attempt 1 truncates the file,
    later attempts append, and each attempt starts with a header line.

    The descriptors are restored on exit, also when an exception escapes, so
    the traceback is written to the log first.

    Args:
        path: Log file; parent directories are created.
        attempt: Attempt number (Snakemake's ``attempt``); 1 overwrites.
        label: Text for the header line (e.g. the rule name).
    """
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    handle = open(log_path, "w" if int(attempt) <= 1 else "a")
    try:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        header = (
            f"=== {label or 'job'}, attempt {int(attempt)}, "
            f"{time.strftime('%F %T')} ==="
        )
        print(header, flush=True)
        try:
            yield log_path
        except BaseException:
            import traceback

            traceback.print_exc()
            raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        handle.close()
