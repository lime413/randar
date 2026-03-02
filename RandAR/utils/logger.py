import logging
import os
import torch.distributed as dist


def _get_rank() -> int:
    """
    Safe rank getter.
    - Returns 0 when torch.distributed is not initialized (single GPU / single process).
    - Returns actual rank when running under DDP.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def create_logger(logging_dir: str):
    """
    Create a logger that writes to a log file and stdout.

    Works for:
    - single GPU / single process (no torch.distributed init)
    - multi-GPU DDP (torch.distributed initialized)
    """
    rank = _get_rank()

    logger = logging.getLogger(__name__)

    # Avoid adding handlers multiple times if create_logger() is called again.
    if getattr(logger, "_randar_configured", False):
        return logger

    if rank == 0:
        os.makedirs(logging_dir, exist_ok=True)
        log_path = os.path.join(logging_dir, "log.txt")

        logger.setLevel(logging.INFO)

        fmt = logging.Formatter(
            fmt='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.propagate = False  # prevent duplicate prints via root logger
    else:
        # Non-zero ranks: no-op logger
        logger.addHandler(logging.NullHandler())
        logger.propagate = False

    logger._randar_configured = True
    return logger