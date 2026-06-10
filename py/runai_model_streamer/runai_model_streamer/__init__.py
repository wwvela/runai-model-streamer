from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
    ObjectStorageModel,
    list_safetensors,
    pull_files,
)
from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.file_streamer.requests_iterator import FileChunks
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    DistributedStreamer,
)

import os
import logging

__all__ = [
    "SafetensorsStreamer",
    "ObjectStorageModel",
    "DistributedStreamer",
    "FileStreamer",
    "FileChunks",
    "list_safetensors",
    "pull_files",
]

logger_level = os.environ.get("RUNAI_STREAMER_LOG_LEVEL", "INFO").upper()

logging.getLogger(__name__).setLevel(logger_level)

_logger = logging.getLogger(__name__)
_logger.setLevel(logger_level)
if os.environ.get("RUNAI_STREAMER_LOG_LEVEL") and not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(levelname)s %(asctime)s %(name)s: %(message)s")
    )
    _logger.addHandler(_handler)
