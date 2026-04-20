"""FanFu - Bidirectional GGUF/HuggingFace converter with weight verification."""

__version__ = "0.1.0"

from .result import ToolResult
from .api import convert_gguf_to_hf, convert_hf_to_gguf, compare_weights
from .errors import (
    FanFuError,
    ConversionError,
    ValidationError,
    FileError,
    QuantizationError,
)
from .constants import APP_NAME, APP_VERSION, SUPPORTED_QUANT_TYPES

__all__ = [
    "__version__",
    "APP_NAME",
    "APP_VERSION",
    "SUPPORTED_QUANT_TYPES",
    "ToolResult",
    "convert_gguf_to_hf",
    "convert_hf_to_gguf",
    "compare_weights",
    "FanFuError",
    "ConversionError",
    "ValidationError",
    "FileError",
    "QuantizationError",
]
