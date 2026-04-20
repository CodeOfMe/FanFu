"""Custom error classes for FanFu."""


class FanFuError(Exception):
    """Base exception for all FanFu errors."""


class ConversionError(FanFuError):
    """Raised when a conversion fails."""


class ValidationError(FanFuError):
    """Raised when input validation fails."""


class FileError(FanFuError):
    """Raised when a file operation fails."""


class QuantizationError(FanFuError):
    """Raised when quantization/dequantization fails."""


class ArchitectureError(FanFuError):
    """Raised when model architecture is unsupported."""
