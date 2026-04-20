"""Public API for FanFu."""

from __future__ import annotations

from fanfu.result import ToolResult
from fanfu.gguf_to_hf import convert_gguf_to_hf as _convert_gguf_to_hf
from fanfu.hf_to_gguf import convert_hf_to_gguf as _convert_hf_to_gguf
from fanfu.compare import compare_weights as _compare_weights


def convert_gguf_to_hf(
    gguf_path: str,
    output_dir: str,
    outtype: str = "f32",
    extract_tokenizer: bool = True,
) -> ToolResult:
    """Convert a GGUF file to HuggingFace format.

    Args:
        gguf_path: Path to the input GGUF file.
        output_dir: Path to the output directory.
        outtype: Output float type (f32, f16, bf16).
        extract_tokenizer: Whether to extract tokenizer from GGUF.

    Returns:
        ToolResult with conversion status and output path.
    """
    return _convert_gguf_to_hf(gguf_path, output_dir, outtype, extract_tokenizer)


def convert_hf_to_gguf(
    hf_path: str,
    output_path: str,
    outtype: str = "f32",
) -> ToolResult:
    """Convert a HuggingFace model directory to GGUF format.

    Args:
        hf_path: Path to the HF model directory.
        output_path: Path to the output GGUF file.
        outtype: Output quantization type (f32, f16, bf16, q8_0, auto).

    Returns:
        ToolResult with conversion status and output path.
    """
    return _convert_hf_to_gguf(hf_path, output_path, outtype)


def compare_weights(
    gguf_path: str,
    hf_path: str,
    tolerance: float = 0.5,
) -> ToolResult:
    """Compare weights between a GGUF file and a HF model directory.

    Args:
        gguf_path: Path to the GGUF file.
        hf_path: Path to the HF model directory.
        tolerance: Maximum allowed absolute difference for matching.

    Returns:
        ToolResult with comparison statistics.
    """
    return _compare_weights(gguf_path, hf_path, tolerance)
