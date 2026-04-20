"""HuggingFace to GGUF converter.

Converts HuggingFace model directories to GGUF format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from fanfu.errors import ConversionError, FileError
from fanfu.result import ToolResult

logger = logging.getLogger("fanfu.hf_to_gguf")


def load_hf_weights(hf_path: Path) -> dict[str, np.ndarray]:
    """Load weights from HF safetensors directory."""
    from safetensors import safe_open

    safetensors_files = list(hf_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileError(f"No safetensors files found in {hf_path}")

    weights = {}
    for st_file in safetensors_files:
        logger.info(f"Loading {st_file.name}...")
        with safe_open(st_file, framework="np") as f:
            for key in f.keys():
                hf_key = key
                if hf_key.startswith("model."):
                    hf_key = hf_key[6:]
                weights[hf_key] = f.get_tensor(key).astype(np.float32)

    return weights


def load_hf_config(hf_path: Path) -> dict[str, Any]:
    """Load model config from HF directory."""
    config_path = hf_path / "config.json"
    if not config_path.exists():
        raise FileError(f"config.json not found in {hf_path}")

    with open(config_path) as f:
        return json.load(f)


def quantize_f32_to_q8_0(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize F32 tensor to Q8_0 format.

    Q8_0: block_size=32, each block = 2 bytes (f16 scale) + 32 bytes (int8) = 34 bytes.

    Returns:
        Tuple of (quantized bytes as uint8 array, number of elements).
    """
    block_size = 32
    total_elements = data.size
    n_blocks = (total_elements + block_size - 1) // block_size

    flat = data.flatten().astype(np.float64)
    padded_size = n_blocks * block_size
    if padded_size > total_elements:
        flat = np.pad(flat, (0, padded_size - total_elements))

    result = np.zeros(n_blocks * 34, dtype=np.uint8)

    for i in range(n_blocks):
        block = flat[i * block_size:(i + 1) * block_size]
        max_abs = np.max(np.abs(block))
        if max_abs == 0:
            d = np.float16(0)
        else:
            d = np.float16(max_abs / 127.0)

        d_bytes = np.array([d], dtype=np.float16).view(np.uint8)
        offset = i * 34
        result[offset:offset + 2] = d_bytes

        if d != 0:
            qs = np.round(block / float(d)).clip(-128, 127).astype(np.int8)
        else:
            qs = np.zeros(block_size, dtype=np.int8)

        result[offset + 2:offset + 34] = qs.view(np.uint8)

    return result, total_elements


def convert_hf_to_gguf(
    hf_path: str,
    output_path: str,
    outtype: str = "f32",
) -> ToolResult:
    """Convert a HuggingFace model directory to GGUF format.

    Args:
        hf_path: Path to the HF model directory.
        output_path: Path to the output GGUF file.
        outtype: Output type (f32, f16, bf16, q8_0, auto).

    Returns:
        ToolResult with conversion status.
    """
    try:
        import gguf
    except ImportError:
        return ToolResult(success=False, error="gguf package not installed. Run: pip install gguf")

    hf = Path(hf_path)
    if not hf.exists():
        return ToolResult(success=False, error=f"HF directory not found: {hf}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        config = load_hf_config(hf)
    except FileError as e:
        return ToolResult(success=False, error=str(e))

    try:
        weights = load_hf_weights(hf)
    except FileError as e:
        return ToolResult(success=False, error=str(e))

    logger.info(f"Loaded {len(weights)} tensors from {hf}")

    arch = config.get("model_type", "llama")
    architectures = config.get("architectures", [])
    if architectures:
        arch_name = architectures[0].replace("ForCausalLM", "").lower()
    else:
        arch_name = arch

    n_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
    n_embd = config.get("hidden_size", config.get("n_embd", 0))
    n_head = config.get("num_attention_heads", config.get("n_head", 0))
    n_ffn = config.get("intermediate_size", config.get("n_inner", 0))
    vocab_size = config.get("vocab_size", 0)

    writer = gguf.GGUFWriter(str(output), arch_name)

    writer.add_uint32(gguf.Keys.General.ALIGNMENT, 32)

    if vocab_size:
        writer.add_uint32(gguf.Keys.Tokenizer.MODEL_MAX_LENGTH, vocab_size)

    if n_embd:
        writer.add_uint32(f"{arch}.embedding_length", n_embd)
    if n_head:
        writer.add_uint32(f"{arch}.attention.head_count", n_head)
    if n_layers:
        writer.add_uint32(f"{arch}.block_count", n_layers)
    if n_ffn:
        writer.add_uint32(f"{arch}.feed_forward_length", n_ffn)

    n_kv_heads = config.get("num_key_value_heads")
    if n_kv_heads is not None:
        if isinstance(n_kv_heads, list):
            writer.add_uint32_array(f"{arch}.attention.head_count_kv", n_kv_heads)
        else:
            writer.add_uint32(f"{arch}.attention.head_count_kv", n_kv_heads)

    rope_theta = config.get("rope_theta")
    if rope_theta:
        writer.add_float32(f"{arch}.rope.freq_base", float(rope_theta))

    rms_norm_eps = config.get("rms_norm_eps", config.get("layer_norm_eps"))
    if rms_norm_eps is not None:
        writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", float(rms_norm_eps))

    head_dim = config.get("head_dim")
    if head_dim:
        writer.add_uint32(f"{arch}.attention.key_length", head_dim)
        writer.add_uint32(f"{arch}.attention.value_length", head_dim)

    eos_token_id = config.get("eos_token_id")
    if eos_token_id is not None:
        writer.add_uint32(gguf.Keys.Tokenizer.EOS_ID, eos_token_id)

    bos_token_id = config.get("bos_token_id")
    if bos_token_id is not None:
        writer.add_uint32(gguf.Keys.Tokenizer.BOS_ID, bos_token_id)

    logger.info("Writing tensors...")
    for name, data in weights.items():
        if outtype == "q8_0" and data.ndim >= 2 and data.dtype == np.float32:
            quant_data, _ = quantize_f32_to_q8_0(data)
            writer.add_tensor(name, quant_data, tensor_type=gguf.GGMLQuantizationType.Q8_0)
        elif outtype == "f16" and data.ndim >= 2:
            writer.add_tensor(name, data.astype(np.float16))
        elif outtype == "bf16" and data.ndim >= 2:
            writer.add_tensor(name, data.astype(np.float32).view(np.uint32) >> 16)
        else:
            writer.add_tensor(name, data.astype(np.float32))

    logger.info(f"Saved GGUF file: {output}")
    return ToolResult(
        success=True,
        data={"output_path": str(output), "tensors": len(weights)},
    )
