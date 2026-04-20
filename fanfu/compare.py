"""Weight comparison between GGUF and HuggingFace models."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from fanfu.errors import ConversionError, FileError, QuantizationError
from fanfu.result import ToolResult
from fanfu.gguf_to_hf import dequantize_tensor

logger = logging.getLogger("fanfu.compare")

HF_TO_GGUF_MAPPING: dict[str, str] = {}


def _build_mapping():
    """Build HF to GGUF name mapping for common architectures."""
    for layer_id in range(100):
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.input_layernorm.weight"] = f"blk.{layer_id}.attn_norm.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.post_attention_layernorm.weight"] = f"blk.{layer_id}.post_attention_norm.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.self_attn.qkv_proj.weight"] = f"blk.{layer_id}.attn_qkv.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.self_attn.gate_proj.weight"] = f"blk.{layer_id}.attn_gate.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mlp.gate_proj.weight"] = f"blk.{layer_id}.ffn_gate.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mlp.up_proj.weight"] = f"blk.{layer_id}.ffn_up.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mlp.down_proj.weight"] = f"blk.{layer_id}.ffn_down.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mamba.norm.weight"] = f"blk.{layer_id}.ssm_norm.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mamba.alpha.weight"] = f"blk.{layer_id}.ssm_alpha.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mamba.beta.weight"] = f"blk.{layer_id}.ssm_beta.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mamba.conv1d.weight"] = f"blk.{layer_id}.ssm_conv1d.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mamba.out_proj.weight"] = f"blk.{layer_id}.ssm_out.weight"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mamba.A_log"] = f"blk.{layer_id}.ssm_A"
        HF_TO_GGUF_MAPPING[f"layers.{layer_id}.mamba.dt_proj.bias"] = f"blk.{layer_id}.ssm_dt"

    HF_TO_GGUF_MAPPING["embed_tokens.weight"] = "token_embd.weight"
    HF_TO_GGUF_MAPPING["lm_head.weight"] = "output.weight"


_build_mapping()


def map_hf_to_gguf_name(hf_name: str) -> str | None:
    """Map HF tensor name to GGUF tensor name."""
    if hf_name in HF_TO_GGUF_MAPPING:
        return HF_TO_GGUF_MAPPING[hf_name]

    layer_match = re.match(r"layers\.(\d+)\.(.+)", hf_name)
    if layer_match:
        layer_id = int(layer_match.group(1))
        sub_name = layer_match.group(2)
        key = f"layers.{layer_id}.{sub_name}"
        if key in HF_TO_GGUF_MAPPING:
            return HF_TO_GGUF_MAPPING[key]

    return None


def load_gguf_weights(gguf_path: Path) -> dict[str, np.ndarray]:
    """Load weights from GGUF file."""
    try:
        import gguf
    except ImportError:
        raise ImportError("gguf package not installed. Run: pip install gguf")

    reader = gguf.GGUFReader(str(gguf_path), "r")
    weights = {}
    for tensor in reader.tensors:
        name = tensor.name
        data = dequantize_tensor(tensor)
        weights[name] = data

    return weights


def load_hf_weights(hf_path: Path) -> dict[str, np.ndarray]:
    """Load weights from HF safetensors directory."""
    from safetensors import safe_open

    safetensors_files = list(hf_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileError(f"No safetensors files found in {hf_path}")

    weights = {}
    for st_file in safetensors_files:
        with safe_open(st_file, framework="np") as f:
            for key in f.keys():
                hf_key = key
                if hf_key.startswith("model."):
                    hf_key = hf_key[6:]
                weights[hf_key] = f.get_tensor(key).astype(np.float32)

    return weights


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
    gguf_p = Path(gguf_path)
    hf_p = Path(hf_path)

    if not gguf_p.exists():
        return ToolResult(success=False, error=f"GGUF file not found: {gguf_p}")
    if not hf_p.exists():
        return ToolResult(success=False, error=f"HF directory not found: {hf_p}")

    try:
        logger.info("Loading GGUF weights...")
        gguf_weights = load_gguf_weights(gguf_p)
        logger.info(f"Loaded {len(gguf_weights)} tensors")

        logger.info("Loading HF weights...")
        hf_weights = load_hf_weights(hf_p)
        logger.info(f"Loaded {len(hf_weights)} tensors")
    except Exception as e:
        return ToolResult(success=False, error=str(e))

    matched = 0
    mismatched = 0
    shape_mismatch = 0
    gguf_only = 0
    hf_only = 0
    max_diff = 0.0
    max_diff_name = ""
    diff_details = []

    gguf_keys = set(gguf_weights.keys())
    hf_keys = set(hf_weights.keys())
    matched_pairs: dict[str, str] = {}

    for gguf_name in sorted(gguf_keys):
        if gguf_name in hf_keys:
            hf_name = gguf_name
        else:
            hf_name = None
            for hkey in hf_keys:
                mapped = map_hf_to_gguf_name(hkey)
                if mapped == gguf_name:
                    hf_name = hkey
                    break

        if hf_name is None:
            gguf_only += 1
            continue

        matched_pairs[gguf_name] = hf_name
        gguf_data = gguf_weights[gguf_name]
        hf_data = hf_weights[hf_name]

        if gguf_data.shape != hf_data.shape:
            shape_mismatch += 1
            diff_details.append({
                "tensor": gguf_name,
                "issue": "shape_mismatch",
                "gguf_shape": list(gguf_data.shape),
                "hf_shape": list(hf_data.shape),
            })
            continue

        diff = np.abs(gguf_data - hf_data)
        max_d = float(np.max(diff))
        mean_d = float(np.mean(diff))

        if max_d > tolerance:
            mismatched += 1
            if max_d > max_diff:
                max_diff = max_d
                max_diff_name = gguf_name
            diff_details.append({
                "tensor": gguf_name,
                "issue": "value_mismatch",
                "max_diff": max_d,
                "mean_diff": mean_d,
            })
        else:
            matched += 1

    matched_hf = set(matched_pairs.values())
    for hf_name in hf_keys:
        if hf_name not in matched_hf:
            hf_only += 1

    total_compared = matched + mismatched + shape_mismatch
    accuracy = matched / total_compared * 100 if total_compared > 0 else 0

    result_data = {
        "matched": matched,
        "mismatched": mismatched,
        "shape_mismatch": shape_mismatch,
        "gguf_only": gguf_only,
        "hf_only": hf_only,
        "max_diff": max_diff,
        "max_diff_name": max_diff_name,
        "accuracy": accuracy,
        "diff_details": diff_details[:20],
    }

    return ToolResult(success=True, data=result_data)
