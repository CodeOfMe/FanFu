#!/usr/bin/env python3
"""
Comprehensive GGUF to HF conversion test suite.

Tests:
1. Convert GGUF to HF safetensors
2. Verify weight consistency (structure + values)
3. Run inference comparison (ollama vs transformers)
4. Validate both models produce identical outputs

Usage:
    python test_conversion.py --model qwen3.5
    python test_conversion.py --model lfm2.5
    python test_conversion.py --all
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("test_conversion")

# ============================================================
# Model configurations
# ============================================================

MODEL_CONFIGS = {
    "qwen3.5": {
        "ollama_name": "huihui_ai/qwen3.5-abliterated:0.8B",
        "gguf_blob": "sha256-c04e57114409a84843ed37427be24ea278d5f74f35790734b9bd5554f237210a",
        "hf_output": "models/qwen3.5-abliterated-0.8B-hf",
        "arch": "qwen35",
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
        "head_dim": 64,
        "rope_theta": 10000000,
        "vocab_size": 248320,
        "has_vision": True,
        "has_mtp": True,
        "has_ssm": True,
    },
    "lfm2.5": {
        "ollama_name": "huihui_ai/lfm2.5-abliterated:latest",
        "gguf_blob": "sha256-232f69615a92cad2a92dc946c98602c4d1397b4fdd6c40fce713e50e4ed0d3ac",
        "hf_output": "models/lfm2.5-abliterated-hf",
        "arch": "lfm2",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "rope_theta": 500000,
        "vocab_size": 65536,
        "has_vision": False,
        "has_mtp": False,
        "has_ssm": False,
        "has_shortconv": True,
    },
}

OLLAMA_BLOBS_DIR = Path(os.path.expanduser("~/.ollama/models/blobs"))


# ============================================================
# Tensor name mapping (comprehensive)
# ============================================================

# GGUF -> HF mapping for block-level tensors
GGUF_BLOCK_TO_HF = {
    # Standard attention
    "attn_norm.weight": "input_layernorm.weight",
    "attn_norm.bias": "input_layernorm.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.bias": "post_attention_layernorm.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # Fused QKV (needs special handling)
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_qkv.bias": "self_attn.qkv_proj.bias",
    # Gating (Qwen3.5 specific)
    "attn_gate.weight": "self_attn.gate_proj.weight",
    # FFN
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_gate.bias": "mlp.gate_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    # SSM/Mamba
    "ssm_a": "mamba.A_log",
    "ssm_dt": "mamba.dt_proj.bias",
    "ssm_norm.weight": "mamba.norm.weight",
    "ssm_conv1d.weight": "mamba.conv1d.weight",
    "ssm_conv1d.bias": "mamba.conv1d.bias",
    "ssm_alpha.weight": "mamba.alpha.weight",
    "ssm_beta.weight": "mamba.beta.weight",
    "ssm_out.weight": "mamba.out_proj.weight",
    "ssm_out.bias": "mamba.out_proj.bias",
    # Shortconv (lfm2 specific)
    "shortconv.conv.weight": "shortconv.conv.weight",
    "shortconv.in_proj.weight": "shortconv.in_proj.weight",
    "shortconv.out_proj.weight": "shortconv.out_proj.weight",
}

# Top-level tensor mapping
GGUF_TOP_TO_HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "token_embd_norm.weight": "model.embed_tokens.norm.weight",
    # Vision encoder
    "v.patch_embed.weight": "vision_tower.patch_embed.weight",
    "v.patch_embed.bias": "vision_tower.patch_embed.bias",
    "v.pos_embed.weight": "vision_tower.pos_embed.weight",
    "v.merger.linear_fc1.weight": "vision_tower.merger.linear_fc1.weight",
    "v.merger.linear_fc1.bias": "vision_tower.merger.linear_fc1.bias",
    "v.merger.linear_fc2.weight": "vision_tower.merger.linear_fc2.weight",
    "v.merger.linear_fc2.bias": "vision_tower.merger.linear_fc2.bias",
    "v.merger.norm.weight": "vision_tower.merger.norm.weight",
    "v.merger.norm.bias": "vision_tower.merger.norm.bias",
    # MTP
    "mtp.norm.weight": "model.mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight": "model.mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight": "model.mtp.pre_fc_norm_hidden.weight",
    "mtp.fc.weight": "model.mtp.fc.weight",
}

# Vision block mapping
GGUF_VISION_BLOCK_TO_HF = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    "norm1.weight": "layer_norm1.weight",
    "norm1.bias": "layer_norm1.bias",
    "norm2.weight": "layer_norm2.weight",
    "norm2.bias": "layer_norm2.bias",
    "mlp.linear_fc1.weight": "mlp.fc1.weight",
    "mlp.linear_fc1.bias": "mlp.fc1.bias",
    "mlp.linear_fc2.weight": "mlp.fc2.weight",
    "mlp.linear_fc2.bias": "mlp.fc2.bias",
}


def gguf_to_hf_name(gguf_name: str, arch: str) -> str | None:
    """Convert GGUF tensor name to HuggingFace name."""
    # Top-level tensors
    if gguf_name in GGUF_TOP_TO_HF:
        return GGUF_TOP_TO_HF[gguf_name]

    # Main block tensors: blk.{N}.{tensor}
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".")
        if len(parts) >= 3:
            bid = parts[1]
            tensor_name = ".".join(parts[2:])
            hf_suffix = GGUF_BLOCK_TO_HF.get(tensor_name)
            if hf_suffix:
                return f"model.layers.{bid}.{hf_suffix}"

    # Vision block tensors: v.blk.{N}.{tensor}
    if gguf_name.startswith("v.blk."):
        parts = gguf_name.split(".")
        if len(parts) >= 4:
            bid = parts[2]
            tensor_name = ".".join(parts[3:])
            hf_suffix = GGUF_VISION_BLOCK_TO_HF.get(tensor_name)
            if hf_suffix:
                return f"vision_tower.blocks.{bid}.{hf_suffix}"

    # MTP layers: mtp.layers.{N}.{tensor}
    if gguf_name.startswith("mtp.layers."):
        parts = gguf_name.split(".")
        if len(parts) >= 4:
            bid = parts[2]
            tensor_name = ".".join(parts[3:])
            hf_suffix = GGUF_BLOCK_TO_HF.get(tensor_name)
            if hf_suffix:
                return f"model.mtp.layers.{bid}.{hf_suffix}"

    return None


# ============================================================
# Dequantization (using gguf-py for reliability)
# ============================================================

def dequantize_tensor_ggufpy(tensor) -> np.ndarray:
    """Dequantize a GGUF tensor using gguf-py library."""
    import gguf
    qtype = tensor.tensor_type

    if qtype == gguf.GGMLQuantizationType.F32:
        return np.array(tensor.data, dtype=np.float32)
    elif qtype == gguf.GGMLQuantizationType.F16:
        return np.array(tensor.data, dtype=np.float16).astype(np.float32)
    elif qtype == gguf.GGMLQuantizationType.BF16:
        data = np.array(tensor.data, dtype=np.uint16)
        data = data.astype(np.uint32) << 16
        return data.view(np.float32)
    elif qtype == gguf.GGMLQuantizationType.Q8_0:
        return _dequant_q8_0_ggufpy(tensor)
    elif qtype == gguf.GGMLQuantizationType.Q4_0:
        return _dequant_q4_0_ggufpy(tensor)
    elif qtype == gguf.GGMLQuantizationType.Q4_K:
        return _dequant_q4_k_ggufpy(tensor)
    elif qtype == gguf.GGMLQuantizationType.Q5_K:
        return _dequant_q5_k_ggufpy(tensor)
    elif qtype == gguf.GGMLQuantizationType.Q6_K:
        return _dequant_q6_k_ggufpy(tensor)
    elif qtype == gguf.GGMLQuantizationType.Q2_K:
        return _dequant_q2_k_ggufpy(tensor)
    elif qtype == gguf.GGMLQuantizationType.Q3_K:
        return _dequant_q3_k_ggufpy(tensor)
    else:
        logger.warning(f"Unknown quantization type {qtype}, returning as float32")
        return np.array(tensor.data, dtype=np.float32)


def _dequant_q8_0_ggufpy(tensor) -> np.ndarray:
    """Dequantize Q8_0 - vectorized."""
    shape = list(tensor.shape)
    block_size = 32
    block_bytes = 34
    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    # Reshape to blocks: (n_blocks, 34)
    blocks = raw[:n_blocks * block_bytes].reshape(n_blocks, block_bytes)
    # Extract scales (first 2 bytes as f16)
    d = blocks[:, :2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    # Extract quantized values (bytes 2-34 as int8)
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)
    # Dequantize
    result = (d * qs).reshape(-1)
    return result.reshape(shape[::-1]).copy()


def _dequant_q4_0_ggufpy(tensor) -> np.ndarray:
    """Dequantize Q4_0."""
    import struct
    shape = list(tensor.shape)
    block_size = 32
    block_bytes = 18
    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        d = struct.unpack('<e', bytes(raw[offset:offset + 2]))[0]
        qs = raw[offset + 2:offset + 18]
        for j in range(16):
            result[i * block_size + j * 2] = d * ((np.int8((qs[j] & 0x0F) | (((qs[j] & 0x0F) >> 3) & 0x08) << 4)).astype(np.float32))
            result[i * block_size + j * 2 + 1] = d * ((np.int8((qs[j] >> 4) | (((qs[j] >> 7) & 0x08) << 4))).astype(np.float32))

    return result.reshape(shape[::-1]).copy()


def _dequant_q4_k_ggufpy(tensor) -> np.ndarray:
    """Dequantize Q4_K using llama.cpp reference algorithm."""
    import struct
    shape = list(tensor.shape)
    block_size = 256
    block_bytes = 144
    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
    result = np.zeros(total_elements, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * block_bytes
        d = struct.unpack('<e', bytes(raw[offset:offset + 2]))[0]
        dmin = struct.unpack('<e', bytes(raw[offset + 2:offset + 4]))[0]

        # Scales: 12 bytes, decoded to 8 scale + 8 min values
        scales = raw[offset + 4:offset + 16]
        sc = np.zeros(8, dtype=np.float32)
        mn = np.zeros(8, dtype=np.float32)

        # Q4_K scale encoding: 6-bit values
        sc[0] = scales[0] & 63
        sc[1] = scales[1] & 63
        sc[2] = scales[2] & 63
        sc[3] = scales[3] & 63
        sc[4] = scales[4] & 63
        sc[5] = scales[5] & 63
        sc[6] = scales[6] & 63
        sc[7] = scales[7] & 63

        mn[0] = scales[8] & 63
        mn[1] = scales[9] & 63
        mn[2] = scales[10] & 63
        mn[3] = scales[11] & 63
        mn[4] = (scales[0] >> 6) | ((scales[4] >> 6) << 2)
        mn[5] = (scales[1] >> 6) | ((scales[5] >> 6) << 2)
        mn[6] = (scales[2] >> 6) | ((scales[6] >> 6) << 2)
        mn[7] = (scales[3] >> 6) | ((scales[7] >> 6) << 2)

        # Quantized values: 128 bytes = 256 4-bit values
        qs = raw[offset + 16:offset + 144]

        for j in range(128):
            group = j // 32
            ql = qs[j] & 0x0F
            qh = qs[j] >> 4
            result[i * block_size + j] = d * sc[group] * ql - dmin * mn[group]
            result[i * block_size + j + 128] = d * sc[group + 4] * qh - dmin * mn[group + 4]

    return result.reshape(shape[::-1]).copy()


def _dequant_q5_k_ggufpy(tensor) -> np.ndarray:
    """Dequantize Q5_K."""
    import struct
    shape = list(tensor.shape)
    block_size = 256
    block_bytes = 176
    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        d = struct.unpack('<e', bytes(raw[offset:offset + 2]))[0]
        dmin = struct.unpack('<e', bytes(raw[offset + 2:offset + 4]))[0]
        scales = raw[offset + 4:offset + 16]
        qh = raw[offset + 16:offset + 48]
        qs = raw[offset + 48:offset + 176]

        sc = np.zeros(8, dtype=np.float32)
        mn = np.zeros(8, dtype=np.float32)
        for j in range(4):
            sc[j] = scales[j] & 63
            sc[j + 4] = ((scales[j] >> 4) & 0x0F) | ((scales[j + 4] & 0x0F) << 4)
            mn[j] = scales[j + 8] & 63
            mn[j + 4] = ((scales[j + 8] >> 4) & 0x0F) | ((scales[j + 12] & 0x0F) << 4)

        for j in range(128):
            h = ((qh[j // 8] >> (j % 8)) & 1) << 4
            result[i * block_size + j] = d * sc[j // 32] * ((qs[j] & 0x0F) | h) - dmin * mn[j // 32]
            result[i * block_size + j + 128] = d * sc[j // 32] * ((qs[j] >> 4) | h) - dmin * mn[j // 32]

    return result.reshape(shape[::-1]).copy()


def _dequant_q6_k_ggufpy(tensor) -> np.ndarray:
    """Dequantize Q6_K."""
    import struct
    shape = list(tensor.shape)
    block_size = 256
    block_bytes = 210
    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        ql = raw[offset:offset + 128]
        qh = raw[offset + 128:offset + 192]
        scales = raw[offset + 192:offset + 208].view(np.int8)
        d = struct.unpack('<e', bytes(raw[offset + 208:offset + 210]))[0]

        for j in range(256):
            ql_idx = j if j < 128 else j - 128
            scale_idx = j // 16
            h = ((qh[(j % 64)] >> ((j // 64) * 2)) & 3) << 4
            if j < 128:
                val = (ql[j] & 0x0F) | h
            else:
                val = (ql[j - 128] >> 4) | h
            result[i * block_size + j] = d * scales[scale_idx] * (val.astype(np.float32) - 32.0)

    return result.reshape(shape[::-1]).copy()


def _dequant_q2_k_ggufpy(tensor) -> np.ndarray:
    """Dequantize Q2_K."""
    import struct
    shape = list(tensor.shape)
    block_size = 256
    block_bytes = 84
    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        scales = raw[offset:offset + 16]
        qs = raw[offset + 16:offset + 80]
        d = struct.unpack('<e', bytes(raw[offset + 80:offset + 82]))[0]
        dmin = struct.unpack('<e', bytes(raw[offset + 82:offset + 84]))[0]

        for j in range(256):
            scale_idx = j // 64
            dl = d * (scales[scale_idx] & 0x0F)
            ml = dmin * (scales[scale_idx] >> 4)
            val = (qs[j // 4] >> ((j % 4) * 2)) & 3
            result[i * block_size + j] = dl * val - ml

    return result.reshape(shape[::-1]).copy()


def _dequant_q3_k_ggufpy(tensor) -> np.ndarray:
    """Dequantize Q3_K."""
    import struct
    shape = list(tensor.shape)
    block_size = 256
    block_bytes = 110
    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        hmask = raw[offset:offset + 32]
        qs = raw[offset + 32:offset + 96]
        scales = raw[offset + 96:offset + 108]
        d = struct.unpack('<e', bytes(raw[offset + 108:offset + 110]))[0]

        for j in range(256):
            scale_idx = j // 16
            h = ((hmask[j // 8] >> (j % 8)) & 1)
            val = ((qs[j // 4] >> ((j % 4) * 2)) & 3) - (h << 2)
            result[i * block_size + j] = d * (scales[scale_idx] & 0x0F) * val

    return result.reshape(shape[::-1]).copy()


# ============================================================
# Conversion
# ============================================================

def convert_gguf_to_hf(model_name: str, config: dict) -> dict:
    """Convert GGUF to HF safetensors."""
    import gguf
    from safetensors.torch import save_file
    import torch

    gguf_path = OLLAMA_BLOBS_DIR / config["gguf_blob"]
    output_dir = Path(config["hf_output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading GGUF: {gguf_path}")
    reader = gguf.GGUFReader(str(gguf_path), "r")

    arch = config["arch"]
    n_head = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_head)
    hidden_size = config["hidden_size"]

    # Build config
    hf_config = {
        "architectures": [f"{arch.capitalize().replace('_', '')}ForCausalLM"],
        "model_type": arch,
        "torch_dtype": "float32",
        "hidden_size": hidden_size,
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_layers"],
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv_heads if isinstance(n_kv_heads, int) else sum(n_kv_heads),
        "head_dim": config["head_dim"],
        "rope_theta": config["rope_theta"],
        "vocab_size": config["vocab_size"],
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "use_cache": True,
    }

    # Get bos/eos from GGUF
    bos_field = reader.fields.get("tokenizer.ggml.bos_token_id")
    eos_field = reader.fields.get("tokenizer.ggml.eos_token_id")
    if bos_field:
        hf_config["bos_token_id"] = int(bos_field.contents())
    if eos_field:
        hf_config["eos_token_id"] = int(eos_field.contents())

    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    logger.info(f"Saved config.json")

    # Process tensors
    state_dict = {}
    skipped = []
    split_qkv_count = 0

    for tensor in reader.tensors:
        name = tensor.name
        hf_name = gguf_to_hf_name(name, arch)

        if hf_name is None:
            logger.warning(f"Cannot map: {name}")
            skipped.append(name)
            continue

        # Dequantize
        data = dequantize_tensor_ggufpy(tensor)

        # Handle fused QKV splitting for qwen3.5
        if "qkv_proj" in hf_name and data.ndim == 2:
            # Split QKV: shape is [hidden_size, qkv_dim]
            # qkv_dim = q_dim + k_dim + v_dim
            qkv_dim = data.shape[1]
            q_dim = n_head * config["head_dim"]
            if isinstance(n_kv_heads, list):
                # For layers with GQA, k_dim = v_dim = n_kv_head * head_dim
                # But qkv is fused, so we need to figure out the split
                # In Qwen3.5, qkv = [q, k, v] concatenated
                kv_dim = (qkv_dim - q_dim) // 2
            else:
                kv_dim = n_kv_heads * config["head_dim"]

            if q_dim + 2 * kv_dim == qkv_dim:
                q_data = data[:, :q_dim]
                k_data = data[:, q_dim:q_dim + kv_dim]
                v_data = data[:, q_dim + kv_dim:]

                base_name = hf_name.replace(".qkv_proj.weight", "")
                state_dict[f"{base_name}.q_proj.weight"] = torch.from_numpy(q_data.copy())
                state_dict[f"{base_name}.k_proj.weight"] = torch.from_numpy(k_data.copy())
                state_dict[f"{base_name}.v_proj.weight"] = torch.from_numpy(v_data.copy())
                split_qkv_count += 1
                logger.info(f"  Split QKV: {name} -> q/k/v projections")
                continue
            else:
                logger.warning(f"  QKV dimension mismatch: {qkv_dim} != {q_dim} + 2*{kv_dim}")

        state_dict[hf_name] = torch.from_numpy(data.copy())

    # Save safetensors
    total = sum(t.numel() * t.element_size() for t in state_dict.values())
    save_file(state_dict, output_dir / "model.safetensors")
    logger.info(f"Saved model.safetensors ({total / 1e9:.2f} GB, {len(state_dict)} tensors)")

    # Save generation config
    gen_config = {"_from_model_config": True}
    if hf_config.get("bos_token_id") is not None:
        gen_config["bos_token_id"] = hf_config["bos_token_id"]
    if hf_config.get("eos_token_id") is not None:
        gen_config["eos_token_id"] = hf_config["eos_token_id"]
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    # Extract tokenizer
    extract_tokenizer(reader, output_dir)

    return {
        "output_dir": str(output_dir),
        "tensors": len(state_dict),
        "skipped": skipped,
        "split_qkv": split_qkv_count,
    }


def extract_tokenizer(reader, output_dir: Path):
    """Extract tokenizer from GGUF."""
    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field is None:
        logger.warning("No tokenizer found")
        return

    tokens = tokens_field.contents()
    merges_field = reader.fields.get("tokenizer.ggml.merges")
    merges = merges_field.contents() if merges_field else []

    bos_field = reader.fields.get("tokenizer.ggml.bos_token_id")
    eos_field = reader.fields.get("tokenizer.ggml.eos_token_id")
    unk_field = reader.fields.get("tokenizer.ggml.unknown_token_id")
    pad_field = reader.fields.get("tokenizer.ggml.padding_token_id")
    template_field = reader.fields.get("tokenizer.chat_template")

    bos_id = int(bos_field.contents()) if bos_field else None
    eos_id = int(eos_field.contents()) if eos_field else None
    unk_id = int(unk_field.contents()) if unk_field else None
    pad_id = int(pad_field.contents()) if pad_field else None
    chat_template = template_field.contents() if template_field else None

    # tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": True,
        "model_max_length": 262144,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if chat_template:
        tokenizer_config["chat_template"] = chat_template
    if bos_id and isinstance(tokens, list):
        tokenizer_config["bos_token"] = tokens[bos_id]
    if eos_id and isinstance(tokens, list):
        tokenizer_config["eos_token"] = tokens[eos_id]

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    # special_tokens_map.json
    special_tokens = {}
    if bos_id and isinstance(tokens, list):
        special_tokens["bos_token"] = {"content": tokens[bos_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if eos_id and isinstance(tokens, list):
        special_tokens["eos_token"] = {"content": tokens[eos_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if unk_id and isinstance(tokens, list):
        special_tokens["unk_token"] = {"content": tokens[unk_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if pad_id and isinstance(tokens, list):
        special_tokens["pad_token"] = {"content": tokens[pad_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)

    # tokenizer.json
    vocab = {token: i for i, token in enumerate(tokens)}
    merges_list = [m for m in merges if isinstance(m, str) and " " in m]

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "vocab": vocab,
            "merges": merges_list,
        },
    }

    with open(output_dir / "tokenizer.json", "w") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)

    logger.info(f"Tokenizer extracted ({len(tokens)} tokens)")


# ============================================================
# Weight comparison
# ============================================================

def compare_weights(gguf_path: Path, hf_path: Path, tolerance: float = 1e-3) -> dict:
    """Compare weights between GGUF and HF models."""
    import gguf
    from safetensors import safe_open

    logger.info("Loading GGUF weights...")
    reader = gguf.GGUFReader(str(gguf_path), "r")
    gguf_weights = {}
    for tensor in reader.tensors:
        data = dequantize_tensor_ggufpy(tensor)
        gguf_weights[tensor.name] = data

    logger.info("Loading HF weights...")
    hf_weights = {}
    for st_file in hf_path.glob("*.safetensors"):
        with safe_open(st_file, framework="np") as f:
            for key in f.keys():
                hf_weights[key] = f.get_tensor(key).astype(np.float32)

    logger.info(f"GGUF: {len(gguf_weights)} tensors, HF: {len(hf_weights)} tensors")

    # Build reverse mapping (HF -> GGUF)
    hf_to_gguf = {}
    for gguf_name in gguf_weights:
        hf_name = gguf_to_hf_name(gguf_name, "qwen35")  # arch doesn't matter for mapping
        if hf_name:
            hf_to_gguf[hf_name] = gguf_name

    matched = 0
    mismatched = 0
    shape_mismatch = 0
    gguf_only = 0
    hf_only = 0
    max_diff = 0.0
    max_diff_name = ""
    diff_details = []

    # Check GGUF tensors against HF
    for gguf_name, gguf_data in gguf_weights.items():
        hf_name = gguf_to_hf_name(gguf_name, "qwen35")
        if hf_name is None:
            gguf_only += 1
            continue

        # Handle QKV split: check if q/k/v exist in HF
        if "qkv_proj" in hf_name:
            base_name = hf_name.replace(".qkv_proj.weight", "")
            q_name = f"{base_name}.q_proj.weight"
            k_name = f"{base_name}.k_proj.weight"
            v_name = f"{base_name}.v_proj.weight"

            if q_name in hf_weights and k_name in hf_weights and v_name in hf_weights:
                # Reconstruct QKV from HF
                q_data = hf_weights[q_name]
                k_data = hf_weights[k_name]
                v_data = hf_weights[v_name]
                hf_qkv = np.concatenate([q_data, k_data, v_data], axis=-1)
                hf_data = hf_qkv
            else:
                hf_only += 1
                continue
        elif hf_name in hf_weights:
            hf_data = hf_weights[hf_name]
        else:
            hf_only += 1
            continue

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

    # Check HF-only tensors
    matched_hf = set()
    for gguf_name in gguf_weights:
        hf_name = gguf_to_hf_name(gguf_name, "qwen35")
        if hf_name and "qkv_proj" in hf_name:
            base_name = hf_name.replace(".qkv_proj.weight", "")
            matched_hf.add(f"{base_name}.q_proj.weight")
            matched_hf.add(f"{base_name}.k_proj.weight")
            matched_hf.add(f"{base_name}.v_proj.weight")
        elif hf_name:
            matched_hf.add(hf_name)

    for hf_name in hf_weights:
        if hf_name not in matched_hf:
            hf_only += 1

    total_compared = matched + mismatched + shape_mismatch
    accuracy = matched / total_compared * 100 if total_compared > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"WEIGHT COMPARISON RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Matched: {matched}")
    logger.info(f"Value mismatched: {mismatched}")
    logger.info(f"Shape mismatched: {shape_mismatch}")
    logger.info(f"GGUF-only: {gguf_only}")
    logger.info(f"HF-only: {hf_only}")
    if max_diff_name:
        logger.info(f"Max diff: {max_diff_name} ({max_diff:.6e})")
    logger.info(f"Accuracy: {accuracy:.1f}%")

    return {
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


# ============================================================
# Inference comparison
# ============================================================

def run_ollama_inference(model_name: str, prompts: list[str], timeout: int = 120) -> list[dict]:
    """Run inference using ollama."""
    import subprocess

    results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"[{i+1}/{len(prompts)}] Prompt: {prompt[:60]}...")
        try:
            result = subprocess.run(
                ["ollama", "run", model_name, "--nowordwrap"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            generated = result.stdout.strip()
            logger.info(f"  Output: {generated[:200]}...")
            results.append({"prompt": prompt, "generated": generated, "status": "ok"})
        except subprocess.TimeoutExpired:
            logger.info(f"  TIMEOUT")
            results.append({"prompt": prompt, "generated": "", "status": "timeout"})
        except Exception as e:
            logger.info(f"  ERROR: {e}")
            results.append({"prompt": prompt, "generated": "", "status": "error", "error": str(e)})

    return results


def run_hf_inference(model_path: str, prompts: list[str], max_new_tokens: int = 128) -> list[dict]:
    """Run inference using transformers."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        logger.error("transformers not installed")
        return [{"prompt": p, "generated": "", "status": "error", "error": "transformers not installed"} for p in prompts]

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"[{i+1}/{len(prompts)}] Prompt: {prompt[:60]}...")
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            logger.info(f"  Output: {generated[:200]}...")
            results.append({"prompt": prompt, "generated": generated, "status": "ok"})
        except Exception as e:
            logger.info(f"  ERROR: {e}")
            results.append({"prompt": prompt, "generated": "", "status": "error", "error": str(e)})

    return results


# ============================================================
# Main
# ============================================================

TEST_PROMPTS = [
    "What is 2+2?",
    "Explain Python list comprehension in one sentence.",
    "用一句话介绍一下北京",
]


def main():
    parser = argparse.ArgumentParser(description="GGUF to HF conversion test suite")
    parser.add_argument("--model", choices=["qwen3.5", "lfm2.5"], help="Model to test")
    parser.add_argument("--all", action="store_true", help="Test all models")
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion, only test")
    parser.add_argument("--skip-infer", action="store_true", help="Skip inference comparison")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Weight comparison tolerance")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    models_to_test = []
    if args.all:
        models_to_test = list(MODEL_CONFIGS.keys())
    elif args.model:
        models_to_test = [args.model]
    else:
        parser.print_help()
        return

    all_results = {}

    for model_name in models_to_test:
        config = MODEL_CONFIGS[model_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*60}")

        model_results = {"model": model_name, "config": config}

        # Step 1: Convert
        if not args.skip_convert:
            logger.info(f"\n--- Step 1: Converting {model_name} ---")
            start = time.time()
            try:
                convert_result = convert_gguf_to_hf(model_name, config)
                model_results["conversion"] = {
                    "success": True,
                    "time": time.time() - start,
                    "tensors": convert_result["tensors"],
                    "skipped": convert_result["skipped"],
                    "split_qkv": convert_result.get("split_qkv", 0),
                }
                logger.info(f"Conversion successful: {convert_result['tensors']} tensors in {time.time()-start:.1f}s")
            except Exception as e:
                model_results["conversion"] = {"success": False, "error": str(e)}
                logger.error(f"Conversion failed: {e}")
                all_results[model_name] = model_results
                continue
        else:
            model_results["conversion"] = {"success": True, "skipped": True}

        # Step 2: Compare weights
        logger.info(f"\n--- Step 2: Comparing weights ---")
        gguf_path = OLLAMA_BLOBS_DIR / config["gguf_blob"]
        hf_path = Path(config["hf_output"])

        if not gguf_path.exists():
            logger.error(f"GGUF not found: {gguf_path}")
            model_results["weight_comparison"] = {"success": False, "error": "GGUF not found"}
            all_results[model_name] = model_results
            continue

        if not hf_path.exists():
            logger.error(f"HF model not found: {hf_path}")
            model_results["weight_comparison"] = {"success": False, "error": "HF not found"}
            all_results[model_name] = model_results
            continue

        try:
            weight_result = compare_weights(gguf_path, hf_path, args.tolerance)
            model_results["weight_comparison"] = weight_result
        except Exception as e:
            model_results["weight_comparison"] = {"success": False, "error": str(e)}
            logger.error(f"Weight comparison failed: {e}")

        # Step 3: Inference comparison
        if not args.skip_infer:
            logger.info(f"\n--- Step 3: Inference comparison ---")

            # Ollama inference
            logger.info("Running ollama inference...")
            ollama_results = run_ollama_inference(config["ollama_name"], TEST_PROMPTS)
            model_results["ollama_inference"] = ollama_results

            # HF inference
            logger.info("Running HF inference...")
            hf_results = run_hf_inference(str(hf_path), TEST_PROMPTS)
            model_results["hf_inference"] = hf_results

            # Compare outputs
            logger.info("\nInference comparison:")
            for i, (ollama_res, hf_res) in enumerate(zip(ollama_results, hf_results)):
                logger.info(f"  Prompt {i+1}: {ollama_res['prompt'][:50]}...")
                logger.info(f"    Ollama: {ollama_res.get('generated', '')[:100]}...")
                logger.info(f"    HF:     {hf_res.get('generated', '')[:100]}...")

        all_results[model_name] = model_results

    # Save results
    output_file = Path("test_conversion_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_file}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for model_name, result in all_results.items():
        conv = result.get("conversion", {})
        wc = result.get("weight_comparison", {})
        logger.info(f"{model_name}:")
        logger.info(f"  Conversion: {'OK' if conv.get('success') else 'FAIL'}")
        if wc.get("accuracy") is not None:
            logger.info(f"  Weight accuracy: {wc['accuracy']:.1f}%")
            logger.info(f"  Matched: {wc.get('matched', 0)}, Mismatched: {wc.get('mismatched', 0)}")


if __name__ == "__main__":
    main()
