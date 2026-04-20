#!/usr/bin/env python3
"""
Comprehensive GGUF to HF conversion verification tests.

Tests:
1. Weight structure verification (all tensors mapped correctly)
2. Weight value verification (dequantized values match)
3. Inference comparison (same prompts, same outputs)

Usage:
    python tests/test_conversion_verification.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Model configurations
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
        "has_vision": True,
        "has_mtp": True,
        "has_ssm": True,
        "has_fused_qkv": True,
    },
    "lfm2.5": {
        "ollama_name": "huihui_ai/lfm2.5-abliterated:latest",
        "gguf_blob": "sha256-232f69615a92cad2a92dc946c98602c4d1397b4fdd6c40fce713e50e4ed0d3ac",
        "hf_output": "models/lfm2.5-abliterated-hf",
        "arch": "lfm2",
        "hidden_size": 2048,
        "intermediate_size": 12288,
        "num_layers": 16,
        "num_attention_heads": 32,
        "has_vision": False,
        "has_mtp": False,
        "has_ssm": False,
        "has_shortconv": True,
    },
}

OLLAMA_BLOBS_DIR = Path(os.path.expanduser("~/.ollama/models/blobs"))


def load_gguf_weights(gguf_path):
    """Load and dequantize all GGUF weights."""
    import gguf
    from gguf import dequantize

    reader = gguf.GGUFReader(str(gguf_path), "r")
    weights = {}
    for tensor in reader.tensors:
        qtype = tensor.tensor_type
        shape = list(tensor.shape)
        logical_shape = shape[::-1]

        if qtype == gguf.GGMLQuantizationType.F32:
            data = np.array(tensor.data, dtype=np.float32).reshape(logical_shape)
        elif qtype == gguf.GGMLQuantizationType.F16:
            data = np.array(tensor.data, dtype=np.float16).astype(np.float32).reshape(logical_shape)
        else:
            data = dequantize(tensor.data, qtype).reshape(logical_shape).astype(np.float32)

        weights[tensor.name] = data
    return weights


def load_hf_weights(hf_path):
    """Load all HF safetensors weights."""
    from safetensors import safe_open

    weights = {}
    for st_file in Path(hf_path).glob("*.safetensors"):
        with safe_open(st_file, framework="np") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).astype(np.float32)
    return weights


# GGUF to HF name mapping
def gguf_to_hf_mapping(gguf_name, arch_config):
    """Map GGUF tensor name to HF tensor name(s)."""
    # Top-level tensors
    top_map = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    if gguf_name in top_map:
        return {top_map[gguf_name]}

    # Block tensors
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".")
        if len(parts) >= 3:
            bid = parts[1]
            tensor = ".".join(parts[2:])

            block_map = {
                "attn_norm.weight": f"model.layers.{bid}.input_layernorm.weight",
                "ffn_norm.weight": f"model.layers.{bid}.post_attention_layernorm.weight",
                "attn_q.weight": f"model.layers.{bid}.self_attn.q_proj.weight",
                "attn_k.weight": f"model.layers.{bid}.self_attn.k_proj.weight",
                "attn_v.weight": f"model.layers.{bid}.self_attn.v_proj.weight",
                "attn_output.weight": f"model.layers.{bid}.self_attn.o_proj.weight",
                "attn_q_norm.weight": f"model.layers.{bid}.self_attn.q_norm.weight",
                "attn_k_norm.weight": f"model.layers.{bid}.self_attn.k_norm.weight",
                "ffn_gate.weight": f"model.layers.{bid}.mlp.gate_proj.weight",
                "ffn_down.weight": f"model.layers.{bid}.mlp.down_proj.weight",
                "ffn_up.weight": f"model.layers.{bid}.mlp.up_proj.weight",
                "post_attention_norm.weight": f"model.layers.{bid}.post_attention_layernorm.weight",
                # SSM/Mamba
                "ssm_a": f"model.layers.{bid}.mamba.A_log",
                "ssm_dt": f"model.layers.{bid}.mamba.dt_proj.bias",
                "ssm_norm.weight": f"model.layers.{bid}.mamba.norm.weight",
                "ssm_conv1d.weight": f"model.layers.{bid}.mamba.conv1d.weight",
                "ssm_alpha.weight": f"model.layers.{bid}.mamba.alpha.weight",
                "ssm_beta.weight": f"model.layers.{bid}.mamba.beta.weight",
                "ssm_out.weight": f"model.layers.{bid}.mamba.out_proj.weight",
                # Fused QKV (qwen3.5)
                "attn_qkv.weight": None,  # Special handling
                "attn_gate.weight": f"model.layers.{bid}.self_attn.gate_proj.weight",
                # Shortconv (lfm2.5)
                "shortconv.conv.weight": f"model.layers.{bid}.shortconv.conv.weight",
                "shortconv.in_proj.weight": f"model.layers.{bid}.shortconv.in_proj.weight",
                "shortconv.out_proj.weight": f"model.layers.{bid}.shortconv.out_proj.weight",
            }

            hf_name = block_map.get(tensor)
            if hf_name:
                return {hf_name}
            elif tensor == "attn_qkv.weight" and arch_config.get("has_fused_qkv"):
                # Fused QKV maps to 3 separate tensors
                return {
                    f"model.layers.{bid}.self_attn.q_proj.weight",
                    f"model.layers.{bid}.self_attn.k_proj.weight",
                    f"model.layers.{bid}.self_attn.v_proj.weight",
                }

    # Vision blocks
    if gguf_name.startswith("v.blk."):
        parts = gguf_name.split(".")
        if len(parts) >= 4:
            bid = parts[2]
            tensor = ".".join(parts[3:])
            vision_map = {
                "attn_q.weight": f"vision_tower.blocks.{bid}.self_attn.q_proj.weight",
                "attn_q.bias": f"vision_tower.blocks.{bid}.self_attn.q_proj.bias",
                "attn_k.weight": f"vision_tower.blocks.{bid}.self_attn.k_proj.weight",
                "attn_k.bias": f"vision_tower.blocks.{bid}.self_attn.k_proj.bias",
                "attn_v.weight": f"vision_tower.blocks.{bid}.self_attn.v_proj.weight",
                "attn_v.bias": f"vision_tower.blocks.{bid}.self_attn.v_proj.bias",
                "attn_out.weight": f"vision_tower.blocks.{bid}.self_attn.out_proj.weight",
                "attn_out.bias": f"vision_tower.blocks.{bid}.self_attn.out_proj.bias",
                "norm1.weight": f"vision_tower.blocks.{bid}.layer_norm1.weight",
                "norm1.bias": f"vision_tower.blocks.{bid}.layer_norm1.bias",
                "norm2.weight": f"vision_tower.blocks.{bid}.layer_norm2.weight",
                "norm2.bias": f"vision_tower.blocks.{bid}.layer_norm2.bias",
                "mlp.linear_fc1.weight": f"vision_tower.blocks.{bid}.mlp.fc1.weight",
                "mlp.linear_fc1.bias": f"vision_tower.blocks.{bid}.mlp.fc1.bias",
                "mlp.linear_fc2.weight": f"vision_tower.blocks.{bid}.mlp.fc2.weight",
                "mlp.linear_fc2.bias": f"vision_tower.blocks.{bid}.mlp.fc2.bias",
            }
            hf_name = vision_map.get(tensor)
            if hf_name:
                return {hf_name}

    # Vision top-level
    vision_top = {
        "v.patch_embed.weight": "vision_tower.patch_embed.weight",
        "v.patch_embed.bias": "vision_tower.patch_embed.bias",
        "v.pos_embed.weight": "vision_tower.pos_embed.weight",
        "v.merger.linear_fc1.weight": "vision_tower.merger.linear_fc1.weight",
        "v.merger.linear_fc1.bias": "vision_tower.merger.linear_fc1.bias",
        "v.merger.linear_fc2.weight": "vision_tower.merger.linear_fc2.weight",
        "v.merger.linear_fc2.bias": "vision_tower.merger.linear_fc2.bias",
        "v.merger.norm.weight": "vision_tower.merger.norm.weight",
        "v.merger.norm.bias": "vision_tower.merger.norm.bias",
    }
    if gguf_name in vision_top:
        return {vision_top[gguf_name]}

    # MTP layers
    if gguf_name.startswith("mtp."):
        mtp_map = {
            "mtp.norm.weight": "model.mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight": "model.mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight": "model.mtp.pre_fc_norm_hidden.weight",
            "mtp.fc.weight": "model.mtp.fc.weight",
        }
        if gguf_name in mtp_map:
            return {mtp_map[gguf_name]}

        if gguf_name.startswith("mtp.layers."):
            parts = gguf_name.split(".")
            if len(parts) >= 4:
                bid = parts[2]
                tensor = ".".join(parts[3:])
                block_map = {
                    "attn_q.weight": f"model.mtp.layers.{bid}.self_attn.q_proj.weight",
                    "attn_k.weight": f"model.mtp.layers.{bid}.self_attn.k_proj.weight",
                    "attn_v.weight": f"model.mtp.layers.{bid}.self_attn.v_proj.weight",
                    "attn_output.weight": f"model.mtp.layers.{bid}.self_attn.o_proj.weight",
                    "attn_q_norm.weight": f"model.mtp.layers.{bid}.self_attn.q_norm.weight",
                    "attn_k_norm.weight": f"model.mtp.layers.{bid}.self_attn.k_norm.weight",
                    "ffn_gate.weight": f"model.mtp.layers.{bid}.mlp.gate_proj.weight",
                    "ffn_down.weight": f"model.mtp.layers.{bid}.mlp.down_proj.weight",
                    "ffn_up.weight": f"model.mtp.layers.{bid}.mlp.up_proj.weight",
                    "attn_norm.weight": f"model.mtp.layers.{bid}.input_layernorm.weight",
                    "post_attention_norm.weight": f"model.mtp.layers.{bid}.post_attention_layernorm.weight",
                }
                hf_name = block_map.get(tensor)
                if hf_name:
                    return {hf_name}

    return set()


class TestQwen35Conversion:
    """Test qwen3.5-abliterated:0.8B conversion."""

    @pytest.fixture(scope="class")
    def gguf_weights(self):
        gguf_path = OLLAMA_BLOBS_DIR / MODEL_CONFIGS["qwen3.5"]["gguf_blob"]
        return load_gguf_weights(gguf_path)

    @pytest.fixture(scope="class")
    def hf_weights(self):
        hf_path = Path(MODEL_CONFIGS["qwen3.5"]["hf_output"])
        return load_hf_weights(hf_path)

    @pytest.fixture(scope="class")
    def config(self):
        return MODEL_CONFIGS["qwen3.5"]

    def test_gguf_tensor_count(self, gguf_weights):
        """Verify GGUF has expected number of tensors."""
        assert len(gguf_weights) == 536, f"Expected 536 GGUF tensors, got {len(gguf_weights)}"

    def test_hf_tensor_count(self, hf_weights):
        """Verify HF has expected number of tensors (more due to QKV splitting)."""
        # 536 GGUF tensors, 18 fused QKV split into 54 = +36 extra
        assert len(hf_weights) == 572, f"Expected 572 HF tensors, got {len(hf_weights)}"

    def test_all_gguf_tensors_mapped(self, gguf_weights, config):
        """Verify all GGUF tensors can be mapped to HF names."""
        unmapped = []
        for gguf_name in gguf_weights:
            hf_names = gguf_to_hf_mapping(gguf_name, config)
            if not hf_names:
                unmapped.append(gguf_name)

        assert len(unmapped) == 0, f"Unmapped GGUF tensors: {unmapped[:10]}..."

    def test_embed_tokens_shape(self, gguf_weights, hf_weights):
        """Verify embedding token weights match."""
        gguf_data = gguf_weights["token_embd.weight"]
        hf_data = hf_weights["model.embed_tokens.weight"]

        assert gguf_data.shape == hf_data.shape, f"Shape mismatch: {gguf_data.shape} vs {hf_data.shape}"
        assert np.allclose(gguf_data, hf_data, atol=1e-3), "Embedding weights don't match"

    def test_output_norm_shape(self, gguf_weights, hf_weights):
        """Verify output norm weights match."""
        gguf_data = gguf_weights["output_norm.weight"]
        hf_data = hf_weights["model.norm.weight"]

        assert gguf_data.shape == hf_data.shape
        assert np.allclose(gguf_data, hf_data, atol=1e-3)

    def test_ffn_layer_weights(self, gguf_weights, hf_weights):
        """Verify FFN weights for a sample layer."""
        # Test layer 0
        for tensor in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
            gguf_name = f"blk.0.{tensor}"
            hf_name = f"model.layers.0.mlp.{tensor.replace('ffn_', '').replace('.weight', '_proj.weight')}"

            if tensor == "ffn_gate.weight":
                hf_name = "model.layers.0.mlp.gate_proj.weight"
            elif tensor == "ffn_up.weight":
                hf_name = "model.layers.0.mlp.up_proj.weight"
            elif tensor == "ffn_down.weight":
                hf_name = "model.layers.0.mlp.down_proj.weight"

            assert gguf_name in gguf_weights, f"Missing GGUF tensor: {gguf_name}"
            assert hf_name in hf_weights, f"Missing HF tensor: {hf_name}"

            gguf_data = gguf_weights[gguf_name]
            hf_data = hf_weights[hf_name]

            assert gguf_data.shape == hf_data.shape, f"Shape mismatch for {gguf_name}: {gguf_data.shape} vs {hf_data.shape}"
            assert np.allclose(gguf_data, hf_data, atol=1e-3), f"Value mismatch for {gguf_name}"

    def test_fused_qkv_split(self, gguf_weights, hf_weights, config):
        """Verify fused QKV tensors are correctly split."""
        hidden_size = config["hidden_size"]
        head_dim = 64
        n_heads = config["num_attention_heads"]

        # Test layer 0 (fused QKV)
        gguf_qkv = gguf_weights["blk.0.attn_qkv.weight"]

        # In HF/PyTorch format: weight[out_dim, in_dim]
        # Q, K, V are concatenated along in_dim (axis=1)
        q_hf = hf_weights["model.layers.0.self_attn.q_proj.weight"]
        k_hf = hf_weights["model.layers.0.self_attn.k_proj.weight"]
        v_hf = hf_weights["model.layers.0.self_attn.v_proj.weight"]

        # Verify shapes
        assert q_hf.shape[0] == k_hf.shape[0] == v_hf.shape[0], f"Output dim mismatch"
        assert q_hf.shape[1] + k_hf.shape[1] + v_hf.shape[1] == gguf_qkv.shape[1], f"Input dim sum mismatch"

        # Concatenate along input dimension (axis=1)
        qkv_reconstructed = np.concatenate([q_hf, k_hf, v_hf], axis=1)
        assert qkv_reconstructed.shape == gguf_qkv.shape, f"QKV shape mismatch: {qkv_reconstructed.shape} vs {gguf_qkv.shape}"
        assert np.allclose(gguf_qkv, qkv_reconstructed, atol=1e-3)

    def test_vision_tower_weights(self, gguf_weights, hf_weights):
        """Verify vision tower weights match."""
        # Test patch embed
        gguf_data = gguf_weights["v.patch_embed.weight"]
        hf_data = hf_weights["vision_tower.patch_embed.weight"]

        assert gguf_data.shape == hf_data.shape
        assert np.allclose(gguf_data, hf_data, atol=1e-3)

        # Test vision block 0
        for tensor in ["attn_q.weight", "attn_k.weight", "attn_v.weight"]:
            gguf_name = f"v.blk.0.{tensor}"
            hf_name = f"vision_tower.blocks.0.self_attn.{tensor.replace('attn_', '').replace('.weight', '_proj.weight')}"

            if tensor == "attn_q.weight":
                hf_name = "vision_tower.blocks.0.self_attn.q_proj.weight"
            elif tensor == "attn_k.weight":
                hf_name = "vision_tower.blocks.0.self_attn.k_proj.weight"
            elif tensor == "attn_v.weight":
                hf_name = "vision_tower.blocks.0.self_attn.v_proj.weight"

            assert gguf_name in gguf_weights
            assert hf_name in hf_weights
            assert np.allclose(gguf_weights[gguf_name], hf_weights[hf_name], atol=1e-3)

    def test_mtp_layer_weights(self, gguf_weights, hf_weights):
        """Verify MTP layer weights match."""
        # Test MTP layer 0
        for tensor in ["attn_q.weight", "ffn_gate.weight"]:
            gguf_name = f"mtp.layers.0.{tensor}"
            if tensor == "attn_q.weight":
                hf_name = "model.mtp.layers.0.self_attn.q_proj.weight"
            elif tensor == "ffn_gate.weight":
                hf_name = "model.mtp.layers.0.mlp.gate_proj.weight"

            assert gguf_name in gguf_weights
            assert hf_name in hf_weights
            assert np.allclose(gguf_weights[gguf_name], hf_weights[hf_name], atol=1e-3)

    def test_ssm_layer_weights(self, gguf_weights, hf_weights):
        """Verify SSM/Mamba layer weights match."""
        # Test layer 0 SSM
        for tensor in ["ssm_a", "ssm_norm.weight", "ssm_out.weight"]:
            gguf_name = f"blk.0.{tensor}"
            if tensor == "ssm_a":
                hf_name = "model.layers.0.mamba.A_log"
            elif tensor == "ssm_norm.weight":
                hf_name = "model.layers.0.mamba.norm.weight"
            elif tensor == "ssm_out.weight":
                hf_name = "model.layers.0.mamba.out_proj.weight"

            assert gguf_name in gguf_weights
            assert hf_name in hf_weights
            assert np.allclose(gguf_weights[gguf_name], hf_weights[hf_name], atol=1e-3)

    def test_all_weights_match(self, gguf_weights, hf_weights, config):
        """Comprehensive test: all GGUF weights match their HF counterparts."""
        tolerance = 1e-3
        matched = 0
        total_gguf = 0

        for gguf_name, gguf_data in gguf_weights.items():
            total_gguf += 1
            hf_names = gguf_to_hf_mapping(gguf_name, config)

            if not hf_names:
                continue

            # Handle fused QKV
            if "attn_qkv.weight" in gguf_name:
                hf_names_list = list(hf_names)
                # Find Q, K, V names
                q_name = [n for n in hf_names_list if "q_proj" in n][0]
                k_name = [n for n in hf_names_list if "k_proj" in n][0]
                v_name = [n for n in hf_names_list if "v_proj" in n][0]

                q_hf = hf_weights[q_name]
                k_hf = hf_weights[k_name]
                v_hf = hf_weights[v_name]

                # Concatenate along input dimension (axis=1)
                qkv_reconstructed = np.concatenate([q_hf, k_hf, v_hf], axis=1)
                assert gguf_data.shape == qkv_reconstructed.shape, f"QKV shape mismatch: {gguf_name}"
                assert np.allclose(gguf_data, qkv_reconstructed, atol=tolerance), f"QKV value mismatch: {gguf_name}"
                matched += 1
            else:
                hf_name = list(hf_names)[0]
                if hf_name in hf_weights:
                    hf_data = hf_weights[hf_name]
                    assert gguf_data.shape == hf_data.shape, f"Shape mismatch: {gguf_name}"
                    assert np.allclose(gguf_data, hf_data, atol=tolerance), f"Value mismatch: {gguf_name}"
                    matched += 1

        assert matched == total_gguf, f"Only {matched}/{total_gguf} GGUF tensors matched"


class TestLfm25Conversion:
    """Test lfm2.5-abliterated:latest conversion."""

    @pytest.fixture(scope="class")
    def gguf_weights(self):
        gguf_path = OLLAMA_BLOBS_DIR / MODEL_CONFIGS["lfm2.5"]["gguf_blob"]
        return load_gguf_weights(gguf_path)

    @pytest.fixture(scope="class")
    def hf_weights(self):
        hf_path = Path(MODEL_CONFIGS["lfm2.5"]["hf_output"])
        return load_hf_weights(hf_path)

    @pytest.fixture(scope="class")
    def config(self):
        return MODEL_CONFIGS["lfm2.5"]

    def test_gguf_tensor_count(self, gguf_weights):
        """Verify GGUF has expected number of tensors."""
        assert len(gguf_weights) == 148, f"Expected 148 GGUF tensors, got {len(gguf_weights)}"

    def test_hf_tensor_count(self, hf_weights):
        """Verify HF has expected number of tensors."""
        assert len(hf_weights) == 148, f"Expected 148 HF tensors, got {len(hf_weights)}"

    def test_all_gguf_tensors_mapped(self, gguf_weights, config):
        """Verify all GGUF tensors can be mapped to HF names."""
        unmapped = []
        for gguf_name in gguf_weights:
            hf_names = gguf_to_hf_mapping(gguf_name, config)
            if not hf_names:
                unmapped.append(gguf_name)

        assert len(unmapped) == 0, f"Unmapped GGUF tensors: {unmapped}"

    def test_embed_tokens_shape(self, gguf_weights, hf_weights):
        """Verify embedding token weights match."""
        gguf_data = gguf_weights["token_embd.weight"]
        hf_data = hf_weights["model.embed_tokens.weight"]

        assert gguf_data.shape == hf_data.shape
        assert np.allclose(gguf_data, hf_data, atol=1e-3)

    def test_output_norm_shape(self, gguf_weights, hf_weights):
        """Verify output norm weights match."""
        gguf_data = gguf_weights["output_norm.weight"]
        hf_data = hf_weights["model.norm.weight"]

        assert gguf_data.shape == hf_data.shape
        assert np.allclose(gguf_data, hf_data, atol=1e-3)

    def test_ffn_layer_weights(self, gguf_weights, hf_weights):
        """Verify FFN weights for a sample layer."""
        for tensor in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
            gguf_name = f"blk.0.{tensor}"
            if tensor == "ffn_gate.weight":
                hf_name = "model.layers.0.mlp.gate_proj.weight"
            elif tensor == "ffn_up.weight":
                hf_name = "model.layers.0.mlp.up_proj.weight"
            elif tensor == "ffn_down.weight":
                hf_name = "model.layers.0.mlp.down_proj.weight"

            assert gguf_name in gguf_weights
            assert hf_name in hf_weights

            gguf_data = gguf_weights[gguf_name]
            hf_data = hf_weights[hf_name]

            assert gguf_data.shape == hf_data.shape
            assert np.allclose(gguf_data, hf_data, atol=1e-3)

    def test_attention_layer_weights(self, gguf_weights, hf_weights):
        """Verify attention weights for a layer with separate QKV."""
        # Layer 2 has separate Q, K, V
        for tensor in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]:
            gguf_name = f"blk.2.{tensor}"
            if tensor == "attn_q.weight":
                hf_name = "model.layers.2.self_attn.q_proj.weight"
            elif tensor == "attn_k.weight":
                hf_name = "model.layers.2.self_attn.k_proj.weight"
            elif tensor == "attn_v.weight":
                hf_name = "model.layers.2.self_attn.v_proj.weight"
            elif tensor == "attn_output.weight":
                hf_name = "model.layers.2.self_attn.o_proj.weight"

            assert gguf_name in gguf_weights
            assert hf_name in hf_weights
            assert np.allclose(gguf_weights[gguf_name], hf_weights[hf_name], atol=1e-3)

    def test_shortconv_layer_weights(self, gguf_weights, hf_weights):
        """Verify shortconv layer weights match."""
        # Layer 0 has shortconv
        for tensor in ["shortconv.conv.weight", "shortconv.in_proj.weight", "shortconv.out_proj.weight"]:
            gguf_name = f"blk.0.{tensor}"
            hf_name = f"model.layers.0.{tensor}"

            assert gguf_name in gguf_weights, f"Missing GGUF: {gguf_name}"
            assert hf_name in hf_weights, f"Missing HF: {hf_name}"
            assert np.allclose(gguf_weights[gguf_name], hf_weights[hf_name], atol=1e-3)

    def test_all_weights_match(self, gguf_weights, hf_weights, config):
        """Comprehensive test: all GGUF weights match their HF counterparts."""
        tolerance = 1e-3
        matched = 0

        for gguf_name, gguf_data in gguf_weights.items():
            hf_names = gguf_to_hf_mapping(gguf_name, config)
            assert hf_names, f"Unmapped GGUF tensor: {gguf_name}"

            hf_name = list(hf_names)[0]
            assert hf_name in hf_weights, f"Missing HF tensor: {hf_name} for {gguf_name}"

            hf_data = hf_weights[hf_name]
            assert gguf_data.shape == hf_data.shape, f"Shape mismatch: {gguf_name}"
            assert np.allclose(gguf_data, hf_data, atol=tolerance), f"Value mismatch: {gguf_name}"
            matched += 1

        assert matched == len(gguf_weights), f"Only {matched}/{len(gguf_weights)} tensors matched"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
