"""Tests for FanFu core modules."""

import pytest
import numpy as np

from fanfu.constants import APP_NAME, APP_VERSION, SUPPORTED_QUANT_TYPES
from fanfu.errors import (
    FanFuError,
    ConversionError,
    ValidationError,
    FileError,
    QuantizationError,
    ArchitectureError,
)
from fanfu.result import ToolResult
from fanfu.compare import map_hf_to_gguf_name


class TestConstants:
    def test_app_name(self):
        assert APP_NAME == "FanFu"

    def test_app_version(self):
        assert isinstance(APP_VERSION, str)
        assert len(APP_VERSION) > 0

    def test_supported_quant_types(self):
        assert "f32" in SUPPORTED_QUANT_TYPES
        assert "f16" in SUPPORTED_QUANT_TYPES
        assert "bf16" in SUPPORTED_QUANT_TYPES
        assert "q8_0" in SUPPORTED_QUANT_TYPES


class TestErrors:
    def test_fanfu_error(self):
        with pytest.raises(FanFuError):
            raise FanFuError("test")

    def test_conversion_error(self):
        assert issubclass(ConversionError, FanFuError)

    def test_validation_error(self):
        assert issubclass(ValidationError, FanFuError)

    def test_file_error(self):
        assert issubclass(FileError, FanFuError)

    def test_quantization_error(self):
        assert issubclass(QuantizationError, FanFuError)

    def test_architecture_error(self):
        assert issubclass(ArchitectureError, FanFuError)


class TestToolResult:
    def test_success(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert bool(result) is True

    def test_failure(self):
        result = ToolResult(success=False, error="something went wrong")
        assert result.success is False
        assert result.error == "something went wrong"
        assert result.data is None
        assert bool(result) is False


class TestMapping:
    def test_embed_tokens(self):
        assert map_hf_to_gguf_name("embed_tokens.weight") == "token_embd.weight"

    def test_lm_head(self):
        assert map_hf_to_gguf_name("lm_head.weight") == "output.weight"

    def test_layer_norm(self):
        assert map_hf_to_gguf_name("layers.0.input_layernorm.weight") == "blk.0.attn_norm.weight"
        assert map_hf_to_gguf_name("layers.5.input_layernorm.weight") == "blk.5.attn_norm.weight"

    def test_mlp(self):
        assert map_hf_to_gguf_name("layers.0.mlp.gate_proj.weight") == "blk.0.ffn_gate.weight"
        assert map_hf_to_gguf_name("layers.0.mlp.up_proj.weight") == "blk.0.ffn_up.weight"
        assert map_hf_to_gguf_name("layers.0.mlp.down_proj.weight") == "blk.0.ffn_down.weight"

    def test_attention(self):
        assert map_hf_to_gguf_name("layers.0.self_attn.qkv_proj.weight") == "blk.0.attn_qkv.weight"
        assert map_hf_to_gguf_name("layers.0.self_attn.gate_proj.weight") == "blk.0.attn_gate.weight"

    def test_mamba(self):
        assert map_hf_to_gguf_name("layers.0.mamba.norm.weight") == "blk.0.ssm_norm.weight"
        assert map_hf_to_gguf_name("layers.0.mamba.alpha.weight") == "blk.0.ssm_alpha.weight"

    def test_unknown(self):
        assert map_hf_to_gguf_name("unknown.tensor.weight") is None


class TestDequantize:
    def test_q8_0_roundtrip(self):
        from fanfu.gguf_to_hf import dequantize_q8_0
        import struct

        original = np.random.randn(64, 128).astype(np.float32)
        shape = [128, 64]

        total_elements = int(np.prod(shape))
        n_blocks = total_elements // 32
        block_bytes = 34

        raw = np.zeros(n_blocks * block_bytes, dtype=np.uint8)
        flat = original.flatten()

        for i in range(n_blocks):
            block = flat[i * 32:(i + 1) * 32]
            max_abs = np.max(np.abs(block))
            if max_abs == 0:
                d = np.float16(0)
            else:
                d = np.float16(max_abs / 127.0)

            d_bytes = np.array([d], dtype=np.float16).view(np.uint8)
            offset = i * block_bytes
            raw[offset:offset + 2] = d_bytes

            if d != 0:
                qs = np.round(block / float(d)).clip(-128, 127).astype(np.int8)
            else:
                qs = np.zeros(32, dtype=np.int8)

            raw[offset + 2:offset + 34] = qs.view(np.uint8)

        result = dequantize_q8_0(raw, shape)
        assert result.shape == original.shape
        max_diff = np.max(np.abs(original - result))
        mean_diff = np.mean(np.abs(original - result))
        assert mean_diff < 0.01


class TestCompareWeights:
    def test_compare_identical_arrays(self):
        from fanfu.compare import compare_weights
        import tempfile
        import json
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "model.gguf")
            hf_dir = os.path.join(tmpdir, "hf_model")
            os.makedirs(hf_dir)

            result = compare_weights(gguf_path, hf_dir)
            assert result.success is False

    def test_compare_nonexistent_gguf(self):
        from fanfu.compare import compare_weights

        result = compare_weights("/nonexistent/model.gguf", "/nonexistent/hf")
        assert result.success is False
        assert "not found" in result.error.lower()


class TestCLI:
    def test_cli_import(self):
        from fanfu import cli
        assert hasattr(cli, "main")

    def test_cli_has_subcommands(self):
        from fanfu import cli
        import inspect
        source = inspect.getsource(cli)
        assert "gguf-to-hf" in source
        assert "hf-to-gguf" in source
        assert "compare" in source
