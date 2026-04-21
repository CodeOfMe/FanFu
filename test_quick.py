#!/usr/bin/env python3
"""Quick test script for FanFu - runs basic functionality checks."""

import sys
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported."""
    try:
        from fanfu import (
            convert_gguf_to_hf,
            convert_hf_to_gguf,
            compare_weights,
            ToolResult,
            FanFuError,
            ConversionError,
            ValidationError,
            FileError,
            QuantizationError,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_tool_result():
    """Test ToolResult class."""
    from fanfu import ToolResult

    # Success case
    result = ToolResult(success=True, data={"key": "value"})
    assert result.success is True
    assert result.data == {"key": "value"}
    assert result.error is None
    assert bool(result) is True
    print("✅ ToolResult success case passed")

    # Failure case
    result = ToolResult(success=False, error="something went wrong")
    assert result.success is False
    assert result.error == "something went wrong"
    assert result.data is None
    assert bool(result) is False
    print("✅ ToolResult failure case passed")

    return True


def test_errors():
    """Test custom exceptions."""
    from fanfu import (
        FanFuError,
        ConversionError,
        ValidationError,
        FileError,
        QuantizationError,
        ArchitectureError,
    )

    assert issubclass(ConversionError, FanFuError)
    assert issubclass(ValidationError, FanFuError)
    assert issubclass(FileError, FanFuError)
    assert issubclass(QuantizationError, FanFuError)
    assert issubclass(ArchitectureError, FanFuError)
    print("✅ All error classes verified")

    return True


def test_constants():
    """Test constants."""
    from fanfu.constants import APP_NAME, APP_VERSION, SUPPORTED_QUANT_TYPES

    assert APP_NAME == "FanFu"
    assert isinstance(APP_VERSION, str)
    assert "f32" in SUPPORTED_QUANT_TYPES
    assert "f16" in SUPPORTED_QUANT_TYPES
    assert "bf16" in SUPPORTED_QUANT_TYPES
    print(f"✅ Constants verified: {APP_NAME} v{APP_VERSION}")

    return True


def test_mapping():
    """Test tensor name mapping."""
    from fanfu.compare import map_hf_to_gguf_name

    assert map_hf_to_gguf_name("embed_tokens.weight") == "token_embd.weight"
    assert map_hf_to_gguf_name("lm_head.weight") == "output.weight"
    assert map_hf_to_gguf_name("layers.0.input_layernorm.weight") == "blk.0.attn_norm.weight"
    assert map_hf_to_gguf_name("layers.0.mlp.gate_proj.weight") == "blk.0.ffn_gate.weight"
    print("✅ Tensor name mapping verified")

    return True


def test_files_exist():
    """Test that key files exist."""
    key_files = [
        "fanfu/__init__.py",
        "fanfu/api.py",
        "fanfu/cli.py",
        "fanfu/gguf_to_hf.py",
        "fanfu/compare.py",
        "tests/test_core.py",
        "tests/test_conversion_verification.py",
        "pyproject.toml",
        "README.md",
    ]

    for f in key_files:
        if not Path(f).exists():
            print(f"❌ Missing file: {f}")
            return False

    print(f"✅ All {len(key_files)} key files exist")
    return True


def main():
    """Run all quick tests."""
    print("FanFu Quick Test")
    print("=" * 40)

    tests = [
        test_imports,
        test_tool_result,
        test_errors,
        test_constants,
        test_mapping,
        test_files_exist,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All quick tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
