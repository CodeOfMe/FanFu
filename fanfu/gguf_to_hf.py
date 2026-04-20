"""GGUF to HuggingFace converter.

Converts GGUF files to HuggingFace safetensors format with optional tokenizer extraction.
"""

from __future__ import annotations

import json
import logging
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fanfu.errors import ConversionError, FileError, QuantizationError
from fanfu.result import ToolResult

logger = logging.getLogger("fanfu.gguf_to_hf")

GGML_QUANT_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4),   # F32
    1: (1, 2),   # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    24: (1, 1),   # I8
    25: (1, 2),   # I16
    26: (1, 4),   # I32
    27: (1, 8),   # I64
    28: (1, 8),   # F64
    30: (1, 2),   # BF16
}

QK_K = 256


def dequantize_q8_0(raw_data: np.ndarray, shape: list[int]) -> np.ndarray:
    """Dequantize Q8_0 tensor. GGUF Q8_0: block_size=32, block_bytes=34 (2 f16 scale + 32 int8)."""
    block_size = 32
    block_bytes = 34

    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(raw_data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        d = struct.unpack('<e', bytes(raw[offset:offset + 2]))[0]
        qs = raw[offset + 2:offset + 34].view(np.int8).astype(np.float32)
        result[i * block_size:(i + 1) * block_size] = d * qs

    return result.reshape(shape[::-1]).copy()


def dequantize_q4_0(raw_data: np.ndarray, shape: list[int]) -> np.ndarray:
    """Dequantize Q4_0 tensor."""
    block_size = 32
    block_bytes = 18

    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(raw_data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        d = struct.unpack('<e', bytes(raw[offset:offset + 2]))[0]
        qs = raw[offset + 2:offset + 18]
        for j in range(16):
            result[i * block_size + j * 2] = d * (np.int8((qs[j] & 0x0F) | (((qs[j] & 0x0F) >> 3) & 0x08) << 4)).astype(np.float32)
            result[i * block_size + j * 2 + 1] = d * (np.int8((qs[j] >> 4) | (((qs[j] >> 7) & 0x08) << 4))).astype(np.float32)

    return result.reshape(shape[::-1]).copy()


def dequantize_tensor(tensor) -> np.ndarray:
    """Dequantize a GGUF tensor to float32."""
    import gguf
    qtype = tensor.tensor_type

    if qtype == gguf.GGMLQuantizationType.F32:
        data = np.array(tensor.data, dtype=np.float32)
    elif qtype == gguf.GGMLQuantizationType.F16:
        data = np.array(tensor.data, dtype=np.float16).astype(np.float32)
    elif qtype == gguf.GGMLQuantizationType.BF16:
        data = np.array(tensor.data, dtype=np.uint16).view(np.uint16)
        data = data.astype(np.uint32) << 16
        data = data.view(np.float32)
    elif qtype == gguf.GGMLQuantizationType.Q8_0:
        data = dequantize_q8_0(tensor.data, tensor.shape)
    elif qtype == gguf.GGMLQuantizationType.Q4_0:
        data = dequantize_q4_0(tensor.data, tensor.shape)
    else:
        data = np.array(tensor.data, dtype=np.float32)

    return data


def extract_tokenizer_from_gguf(gguf_path: Path, output_dir: Path) -> None:
    """Extract full tokenizer from GGUF and create HF-compatible tokenizer files."""
    import gguf

    reader = gguf.GGUFReader(str(gguf_path), "r")

    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field is None:
        logger.warning("No tokenizer.ggml.tokens field found, skipping tokenizer extraction")
        return

    tokens = tokens_field.contents()
    logger.info(f"Extracting {len(tokens)} tokens")

    merges_field = reader.fields.get("tokenizer.ggml.merges")
    merges = []
    if merges_field is not None:
        merges = merges_field.contents()

    eos_field = reader.fields.get("tokenizer.ggml.eos_token_id")
    bos_field = reader.fields.get("tokenizer.ggml.bos_token_id")
    unk_field = reader.fields.get("tokenizer.ggml.unknown_token_id")
    pad_field = reader.fields.get("tokenizer.ggml.padding_token_id")

    eos_token_id = int(eos_field.contents()) if eos_field is not None else None
    bos_token_id = int(bos_field.contents()) if bos_field is not None else None
    unk_token_id = int(unk_field.contents()) if unk_field is not None else None
    pad_token_id = int(pad_field.contents()) if pad_field is not None else None

    template_field = reader.fields.get("tokenizer.chat_template")
    chat_template = None
    if template_field is not None:
        chat_template = template_field.contents()

    tokenizer_config: dict[str, Any] = {
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": True,
        "model_max_length": 262144,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if chat_template:
        tokenizer_config["chat_template"] = chat_template
    if bos_token_id is not None and isinstance(tokens, list):
        tokenizer_config["bos_token"] = tokens[bos_token_id]
    if eos_token_id is not None and isinstance(tokens, list):
        tokenizer_config["eos_token"] = tokens[eos_token_id]
    if unk_token_id is not None and isinstance(tokens, list):
        tokenizer_config["unk_token"] = tokens[unk_token_id]
    if pad_token_id is not None and isinstance(tokens, list):
        tokenizer_config["pad_token"] = tokens[pad_token_id]

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    special_tokens: dict[str, Any] = {}
    if bos_token_id is not None and isinstance(tokens, list):
        special_tokens["bos_token"] = {"content": tokens[bos_token_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if eos_token_id is not None and isinstance(tokens, list):
        special_tokens["eos_token"] = {"content": tokens[eos_token_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if unk_token_id is not None and isinstance(tokens, list):
        special_tokens["unk_token"] = {"content": tokens[unk_token_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if pad_token_id is not None and isinstance(tokens, list):
        special_tokens["pad_token"] = {"content": tokens[pad_token_id], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)

    vocab = {}
    for i, token in enumerate(tokens):
        vocab[token] = i

    merges_list = []
    if isinstance(merges, list):
        for m in merges:
            if isinstance(m, str) and " " in m:
                merges_list.append(m)

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

    logger.info("Tokenizer files saved")


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
        ToolResult with conversion status.
    """
    try:
        import gguf
    except ImportError:
        return ToolResult(success=False, error="gguf package not installed. Run: pip install gguf")

    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        return ToolResult(success=False, error=f"GGUF file not found: {gguf_path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading GGUF file: {gguf_path}")
    reader = gguf.GGUFReader(str(gguf_path), "r")

    config: dict[str, Any] = {}
    arch = None
    for key, field in reader.fields.items():
        if key.startswith("general."):
            short_key = key[len("general."):]
            if short_key == "architecture":
                arch = field.parts[0]
                if hasattr(reader, 'tensors'):
                    try:
                        config["architectures"] = [f"{arch}ForCausalLM"]
                        config["model_type"] = arch
                    except Exception:
                        config["architectures"] = [str(field.contents())]
                        config["model_type"] = str(field.contents())
            elif short_key == "name":
                config["name"] = str(field.contents())

    for key, field in reader.fields.items():
        if key.startswith(f"{arch}.") if arch else False:
            short_key = key[len(arch) + 1:]
            try:
                val = field.contents()
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                config[short_key] = val
            except Exception:
                pass

    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("Saved config.json")

    if extract_tokenizer:
        extract_tokenizer_from_gguf(gguf_path, output)

    logger.info("Loading and dequantizing tensors...")
    state_dict: dict[str, np.ndarray] = {}
    skipped = []

    for tensor in reader.tensors:
        name = tensor.name
        try:
            data = dequantize_tensor(tensor)

            if outtype == "f16":
                if data.dtype == np.float32 and data.ndim >= 2:
                    data = data.astype(np.float16)
            elif outtype == "bf16":
                if data.dtype == np.float32 and data.ndim >= 2:
                    data = data.view(np.uint32)
                    data = (data >> 16).astype(np.uint16).view(np.float16)

            state_dict[name] = data
        except Exception as e:
            skipped.append((name, str(e)))

    logger.info(f"Loaded {len(state_dict)} tensors, skipped {len(skipped)}")

    try:
        from safetensors.torch import save_file
        import torch
    except ImportError:
        return ToolResult(success=False, error="safetensors or torch not installed. Run: pip install safetensors torch")

    torch_state = {}
    for n, d in state_dict.items():
        torch_state[n] = torch.from_numpy(d.copy())

    total = sum(t.numel() * t.element_size() for t in torch_state.values())
    save_file(torch_state, output / "model.safetensors")
    logger.info(f"Saved model.safetensors ({total / 1e9:.2f} GB)")

    gen = {"_from_model_config": True}
    if config.get("eos_token_id") is not None:
        gen["eos_token_id"] = config["eos_token_id"]
    with open(output / "generation_config.json", "w") as f:
        json.dump(gen, f, indent=2)

    return ToolResult(
        success=True,
        data={"output_dir": str(output), "tensors": len(state_dict), "skipped": len(skipped)},
    )
