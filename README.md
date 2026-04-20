# FanFu - GGUF/HuggingFace Bidirectional Converter

Bidirectional converter between GGUF and HuggingFace formats with weight verification.

> **FanFu (反复)** -- Back and forth, roundtrip.

## Features

- **GGUF to HuggingFace** -- Convert GGUF files to HuggingFace safetensors format with automatic dequantization (Q8_0, Q4_0, F16, F32).
- **HuggingFace to GGUF** -- Convert HuggingFace model directories to GGUF format with optional quantization.
- **Weight Verification** -- Compare weights between GGUF and HF models with configurable tolerance, reporting matched/mismatched tensors.
- **Tokenizer Extraction** -- Automatically extract tokenizer from GGUF files and generate HF-compatible tokenizer files (tokenizer.json, tokenizer_config.json, special_tokens_map.json).
- **CLI & Python API** -- Use from the command line or import as a library.

## Requirements

- Python 3.10+
- `gguf`, `safetensors`, `torch`, `numpy`, `rich`

## Installation

### From PyPI (Recommended)

```bash
pip install fanfu
```

### From Source

```bash
git clone https://github.com/CodeOfMe/FanFu.git
cd FanFu
pip install -e .
```

## Usage

### CLI

```bash
# Convert GGUF to HuggingFace
fanfu gguf-to-hf model.gguf -o hf_model/
fanfu gguf-to-hf model.gguf -o hf_model/ -t f16
fanfu gguf-to-hf model.gguf -o hf_model/ --no-tokenizer

# Convert HuggingFace to GGUF
fanfu hf-to-gguf hf_model/ -o model.gguf
fanfu hf-to-gguf hf_model/ -o model.gguf -t q8_0
fanfu hf-to-gguf hf_model/ -o model.gguf -t f16

# Compare weights between GGUF and HF
fanfu compare model.gguf hf_model/
fanfu compare model.gguf hf_model/ --tolerance 0.1 -o results.json

# Show version
fanfu --version
```

### Python API

```python
from fanfu import convert_gguf_to_hf, convert_hf_to_gguf, compare_weights

# GGUF -> HF
result = convert_gguf_to_hf("model.gguf", "hf_model/", outtype="f32")
print(result.data)  # {"output_dir": "hf_model/", "tensors": 536, "skipped": 0}

# HF -> GGUF
result = convert_hf_to_gguf("hf_model/", "model.gguf", outtype="q8_0")
print(result.data)  # {"output_path": "model.gguf", "tensors": 535}

# Compare weights
result = compare_weights("model.gguf", "hf_model/", tolerance=0.5)
print(f"Accuracy: {result.data['accuracy']:.1f}%")
print(f"Matched: {result.data['matched']}, Mismatched: {result.data['mismatched']}")
```

## Project Structure

```
FanFu/
├── pyproject.toml              # Package metadata & build config
├── MANIFEST.in                 # Source distribution manifest
├── LICENSE                     # GPL-3.0-or-later
├── README.md                   # English documentation
├── README_CN.md                # Chinese documentation
├── fanfu/
│   ├── __init__.py             # Package version & public exports
│   ├── __main__.py             # python -m fanfu entry
│   ├── cli.py                  # CLI entry point with subcommands
│   ├── api.py                  # Public Python API
│   ├── constants.py            # App constants & architecture mappings
│   ├── errors.py               # Custom exception classes
│   ├── gguf_to_hf.py           # GGUF -> HuggingFace converter
│   ├── hf_to_gguf.py           # HuggingFace -> GGUF converter
│   └── compare.py              # Weight comparison & verification
├── tests/
│   ├── __init__.py
│   └── test_core.py            # Comprehensive test suite
└── publish.py                  # PyPI publish helper script
```

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=fanfu --cov-report=term-missing
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

## Publishing

```bash
# Build only
python publish.py

# Build + upload to TestPyPI
python publish.py test

# Build + upload to PyPI
python publish.py release
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.
