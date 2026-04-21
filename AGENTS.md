# FanFu - AI Agent Instructions

## Project Overview
FanFu is a bidirectional converter between GGUF (llama.cpp/Ollama) and HuggingFace safetensors formats, with comprehensive weight verification.

## Key Commands

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run quick tests
python test_quick.py

# Run package tests  
python test_package.py

# Run conversion verification only
python -m pytest tests/test_conversion_verification.py -v

# Run core tests only
python -m pytest tests/test_core.py -v
```

### Building & Publishing
```bash
# Build package
python -m build

# Upload to PyPI (Linux/macOS)
bash upload_pypi.sh

# Upload to PyPI (Windows)
upload_pypi.bat
```

### Diagrams
```bash
# Generate all diagrams
python diagrams/generate_diagrams.py
```

## Architecture
- `fanfu/gguf_to_hf.py` - Main GGUF to HuggingFace converter
- `fanfu/hf_to_gguf.py` - HuggingFace to GGUF converter
- `fanfu/compare.py` - Weight comparison utilities
- `fanfu/cli.py` - Command-line interface
- `fanfu/api.py` - Python API

## Testing Standards
- All tests must pass before any commit
- Weight conversion must achieve 100% match rate
- Tolerance for floating point comparison: 1e-3
- Support quantization types: F32, F16, BF16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0

## Code Style
- Follow PEP 8
- Use type hints
- Docstrings for all public functions
- Keep functions focused and testable
