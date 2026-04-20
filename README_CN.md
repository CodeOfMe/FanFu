# FanFu (反复) - GGUF/HuggingFace 双向转换器

GGUF 和 HuggingFace 格式之间的双向转换器，支持权重验证。

> **反复** -- 来回转换，往返验证。

## 功能特性

- **GGUF 转 HuggingFace** -- 将 GGUF 文件转换为 HuggingFace safetensors 格式，支持自动反量化（Q8_0、Q4_0、F16、F32）。
- **HuggingFace 转 GGUF** -- 将 HuggingFace 模型目录转换为 GGUF 格式，支持可选量化。
- **权重验证** -- 比较 GGUF 和 HF 模型之间的权重，支持自定义容差，报告匹配/不匹配的张量。
- **Tokenizer 提取** -- 自动从 GGUF 文件中提取 tokenizer 并生成 HF 兼容的 tokenizer 文件（tokenizer.json、tokenizer_config.json、special_tokens_map.json）。
- **CLI 和 Python API** -- 支持命令行使用和 Python 库导入。

## 环境要求

- Python 3.10+
- `gguf`、`safetensors`、`torch`、`numpy`、`rich`

## 安装

### 从 PyPI 安装（推荐）

```bash
pip install fanfu
```

### 从源码安装

```bash
git clone https://github.com/CodeOfMe/FanFu.git
cd FanFu
pip install -e .
```

## 使用方法

### 命令行

```bash
# GGUF 转 HuggingFace
fanfu gguf-to-hf model.gguf -o hf_model/
fanfu gguf-to-hf model.gguf -o hf_model/ -t f16
fanfu gguf-to-hf model.gguf -o hf_model/ --no-tokenizer

# HuggingFace 转 GGUF
fanfu hf-to-gguf hf_model/ -o model.gguf
fanfu hf-to-gguf hf_model/ -o model.gguf -t q8_0
fanfu hf-to-gguf hf_model/ -o model.gguf -t f16

# 比较 GGUF 和 HF 的权重
fanfu compare model.gguf hf_model/
fanfu compare model.gguf hf_model/ --tolerance 0.1 -o results.json

# 查看版本
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

# 比较权重
result = compare_weights("model.gguf", "hf_model/", tolerance=0.5)
print(f"准确率: {result.data['accuracy']:.1f}%")
print(f"匹配: {result.data['matched']}, 不匹配: {result.data['mismatched']}")
```

## 项目结构

```
FanFu/
├── pyproject.toml              # 包元数据和构建配置
├── MANIFEST.in                 # 源码分发包清单
├── LICENSE                     # GPL-3.0-or-later
├── README.md                   # 英文文档
├── README_CN.md                # 中文文档
├── fanfu/
│   ├── __init__.py             # 包版本和公共导出
│   ├── __main__.py             # python -m fanfu 入口
│   ├── cli.py                  # CLI 入口，包含子命令
│   ├── api.py                  # 公共 Python API
│   ├── constants.py            # 应用常量和架构映射
│   ├── errors.py               # 自定义异常类
│   ├── gguf_to_hf.py           # GGUF -> HuggingFace 转换器
│   ├── hf_to_gguf.py           # HuggingFace -> GGUF 转换器
│   └── compare.py              # 权重比较和验证
├── tests/
│   ├── __init__.py
│   └── test_core.py            # 综合测试套件
└── publish.py                  # PyPI 发布辅助脚本
```

## 测试

```bash
# 运行完整测试套件
python -m pytest tests/ -v

# 运行覆盖率报告
python -m pytest tests/ --cov=fanfu --cov-report=term-missing
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
python -m pytest tests/ -v
```

## 发布

```bash
# 仅构建
python publish.py

# 构建 + 上传到 TestPyPI
python publish.py test

# 构建 + 上传到 PyPI
python publish.py release
```

## 许可证

GPL-3.0-or-later。详见 [LICENSE](LICENSE)。
