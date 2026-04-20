"""Application constants."""

from fanfu import __version__

APP_NAME = "FanFu"
APP_VERSION = __version__

SUPPORTED_QUANT_TYPES = ["f32", "f16", "bf16", "q8_0", "auto"]

GGUF_TO_HF_ARCH_MAP = {
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_5": "Qwen3_5ForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
    "gemma3": "Gemma3ForCausalLM",
    "phi3": "Phi3ForCausalLM",
    "mistral": "MistralForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "command-r": "CohereForCausalLM",
    "falcon": "FalconForCausalLM",
    "mamba": "MambaForCausalLM",
    "bert": "BertModel",
}

HF_TO_GGUF_ARCH_MAP = {v: k for k, v in GGUF_TO_HF_ARCH_MAP.items()}
