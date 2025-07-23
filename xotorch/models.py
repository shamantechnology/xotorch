from xotorch.inference.shard import Shard
from typing import Optional, List

model_cards = {
  ### llama
  "llama-3.3-70b": {
    "layers": 80,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Llama-3.3-70B-Instruct",
    },
  },
  "llama-3.2-1b": {
    "layers": 16,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct"
    },
  },
  "llama-3.2-3b": {
    "layers": 28,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct"
    },
  },
  "llama-3.1-8b": {
    "layers": 32,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Meta-Llama-3.1-8B-Instruct",
    },
  },
  "llama-3.1-70b": {
    "layers": 80,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Meta-Llama-3.1-70B-Instruct",
    },
  },
  "llama-3-8b": {
    "layers": 32,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/llama-3-8b",
    },
  },
  "llama-3-70b": {
    "layers": 80,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/llama-3-70b-bnb-4bit",
    },
  },
  "llama-3.1-405b": {
    "layers": 126,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
    },
  },
  "llama-3.1-405b-8bit": {
    "layers": 126,
    "repo": {"TorchDynamicShardInferenceEngine": "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",},
  },
  ### mistral
  "mistral-nemo": {
    "layers": 40,
    "repo": {"TorchDynamicShardInferenceEngine": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",},
  },
  "mistral-large": {
    "layers": 88,
    "repo": {"TorchDynamicShardInferenceEngine": "unsloth/Mistral-Large-Instruct-2407-bnb-4bit",},
  },
  ### deepseek
  "deepseek-coder-v2-lite": { "layers": 27, "repo": { "TorchDynamicShardInferenceEngine": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", }, },
  "deepseek-v3": { "layers": 61, "repo": { "TorchDynamicShardInferenceEngine": "unsloth/DeepSeek-V3-bf16", }, },
  "deepseek-r1": { "layers": 61, "repo": { "TorchDynamicShardInferenceEngine": "deepseek-ai/DeepSeek-R1", }, },
  ### deepseek distills
  "deepseek-r1-distill-qwen-1.5b": { "layers": 28, "repo": { "TorchDynamicShardInferenceEngine": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B", }, },
  "deepseek-r1-distill-qwen-7b": { "layers": 28, "repo": { "TorchDynamicShardInferenceEngine": "unsloth/DeepSeek-R1-Distill-Qwen-7B", }, },
  "deepseek-r1-distill-qwen-14b": { "layers": 48, "repo": { "TorchDynamicShardInferenceEngine": "unsloth/DeepSeek-R1-Distill-Qwen-14B", }, },
  "deepseek-r1-distill-qwen-32b": { "layers": 64, "repo": { "TorchDynamicShardInferenceEngine": "unsloth/DeepSeek-R1-Distill-Qwen-32B", }, },
  "deepseek-r1-distill-llama-8b": { "layers": 32, "repo": { "TorchDynamicShardInferenceEngine": "unsloth/DeepSeek-R1-Distill-Llama-8B", }, },
  "deepseek-r1-distill-llama-70b": { "layers": 80, "repo": { "TorchDynamicShardInferenceEngine": "unsloth/DeepSeek-R1-Distill-Llama-70B", }, },
  ### llava
  "llava-1.5-7b-hf": {
    "layers": 32,
    "repo": {"TorchDynamicShardInferenceEngine": "llava-hf/llava-1.5-7b-hf",},
  },
  ### qwen
  "qwen-2.5-0.5b": {
    "layers": 28,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-0.5B-Instruct"
    },
  },
  "qwen-2.5-1.5b": {
    "layers": 28,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-1.5B-Instruct"
    },
  },
  "qwen-2.5-coder-1.5b": {
    "layers": 28,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-Coder-1.5B-Instruct"
    },
  },
  "qwen-2.5-3b": {
    "layers": 36,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-3B-Instruct"
    },
  },
  "qwen-2.5-coder-3b": {
    "layers": 36,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-Coder-3B-Instruct"
    },
  },
  "qwen-2.5-7b": {
    "layers": 28,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-7B-Instruct"
    },
  },
  "qwen-2.5-coder-7b": {
    "layers": 28,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-Coder-7B-Instruct"
    },
  },
  "qwen-2.5-14b": {
    "layers": 48,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-14B-Instruct"
    },
  },
  "qwen-2.5-coder-14b": {
    "layers": 48,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen2.5-Coder-14B-Instruct"
    },
  },
  "qwen-2.5-32b": {
    "layers": 64,
    "repo": {
      "TorchDynamicShardInferenceEngine": "Qwen/Qwen2.5-32B-Instruct"
    },
  },
  "qwen-3-32b": {
    "layers": 64,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen3-32B-unsloth-bnb-4bit"
    },
  },
  "qwen-3-30b-a3b": {
    "layers": 48,
    "repo": {
      "TorchDynamicShardInferenceEngine": "unsloth/Qwen3-30B-A3B-bnb-4bit"
    },
  },
  "qwen-2.5-coder-32b": {
    "layers": 64,
    "repo": {
      "TorchDynamicShardInferenceEngine": "Qwen/Qwen2.5-Coder-32B-Instruct"
    },
  },
  "qwen-2.5-72b": {
    "layers": 80,
    "repo": {
      "TorchDynamicShardInferenceEngine": "Qwen/Qwen2.5-72B-Instruct"
    },
  },
  "qwen-2.5-math-72b": {
    "layers": 80,
    "repo": {
      "TorchDynamicShardInferenceEngine": "Qwen/Qwen2.5-Math-72B-Instruct"
    },
  },
  ### nemotron
  "nemotron-70b": {
    "layers": 80,
    "repo": {"TorchDynamicShardInferenceEngine": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",},
  },
  # stable diffusion
  # "stable-diffusion-2-1-base": {"layers": 31, "repo": {"TorchDynamicShardInferenceEngine": "stabilityai/stable-diffusion-2-1-base"}},
  # phi
  "phi-4-mini-instruct": {
    "layers": 40,
    "repo": {"TorchDynamicShardInferenceEngine": "microsoft/Phi-4-mini-instruct",},
  },
  # dummy
  "dummy": {
    "layers": 8,
    "repo": {"DummyInferenceEngine": "dummy",},
  },
}

pretty_name = {
  "llama-3.3-70b": "Llama 3.3 70B",
  "llama-3.2-1b": "Llama 3.2 1B",
  "llama-3.2-1b-8bit": "Llama 3.2 1B (8-bit)",
  "llama-3.2-3b": "Llama 3.2 3B",
  "llama-3.2-3b-8bit": "Llama 3.2 3B (8-bit)",
  "llama-3.2-3b-bf16": "Llama 3.2 3B (BF16)",
  "llama-3.1-8b": "Llama 3.1 8B",
  "llama-3.1-70b": "Llama 3.1 70B",
  "llama-3.1-70b-bf16": "Llama 3.1 70B (BF16)",
  "llama-3.1-405b": "Llama 3.1 405B",
  "llama-3.1-405b-8bit": "Llama 3.1 405B (8-bit)",
  "gemma2-9b": "Gemma2 9B",
  "gemma2-27b": "Gemma2 27B",
  "nemotron-70b": "Nemotron 70B",
  "mistral-nemo": "Mistral Nemo",
  "mistral-large": "Mistral Large",
  "deepseek-coder-v2-lite": "Deepseek Coder V2 Lite",
  "deepseek-coder-v2.5": "Deepseek Coder V2.5",
  "deepseek-v3": "Deepseek V3 (4-bit)",
  "deepseek-v3-3bit": "Deepseek V3 (3-bit)",
  "deepseek-r1": "Deepseek R1 (4-bit)",
  "deepseek-r1-3bit": "Deepseek R1 (3-bit)",
  "llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
  "qwen-2.5-0.5b": "Qwen 2.5 0.5B",
  "qwen-2.5-1.5b": "Qwen 2.5 1.5B",
  "qwen-2.5-coder-1.5b": "Qwen 2.5 Coder 1.5B",
  "qwen-2.5-3b": "Qwen 2.5 3B",
  "qwen-2.5-coder-3b": "Qwen 2.5 Coder 3B",
  "qwen-2.5-7b": "Qwen 2.5 7B",
  "qwen-2.5-coder-7b": "Qwen 2.5 Coder 7B",
  "qwen-2.5-math-7b": "Qwen 2.5 7B (Math)",
  "qwen-2.5-14b": "Qwen 2.5 14B",
  "qwen-2.5-coder-14b": "Qwen 2.5 Coder 14B",
  "qwen-2.5-32b": "Qwen 2.5 32B",
  "qwen-2.5-coder-32b": "Qwen 2.5 Coder 32B",
  "qwen-2.5-72b": "Qwen 2.5 72B",
  "qwen-2.5-math-72b": "Qwen 2.5 72B (Math)",
  "qwen-3-32b": "Qwen 3 32B",
  "qwen-3-30b-a3b": "Qwen 3 30B (A3B)",
  "phi-4-mini-instruct": "Phi-4 Mini Instruct",
  "llama-3-8b": "Llama 3 8B",
  "llama-3-70b": "Llama 3 70B",
  "stable-diffusion-2-1-base": "Stable Diffusion 2.1",
  "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1 Distill Qwen 1.5B",
  "deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill Qwen 7B",
  "deepseek-r1-distill-qwen-14b": "DeepSeek R1 Distill Qwen 14B",
  "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
  "deepseek-r1-distill-llama-8b": "DeepSeek R1 Distill Llama 8B",
  "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
}

def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
  return model_cards.get(model_id, {}).get("repo", {}).get(inference_engine_classname, None)

def get_pretty_name(model_id: str) -> Optional[str]:
  return pretty_name.get(model_id, None)

def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  repo = get_repo(model_id, inference_engine_classname)
  n_layers = model_cards.get(model_id, {}).get("layers", 0)
  if repo is None or n_layers < 1:
    return None
  return Shard(model_id, 0, 0, n_layers)

def build_full_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  base_shard = build_base_shard(model_id, inference_engine_classname)
  if base_shard is None: return None
  return Shard(base_shard.model_id, 0, base_shard.n_layers - 1, base_shard.n_layers)

def get_supported_models(supported_inference_engine_lists: Optional[List[List[str]]] = None) -> List[str]:
  if not supported_inference_engine_lists:
    return list(model_cards.keys())

  from xotorch.inference.inference_engine import inference_engine_classes
  supported_inference_engine_lists = [[inference_engine_classes[engine] if engine in inference_engine_classes else engine for engine in engine_list]
                                      for engine_list in supported_inference_engine_lists]

  def has_any_engine(model_info: dict, engine_list: List[str]) -> bool:
    return any(engine in model_info.get("repo", {}) for engine in engine_list)

  def supports_all_engine_lists(model_info: dict) -> bool:
    return all(has_any_engine(model_info, engine_list) for engine_list in supported_inference_engine_lists)

  return [model_id for model_id, model_info in model_cards.items() if supports_all_engine_lists(model_info)]
