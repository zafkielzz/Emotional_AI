# -*- coding: utf-8 -*-
"""
PhoneFarm Core Configuration
Hardware target: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
Model: Qwen 3 8B (INT4 NF4 Quantization via BitsAndBytes)
"""

import os
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ModelConfig:
    model_path: str = os.environ.get("MODEL_PATH", "/media/zafkiel/WORK_SPACE2/models/Qwen3-8B")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = True
    default_max_new_tokens: int = 420
    default_temperature: float = 0.3
    top_p: float = 0.9

CONFIG = ModelConfig()
