# -*- coding: utf-8 -*-
"""
PhoneFarm LLM Backend Singleton
Loads and manages Qwen 3 8B (INT4 NF4) in GPU memory without duplicate allocations.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.core.config import CONFIG, ModelConfig


class LLMBackend:
    _instance: LLMBackend | None = None

    def __init__(self, cfg: ModelConfig = CONFIG):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._load_model()

    @classmethod
    def get_instance(cls, cfg: ModelConfig = CONFIG) -> LLMBackend:
        if cls._instance is None:
            cls._instance = cls(cfg)
        return cls._instance

    def _load_model(self):
        print(f"[*] [LLMBackend] Loading model from: {self.cfg.model_path} in INT4 NF4...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, local_files_only=True)
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.load_in_4bit,
            bnb_4bit_quant_type=self.cfg.quant_type,
            bnb_4bit_compute_dtype=self.cfg.compute_dtype,
            bnb_4bit_use_double_quant=self.cfg.use_double_quant,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            quantization_config=bnb_config,
            device_map=self.cfg.device,
            local_files_only=True,
        )
        t1 = time.time()
        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        print(f"[+] [LLMBackend] Loaded in {t1 - t0:.2f}s | VRAM Allocated: {vram_gb:.2f} GB")

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        enable_thinking: bool = False,
    ) -> tuple[str, str, float]:
        """
        Generates completion given chat messages.
        Returns:
            (thinking_trace, response_text, latency_ms)
        """
        max_tokens = max_new_tokens or self.cfg.default_max_new_tokens
        temp = temperature if temperature is not None else self.cfg.default_temperature
        p = top_p if top_p is not None else self.cfg.top_p

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.model.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=p,
                do_sample=(temp > 0.0),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.time() - t0) * 1000

        gen_tokens = out[0][len(inputs.input_ids[0]):]
        full_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Release tensor references
        del inputs
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        thinking = ""
        text = full_text
        if "<think>" in full_text and "</think>" in full_text:
            parts = full_text.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            text = parts[1].strip()
        elif "<think>" in full_text:
            thinking = full_text.replace("<think>", "").strip()
            text = ""

        return thinking, text, latency_ms
