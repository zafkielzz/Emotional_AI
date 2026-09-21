# -*- coding: utf-8 -*-
"""
Module 3: Dense Memory Embedder
Utilizes sentence-transformers/all-MiniLM-L6-v2 (384-dim) for high-accuracy semantic similarity.
Optimized for Host Laptop: Runs on CPU by default to preserve 100% of GPU VRAM for Qwen 3 8B.
"""

from __future__ import annotations

import glob
import os
from typing import Sequence
import numpy as np


class DenseMemoryEmbedder:
    """
    Offline Dense Text Embedder based on all-MiniLM-L6-v2.
    Produces 384-dimensional unit-normalized semantic vectors.
    """
    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.dimension = 384
        self.tokenizer = None
        self.model = None
        self.is_transformer = False
        
        self._initialize_model(model_path)

    def _initialize_model(self, custom_path: str | None):
        snapshot_path = custom_path
        if snapshot_path is None:
            # Check default HuggingFace cache
            pattern = os.path.expanduser(
                "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
            )
            matches = glob.glob(pattern)
            if matches:
                snapshot_path = sorted(matches)[-1]

        if snapshot_path and os.path.exists(snapshot_path):
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
                self.model = AutoModel.from_pretrained(snapshot_path, local_files_only=True)
                self.model.to(self.device)
                self.model.eval()
                self.is_transformer = True
                print(f"[+] [DenseMemoryEmbedder] Loaded all-MiniLM-L6-v2 (384-d) from {snapshot_path} on {self.device}")
                return
            except Exception as e:
                print(f"[!] [DenseMemoryEmbedder] Transformer load failed ({e}), falling back to deterministic vectorizer.")

        print("[!] [DenseMemoryEmbedder] No local all-MiniLM-L6-v2 snapshot found, using fallback vectorizer.")
        self.is_transformer = False

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Encodes a sequence of texts into 384-dimensional normalized vector embeddings.
        """
        if not texts:
            return []

        if self.is_transformer and self.model is not None and self.tokenizer is not None:
            import torch
            inputs = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**inputs)
                # Mean pooling with attention mask
                token_embeddings = out.last_hidden_state
                input_mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                # L2 normalize to unit sphere
                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                return normalized.cpu().numpy().tolist()

        # Deterministic hashing fallback for offline test environments without torch/transformers
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                idx = abs(hash(w)) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors

    @staticmethod
    def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
        """Computes cosine similarity between two unit-normalized vectors."""
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def cosine_similarity_batch(query_vec: Sequence[float], matrix: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Vectorized matrix-vector cosine similarity.
        Matrix shape: (N, 384), Query shape: (384,)
        """
        if not matrix:
            return np.array([], dtype=np.float32)
        q = np.asarray(query_vec, dtype=np.float32)
        m = np.asarray(matrix, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        m_norms = np.linalg.norm(m, axis=1)
        
        # Avoid zero division
        denominator = q_norm * m_norms + 1e-8
        return np.dot(m, q) / denominator
