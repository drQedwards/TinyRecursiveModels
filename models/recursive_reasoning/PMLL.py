#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMLL.py — Persistent Memory Logic Loop
Author: Dr. Josef Kurk Edwards (Dr. Q) & John Trompeter
Implements the core Persistent Memory Logic Loop for recursive transformer models (RTM).
"""

import os
import json
import time
import uuid
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from hashlib import blake2b
from datetime import datetime
from pathlib import Path
from numpy.linalg import norm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryLine:
    anchor: str
    key: List[float]
    value: List[float]
    ctx: Dict[str, Any]
    timestamp: float
    recency: float
    novelty: float
    similarity: float


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def vector_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))


def hash_anchor(prev_anchor: str, key_vec: np.ndarray, value_vec: np.ndarray, ctx_hash: str) -> str:
    """Deterministic anchor generation via 64-bit chained hash."""
    h = blake2b(digest_size=8)
    h.update((prev_anchor + str(np.sum(key_vec)) + str(np.sum(value_vec)) + ctx_hash).encode())
    return h.hexdigest()


def now_ms() -> float:
    return time.time() * 1000


# ---------------------------------------------------------------------------
# Core Class
# ---------------------------------------------------------------------------

class PersistentMemoryLogicLoop:
    """
    Core class implementing the PMLL concept.
    Stores embeddings and context as persistent “lines” with recency/novelty weighting.
    """

    def __init__(self, storage_path: str = "pmll_store.json", dim: int = 768):
        self.storage_path = Path(storage_path)
        self.dim = dim
        self.memory: List[MemoryLine] = []
        self.index = None
        self.alpha, self.beta, self.gamma = 0.35, 0.50, 0.08
        self.recency_half_life_ms = 6 * 3600 * 1000  # 6 hours
        self.sim_floor, self.novelty_thresh = 0.15, 0.45

        if self.storage_path.exists():
            self.load()
        if FAISS_AVAILABLE:
            self._init_faiss()

    # -----------------------------------------------------------------------
    # Index + Persistence
    # -----------------------------------------------------------------------

    def _init_faiss(self):
        self.index = faiss.IndexFlatL2(self.dim)
        if len(self.memory) > 0:
            mat = np.array([m.key for m in self.memory], dtype=np.float32)
            self.index.add(mat)

    def save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.memory], f, indent=2)

    def load(self):
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.memory = [MemoryLine(**d) for d in data]

    # -----------------------------------------------------------------------
    # Core Methods
    # -----------------------------------------------------------------------

    def put(self, key: np.ndarray, value: np.ndarray, ctx: Dict[str, Any], prev_anchor: Optional[str] = "") -> str:
        """Insert a new line into the lattice."""
        ctx_hash = blake2b(json.dumps(ctx, sort_keys=True).encode(), digest_size=8).hexdigest()
        anchor = hash_anchor(prev_anchor, key, value, ctx_hash)
        sim = 0.0
        nov = 1.0

        if len(self.memory) > 0:
            sims = [vector_similarity(key, np.array(m.key)) for m in self.memory]
            sim = max(sims)
            nov = 1 - sim

        ts = now_ms()
        rec = 1.0
        new_line = MemoryLine(anchor, key.tolist(), value.tolist(), ctx, ts, rec, nov, sim)
        self.memory.append(new_line)

        if FAISS_AVAILABLE:
            self.index.add(np.expand_dims(key.astype(np.float32), 0))
        self.save()

        return anchor

    def query(self, query_vec: np.ndarray, k: int = 5) -> List[MemoryLine]:
        """Return top-k relevant memory lines."""
        if len(self.memory) == 0:
            return []

        if FAISS_AVAILABLE:
            D, I = self.index.search(np.expand_dims(query_vec.astype(np.float32), 0), k)
            results = [self.memory[i] for i in I[0]]
        else:
            sims = [(vector_similarity(query_vec, np.array(m.key)), m) for m in self.memory]
            sims.sort(key=lambda x: x[0], reverse=True)
            results = [m for _, m in sims[:k]]

        # Recency decay
        now = now_ms()
        for m in results:
            dt = now - m.timestamp
            m.recency = np.exp(-dt / self.recency_half_life_ms)

        return results

    def bias_vector(self, query_vec: np.ndarray, k: int = 5) -> np.ndarray:
        """Return weighted sum of stored value vectors for bias application."""
        lines = self.query(query_vec, k)
        if not lines:
            return np.zeros(self.dim)

        weights = np.array([self.alpha * l.recency + self.beta * l.similarity + self.gamma * l.novelty for l in lines])
        values = np.array([l.value for l in lines])
        wnorm = weights / (np.sum(weights) + 1e-9)
        return np.sum(values * wnorm[:, None], axis=0)


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pmll = PersistentMemoryLogicLoop(dim=8)
    vecA = np.random.randn(8)
    vecB = np.random.randn(8)
    ctxA = {"topic": "recursion", "step": 1}
    anchorA = pmll.put(vecA, vecB, ctxA)
    print("Anchor A:", anchorA)

    q = np.random.randn(8)
    retrieved = pmll.query(q)
    print(f"Queried {len(retrieved)} memory lines.")
    print("Bias Vector:", pmll.bias_vector(q))
