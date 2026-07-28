"""
Agent 2: Cache Profiler Agent (CPA)
TokenLeak Framework

Simulates Flush+Reload-style LLC cache profiling for transformer
embedding lookups. Each token maps to a set of L cache lines
(embedding matrix rows). In simulation, cache access bitmaps
are synthesized with controlled noise.
"""

import numpy as np


class CacheProfilerAgent:
    """
    Agent 2 (CPA): Profiles LLC cache access patterns per token.

    Real deployment: Flush+Reload on shared embedding matrix pages.
    Simulation: Binary bitmap c_tau synthesized from token identity.
    """

    def __init__(self, n_cache_sets: int = 256,
                 hit_prob: float = 0.90,
                 noise_prob: float = 0.05,
                 seed: int = 42):
        """
        Parameters
        ----------
        n_cache_sets : number of monitored cache set addresses (L)
        hit_prob     : probability that an accessed cache line is detected
        noise_prob   : probability of a false-positive cache hit per set
        """
        self.L = n_cache_sets
        self.hit_prob = hit_prob
        self.noise_prob = noise_prob
        self.rng = np.random.default_rng(seed)
        self.fingerprints: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def _token_cache_pattern(self, token_id: int) -> np.ndarray:
        """
        Generate a reproducible cache line access pattern for a token.
        Uses a deterministic hash-based assignment.
        """
        rng_tok = np.random.default_rng(token_id + 12345)
        n_accessed = max(1, int(np.clip(rng_tok.normal(12, 3), 3, 30)))
        pattern = np.zeros(self.L, dtype=np.float32)
        indices = rng_tok.choice(self.L, size=n_accessed, replace=False)
        pattern[indices] = 1.0
        return pattern

    def build_profile(self, token_ids: list[int],
                      n_samples: int = 50) -> None:
        """
        Build cache fingerprint library for all tokens.
        Averages n_samples noisy observations per token.
        """
        for tok in token_ids:
            true_pattern = self._token_cache_pattern(tok)
            noisy_sum = np.zeros(self.L, dtype=np.float32)
            for _ in range(n_samples):
                obs = np.zeros(self.L, dtype=np.float32)
                for j in range(self.L):
                    if true_pattern[j] == 1.0:
                        if self.rng.random() < self.hit_prob:
                            obs[j] = 1.0
                    else:
                        if self.rng.random() < self.noise_prob:
                            obs[j] = 1.0
                noisy_sum += obs
            self.fingerprints[tok] = noisy_sum / n_samples

    def observe(self, true_token_ids: list[int]) -> list[np.ndarray]:
        """
        Simulate cache observations for a target token sequence.
        Returns a list of noisy binary bitmaps c_j.
        """
        observations = []
        for tok in true_token_ids:
            true_pattern = self._token_cache_pattern(tok)
            obs = np.zeros(self.L, dtype=np.float32)
            for j in range(self.L):
                if true_pattern[j] == 1.0:
                    if self.rng.random() < self.hit_prob:
                        obs[j] = 1.0
                else:
                    if self.rng.random() < self.noise_prob:
                        obs[j] = 1.0
            observations.append(obs)
        return observations

    def get_profile_dict(self) -> dict:
        """Export fingerprint library for TRCS scoring."""
        return {tok: fp for tok, fp in self.fingerprints.items()}
