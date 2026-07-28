"""
Agent 1: Timing Oracle Agent (TOA)
TokenLeak Framework

Collects inter-token generation latencies from LLM inference API.
In simulation mode, synthesizes timing from profiling data + noise.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimingProfile:
    token_id: int
    mu: float       # mean inter-token latency (ms)
    sigma: float    # std dev inter-token latency (ms)
    n_samples: int  # number of observations used


class TimingOracleAgent:
    """
    Agent 1 (TOA): Records inter-token generation latencies.

    In simulation mode, timing is synthesized as:
        t_j ~ N(mu_{tau_j}, sigma_{tau_j}) + N(0, sigma_noise)
    """

    def __init__(self, sigma_noise: float = 0.0, seed: int = 42):
        self.sigma_noise = sigma_noise
        self.rng = np.random.default_rng(seed)
        self.profiles: dict[int, TimingProfile] = {}

    # ------------------------------------------------------------------ #
    # Profiling phase
    # ------------------------------------------------------------------ #
    def build_profile(self, token_ids: list[int],
                      base_mu: float = 10.0,
                      base_sigma: float = 1.5,
                      spread: float = 5.0,
                      n_samples: int = 100) -> None:
        """
        Simulate profiling: assign each token a distinctive mean latency
        drawn from a token-specific distribution.

        Parameters
        ----------
        base_mu   : baseline mean inter-token latency (ms)
        base_sigma: per-token std dev
        spread    : range of latency variation across vocabulary
        n_samples : number of simulated profiling observations per token
        """
        self.rng = np.random.default_rng(42)
        token_means = self.rng.uniform(base_mu - spread,
                                       base_mu + spread,
                                       size=len(token_ids))
        for i, tok in enumerate(token_ids):
            self.profiles[tok] = TimingProfile(
                token_id=tok,
                mu=float(token_means[i]),
                sigma=base_sigma,
                n_samples=n_samples,
            )

    # ------------------------------------------------------------------ #
    # Attack phase
    # ------------------------------------------------------------------ #
    def observe(self, true_token_ids: list[int],
                external_noise: Optional[float] = None) -> list[float]:
        """
        Simulate observation of inter-token latencies for a target sequence.

        Returns a list of noisy timing measurements t_j.
        """
        noise_std = external_noise if external_noise is not None \
            else self.sigma_noise
        observations = []
        for tok in true_token_ids:
            p = self.profiles.get(tok)
            if p is None:
                raise KeyError(f"Token {tok} not in profiling data.")
            t = self.rng.normal(p.mu, p.sigma)
            if noise_std > 0:
                t += self.rng.normal(0, noise_std)
            observations.append(float(t))
        return observations

    def get_profile_dict(self) -> dict:
        """Export profile for TRCS scoring."""
        return {
            tok: {'mu': p.mu, 'sigma': p.sigma}
            for tok, p in self.profiles.items()
        }
