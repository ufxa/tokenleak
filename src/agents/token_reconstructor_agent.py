"""
Agent 3: Token Reconstructor Agent (TRA)
TokenLeak Framework

Fuses timing and cache signals via TRCS to reconstruct private prompts.
"""

import numpy as np
from src.metrics.trcs import TRCS


class TokenReconstructorAgent:
    """
    Agent 3 (TRA): Computes TRCS scores and reconstructs token sequences.
    """

    def __init__(self, trcs: TRCS, top_k: int = 5,
                 prune_sigma: float = 3.0):
        """
        Parameters
        ----------
        trcs      : fitted TRCS metric instance
        top_k     : number of top candidates to return per position
        prune_sigma: vocabulary pruning threshold (multiples of sigma)
        """
        self.trcs = trcs
        self.top_k = top_k
        self.prune_sigma = prune_sigma

    def _prune_vocab(self, t_obs: float,
                     timing_profile: dict,
                     vocab: list[int]) -> list[int]:
        """
        Prune vocabulary to tokens whose profiled mean is within
        prune_sigma standard deviations of the observed timing.
        """
        candidates = []
        for tok in vocab:
            p = timing_profile.get(tok)
            if p is None:
                continue
            if abs(t_obs - p['mu']) <= self.prune_sigma * p['sigma']:
                candidates.append(tok)
        return candidates if candidates else vocab

    def reconstruct(self,
                    timing_obs: list[float],
                    cache_obs: list[np.ndarray],
                    timing_profile: dict,
                    cache_profile: dict,
                    vocab: list[int]) -> list[list[int]]:
        """
        Reconstruct top-k token candidates per position.

        Returns
        -------
        List of length n, each entry is a list of top_k token ids.
        """
        assert len(timing_obs) == len(cache_obs), \
            "Timing and cache observation lengths must match."
        result = []
        for t_obs, c_obs in zip(timing_obs, cache_obs):
            candidates = self._prune_vocab(t_obs, timing_profile, vocab)
            merged_profile = {
                tok: {
                    'mu': timing_profile[tok]['mu'],
                    'sigma': timing_profile[tok]['sigma'],
                    'cache': cache_profile[tok],
                }
                for tok in candidates
                if tok in timing_profile and tok in cache_profile
            }
            ranked = self.trcs.rank(t_obs, c_obs, merged_profile)
            result.append(ranked[:self.top_k])
        return result

    @staticmethod
    def top_k_accuracy(predictions: list[list[int]],
                       true_tokens: list[int],
                       k: int = 1) -> float:
        """Compute Top-k accuracy across all positions."""
        correct = sum(
            1 for pred, true in zip(predictions, true_tokens)
            if true in pred[:k]
        )
        return correct / len(true_tokens) if true_tokens else 0.0
