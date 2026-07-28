"""
TokenLeak Evaluation Script
Reproduces the simulation results reported in Sections VI of the paper.

Usage:
    python -m src.evaluation.evaluate

Outputs CSV files to data/results/.
Seed: 42 (all experiments).
"""

import numpy as np
import pandas as pd
import json
import os
from scipy import stats

from src.agents.timing_oracle_agent import TimingOracleAgent
from src.agents.cache_profiler_agent import CacheProfilerAgent
from src.agents.token_reconstructor_agent import TokenReconstructorAgent
from src.metrics.trcs import TRCS

SEED = 42
N_TOKENS = 500       # vocabulary size for simulation (subset of 50,257)
N_PROMPTS = 500      # prompts per evaluation round
PROMPT_LEN = 20      # tokens per prompt
N_ROUNDS = 10
NOISE_LEVELS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
TOP_K = 5
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

rng_main = np.random.default_rng(SEED)


def make_vocab(n: int) -> list[int]:
    return list(range(n))


def make_prompts(vocab: list[int], n: int, length: int,
                 rng: np.random.Generator) -> list[list[int]]:
    return [list(rng.choice(vocab, size=length)) for _ in range(n)]


def run_experiment(noise_std: float, vocab: list[int],
                   prompts: list[list[int]],
                   toa: TimingOracleAgent,
                   cpa: CacheProfilerAgent) -> dict:
    trcs_metric = TRCS(w1=0.55, w2=0.45)
    timing_profile = toa.get_profile_dict()
    cache_profile = cpa.get_profile_dict()
    tra = TokenReconstructorAgent(trcs_metric, top_k=TOP_K)

    top1_list, top5_list = [], []
    for prompt in prompts:
        t_obs = toa.observe(prompt, external_noise=noise_std)
        c_obs = cpa.observe(prompt)
        preds = tra.reconstruct(t_obs, c_obs, timing_profile,
                                cache_profile, vocab)
        top1_list.append(tra.top_k_accuracy(preds, prompt, k=1))
        top5_list.append(tra.top_k_accuracy(preds, prompt, k=5))

    return {
        'noise_std': noise_std,
        'top1_mean': float(np.mean(top1_list)),
        'top1_std':  float(np.std(top1_list)),
        'top1_ci95': float(stats.sem(top1_list) * 1.96),
        'top5_mean': float(np.mean(top5_list)),
        'top5_std':  float(np.std(top5_list)),
        'top5_ci95': float(stats.sem(top5_list) * 1.96),
    }


def main() -> None:
    vocab = make_vocab(N_TOKENS)
    prompts_all = make_prompts(vocab, N_PROMPTS, PROMPT_LEN, rng_main)

    toa = TimingOracleAgent(sigma_noise=0.0, seed=SEED)
    toa.build_profile(vocab)

    cpa = CacheProfilerAgent(n_cache_sets=256, seed=SEED)
    cpa.build_profile(vocab)

    all_results = []
    for noise in NOISE_LEVELS:
        print(f"[*] noise_std={noise:.1f} ms ...")
        round_results = []
        for r in range(N_ROUNDS):
            rng_r = np.random.default_rng(SEED + r)
            prompts = make_prompts(vocab, N_PROMPTS, PROMPT_LEN, rng_r)
            res = run_experiment(noise, vocab, prompts, toa, cpa)
            res['round'] = r
            round_results.append(res)

        df_r = pd.DataFrame(round_results)
        agg = {
            'noise_std': noise,
            'top1_mean': df_r['top1_mean'].mean(),
            'top1_ci95': df_r['top1_mean'].std() * 1.96 / N_ROUNDS ** 0.5,
            'top5_mean': df_r['top5_mean'].mean(),
            'top5_ci95': df_r['top5_mean'].std() * 1.96 / N_ROUNDS ** 0.5,
        }
        all_results.append(agg)
        print(f"    Top-1: {agg['top1_mean']:.3f} +/- {agg['top1_ci95']:.3f}")

    df = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "accuracy_vs_noise.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Results saved to {out_path}")

    with open(os.path.join(RESULTS_DIR, "metadata.json"), "w") as f:
        json.dump({'seed': SEED, 'n_tokens': N_TOKENS,
                   'n_prompts': N_PROMPTS, 'prompt_len': PROMPT_LEN,
                   'n_rounds': N_ROUNDS}, f, indent=2)


if __name__ == "__main__":
    main()
