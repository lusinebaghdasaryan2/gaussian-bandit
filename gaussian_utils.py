# gaussian_utils.py
"""Thompson sampling for Gaussian rewards with a known noise standard deviation.

Model
-----
Reward when alternative i is selected:   R | A = i  ~  Normal(mu_i, sigma^2)
- sigma is KNOWN and shared by all alternatives (reward precision = 1 / sigma^2)
- mu_i is UNKNOWN to the learner; every mu_i gets the same Normal(0, 25) prior

Conjugate update (Normal prior + Normal likelihood with known variance):
    posterior precision = prior precision + 1 / sigma^2          (per observation)
    posterior mean      = (prior precision * prior mean + reward / sigma^2)
                          / posterior precision
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 25.0  # Normal(0, 25): variance 25, i.e. standard deviation 5
N_ALTERNATIVES = 3


@dataclass
class GaussianBanditResult:
    true_means: List[float]
    sigma: float
    n_rounds: int
    seed: int
    posterior_means: List[float]
    posterior_variances: List[float]
    selection_counts: List[int]
    total_reward: float
    recommendation: int  # 0-based index of the largest posterior mean
    choices: List[int]  # arm chosen in every round
    rewards: List[float]  # reward observed in every round


def validate_config(true_means: Sequence[float], sigma: float, n_rounds) -> List[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors = []
    if len(true_means) != N_ALTERNATIVES:
        errors.append(f"Exactly {N_ALTERNATIVES} true means are required.")
    if any(not np.isfinite(m) for m in true_means):
        errors.append("True means must be finite numbers.")
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        errors.append("Known sigma must be a positive number (sigma > 0).")
    if (
        n_rounds is None
        or isinstance(n_rounds, bool)
        or int(n_rounds) != n_rounds
        or n_rounds < 1
    ):
        errors.append("Number of rounds must be a positive integer.")
    return errors


def update_posterior(post_mean: float, post_prec: float, reward: float, sigma: float):
    """One-observation conjugate update for a single alternative.

    post_mean, post_prec: current posterior mean and precision (1 / variance).
    Returns the new (mean, precision).
    """
    reward_prec = 1.0 / sigma**2
    new_prec = post_prec + reward_prec
    new_mean = (post_prec * post_mean + reward_prec * reward) / new_prec
    return new_mean, new_prec


def run_gaussian_thompson(
    true_means: Sequence[float], sigma: float, n_rounds: int, seed: int
) -> GaussianBanditResult:
    errors = validate_config(true_means, sigma, n_rounds)
    if errors:
        raise ValueError(" ".join(errors))

    n_rounds = int(n_rounds)
    rng = np.random.default_rng(int(seed))  # one seeded generator => reproducible

    k = len(true_means)
    post_mean = np.full(k, PRIOR_MEAN, dtype=float)          # prior mean 0
    post_prec = np.full(k, 1.0 / PRIOR_VARIANCE, dtype=float)  # prior precision 1/25
    counts = np.zeros(k, dtype=int)
    choices: List[int] = []
    rewards: List[float] = []

    for _ in range(n_rounds):
        # 1) sample a possible mean from each CURRENT posterior (never the true means)
        sampled = rng.normal(post_mean, np.sqrt(1.0 / post_prec))
        # 2) select the largest sample
        arm = int(np.argmax(sampled))
        # 3) simulator generates one continuous Gaussian reward (may be negative)
        reward = float(rng.normal(true_means[arm], sigma))
        # 4) update ONLY the selected alternative
        post_mean[arm], post_prec[arm] = update_posterior(
            post_mean[arm], post_prec[arm], reward, sigma
        )
        counts[arm] += 1
        choices.append(arm)
        rewards.append(reward)

    return GaussianBanditResult(
        true_means=[float(m) for m in true_means],
        sigma=float(sigma),
        n_rounds=n_rounds,
        seed=int(seed),
        posterior_means=post_mean.tolist(),
        posterior_variances=(1.0 / post_prec).tolist(),
        selection_counts=counts.tolist(),
        total_reward=float(np.sum(rewards)),
        recommendation=int(np.argmax(post_mean)),
        choices=choices,
        rewards=rewards,
    )
