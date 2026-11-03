"""Monte Carlo Shapley estimator and studentized circular block bootstrap.

Implements Section 2.2.2 (objective-level Shapley attribution) and
Section 2.2.5 (statistical inference) of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Monte Carlo Shapley value estimation
# ---------------------------------------------------------------------------

def monte_carlo_shapley(value_fn: Callable[[np.ndarray], float],
                         n_features: int,
                         n_coalitions: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Estimate feature-level Shapley values of ``value_fn`` with permutation
    sampling. Returns an array of length ``n_features``.
    """
    phi = np.zeros(n_features)
    counts = np.zeros(n_features)
    for _ in range(n_coalitions):
        perm = rng.permutation(n_features)
        lam = np.zeros(n_features)
        prev = value_fn(lam)
        for i in perm:
            lam[i] = 1.0
            cur = value_fn(lam)
            phi[i] += cur - prev
            counts[i] += 1
            prev = cur
    counts = np.where(counts == 0, 1, counts)
    return phi / counts


def shapley_main_effects(value_fn: Callable[[np.ndarray], float],
                          n_features: int) -> np.ndarray:
    """Main-effect term v({i}) - v(emptyset)."""
    empty = value_fn(np.zeros(n_features))
    main = np.zeros(n_features)
    for i in range(n_features):
        lam = np.zeros(n_features); lam[i] = 1.0
        main[i] = value_fn(lam) - empty
    return main


# ---------------------------------------------------------------------------
# Studentized circular block bootstrap for Sharpe attribution
# ---------------------------------------------------------------------------

@dataclass
class SharpeAttributionCI:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    significant: bool


def _sharpe_from_moments(mu_c: float, mu_ci: float,
                           gamma_c: float, gamma_ci: float) -> float:
    """Equation (35): phi_i(SR_c) = mu_{c,i}/sigma - mu_c * (gamma_{c,i}-mu_c*mu_{c,i}) / sigma^3."""
    sigma2 = gamma_c - mu_c * mu_c
    if sigma2 <= 0:
        return 0.0
    sigma = np.sqrt(sigma2)
    cov = gamma_ci - mu_c * mu_ci
    return mu_ci / sigma - mu_c * cov / (sigma ** 3)


def circular_block_bootstrap_sharpe_attr(ret: np.ndarray, phi_ret: np.ndarray,
                                           block: int, B: int,
                                           rng: np.random.Generator
                                           ) -> SharpeAttributionCI:
    """Studentized percentile-t confidence interval for the local Sharpe
    attribution of a single feature.

    ``ret``     — length-T daily portfolio return.
    ``phi_ret`` — length-T feature return contribution series.
    """
    T = len(ret)
    assert len(phi_ret) == T
    mu_c = ret.mean()
    mu_ci = phi_ret.mean()
    gamma_c = (ret ** 2).mean()
    gamma_ci = (ret * phi_ret).mean()
    theta_hat = _sharpe_from_moments(mu_c, mu_ci, gamma_c, gamma_ci) * np.sqrt(252.0)

    # Delta-method asymptotic variance via plug-in block LRV.
    f_t = np.stack([ret, phi_ret, ret ** 2, ret * phi_ret], axis=1)
    f_bar = f_t.mean(axis=0)
    f_dev = f_t - f_bar
    # Simple block LRV with block length b (equation (37)-(38)).
    k = T // block
    if k == 0:
        return SharpeAttributionCI(theta_hat, np.nan, np.nan, np.nan, False)
    block_sums = np.array([f_dev[i * block:(i + 1) * block].sum(axis=0) for i in range(k)])
    Omega = (block_sums[:, :, None] * block_sums[:, None, :]).mean(axis=0) / block

    # Gradient of theta w.r.t. (mu_c, mu_ci, gamma_c, gamma_ci).
    sigma2 = gamma_c - mu_c * mu_c
    if sigma2 <= 0:
        return SharpeAttributionCI(theta_hat, np.nan, np.nan, np.nan, False)
    sigma = np.sqrt(sigma2); sigma3 = sigma ** 3; sigma5 = sigma ** 5
    cov = gamma_ci - mu_c * mu_ci
    # dtheta/d mu_c, dtheta/d mu_ci, dtheta/d gamma_c, dtheta/d gamma_ci
    g = np.array([
        -mu_ci * mu_c / sigma3 - cov / sigma3 + 3 * mu_c ** 2 * cov / sigma5,
        1.0 / sigma + mu_c ** 2 / sigma3,
        0.5 * mu_ci * mu_c / sigma3 + 1.5 * mu_c * cov / sigma5 * (-1.0),
        -mu_c / sigma3,
    ])
    var_theta = float(g @ Omega @ g) / T
    se = np.sqrt(max(var_theta, 0.0)) * np.sqrt(252.0)

    # Circular block bootstrap replications.
    boot_t = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, T, size=k)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1) % T
        rb = ret[idx]; pb = phi_ret[idx]
        mub = rb.mean(); mubi = pb.mean()
        gab = (rb ** 2).mean(); gabi = (rb * pb).mean()
        tb = _sharpe_from_moments(mub, mubi, gab, gabi) * np.sqrt(252.0)
        # Bootstrap SE via same block LRV plug-in.
        fb = np.stack([rb, pb, rb ** 2, rb * pb], axis=1)
        fbd = fb - fb.mean(axis=0)
        kb = len(rb) // block
        if kb == 0:
            boot_t[b] = 0.0; continue
        bsums = np.array([fbd[i * block:(i + 1) * block].sum(axis=0) for i in range(kb)])
        Ob = (bsums[:, :, None] * bsums[:, None, :]).mean(axis=0) / block
        sig2b = gab - mub * mub
        if sig2b <= 0:
            boot_t[b] = 0.0; continue
        sigb = np.sqrt(sig2b)
        covb = gabi - mub * mubi
        gb = np.array([
            -mubi * mub / sigb ** 3 - covb / sigb ** 3 + 3 * mub ** 2 * covb / sigb ** 5,
            1.0 / sigb + mub ** 2 / sigb ** 3,
            0.5 * mubi * mub / sigb ** 3 - 1.5 * mub * covb / sigb ** 5,
            -mub / sigb ** 3,
        ])
        vb = float(gb @ Ob @ gb) / len(rb)
        seb = np.sqrt(max(vb, 0.0)) * np.sqrt(252.0)
        boot_t[b] = (tb - theta_hat) / seb if seb > 0 else 0.0

    q_lo, q_hi = np.quantile(boot_t, [0.025, 0.975])
    ci_low = theta_hat - q_hi * se
    ci_high = theta_hat - q_lo * se
    sig = not (ci_low <= 0.0 <= ci_high)
    return SharpeAttributionCI(theta_hat, se, ci_low, ci_high, sig)


__all__ = [
    "monte_carlo_shapley", "shapley_main_effects",
    "circular_block_bootstrap_sharpe_attr", "SharpeAttributionCI",
]
