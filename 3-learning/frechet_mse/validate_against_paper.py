#!/usr/bin/env python
"""Check the backend-free port against the ICML 2024 reference implementation.

Imports both — ``hdrlib.core.rmt`` and the reference ``src`` of
https://github.com/AmmarMian/icml-rmt-2024 — feeds them the same matrices, and
reports the discrepancy layer by layer, from the pieces that should agree to
machine precision up to the estimate the figures actually plot.

The point is not to prove the port correct in the abstract but to say *where*
and *by how much* it departs, because two of the substitutions are not
bit-exact by construction:

* the reference factorises with LAPACK ``dpptrf`` (packed Cholesky) and inverts
  the factor with ``dtrtri`` (triangular inverse); the port uses ``cholesky``
  and a general ``inv``. Same objects, different rounding;
* the analytical shrinkage estimator uses ``eigh`` here and ``eig`` there.

Everything else is a term-for-term transposition and should agree to ~1e-12 in
float64.

Run:
    uv run --group reference python 3-learning/frechet_mse/validate_against_paper.py \\
        --reference /path/to/icml-rmt-2024/code
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hdrlib.core import rmt


def relative_error(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(b)
    if denominator == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denominator)


def report(name: str, error: float, tolerance: float) -> bool:
    ok = error <= tolerance
    mark = "  ok  " if ok else " FAIL "
    print(f"[{mark}] {name:<52} rel. err = {error:.3e}   (tol {tolerance:.0e})")
    return ok


def random_spd(rng, n_features: int, condition_number: float) -> np.ndarray:
    """Same construction as the reference: random basis, spread eigenvalues.

    The reference draws the basis with ``scipy.stats.ortho_group`` and the
    eigenvalues uniformly between the two extremes it pins to
    ``1/sqrt(cond)`` and ``sqrt(cond)``. Reproduced here so the validation does
    not depend on the reference's own generator, which lives behind pymanopt.
    """
    basis = np.linalg.qr(rng.standard_normal((n_features, n_features)))[0]
    low, high = 1 / np.sqrt(condition_number), np.sqrt(condition_number)
    eigenvalues = rng.uniform(low, high, size=n_features)
    eigenvalues[-2], eigenvalues[-1] = low, high
    return basis @ np.diag(eigenvalues) @ basis.T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=str, required=True,
        help="Path to the 'code' directory of the icml-rmt-2024 checkout.")
    parser.add_argument("--n_features", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--n_matrices", type=int, default=8)
    parser.add_argument("--condition_number", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_trials_mse", type=int, default=20,
        help="Monte-Carlo trials for the end-to-end MSE comparison.")
    parser.add_argument("--backend", type=str, default="numpy",
        help="Backend to validate. The reference is numpy-only, so this "
             "compares the chosen backend against numpy-LAPACK.")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser().resolve()
    if not (reference_path / "src" / "mean.py").exists():
        sys.exit(f"No src/mean.py under {reference_path}")
    sys.path.insert(0, str(reference_path))

    from src.covariance import SCM_estimator                      # noqa: E402
    from src.distance import (                                     # noqa: E402
        RMT_squaredFisherDistance_deterministic_SCMs,
        squaredFisherDistance,
    )
    from src.estimation import analytical_shrinkage_estimator      # noqa: E402
    from src.mean import (                                         # noqa: E402
        RMT_geometric_mean,
        _aux_RMT_mean_cost_grad,
        geometric_mean,
    )

    rng = np.random.default_rng(args.seed)
    d, n, k = args.n_features, args.n_samples, args.n_matrices

    print(f"d = {d}, N = {n}, K = {k}, c = {d / n:.3f}, "
          f"backend = {args.backend}\n")

    # Data: K matrices of n samples each, drawn from covariances scattered
    # around a common SPD centre — the shape the experiment uses.
    centre = random_spd(rng, d, args.condition_number)
    factor = np.linalg.cholesky(centre)
    tangents = rng.standard_normal((k, d, d)) * 0.1
    tangents = (tangents + tangents.swapaxes(-1, -2)) / 2
    tangents -= tangents.mean(axis=0)
    values, vectors = np.linalg.eigh(tangents)
    covariances = factor @ (
        vectors @ (np.exp(values)[..., None] * vectors.swapaxes(-1, -2))
    ) @ factor.T
    data = np.stack([
        rng.multivariate_normal(np.zeros(d), covariances[i], size=n)
        for i in range(k)
    ])

    passed = []

    # Everything the port is given has to live on the backend under test; the
    # reference stays on numpy, which is the whole point of the comparison.
    def on_device(x):
        return rmt.get_data_on_device(np.asarray(x, dtype=float), args.backend)

    # The corrected path needs float64. A backend that cannot hold it — Metal,
    # today — is not a failure of the port and is reported as such rather than
    # allowed to produce a confusing internal error deeper down.
    probe = on_device(np.eye(2))
    try:
        rmt.require_double(probe, "validation")
    except TypeError as exc:
        print(f"[ skip ] backend '{args.backend}' cannot hold float64.\n        {exc}")
        return 0

    # ── layer 1: the sample covariance ────────────────────────────────────
    passed.append(report(
        "scm",
        relative_error(rmt.to_numpy(rmt.scm(on_device(data), args.backend)),
                       SCM_estimator(data)),
        1e-12,
    ))

    scms = SCM_estimator(data)

    # ── layer 2: the two distances ────────────────────────────────────────
    passed.append(report(
        "corrected squared distance",
        relative_error(
            rmt.to_numpy(rmt.rmt_corrected_squared_distance(
                on_device(centre), on_device(scms), n, args.backend)),
            RMT_squaredFisherDistance_deterministic_SCMs(centre, scms, n),
        ),
        1e-10,
    ))
    passed.append(report(
        "plain squared distance",
        relative_error(
            rmt.to_numpy(rmt._plain_cost_grad(
                rmt.get_backend_module(args.backend), args.backend,
                on_device(
                    np.linalg.inv(np.linalg.cholesky(centre)) @ scms
                    @ np.linalg.inv(np.linalg.cholesky(centre)).T),
                d, return_grad=False)) * 2,
            np.mean(squaredFisherDistance(centre, scms)),
        ),
        1e-10,
    ))

    # ── layer 3: cost and gradient, the delicate part ─────────────────────
    # This is where the near-degenerate divisions of the gradient live. It is
    # evaluated at a point away from the optimum, where the terms are largest.
    inverse_factor = np.linalg.inv(np.linalg.cholesky(centre))
    transformed = inverse_factor @ scms @ inverse_factor.T
    reference_cost, reference_grad = _aux_RMT_mean_cost_grad(
        transformed, d, n, d / n, return_grad=True
    )
    be = rmt.get_backend_module(args.backend)
    port_cost, port_grad = rmt._rmt_cost_grad(
        be, args.backend,
        on_device(transformed),
        d, n, d / n, return_grad=True,
    )
    passed.append(report(
        "corrected cost",
        relative_error(rmt.to_numpy(port_cost), reference_cost), 1e-10))
    passed.append(report(
        "corrected gradient (canonical)",
        relative_error(rmt.to_numpy(port_grad), reference_grad), 1e-8))

    # ── layer 4: the shrinkage baselines ──────────────────────────────────
    passed.append(report(
        "analytical shrinkage (eigh vs eig)",
        relative_error(
            rmt.to_numpy(rmt.analytical_shrinkage(
                on_device(data[0]), args.backend, shrink=0)),
            analytical_shrinkage_estimator(data[0], shrink=0),
        ),
        1e-8,
    ))
    try:
        from sklearn.covariance import OAS, LedoitWolf
        passed.append(report(
            "linear Ledoit-Wolf vs scikit-learn",
            relative_error(
                rmt.to_numpy(rmt.ledoit_wolf_linear(
                    on_device(data[0]), args.backend)),
                LedoitWolf(assume_centered=True).fit(data[0]).covariance_,
            ),
            1e-10,
        ))
        passed.append(report(
            "OAS vs scikit-learn",
            relative_error(
                rmt.to_numpy(rmt.oas(on_device(data[0]), args.backend)),
                OAS(assume_centered=True).fit(data[0]).covariance_,
            ),
            1e-10,
        ))
    except ImportError:
        print("[ skip ] scikit-learn not installed — shrinkage baselines skipped")

    # ── layer 5: the estimates themselves ─────────────────────────────────
    # These are the outputs of an iterative descent with a backtracking line
    # search, and the tolerance is deliberately loose. The two loops agree to
    # 1e-13 on the first iterate and drift by roughly one digit every few
    # iterations: once two costs differ in their last bits, the line search can
    # take one halving more or less, which moves the next iterate by far more
    # than the rounding that caused it. After thirty-odd iterations near a flat
    # minimum the gap settles around 1e-5.
    #
    # That is a property of the algorithm, not of the port — the same thing
    # happens between two runs of the reference on different BLAS builds. What
    # has to be checked instead is that it does not move the quantity the
    # figures plot, which is layer 6.
    passed.append(report(
        "plain Fréchet mean",
        relative_error(
            rmt.to_numpy(rmt.frechet_mean_cholesky(
                on_device(scms), backend=args.backend)[0]),
            geometric_mean(scms),
        ),
        1e-6,
    ))
    passed.append(report(
        "RMT-corrected Fréchet mean",
        relative_error(
            rmt.to_numpy(rmt.rmt_frechet_mean(
                on_device(data), backend=args.backend)[0]),
            RMT_geometric_mean(data),
        ),
        1e-4,
    ))

    # ── layer 6: what the figures actually plot ───────────────────────────
    # The MSE is the squared affine-invariant distance from the true mean to
    # the estimate, averaged over trials. It is the only number the figures
    # show, and it is what must agree — a 1e-5 wobble in the estimate is
    # invisible in a curve whose methods are tens of dB apart.
    def squared_distance(reference_matrix, estimate):
        inverse = np.linalg.inv(np.linalg.cholesky(reference_matrix))
        logs = np.log(np.linalg.eigvalsh(inverse @ estimate @ inverse.T))
        return float(logs @ logs)

    errors_reference, errors_port = [], []
    for trial in range(args.n_trials_mse):
        trial_rng = np.random.default_rng(args.seed + 1000 + trial)
        trial_tangents = trial_rng.standard_normal((k, d, d)) * 0.1
        trial_tangents = (trial_tangents + trial_tangents.swapaxes(-1, -2)) / 2
        trial_tangents -= trial_tangents.mean(axis=0)
        trial_values, trial_vectors = np.linalg.eigh(trial_tangents)
        trial_covariances = factor @ (
            trial_vectors
            @ (np.exp(trial_values)[..., None] * trial_vectors.swapaxes(-1, -2))
        ) @ factor.T
        trial_data = np.stack([
            trial_rng.multivariate_normal(np.zeros(d), trial_covariances[i], size=n)
            for i in range(k)
        ])
        errors_reference.append(
            squared_distance(centre, RMT_geometric_mean(trial_data))
        )
        errors_port.append(squared_distance(centre, rmt.to_numpy(
            rmt.rmt_frechet_mean(
                on_device(trial_data), backend=args.backend)[0]
        )))

    mse_reference = float(np.mean(errors_reference))
    mse_port = float(np.mean(errors_port))
    print(f"         mse reference = {mse_reference:.6f}, "
          f"port = {mse_port:.6f}  ({args.n_trials_mse} trials)")
    passed.append(report(
        f"mse over {args.n_trials_mse} trials",
        abs(mse_port - mse_reference) / mse_reference,
        1e-3,
    ))

    print()
    if all(passed):
        print(f"All {len(passed)} checks passed on backend '{args.backend}'.")
        return 0
    print(f"{passed.count(False)} of {len(passed)} checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
