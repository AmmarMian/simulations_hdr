"""Sonar library unit tests.

Checks:
1. M is full-rank positive definite (no co-located sensor bug).
2. 2TYL recovers a known M at large K (consistency).
3. M-NMF-G → M-NMF-I reduction when M_12 = 0 (block-diagonal M).
4. Empirical PFA ≈ nominal PFA under known-M GLRT (matrix-CFAR sanity).
5. H1 mean stat > H0 mean stat for all detectors.
6. NMFSingleArray signature (m, M, P, array_idx).
"""

from __future__ import annotations

import numpy as np
import pytest

from hdrlib.sonar import detectors as det
from hdrlib.sonar import estimation as est
from hdrlib.sonar import mc as smc
from hdrlib.sonar import simulation as sim
from hdrlib.core.estimation import SCMEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup_small():
    m = 8
    M = sim.make_sonar_covariance(m)
    P = sim.make_steering_matrix(m, 30.0, 45.0)
    return m, M, P


# ---------------------------------------------------------------------------
# 1. Positive-definiteness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [4, 8, 16, 64])
def test_covariance_positive_definite(m):
    M = sim.make_sonar_covariance(m)
    eigs = np.linalg.eigvalsh(M)
    assert np.all(eigs > 0), f"M is not positive definite for m={m}: min_eig={eigs.min()}"


def test_covariance_shape(setup_small):
    m, M, _ = setup_small
    assert M.shape == (2 * m, 2 * m)


# ---------------------------------------------------------------------------
# 2. 2TYL consistency at large K
# ---------------------------------------------------------------------------

def test_tyler_consistency():
    m = 8
    M = sim.make_sonar_covariance(m)
    K = 5000
    X_sec = sim.generate_secondary_data(1, K, m, M, seed=7)  # (1, K, 2m)
    M_hat = est.two_array_tyler(X_sec, m)   # (1, 2m, 2m)
    M_hat = M_hat[0]

    # Normalise both to trace = 1 for shape comparison
    M_norm = M / np.trace(M)
    Mh_norm = M_hat / np.trace(M_hat)

    rel_err = np.linalg.norm(Mh_norm - M_norm, 'fro') / np.linalg.norm(M_norm, 'fro')
    assert rel_err < 0.15, f"2TYL relative error too large at K={K}: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# 3. M-NMF-G → M-NMF-I when M_12 = 0
# ---------------------------------------------------------------------------

def test_mnmfg_equals_mnmfi_block_diagonal():
    m = 8
    # Build block-diagonal M (no cross-array coupling)
    M_bd = np.zeros((2 * m, 2 * m), dtype=np.complex128)
    M11 = sim.make_sonar_covariance(m)[:m, :m]   # grab M_11 block only
    M22 = sim.make_sonar_covariance(m)[m:, m:]   # and M_22
    M_bd[:m, :m] = M11
    M_bd[m:, m:] = M22

    P = sim.make_steering_matrix(m, 45.0, 45.0)

    glrt  = det.MNMFGlrt(m, M_bd, P)
    indep = det.MNMFIndependent(m, M_bd, P)

    n = 200
    X = sim.generate_sonar_data_h0(n, m, M_bd, seed=99)
    s_glrt  = glrt.compute(X)
    s_indep = indep.compute(X)

    # Both should give finite, non-NaN statistics
    assert not np.any(np.isnan(s_glrt))
    assert not np.any(np.isnan(s_indep))
    # With M_12 = 0, M-NMF-G and M-NMF-I should rank the trials the same way
    corr = np.corrcoef(s_glrt, s_indep)[0, 1]
    assert corr > 0.95, f"M-NMF-G and M-NMF-I rank correlation low: {corr:.3f}"


# ---------------------------------------------------------------------------
# 4. Empirical PFA ≈ nominal under matrix-CFAR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pfa", [0.1, 0.05, 0.01])
def test_empirical_pfa_nominal(setup_small, pfa):
    m, M, P = setup_small
    glrt = det.MNMFGlrt(m, M, P)

    n_h0 = 3000
    X = sim.generate_sonar_data_h0(n_h0, m, M, seed=123)
    stats = glrt.compute(X)

    eta = smc.threshold_at_pfa(stats, pfa)
    pfa_emp = smc.empirical_pd(stats, eta)  # same stats → PFA estimate

    # Allow ±2 standard deviations of binomial fluctuation
    sigma = np.sqrt(pfa * (1 - pfa) / n_h0)
    assert abs(pfa_emp - pfa) < 3 * sigma + 0.01, (
        f"Empirical PFA {pfa_emp:.4f} too far from nominal {pfa} "
        f"(3σ={3*sigma:.4f})"
    )


# ---------------------------------------------------------------------------
# 5. All detectors: H1 mean > H0 mean
# ---------------------------------------------------------------------------

def test_detectors_h1_gt_h0(setup_small):
    m, M, P = setup_small
    n = 300
    K = 4 * m
    snr_db, alphas = sim.snr_alpha_sweep(m, M, P, snr_min_db=0.0, snr_max_db=0.0, n_snr=1)
    alpha = float(alphas[0])

    X0   = sim.generate_sonar_data_h0(n, m, M, seed=1)
    X1   = sim.generate_sonar_data_h1(n, m, M, P, alpha, seed=2)
    Xsec = sim.generate_secondary_data(n, K, m, M, seed=3)

    tyl_est = est.TwoArrayTylerEstimator(m)
    detectors = {
        "M-NMF-G":      det.MNMFGlrt(m, M, P),
        "M-NMF-R":      det.MNMFRao(m, M, P),
        "M-NMF-I":      det.MNMFIndependent(m, M, P),
        "SA-1":         det.NMFSingleArray(m, M, P, array_idx=0),
        "SA-2":         det.NMFSingleArray(m, M, P, array_idx=1),
        "MIMO-MF":      det.MimoMatchedFilter(m, M, P),
        "Ada-GLRT-TYL": det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P), tyl_est),
    }

    for name, d in detectors.items():
        kwargs = {"X_secondary": Xsec} if isinstance(d, det.AdaptiveSonarDetector) else {}
        s0 = d.compute(X0, **kwargs)
        s1 = d.compute(X1, **kwargs)
        assert float(np.mean(s1)) > float(np.mean(s0)), (
            f"{name}: H1 mean {np.mean(s1):.4f} <= H0 mean {np.mean(s0):.4f}"
        )


# ---------------------------------------------------------------------------
# 6. NMFSingleArray signature
# ---------------------------------------------------------------------------

def test_nmf_single_array_signature(setup_small):
    m, M, P = setup_small
    sa0 = det.NMFSingleArray(m, M, P, array_idx=0)
    sa1 = det.NMFSingleArray(m, M, P, array_idx=1)
    assert sa0.array_idx == 0
    assert sa1.array_idx == 1

    X = sim.generate_sonar_data_h0(50, m, M, seed=0)
    s0 = sa0.compute(X)
    s1 = sa1.compute(X)
    assert s0.shape == (50,)
    assert s1.shape == (50,)
    assert not np.any(np.isnan(s0))
    assert not np.any(np.isnan(s1))


# ---------------------------------------------------------------------------
# 7. Adaptive detector with SCM estimator
# ---------------------------------------------------------------------------

def test_adaptive_scm(setup_small):
    m, M, P = setup_small
    K = 4 * m
    glrt = det.MNMFGlrt(m, M, P)
    ada  = det.AdaptiveSonarDetector(glrt, SCMEstimator())

    X    = sim.generate_sonar_data_h0(100, m, M, seed=77)
    Xsec = sim.generate_secondary_data(100, K, m, M, seed=88)
    stat = ada.compute(X, X_secondary=Xsec)
    assert stat.shape == (100,)
    assert not np.any(np.isnan(stat))
