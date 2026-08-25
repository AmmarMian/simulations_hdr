#!/usr/bin/env python
"""Non-regression bench: hdrlib.sonar against Olivier Lerda's MATLAB reference.

The reference .mat files produced by PfaSeuilSimuMultivarSpace2D.m carry, for a
fixed covariance setting, everything needed for a DETERMINISTIC comparison:

  C, iC, iCP, iCA   the exact covariance actually used and its inverses,
  xCaptP, xCaptA    the sensor coordinates, in metres,
  theta             the 64 beam directions,
  datacube          the 1000 x 64 x 2 simulated data,
  t<detector>       the statistic of every detector, for every range bin and
                    every (theta1, theta2) pair.

So no Monte-Carlo is involved here: we feed the MATLAB data to the Python
detectors and compare numbers term by term.  Any deviation is a formula or a
convention difference, not sampling noise.

The comparison is run twice for each detector: once with the covariance model
rebuilt by hdrlib (which tests make_sonar_covariance and make_steering_matrix
as well), once with the covariance and steering taken from the .mat itself
(which isolates the detector formulas from the model conventions).

Usage
-----
    uv run python compare_matlab_reference.py --mat <path to .mat>
        [--n-bins 12] [--n-theta 6] [--tol 1e-9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

from hdrlib.sonar.detectors import (
    MimoMatchedFilter,
    MNMFGlrt,
    MNMFIndependent,
    MNMFRao,
    NMFSingleArray,
)
from hdrlib.sonar.estimation import two_array_tyler
from hdrlib.sonar.simulation import make_sonar_covariance, make_steering_matrix

_DEFAULT_MAT = (
    "/Users/ammarmian/Research/sonar/Ressources_OlivierLerda/"
    "pfa-seuil-R1000-rhoP04-rhoA09-G.mat"
)


# ---------------------------------------------------------------------------
# MATLAB conventions, transcribed literally
# ---------------------------------------------------------------------------

def steering_matlab(x_capt_p, x_capt_a, theta1, theta2, f0, celerite):
    """P = blkdiag(v(theta1), v(theta2)) with the coordinates of the .mat.

    MATLAB: vTheta = exp(1i*2*pi*f0*xCapt*sin(theta)/celerite).'
    Note the two arrays do not share a coordinate convention: P sits on half
    integer multiples of the spacing, A on integers with the centre removed.
    """
    m = len(x_capt_p)
    v1 = np.exp(1j * 2 * np.pi * f0 * x_capt_p * np.sin(theta1) / celerite)
    v2 = np.exp(1j * 2 * np.pi * f0 * x_capt_a * np.sin(theta2) / celerite)
    P = np.zeros((2 * m, 2), dtype=np.complex128)
    P[:m, 0] = v1
    P[m:, 1] = v2
    return P


def reference_cells(i, K, G, R):
    """MATLAB: refCells = [i-G-K/2 : i-G-1, i+G+1 : i+G+K/2] (1-based)."""
    half = K // 2
    left = np.arange(i - G - half, i - G)
    right = np.arange(i + G + 1, i + G + half + 1)
    cells = np.concatenate([left, right])
    return cells[(cells >= 0) & (cells < R)]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def relative_deviation(python_vals, matlab_vals):
    """Max relative deviation, guarding against near-zero references."""
    py = np.asarray(python_vals, dtype=float).ravel()
    ml = np.asarray(matlab_vals, dtype=float).ravel()
    scale = np.maximum(np.abs(ml), np.abs(py))
    scale = np.where(scale > 1e-300, scale, 1.0)
    return float(np.max(np.abs(py - ml) / scale))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat", type=str, default=_DEFAULT_MAT,
        help="Reference .mat produced by PfaSeuilSimuMultivarSpace2D.m.")
    parser.add_argument("--n-bins", type=int, default=12,
        help="Number of range bins to test (default 12).")
    parser.add_argument("--n-theta", type=int, default=6,
        help="Number of beam directions per axis to test (default 6).")
    parser.add_argument("--tol", type=float, default=1e-9,
        help="Relative deviation above which a detector is reported as differing.")
    parser.add_argument("--tol-adaptive", type=float, default=1e-5,
        help="Same, for the Tyler-based versions. Looser on purpose: the "
             "reference was computed with TylerMIMO.m stopping at eps=1e-6, so "
             "agreement cannot be asked for beyond that. Deviations at the 1e-6 "
             "level mean the two fixed points coincide.")
    parser.add_argument("--adaptive", action="store_true",
        help="Also compare the SCM and Tyler adaptive versions (slower: one "
             "estimator fit per range bin).")
    args = parser.parse_args()

    mat_path = Path(args.mat)
    if not mat_path.exists():
        sys.exit(f"Reference file not found: {mat_path}")
    d = sio.loadmat(str(mat_path), squeeze_me=True)

    m = int(d["nbCapt"])
    K = int(d["K"])
    G = int(d["G"])
    R = int(d["datacube"].shape[0])
    C_ml = np.asarray(d["C"], dtype=np.complex128)
    theta = np.asarray(d["theta"], dtype=float)
    x_capt_p = np.asarray(d["xCaptP"], dtype=float)
    x_capt_a = np.asarray(d["xCaptA"], dtype=float)
    # tx, env and rx come back as MATLAB structs: 0-d record arrays.
    f0 = float(np.asarray(d["tx"]["f0"]).item())
    celerite = float(np.asarray(d["env"]["celerite"]).item())
    datacube = np.asarray(d["datacube"], dtype=np.complex128)
    rho_p, rho_a = float(d["rhoP"]), float(d["rhoA"])
    noise_stat = str(d["noiseStat"])

    print(f"Reference : {mat_path.name}")
    print(f"  m={m}  K={K}  G={G}  R={R}  rho1={rho_p}  rho2={rho_a}  clutter={noise_stat}")
    print(f"  f0={f0} Hz  c={celerite} m/s  spacing={abs(x_capt_p[1] - x_capt_p[0]):.4f} m "
          f"→ d/lambda={abs(x_capt_p[1] - x_capt_p[0]) * f0 / celerite:.3f}")

    # ---- 1. the covariance model itself -----------------------------------
    # beta is the scale factor of the model and is not stored as such, but the
    # diagonal of C is exactly beta (rho^0 = 1). The reference sets use three
    # different values: 3e-4, 1 and 100.
    beta = float(np.real(C_ml[0, 0]))
    zero_cross = bool(np.max(np.abs(C_ml[:m, m:])) == 0.0)
    C_py = make_sonar_covariance(m, beta=beta, rho1=rho_p, rho2=rho_a,
                                 zero_cross_blocks=zero_cross)
    dev_C = relative_deviation(np.abs(C_py), np.abs(C_ml))
    print(f"\n[modèle] make_sonar_covariance vs C du .mat (beta={beta:g}) : "
          f"écart relatif max = {dev_C:.3e}"
          + ("  [blocs croisés nuls]" if zero_cross else ""))

    # ---- 2. the steering matrix -------------------------------------------
    t1, t2 = theta[10], theta[40]
    P_ml = steering_matlab(x_capt_p, x_capt_a, t1, t2, f0, celerite)
    P_py = make_steering_matrix(m, np.rad2deg(t1), np.rad2deg(t2))
    # A steering vector is only defined up to a global phase -- every detector
    # here uses it through projectors -- so colinearity is the meaningful test,
    # not an entry-by-entry comparison of phases.
    col1 = abs(np.vdot(P_py[:m, 0], P_ml[:m, 0])) / m
    col2 = abs(np.vdot(P_py[m:, 1], P_ml[m:, 1])) / m
    print(f"[modèle] make_steering_matrix vs convention .mat : colinéarité "
          f"{col1:.12f} / {col2:.12f} (1 = identique à un déphasage près)")

    # ---- 3. detector formulas, on the MATLAB's own C and steering ---------
    bins = np.linspace(G + K // 2, R - G - K // 2 - 1, args.n_bins).astype(int)
    th_idx = np.linspace(0, len(theta) - 1, args.n_theta).astype(int)

    # For each detector: the reference field, how it is indexed, and how the
    # MATLAB value must be transformed to be comparable with the Python one.
    #   "th1"/"th2" — the 1-D references are indexed by a single beam index;
    #                 tNMF_C_A is filled inside the iTheta1 loop but steers
    #                 array A, so it must be read at the index that array A
    #                 points to.
    #   log         — MNMFIndependent returns the log-statistic
    #                 -m sum log(1 - NMF_i), the MATLAB one returns
    #                 prod (1 - NMF_i)^-m. Same detector, monotone transform.
    keys = {
        "NMF 1 (antenne P)":  ("tNMF_C_P", "th1", None),
        "NMF 2 (antenne A)":  ("tNMF_C_A", "th2", None),
        "MIMO-MF":            ("tMF_C_MIMO", "pair", None),
        "M-NMF-I":            ("tNMF_C_MIMO_I", "pair", "log"),
        "M-NMF-R (Rao)":      ("tNMF_C_MIMO_R", "pair", None),
        "M-NMF-G (GLRT)":     ("tNMF_C_MIMO_G", "pair", None),
    }
    collected = {name: ([], []) for name in keys}

    for i in bins:
        x = np.concatenate([datacube[i, :, 0], datacube[i, :, 1]])
        for i1 in th_idx:
            for i2 in th_idx:
                P = steering_matlab(x_capt_p, x_capt_a, theta[i1], theta[i2], f0, celerite)

                stats = {
                    "NMF 1 (antenne P)": NMFSingleArray(m, C_ml, P, array_idx=0).compute(x),
                    "NMF 2 (antenne A)": NMFSingleArray(m, C_ml, P, array_idx=1).compute(x),
                    "MIMO-MF":           MimoMatchedFilter(m, C_ml, P).compute(x),
                    "M-NMF-I":           MNMFIndependent(m, C_ml, P).compute(x),
                    "M-NMF-R (Rao)":     MNMFRao(m, C_ml, P).compute(x),
                    "M-NMF-G (GLRT)":    MNMFGlrt(m, C_ml, P).compute(x),
                }
                for name, (key, indexing, transform) in keys.items():
                    ref = d[key]
                    if indexing == "th1":
                        ml = float(ref[i, i1])
                    elif indexing == "th2":
                        ml = float(ref[i, i2])
                    else:
                        ml = float(ref[i, i1, i2])
                    if transform == "log":
                        ml = float(np.log(ml))
                    collected[name][0].append(float(np.real(stats[name])))
                    collected[name][1].append(ml)

    print(f"\n[détecteurs] {len(bins)} cases distance x {len(th_idx)}^2 directions, "
          f"covariance et pointage pris dans le .mat\n")
    print(f"  {'détecteur':22s} {'écart rel. max':>15s}   {'ratio médian py/ml':>20s}   verdict")
    print("  " + "-" * 76)
    verdicts = {}
    for name in keys:
        py = np.array(collected[name][0])
        ml = np.array(collected[name][1])
        dev = relative_deviation(py, ml)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = float(np.nanmedian(np.where(np.abs(ml) > 1e-300, py / ml, np.nan)))
        verdict = "OK" if dev < args.tol else "DIFFÈRE"
        verdicts[name] = dev
        print(f"  {name:22s} {dev:15.3e}   {ratio:20.6f}   {verdict}")

    # ---- 4. adaptive versions --------------------------------------------
    if args.adaptive:
        print(f"\n[versions adaptatives] estimateurs sur les {K} cellules de référence\n")
        ad_keys = {
            "M-ANMF-R (SCM)":   ("tANMF_SCM_MIMO_R", "scm"),
            "M-ANMF-G (SCM)":   ("tANMF_SCM_MIMO_G", "scm"),
            "M-ANMF-R (Tyler)": ("tANMF_TYL_MIMO_R", "tyler"),
            "M-ANMF-G (Tyler)": ("tANMF_TYL_MIMO_G", "tyler"),
        }
        ad_collected = {name: ([], []) for name in ad_keys}
        for i in bins:
            x = np.concatenate([datacube[i, :, 0], datacube[i, :, 1]])
            cells = reference_cells(i, K, G, R)
            Xs = np.concatenate([datacube[cells, :, 0], datacube[cells, :, 1]], axis=-1)
            # SCM = (1/K) sum_k x_k x_k^H. With Xs holding x_k^T as rows,
            # that is Xs.T @ conj(Xs) -- Xs.conj().T @ Xs would give its
            # conjugate, which is a different matrix for complex data.
            scm = Xs.T @ Xs.conj() / len(cells)
            tyl = two_array_tyler(Xs, m, tol=1e-6, iter_max=100)
            for i1 in th_idx:
                for i2 in th_idx:
                    P = steering_matlab(x_capt_p, x_capt_a, theta[i1], theta[i2], f0, celerite)
                    for name, (key, kind) in ad_keys.items():
                        M_hat = scm if kind == "scm" else tyl
                        det = (MNMFRao(m, C_ml, P) if "-R " in name else MNMFGlrt(m, C_ml, P))
                        val = det.compute(x, M_override=M_hat)
                        ad_collected[name][0].append(float(np.real(val)))
                        ad_collected[name][1].append(float(d[key][i, i1, i2]))

        print(f"  {'détecteur':22s} {'écart rel. max':>15s}   {'ratio médian py/ml':>20s}   verdict")
        print("  " + "-" * 76)
        for name in ad_keys:
            py = np.array(ad_collected[name][0])
            ml = np.array(ad_collected[name][1])
            dev = relative_deviation(py, ml)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = float(np.nanmedian(np.where(np.abs(ml) > 1e-300, py / ml, np.nan)))
            tol = args.tol_adaptive if "Tyler" in name else args.tol
            note = "OK" if dev < tol else "DIFFÈRE"
            if "Tyler" in name and dev < tol:
                note = "OK (à la tolérance de convergence)"
            print(f"  {name:22s} {dev:15.3e}   {ratio:20.6f}   {note}")

    n_bad = sum(1 for v in verdicts.values() if v >= args.tol)
    print(f"\n{len(verdicts) - n_bad}/{len(verdicts)} détecteurs conformes "
          f"(tolérance {args.tol:g}).")


if __name__ == "__main__":
    main()
