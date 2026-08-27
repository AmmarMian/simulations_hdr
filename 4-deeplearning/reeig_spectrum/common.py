# Shared measurements for the ReEig study.
#
# Both entry points of this directory — main.py on simulated CovPool matrices,
# real_data.py on the datasets of the chapter — report the same three
# quantities, so that the simulated figure and the real one can be read on the
# same axes.
#
# The quantity that matters is the last one. Backpropagating through a spectral
# layer multiplies the incoming error by the Loewner matrix of
# prop:spdnet-diffm,
#
#     G_ij = (h(lambda_i) - h(lambda_j)) / (lambda_i - lambda_j),
#
# whose largest entry is what makes or breaks the gradient. For h = log, the
# mean value theorem gives |G_ij| <= 1 / min(lambda_i, lambda_j) = 1 / lambda_min,
# so the instability of prop:spdnet-gradevd is governed by the *smallest*
# eigenvalue and nothing else. A ReEig layer of threshold eps sets
# lambda_min >= eps, hence caps the Loewner matrix at 1/eps. That bound is what
# these scripts measure: it turns rem:spdnet-reeig-retrecissement from an
# analogy with shrinkage into a statement one can check.

import torch

from yetanotherspdnet.nn.base import ReEig


def covpool(features):
    """The CovPool layer of eq:spdnet-covpool, written out.

    Parameters
    ----------
    features : torch.Tensor of shape (..., n_filters, n_pixels)
        Filter maps, each flattened over the spatial positions.

    Returns
    -------
    torch.Tensor of shape (..., n_filters, n_filters)
        The empirical covariance of the channels over the positions. This is
        ``T J T^T`` with ``J`` the centring matrix, i.e. exactly
        def:spdnet-covpool and not a plain second moment.
    """
    n_pixels = features.shape[-1]
    centred = features - features.mean(dim=-1, keepdim=True)
    return centred @ centred.transpose(-1, -2) / n_pixels


def loewner_max_log(eigenvalues):
    """Largest entry of the Loewner matrix of ``logm``, per matrix.

    Computed in closed form rather than by forming the matrix: for ``h = log``
    the largest entry of ``G`` is always ``1 / lambda_min``, attained on the
    diagonal entry of the smallest eigenvalue. Forming the full ``(p, p)``
    matrix for every trial would dominate the runtime and change nothing.
    """
    return 1.0 / eigenvalues.min(dim=-1).values


def spectral_summary(covariances, eps):
    """Everything the figures need, for one batch of SPD matrices.

    Returns a dict of 1-D tensors, one entry per matrix of the batch, holding
    the quantities before and after a ReEig layer of threshold ``eps``.
    """
    eigenvalues = torch.linalg.eigvalsh(covariances)
    rectified = torch.linalg.eigvalsh(ReEig(eps=eps)(covariances))

    return {
        "eigenvalues": eigenvalues,
        "lambda_min": eigenvalues.min(dim=-1).values,
        "lambda_max": eigenvalues.max(dim=-1).values,
        "condition": eigenvalues.max(dim=-1).values / eigenvalues.min(dim=-1).values,
        "condition_reeig": rectified.max(dim=-1).values / rectified.min(dim=-1).values,
        # Fraction of the spectrum the layer actually rectifies. This is the
        # honest measure of how much ReEig is doing: at a ratio where nothing
        # is clamped the layer is a no-op, and the bound below is vacuous.
        "fraction_clamped": (eigenvalues <= eps).to(eigenvalues.dtype).mean(dim=-1),
        "loewner_max": loewner_max_log(eigenvalues),
        "loewner_max_reeig": loewner_max_log(rectified),
    }


def decaying_covariance(n_filters, decay, device, dtype):
    """A true covariance whose spectrum decays geometrically.

    Real feature maps and real spectral images have decaying spectra; the
    ``random_SPD`` of yetanotherspdnet draws its eigenvalues uniformly over a
    wide interval instead, which puts a single eigenvalue at the bottom and
    leaves the rest well separated. In that spectrum a ReEig layer clamps one
    eigenvalue per matrix whatever its threshold, so the regime this study is
    about never occurs. Hence a geometric decay here, with ``decay`` the ratio
    between the largest and the smallest eigenvalue.
    """
    exponents = torch.linspace(0, 1, n_filters, device=device, dtype=dtype)
    return torch.diag(decay ** (-exponents))


def sample_covpool(true_covariance, n_pixels, n_trials, generator, device, dtype):
    """Draw ``n_trials`` CovPool matrices at a given number of positions.

    The columns of the filter map are drawn i.i.d. Gaussian with the given true
    covariance, which is the model under which eq:spdnet-covpool is an
    empirical covariance estimated from ``n_pixels`` samples for
    ``n_filters`` variables — the dimensional regime of
    rem:spdnet-covpool-regime.
    """
    n_filters = true_covariance.shape[-1]
    noise = torch.randn(
        (n_trials, n_filters, n_pixels),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    root = torch.linalg.cholesky(true_covariance)
    return covpool(root @ noise)


def resolve_device(name):
    """``--device`` handling, with the reason for a refusal spelled out.

    MPS is rejected rather than silently downgraded: it has no float64 and no
    ``linalg.eigh``, so every spectral layer of this study would fall back to
    the CPU anyway, one operation at a time and with a transfer around each.
    """
    if name == "mps":
        raise SystemExit(
            "MPS cannot run this study: it has no float64, and linalg.eigh is "
            "not implemented for it, so ReEig and LogEig would silently fall "
            "back to the CPU. Use --device cpu or --device cuda."
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda asked for, but torch.cuda is unavailable.")
    return torch.device(name)
