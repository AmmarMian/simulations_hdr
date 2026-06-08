# SAR wavelet decomposition utilities
# Original decompose_image_wavelet by Ammar Mian, CentraleSupélec 2018
# Licensed under Apache 2.0
# Vectorized apply_wavelet_to_sits added for batch processing of SITS data.

import numpy as np
import matplotlib.pyplot as plt

# Physical defaults for Sentinel-1 SLC 1x1 data (L-band)
DEFAULT_CENTER_FREQUENCY = 1.26e9  # Hz
DEFAULT_BANDWIDTH = 80.0e6  # Hz
DEFAULT_RANGE_RESOLUTION = 1.66551366  # m
DEFAULT_AZIMUTH_RESOLUTION = 0.6  # m


def gbellmf(x, a, b, c):
    """Generalized Bell function fuzzy membership generator.

    Parameters
    ----------
    x : array
        Independent variable.
    a : float
        Width parameter.
    b : float
        Slope parameter.
    c : float
        Center parameter.

    Returns
    -------
    array
        y(x) = 1 / (1 + |[(x - c) / a]|^{2b})
    """
    return 1.0 / (1.0 + np.abs((x - c) / a) ** (2 * b))


# ---------------------------------------------------------------------------
# Private helpers shared by computation and visualization
# ---------------------------------------------------------------------------

def _build_frequency_grids(
    n_rows, n_cols, bandwidth, range_resolution, azimuth_resolution, center_frequency, R, L
):
    """Build wavenumber grids and filter-bank width parameters.

    Returns
    -------
    kappa : ndarray, shape (n_rows, n_cols)
    theta : ndarray, shape (n_rows, n_cols)
    width_k : float   — per-sub-band width in the kappa direction
    width_t : float   — per-sub-band width in the theta direction
    """
    c = 3e8
    kappa_0 = 2 * center_frequency / c
    k_range_vec = kappa_0 + (2 * bandwidth / c) * np.linspace(-0.5, 0.5, n_cols)
    k_az_vec = np.linspace(
        -1 / (2 * azimuth_resolution),
        1 / (2 * azimuth_resolution) - 1 / (2 * n_rows * azimuth_resolution),
        n_rows,
    )
    KX, KY = np.meshgrid(k_range_vec, k_az_vec)
    kappa = np.sqrt(KX**2 + KY**2)
    theta = np.arctan2(KY, KX)
    width_k = (kappa.max() - kappa.min()) / R
    width_t = (theta.max() - theta.min()) / L
    return kappa, theta, width_k, width_t


def _wavelet_filter(kappa, theta, m, n, width_k, width_t, d_1, d_2, L):
    """Compute the (m, n) generalized-bell wavelet filter."""
    c_k = kappa.min() + width_k / 2 + m * width_k
    c_t = theta.min() + width_t / 2 + n * width_t
    return gbellmf(kappa, width_k / 2, d_1, c_k) * gbellmf(theta, width_t / 2, d_2, c_t)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def decompose_image_wavelet(
    image,
    bandwidth,
    range_resolution,
    azimuth_resolution,
    center_frequency,
    R,
    L,
    d_1,
    d_2,
    shift=True,
):
    """Wavelet decomposition of a single 2-D SAR image.

    Parameters
    ----------
    image : ndarray, shape (n_rows, n_cols)
        Complex SAR image.
    bandwidth : float
        Sensor bandwidth in Hz.
    range_resolution : float
        Range pixel spacing in metres.
    azimuth_resolution : float
        Azimuth pixel spacing in metres.
    center_frequency : float
        Carrier frequency in Hz.
    R, L : int
        Number of range / azimuth sub-bands.
    d_1, d_2 : float
        Shape parameters of the generalized bell filters.
    shift : bool
        Use fftshift convention. Default True.

    Returns
    -------
    C : ndarray, shape (n_rows, n_cols, R*L), complex
        Wavelet decomposition into R*L sub-bands.
    """
    n_rows, n_cols = image.shape[:2]
    kappa, theta, width_k, width_t = _build_frequency_grids(
        n_rows, n_cols, bandwidth, range_resolution, azimuth_resolution, center_frequency, R, L
    )
    spectre = np.fft.fftshift(np.fft.fft2(image)) if shift else np.fft.fft2(image)

    C = np.zeros((n_rows, n_cols, R * L), dtype=complex)
    for m in range(R):
        for n in range(L):
            H = _wavelet_filter(kappa, theta, m, n, width_k, width_t, d_1, d_2, L)
            filtered = spectre * H
            if shift:
                C[:, :, m * L + n] = np.fft.ifft2(np.fft.fftshift(filtered))
            else:
                C[:, :, m * L + n] = np.fft.ifft2(filtered)
    return C


def apply_wavelet_to_sits(
    sits_data,
    R=3,
    L=3,
    d_1=10.0,
    d_2=10.0,
    center_frequency=DEFAULT_CENTER_FREQUENCY,
    bandwidth=DEFAULT_BANDWIDTH,
    range_resolution=DEFAULT_RANGE_RESOLUTION,
    azimuth_resolution=DEFAULT_AZIMUTH_RESOLUTION,
):
    """Apply wavelet decomposition to a full SITS (Satellite Image Time Series).

    Vectorized implementation: a single FFT is computed over the entire
    (n_rows, n_cols, p, T) stack, then each of the R*L filters is applied
    once via broadcasting over all p polarisations and T dates simultaneously.
    This avoids the O(p*T) FFT calls of the naive per-channel loop.

    Parameters
    ----------
    sits_data : ndarray, shape (n_rows, n_cols, p, T)
        Complex SAR SITS. Each channel sits_data[:,:,i,t] is a 2-D SAR image.
    R : int
        Number of range sub-bands. Default 3.
    L : int
        Number of azimuth sub-bands. Default 3.
    d_1, d_2 : float
        Generalized-bell shape parameters for the filters. Default 10.
    center_frequency : float
        Carrier frequency in Hz. Default: Sentinel-1 L-band (1.26 GHz).
    bandwidth : float
        Sensor bandwidth in Hz. Default: Sentinel-1 (80 MHz).
    range_resolution : float
        Range pixel spacing in metres. Default: Sentinel-1 SLC 1x1.
    azimuth_resolution : float
        Azimuth pixel spacing in metres. Default: Sentinel-1 SLC 1x1.

    Returns
    -------
    ndarray, shape (n_rows, n_cols, p*R*L, T), complex
        Each original polarisation i is expanded to R*L sub-bands placed at
        indices [i*R*L : (i+1)*R*L] in the feature dimension.
    """
    n_rows, n_cols, p, T = sits_data.shape
    kappa, theta, width_k, width_t = _build_frequency_grids(
        n_rows, n_cols, bandwidth, range_resolution, azimuth_resolution, center_frequency, R, L
    )

    # Single FFT over the full stack: (n_rows, n_cols, p, T)
    spectre = np.fft.fftshift(np.fft.fft2(sits_data, axes=(0, 1)), axes=(0, 1))

    # Output buffer: (n_rows, n_cols, p, R*L, T)
    result = np.zeros((n_rows, n_cols, p, R * L, T), dtype=complex)
    for m in range(R):
        for n in range(L):
            H = _wavelet_filter(kappa, theta, m, n, width_k, width_t, d_1, d_2, L)
            filtered = spectre * H[:, :, None, None]  # broadcast over (p, T)
            result[:, :, :, m * L + n, :] = np.fft.ifft2(
                np.fft.fftshift(filtered, axes=(0, 1)), axes=(0, 1)
            )

    # Merge p and R*L dims: (n_rows, n_cols, p, R*L, T) → (n_rows, n_cols, p*R*L, T)
    return result.reshape(n_rows, n_cols, p * R * L, T)


# ---------------------------------------------------------------------------
# Visualization (separated from computation)
# ---------------------------------------------------------------------------

def plot_wavelet_decomposition(
    image,
    bandwidth,
    range_resolution,
    azimuth_resolution,
    center_frequency,
    R,
    L,
    d_1,
    d_2,
    dyn_dB=50,
    shift=True,
):
    """Visualize the wavelet filter bank applied to a single SAR image.

    Creates two figures: one showing each sub-band's filtered spectrum
    (signal × wavelet), and one showing the spatial decomposition.

    Parameters
    ----------
    image : ndarray, shape (n_rows, n_cols)
    bandwidth, range_resolution, azimuth_resolution, center_frequency : float
        Same as :func:`decompose_image_wavelet`.
    R, L : int
    d_1, d_2 : float
    dyn_dB : float
        Dynamic range in dB for display. Default 50.
    shift : bool
        FFT-shift convention. Default True.

    Returns
    -------
    fig_spectrum : plt.Figure
        R×L grid showing filtered spectra (signal × wavelet).
    fig_decomposition : plt.Figure
        R×L grid showing spatial sub-band images.
    """
    n_rows, n_cols = image.shape[:2]
    kappa, theta, width_k, width_t = _build_frequency_grids(
        n_rows, n_cols, bandwidth, range_resolution, azimuth_resolution, center_frequency, R, L
    )
    spectre = np.fft.fftshift(np.fft.fft2(image)) if shift else np.fft.fft2(image)

    fig_s, axes_s = plt.subplots(R, L, figsize=(20, 17), squeeze=False)
    fig_i, axes_i = plt.subplots(R, L, figsize=(20, 17), squeeze=False)
    fig_s.suptitle("Signal × wavelet", fontsize="x-large")
    fig_i.suptitle("Wavelet decomposition", fontsize="x-large")

    for m in range(R):
        for n in range(L):
            H = _wavelet_filter(kappa, theta, m, n, width_k, width_t, d_1, d_2, L)
            filtered = spectre * H
            sub = np.fft.ifft2(np.fft.fftshift(filtered)) if shift else np.fft.ifft2(filtered)

            tp = 20 * np.log10(np.abs(filtered) + 1e-12)
            axes_s[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
            axes_s[m, n].set_axis_off()

            tp = 20 * np.log10(np.abs(sub) + 1e-12)
            axes_i[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
            axes_i[m, n].set_axis_off()

    return fig_s, fig_i


def plot_wavelet_to_sits_debug(
    sits_data,
    R,
    L,
    d_1=10.0,
    d_2=10.0,
    center_frequency=DEFAULT_CENTER_FREQUENCY,
    bandwidth=DEFAULT_BANDWIDTH,
    range_resolution=DEFAULT_RANGE_RESOLUTION,
    azimuth_resolution=DEFAULT_AZIMUTH_RESOLUTION,
    dyn_dB=50,
    save_path=None,
):
    """Visualize and optionally save wavelet debug plots for a SITS.

    Uses the first polarisation and first date as a representative example.

    Parameters
    ----------
    sits_data : ndarray, shape (n_rows, n_cols, p, T)
    R, L : int
    d_1, d_2 : float
    center_frequency, bandwidth, range_resolution, azimuth_resolution : float
    dyn_dB : float
    save_path : str or None
        If given, saves to ``{save_path}_spectrum.png`` and
        ``{save_path}_decomposition.png`` and closes the figures.

    Returns
    -------
    fig_spectrum : plt.Figure
    fig_decomposition : plt.Figure
    """
    n_rows, n_cols = sits_data.shape[:2]
    kappa, theta, width_k, width_t = _build_frequency_grids(
        n_rows, n_cols, bandwidth, range_resolution, azimuth_resolution, center_frequency, R, L
    )
    spectre_00 = np.fft.fftshift(np.fft.fft2(sits_data[:, :, 0, 0]))

    fig_s, axes_s = plt.subplots(R, L, figsize=(4 * L, 4 * R), squeeze=False)
    fig_i, axes_i = plt.subplots(R, L, figsize=(4 * L, 4 * R), squeeze=False)
    fig_s.suptitle("Signal × wavelet (pol=0, t=0)", fontsize="x-large")
    fig_i.suptitle("Wavelet decomposition (pol=0, t=0)", fontsize="x-large")

    # Run wavelet to get sub-band images for pol=0, t=0
    result = apply_wavelet_to_sits(
        sits_data[:, :, :1, :1], R=R, L=L, d_1=d_1, d_2=d_2,
        center_frequency=center_frequency, bandwidth=bandwidth,
        range_resolution=range_resolution, azimuth_resolution=azimuth_resolution,
    )  # (rows, cols, R*L, 1)

    for m in range(R):
        for n in range(L):
            H = _wavelet_filter(kappa, theta, m, n, width_k, width_t, d_1, d_2, L)

            tp = 20 * np.log10(np.abs(spectre_00 * H) + 1e-12)
            axes_s[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
            axes_s[m, n].set_title(f"R={m} L={n}", fontsize=8)
            axes_s[m, n].set_axis_off()

            tp = 20 * np.log10(np.abs(result[:, :, m * L + n, 0]) + 1e-12)
            axes_i[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
            axes_i[m, n].set_title(f"R={m} L={n}", fontsize=8)
            axes_i[m, n].set_axis_off()

    fig_s.tight_layout()
    fig_i.tight_layout()

    if save_path is not None:
        fig_s.savefig(f"{save_path}_spectrum.png", dpi=150)
        fig_i.savefig(f"{save_path}_decomposition.png", dpi=150)
        plt.close(fig_s)
        plt.close(fig_i)
        print(
            f"  Saved wavelet debug plots: "
            f"{save_path}_spectrum.png, {save_path}_decomposition.png"
        )

    return fig_s, fig_i
