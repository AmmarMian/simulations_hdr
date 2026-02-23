# SAR wavelet decomposition utilities
# Original decompose_image_wavelet by Ammar Mian, CentraleSupélec 2018
# Licensed under Apache 2.0
# Vectorized apply_wavelet_to_sits added for batch processing of SITS data.

import numpy as np


# Physical defaults for Sentinel-1 SLC 1x1 data (L-band)
DEFAULT_CENTER_FREQUENCY = 1.26e9       # Hz
DEFAULT_BANDWIDTH        = 80.0e6       # Hz
DEFAULT_RANGE_RESOLUTION = 1.66551366   # m
DEFAULT_AZIMUTH_RESOLUTION = 0.6        # m


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


def decompose_image_wavelet(
    image, bandwidth, range_resolution, azimuth_resolution,
    center_frequency, R, L, d_1, d_2,
    show_decomposition=False, dyn_dB=50, shift=True,
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
    show_decomposition : bool
        Display intermediate spectra via matplotlib. Default False.
    dyn_dB : float
        Dynamic range (dB) for display. Default 50.
    shift : bool
        Use fftshift convention. Default True.

    Returns
    -------
    C : ndarray, shape (n_rows, n_cols, R*L), complex
        Wavelet decomposition into R*L sub-bands.
    """
    n_rows, n_cols = image.shape[0], image.shape[1]
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

    if shift:
        spectre = np.fft.fftshift(np.fft.fft2(image))
    else:
        spectre = np.fft.fft2(image)

    C = np.zeros((n_rows, n_cols, R * L), dtype=complex)
    kappa_B = kappa.max() - kappa.min()
    theta_B = theta.max() - theta.min()
    width_k = kappa_B / R
    width_t = theta_B / L

    if show_decomposition:
        import matplotlib.pyplot as plt
        fig_s, axes_s = plt.subplots(R, L, figsize=(20, 17))
        fig_i, axes_i = plt.subplots(R, L, figsize=(20, 17))
        fig_s.suptitle("Signal × wavelet", fontsize="x-large")
        fig_i.suptitle("Wavelet decomposition", fontsize="x-large")

    for m in range(R):
        for n in range(L):
            c_k = kappa.min() + width_k / 2 + m * width_k
            c_t = theta.min() + width_t / 2 + n * width_t
            H = gbellmf(kappa, width_k / 2, d_1, c_k) * gbellmf(theta, width_t / 2, d_2, c_t)
            filtered = spectre * H
            if shift:
                C[:, :, m * L + n] = np.fft.ifft2(np.fft.fftshift(filtered))
            else:
                C[:, :, m * L + n] = np.fft.ifft2(filtered)

            if show_decomposition:
                tp = 20 * np.log10(np.abs(filtered))
                axes_s[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
                axes_s[m, n].set_axis_off()
                tp = 20 * np.log10(np.abs(C[:, :, m * L + n]))
                axes_i[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
                axes_i[m, n].set_axis_off()

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
    save_path=None,
    dyn_dB=50,
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
    save_path : str or None
        If given, save debug figures to ``{save_path}_spectrum.png`` and
        ``{save_path}_decomposition.png`` using the first polarisation and
        first date as a representative example. Default None (no saving).
    dyn_dB : float
        Dynamic range (dB) for debug figures. Default 50.

    Returns
    -------
    ndarray, shape (n_rows, n_cols, p*R*L, T), complex
        Each original polarisation i is expanded to R*L sub-bands placed at
        indices [i*R*L : (i+1)*R*L] in the feature dimension.
    """
    import matplotlib.pyplot as plt

    n_rows, n_cols, p, T = sits_data.shape
    c = 3e8
    kappa_0 = 2 * center_frequency / c

    # Build frequency grids (same for every polarisation and date)
    k_range_vec = kappa_0 + (2 * bandwidth / c) * np.linspace(-0.5, 0.5, n_cols)
    k_az_vec = np.linspace(
        -1 / (2 * azimuth_resolution),
        1 / (2 * azimuth_resolution) - 1 / (2 * n_rows * azimuth_resolution),
        n_rows,
    )
    KX, KY = np.meshgrid(k_range_vec, k_az_vec)
    kappa = np.sqrt(KX**2 + KY**2)
    theta = np.arctan2(KY, KX)

    kappa_B = kappa.max() - kappa.min()
    theta_B = theta.max() - theta.min()
    width_k = kappa_B / R
    width_t = theta_B / L

    # Single FFT over the full stack: (n_rows, n_cols, p, T)
    spectre = np.fft.fftshift(
        np.fft.fft2(sits_data, axes=(0, 1)), axes=(0, 1)
    )

    # Output buffer: (n_rows, n_cols, p, R*L, T)
    result = np.zeros((n_rows, n_cols, p, R * L, T), dtype=complex)

    if save_path is not None:
        fig_s, axes_s = plt.subplots(R, L, figsize=(4 * L, 4 * R))
        fig_i, axes_i = plt.subplots(R, L, figsize=(4 * L, 4 * R))
        fig_s.suptitle("Signal × wavelet (pol=0, t=0)", fontsize="x-large")
        fig_i.suptitle("Wavelet decomposition (pol=0, t=0)", fontsize="x-large")
        # Spectrum of the first image for display
        spectre_00 = spectre[:, :, 0, 0]

    for m in range(R):
        for n in range(L):
            c_k = kappa.min() + width_k / 2 + m * width_k
            c_t = theta.min() + width_t / 2 + n * width_t
            H = (
                gbellmf(kappa, width_k / 2, d_1, c_k)
                * gbellmf(theta, width_t / 2, d_2, c_t)
            )  # (n_rows, n_cols)

            # Broadcast H over p and T: (n_rows, n_cols, 1, 1)
            filtered = spectre * H[:, :, None, None]  # (n_rows, n_cols, p, T)
            result[:, :, :, m * L + n, :] = np.fft.ifft2(
                np.fft.fftshift(filtered, axes=(0, 1)), axes=(0, 1)
            )

            if save_path is not None:
                tp = 20 * np.log10(np.abs(spectre_00 * H) + 1e-12)
                axes_s[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
                axes_s[m, n].set_title(f"R={m} L={n}", fontsize=8)
                axes_s[m, n].set_axis_off()
                tp = 20 * np.log10(np.abs(result[:, :, 0, m * L + n, 0]) + 1e-12)
                axes_i[m, n].imshow(tp, cmap="gray", aspect="auto", vmin=tp.max() - dyn_dB)
                axes_i[m, n].set_title(f"R={m} L={n}", fontsize=8)
                axes_i[m, n].set_axis_off()

    if save_path is not None:
        fig_s.tight_layout()
        fig_i.tight_layout()
        fig_s.savefig(f"{save_path}_spectrum.png", dpi=150)
        fig_i.savefig(f"{save_path}_decomposition.png", dpi=150)
        plt.close(fig_s)
        plt.close(fig_i)
        print(f"  Saved wavelet debug plots: {save_path}_spectrum.png, {save_path}_decomposition.png")

    # Merge p and R*L dims: (n_rows, n_cols, p, R*L, T) → (n_rows, n_cols, p*R*L, T)
    # Reshape preserves ordering: features for polarisation i at [i*R*L : (i+1)*R*L]
    return result.reshape(n_rows, n_cols, p * R * L, T)
