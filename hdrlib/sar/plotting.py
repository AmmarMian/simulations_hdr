import numpy as np
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from ..core.backend import Array, get_data_on_device


def plot_pauli(sar_data: Array) -> AxesImage:
    """Plot Pauli representation of an SLC SAR image by mapping:
        * Red <- |HH-VV|
        * Green <- |HV|
        * Blue <- |HH+VV|

    Parameters
    ----------
    sar_data : Array
        data of shape (n_rows, n_cols, 3), with third dimension in order HH/HV/VV

    Returns
    -------
    AxesImage
        The plotted image object thanks to matplotlib
    """
    sar_data = get_data_on_device(sar_data, "numpy")
    R = np.abs(sar_data[:, :, 0] - sar_data[:, :, 2])
    G = np.abs(sar_data[:, :, 1])
    B = np.abs(sar_data[:, :, 0] + sar_data[:, :, 2])

    return plt.imshow(np.dstack([R, G, B]), aspect="auto")
