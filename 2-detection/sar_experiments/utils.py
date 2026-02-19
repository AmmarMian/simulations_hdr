# Utilities for SAR experiments

import sys
from pathlib import Path

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from src.backend import get_backend_module, get_data_on_device, Array
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage


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
    R = np.abs(sar_data[:, :, 0] - sar_data[:, :, 2])
    G = np.abs(sar_data[:, :, 1])
    B = np.abs(sar_data[:, :, 0] + sar_data[:, :, 2])

    return plt.imshow(np.dstack([R, G, B]), aspect="auto")
