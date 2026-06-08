# Utilities for SAR experiments

import numpy as np
from src.backend import Array
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import sys
from pathlib import Path

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def require_time_first(data_path: str) -> str:
    """Return the time-first version of a SITS .npy file, or exit with instructions.

    Looks for a ``<stem>_time_first.npy`` file next to *data_path*.
    If found, returns its path as a string.
    If not found, prints a message telling the user how to create it and exits.

    Parameters
    ----------
    data_path : str
        Path to the original .npy file (rows, cols, features, times).

    Returns
    -------
    str
        Path to the time-first .npy file.
    """
    p = Path(data_path)
    time_first = p.with_stem(p.stem + "_time_first")
    if time_first.exists():
        print(f"Note: using time-first dataset {time_first} instead of {p.name}")
        return str(time_first)
    print(
        f"ERROR: time-first dataset not found: {time_first}\n"
        f"Please convert your data first by running:\n\n"
        f"  uv run sar_experiments/prepare_data.py {data_path}\n"
    )
    sys.exit(1)


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
