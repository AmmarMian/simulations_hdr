# Compatibility layer to handle numpy/torch for compuatations
# Author: Ammar Mian
# Date: 21/10/2025

from types import ModuleType
from typing import Optional
import numpy as np
import numpy.typing as npt
import torch


# Common types for type annotations
type ArrayLike = npt.ArrayLike | torch.Tensor
type Array = npt.NDArray | torch.Tensor


# Backend types
BACKEND_TYPES = {"numpy": np.ndarray, "torch": torch.Tensor}


def get_backend_module(backend_basename: str) -> ModuleType:
    """Return backend module from string.

    Parameters
    ----------
    backend_basename: str
        Name of the backend. Choices are : numpy, torch

    Returns
    -------
    ModuleType
        Module of wanted backend
    """
    assert backend_basename in ["numpy", "torch"], (
        f"Backend basename {backend_basename} unknown."
    )
    if backend_basename == "numpy":
        return np
    else:
        return torch


def get_data_on_device(data: Array, backend_name: str) -> Array:
    """Get data on desired backend by converting when need and also
    loading into torch device as needed.

    Parameters
    -----------
    data: Array
        The data to load onto device.

    backend_name: str
        Name of the backend. Choices are : numpy, torch-cpu, torch-cuda

    Returns
    -------
    Array
        data on desired backend
    """
    assert backend_name in ["numpy", "torch-cpu", "torch-cuda"], (
        f"Backend name {backend_name} unknown."
    )

    # Process device if needed
    if "torch" in backend_name:
        backend_basename, device = backend_name.split("-")
        if device == "cuda":
            assert torch.cuda.is_available(), "Device cuda is not available"

    else:
        backend_basename = "numpy"
        device = None

    # Treat data conversion
    if not isinstance(data, BACKEND_TYPES[backend_basename]):
        if backend_basename == "numpy" and isinstance(data, torch.Tensor):
            return data.detach().numpy()
        else:
            return torch.from_numpy(data).to(device=device)
    else:
        return data


def sample_standard_normal(
    n_samples: int, data_shape: list[int], backend_name: str, seed: Optional[int] = None
) -> Array:
    """Sample normal data given backend. Always use numpy for easier behavior management.

    Parameters
    ----------
    n_trials: int
        number of trials to do.

    data_shape: list[int]
        shape of one data sample

    backend_name: str
        Name of the backend. Choices are : numpy, torch-cpu, torch-cuda

    seed: int
        seed for rng. By default None.


    Returns
    -------
    Array
        values of test statstic
    """
    rng = np.random.default_rng(seed)
    shape = (n_samples,) + tuple(data_shape)
    return get_data_on_device(rng.standard_normal(shape), backend_name)
