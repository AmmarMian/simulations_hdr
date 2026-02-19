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


def get_backend_module(backend_name: str) -> ModuleType:
    """Return backend module from string.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Can be 'numpy', 'torch', 'torch-cpu', or 'torch-cuda'.
        If device info is included (e.g., 'torch-cuda'), only the base name is used.

    Returns
    -------
    ModuleType
        Module of wanted backend (numpy or torch)
    """
    # Handle both "torch" and "torch-cpu"/"torch-cuda" formats
    backend_basename = backend_name.split("-")[0] if "-" in backend_name else backend_name
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
            return data.detach().cpu().numpy()
        else:
            return torch.from_numpy(data).to(device=device)
    else:
        # Data is already correct type, but may need device transfer for torch
        if backend_basename == "torch" and device is not None:
            return data.to(device=device)
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
        sampled data on desired backend
    """
    rng = np.random.default_rng(seed)
    shape = (n_samples,) + tuple(data_shape)
    return get_data_on_device(rng.standard_normal(shape), backend_name)


def expand_dims(backend_name: str, x: Array, axis: int) -> Array:
    """Add a new axis to an array that works for both numpy and torch.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    x: Array
        input array

    axis: int
        position where new axis is to be inserted

    Returns
    -------
    Array
        array with expanded dimensions
    """
    backend_module = get_backend_module(backend_name)
    if backend_module is np:
        return backend_module.expand_dims(x, axis=axis)
    else:
        return x.unsqueeze(dim=axis)


def make_writable_copy(backend_name: str, x: Array) -> Array:
    """Make a writable copy of an array that works for both numpy and torch.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    x: Array
        input array

    Returns
    -------
    Array
        writable copy of input array
    """
    backend_module = get_backend_module(backend_name)
    if backend_module is np:
        return backend_module.array(x)
    else:
        return x.clone()


def batched_eigh(
    backend_name: str, X: Array, max_batch_size: int = 16000
) -> tuple[Array, Array]:
    """Compute eigenvalue decomposition with automatic chunking for large CUDA batches.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    X: Array
        input SPD matrices of shape (..., n, n)

    max_batch_size: int
        maximum number of matrices to process in one batch for CUDA (default: 16000)

    Returns
    -------
    tuple[Array, Array]
        eigenvalues and eigenvectors
    """
    backend_module = get_backend_module(backend_name)

    # For large batches with CUDA, process in chunks to avoid cusolver limits
    if backend_module is torch and X.is_cuda:
        batch_shape = X.shape[:-2]
        n_matrices = torch.prod(torch.tensor(batch_shape)).item()

        if n_matrices > max_batch_size:
            # Flatten batch dimensions, process in chunks, then reshape
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-2], X.shape[-1])
            eigvals_list = []
            eigvecs_list = []

            for i in range(0, X_flat.shape[0], max_batch_size):
                chunk = X_flat[i : i + max_batch_size]
                eigvals, eigvecs = backend_module.linalg.eigh(chunk)
                eigvals_list.append(eigvals)
                eigvecs_list.append(eigvecs)

            eigenvalues = backend_module.cat(eigvals_list, dim=0).reshape(
                batch_shape + (X.shape[-1],)
            )
            eigenvectors = backend_module.cat(eigvecs_list, dim=0).reshape(
                original_shape
            )
            return eigenvalues, eigenvectors

    # Normal path for small batches or numpy
    return backend_module.linalg.eigh(X)


def concatenate(backend_name: str, arrays: list[Array], axis: int = 0) -> Array:
    """Concatenate arrays along an axis, works for both numpy and torch.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    arrays: list[Array]
        list of arrays to concatenate

    axis: int
        axis along which to concatenate (default: 0)

    Returns
    -------
    Array
        concatenated array
    """
    backend_module = get_backend_module(backend_name)
    if backend_module is torch:
        return backend_module.cat(arrays, dim=axis)
    else:
        return backend_module.concatenate(arrays, axis=axis)


def get_diagembed(backend_name: str, x: Array) -> Array:
    """Get diagonal matrices out of a batch of vectors.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    x: Array
        input data of shape (..., n)

    Returns
    -------
    Array
        output data of shape (..., n, n) with diagonal embedding of last dimension of x.
    """
    backend_module = get_backend_module(backend_name)
    if backend_module is np:
        eye_matrix = backend_module.eye(x.shape[-1])
        target_shape = x.shape + (x.shape[-1],)
        eye_broadcasted = backend_module.broadcast_to(eye_matrix, target_shape)
        return backend_module.einsum("...i,...ij->...ij", x, eye_broadcasted)
    else:
        # For torch, diag_embed preserves device automatically
        return backend_module.diag_embed(x)


def batched_trace(backend_name: str, X: Array) -> Array:
    """Compute trace of batched matrices.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    X: Array
        input matrices of shape (..., n, n)

    Returns
    -------
    Array
        traces of shape (...,)
    """
    backend_module = get_backend_module(backend_name)
    if backend_module is np:
        return backend_module.diagonal(X, axis1=-2, axis2=-1).sum(axis=-1)
    else:
        return backend_module.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1)


def batched_det(backend_name: str, X: Array) -> Array:
    """Compute determinant of batched matrices.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    X: Array
        input matrices of shape (..., n, n)

    Returns
    -------
    Array
        determinants of shape (...,)
    """
    backend_module = get_backend_module(backend_name)
    return backend_module.linalg.det(X)


def create_scalar_array(value, dtype, backend_name: str) -> Array:
    """Create a 0-dimensional array with a single value.

    Parameters
    ----------
    value : float or bool
        The scalar value to wrap in an array
    dtype : dtype
        Data type for the array
    backend_name : str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    Returns
    -------
    Array
        0-dimensional array containing the value
    """
    backend_module = get_backend_module(backend_name)
    if backend_module is np:
        result = backend_module.array(value, dtype=dtype)
    else:
        result = backend_module.tensor(value, dtype=dtype)
    return get_data_on_device(result, backend_name)


def normalize_covariance(
    cov: Array, normalization: Optional[str], backend_name: str, n_features: int
) -> Array:
    """Normalize covariance matrices according to specified method.

    Parameters
    ----------
    cov: Array
        input covariance matrices of shape (..., n_features, n_features)

    normalization: str or None
        normalization method:
        - None or 'none': no normalization
        - 'diag': normalize so cov[..., 0, 0] = 1
        - 'trace': normalize so trace(cov) = n_features
        - 'det': normalize so det(cov) = 1

    backend_name: str
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    n_features: int
        number of features (dimension of covariance matrix)

    Returns
    -------
    Array
        normalized covariance matrices of shape (..., n_features, n_features)
    """
    if normalization is None or normalization == 'none':
        return cov

    backend_module = get_backend_module(backend_name)

    if normalization == 'diag':
        # Normalize so first diagonal element = 1
        scale = cov[..., 0, 0]
    elif normalization == 'trace':
        # Normalize so trace = n_features
        trace = batched_trace(backend_name, cov)
        scale = trace / n_features
    elif normalization == 'det':
        # Normalize so det = 1
        det = batched_det(backend_name, cov)
        if backend_module is np:
            scale = backend_module.power(det, 1.0 / n_features)
        else:
            scale = backend_module.pow(det, 1.0 / n_features)
    else:
        raise ValueError(
            f"Unknown normalization method: {normalization}. "
            f"Must be one of: None, 'none', 'diag', 'trace', 'det'"
        )

    # Expand scale dimensions for broadcasting: (...,) -> (..., 1, 1)
    if backend_module is np:
        scale_expanded = scale[..., None, None]
    else:
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)

    return cov / scale_expanded
