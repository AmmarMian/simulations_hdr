# Compatibility layer to handle numpy/torch for compuatations
# Author: Ammar Mian
# Date: 21/10/2025

from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Literal, Union
import numpy as np
import numpy.typing as npt
import torch
from torch.nn import Unfold


# Common types for type annotations
type ArrayLike = npt.ArrayLike | torch.Tensor
type Array = npt.NDArray | torch.Tensor


# Backend types
BACKEND_TYPES = {"numpy": np.ndarray, "torch": torch.Tensor}


# Backend class
@dataclass(frozen=True)
class Backend:
    """Hardware/library backend descriptor.

    Decouples library (numpy vs torch) from device (cpu vs cuda).

    Parameters
    ----------
    lib : Literal["numpy", "torch"]
        The library backend
    device : torch.device
        The device (only used if lib="torch")

    Raises
    ------
    ValueError
        If lib="numpy" and device is not CPU
    """

    lib: Literal["numpy", "torch"]
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        if self.lib == "numpy" and self.device.type != "cpu":
            raise ValueError("numpy backend only supports CPU")

    @staticmethod
    def numpy() -> "Backend":
        return Backend("numpy")

    @staticmethod
    def torch_cpu() -> "Backend":
        return Backend("torch")

    @staticmethod
    def torch_cuda() -> "Backend":
        return Backend("torch", torch.device("cuda"))

    @property
    def is_torch(self) -> bool:
        return self.lib == "torch"

    @property
    def is_cuda(self) -> bool:
        return self.lib == "torch" and self.device.type == "cuda"

    @classmethod
    def from_str(cls, s: str) -> "Backend":
        """Parse legacy string format for backward compatibility."""
        if s == "numpy":
            return cls.numpy()
        elif s == "torch-cpu":
            return cls.torch_cpu()
        elif s == "torch-cuda":
            return cls.torch_cuda()
        raise ValueError(f"Unknown backend string: {s!r}")

    def __str__(self) -> str:
        if self.lib == "numpy":
            return "numpy"
        return f"torch-{self.device.type}"


def _normalize_backend(backend: Union[str, Backend]) -> Backend:
    """Convert string or Backend to Backend object."""
    if isinstance(backend, Backend):
        return backend
    return Backend.from_str(backend)


def get_backend_module(backend: Union[str, Backend]) -> ModuleType:
    """Return backend module from string or Backend.

    Parameters
    ----------
    backend: str or Backend
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object.

    Returns
    -------
    ModuleType
        Module of wanted backend (numpy or torch)
    """
    b = _normalize_backend(backend)
    return np if b.lib == "numpy" else torch


def get_data_on_device(data: Array, backend: Union[str, Backend]) -> Array:
    """Get data on desired backend by converting when needed and loading
    into torch device as needed.

    Parameters
    -----------
    data: Array
        The data to load onto device.

    backend: str or Backend
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object.

    Returns
    -------
    Array
        data on desired backend
    """
    b = _normalize_backend(backend)

    if b.lib == "numpy":
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    else:  # torch
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=b.device)
        else:
            return data.to(device=b.device)


def sample_standard_normal(
    n_samples: int,
    data_shape: list[int],
    backend: Union[str, Backend],
    seed: Optional[int] = None,
) -> Array:
    """Sample standard normal data on the given backend.

    Parameters
    ----------
    n_samples: int
        number of samples to draw.

    data_shape: list[int]
        shape of one data sample

    backend: str or Backend
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object.

    seed: int
        seed for rng. By default None.

    Returns
    -------
    Array
        sampled data on desired backend
    """
    b = _normalize_backend(backend)
    shape = (n_samples,) + tuple(data_shape)
    if b.is_torch:
        gen = torch.Generator(device=b.device)
        if seed is not None:
            gen.manual_seed(seed)
        return torch.randn(*shape, generator=gen, device=b.device)
    else:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape)


def sample_uniform(
    n_samples: int,
    data_shape: list[int],
    backend: Union[str, Backend],
    seed: Optional[int] = None,
    low: float = 0.0,
    high: float = 1.0,
) -> Array:
    """Sample uniform data on the given backend.

    Parameters
    ----------
    n_samples: int
        number of samples to draw.

    data_shape: list[int]
        shape of one data sample

    backend: str or Backend
        Backend specification.

    seed: int
        seed for rng. By default None.

    low: float
        lower bound of uniform distribution

    high: float
        upper bound of uniform distribution

    Returns
    -------
    Array
        sampled data on desired backend
    """
    b = _normalize_backend(backend)
    shape = (n_samples,) + tuple(data_shape)

    if b.is_torch:
        gen = torch.Generator(device=b.device)
        if seed is not None:
            gen.manual_seed(seed)
        # torch.rand gives [0, 1), so rescale
        return low + (high - low) * torch.rand(*shape, generator=gen, device=b.device)
    else:
        rng = np.random.default_rng(seed)
        return rng.uniform(low=low, high=high, size=shape)


def expand_dims(backend: Union[str, Backend], x: Array, axis: int) -> Array:
    """Add a new axis to an array that works for both numpy and torch.

    Parameters
    ----------
    backend: str or Backend
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object.

    x: Array
        input array

    axis: int
        position where new axis is to be inserted

    Returns
    -------
    Array
        array with expanded dimensions
    """
    backend_module = get_backend_module(backend)
    if backend_module is np:
        return backend_module.expand_dims(x, axis=axis)
    else:
        return x.unsqueeze(dim=axis)


def make_writable_copy(backend: Union[str, Backend], x: Array) -> Array:
    """Make a writable copy of an array that works for both numpy and torch.

    Parameters
    ----------
    backend: Union[str, Backend]
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    x: Array
        input array

    Returns
    -------
    Array
        writable copy of input array
    """
    backend_module = get_backend_module(backend)
    if backend_module is np:
        return backend_module.array(x)
    else:
        return x.clone()


def batched_eigh(
    backend: Union[str, Backend], X: Array, max_batch_size: int = 16000
) -> tuple[Array, Array]:
    """Compute eigenvalue decomposition with automatic chunking for large CUDA batches.

    Parameters
    ----------
    backend: Union[str, Backend]
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
    backend_module = get_backend_module(backend)

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


def concatenate(
    backend: Union[str, Backend], arrays: list[Array], axis: int = 0
) -> Array:
    """Concatenate arrays along an axis, works for both numpy and torch.

    Parameters
    ----------
    backend: Union[str, Backend]
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
    backend_module = get_backend_module(backend)
    if backend_module is torch:
        return backend_module.cat(arrays, dim=axis)
    else:
        return backend_module.concatenate(arrays, axis=axis)


def get_diagembed(backend: Union[str, Backend], x: Array) -> Array:
    """Get diagonal matrices out of a batch of vectors.

    Parameters
    ----------
    backend: Union[str, Backend]
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    x: Array
        input data of shape (..., n)

    Returns
    -------
    Array
        output data of shape (..., n, n) with diagonal embedding of last dimension of x.
    """
    backend_module = get_backend_module(backend)
    if backend_module is np:
        eye_matrix = backend_module.eye(x.shape[-1])
        target_shape = x.shape + (x.shape[-1],)
        eye_broadcasted = backend_module.broadcast_to(eye_matrix, target_shape)
        return backend_module.einsum("...i,...ij->...ij", x, eye_broadcasted)
    else:
        # For torch, diag_embed preserves device automatically
        return backend_module.diag_embed(x)


def batched_trace(backend: Union[str, Backend], X: Array) -> Array:
    """Compute trace of batched matrices.

    Parameters
    ----------
    backend: Union[str, Backend]
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    X: Array
        input matrices of shape (..., n, n)

    Returns
    -------
    Array
        traces of shape (...,)
    """
    backend_module = get_backend_module(backend)
    if backend_module is np:
        return backend_module.diagonal(X, axis1=-2, axis2=-1).sum(axis=-1)
    else:
        return backend_module.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1)


def batched_det(backend: Union[str, Backend], X: Array) -> Array:
    """Compute determinant of batched matrices.

    Parameters
    ----------
    backend: Union[str, Backend]
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    X: Array
        input matrices of shape (..., n, n)

    Returns
    -------
    Array
        determinants of shape (...,)
    """
    backend_module = get_backend_module(backend)
    return backend_module.linalg.det(X)


def is_complex(backend: Union[str, Backend], X: Array) -> bool:
    """Check whether an array contains complex-valued data.

    Parameters
    ----------
    backend: Union[str, Backend]
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    X: Array
        input array

    Returns
    -------
    bool
        True if the array has a complex dtype, False otherwise
    """
    backend_module = get_backend_module(backend)
    if backend_module is np:
        return np.iscomplex(X).all()
    else:
        return torch.is_complex(X.flatten()[0]) if X.numel() > 0 else False


def create_scalar_array(value, dtype, backend: Union[str, Backend]) -> Array:
    """Create a 0-dimensional array with a single value.

    Parameters
    ----------
    value : float or bool
        The scalar value to wrap in an array
    dtype : dtype
        Data type for the array
    backend : str or Backend
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object.

    Returns
    -------
    Array
        0-dimensional array containing the value
    """
    backend_module = get_backend_module(backend)
    if backend_module is np:
        result = backend_module.array(value, dtype=dtype)
    else:
        result = backend_module.tensor(value, dtype=dtype)
    return get_data_on_device(result, backend)


def to_dtype(X: Array, dtype, backend: Union[str, Backend]) -> Array:
    """Convert array to target dtype, handling both numpy and torch dtypes.

    Parameters
    ----------
    X: Array
        input array

    dtype : np.dtype or torch.dtype
        Target data type. Can be a numpy dtype (e.g., np.float32) or
        torch dtype (e.g., torch.float32). Function handles conversion
        automatically based on the backend.

    backend: Union[str, Backend]
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    Returns
    -------
    Array
        array converted to target dtype on appropriate backend
    """
    backend_module = get_backend_module(backend)

    if backend_module is np:
        # For numpy, convert torch dtypes to numpy dtypes if needed
        if isinstance(dtype, torch.dtype):
            dtype_map = {
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.float16: np.float16,
                torch.complex64: np.complex64,
                torch.complex128: np.complex128,
                torch.int32: np.int32,
                torch.int64: np.int64,
            }
            dtype = dtype_map.get(dtype, dtype)
        return X.astype(dtype)
    else:
        # For torch, convert numpy dtypes to torch dtypes if needed
        if isinstance(dtype, np.dtype):
            dtype_map = {
                np.float32: torch.float32,
                np.float64: torch.float64,
                np.float16: torch.float16,
                np.complex64: torch.complex64,
                np.complex128: torch.complex128,
                np.int32: torch.int32,
                np.int64: torch.int64,
            }
            dtype = dtype_map.get(dtype, dtype)
        return X.to(dtype=dtype)


def to_scalar(x) -> float:
    """Extract a Python float from a backend scalar (numpy or torch).

    Uses .item() for torch tensors to avoid redundant device-to-host
    synchronisation; falls back to float() for numpy scalars/arrays.
    """
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


def dtype_itemsize(dtype) -> int:
    """Return the size in bytes of one element for a numpy or torch dtype.

    Parameters
    ----------
    dtype : np.dtype or torch.dtype
        Data type to query.

    Returns
    -------
    int
        Number of bytes per element.
    """
    if isinstance(dtype, torch.dtype):
        return torch.empty(0, dtype=dtype).element_size()
    return np.dtype(dtype).itemsize


def normalize_covariance(
    cov: Array,
    normalization: Optional[str],
    backend: Union[str, Backend],
    n_features: int,
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

    backend: Union[str, Backend]
        Name of the backend. Choices are: numpy, torch-cpu, torch-cuda

    n_features: int
        number of features (dimension of covariance matrix)

    Returns
    -------
    Array
        normalized covariance matrices of shape (..., n_features, n_features)
    """
    if normalization is None or normalization == "none":
        return cov

    backend_module = get_backend_module(backend)

    if normalization == "diag":
        # Normalize so first diagonal element = 1
        scale = cov[..., 0, 0]
    elif normalization == "trace":
        # Normalize so trace = n_features
        trace = batched_trace(backend, cov)
        scale = trace / n_features
    elif normalization == "det":
        # Normalize so det = 1
        det = batched_det(backend, cov)
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


class Unfold2D:
    """Backend-agnostic 2D sliding window extractor.

    Extracts local patches from a 4D array and returns them in a format
    compatible with the detection pipeline.

    Input shape:  (n_times, n_channels, height, width)
    Output shape: (n_windows, n_times, kernel_size², n_channels)

    Uses torch.nn.Unfold for torch backends and numpy.lib.stride_tricks
    for numpy.
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self._torch_unfold = Unfold(kernel_size=kernel_size, stride=stride)

    def __call__(self, data: Array, backend: Union[str, Backend]) -> Array:
        b = _normalize_backend(backend)
        if b.is_torch:
            return self._call_torch(data)
        return self._call_numpy(data)

    def _call_torch(self, data: torch.Tensor) -> torch.Tensor:
        T, C, _, _ = data.shape
        k = self.kernel_size
        patches = self._torch_unfold(data)  # (T, C*k², L)
        return (
            patches.view(T, C, k * k, -1)  # (T, C, k², L)
            .permute(3, 0, 2, 1)  # (L, T, k², C)
            .contiguous()
        )

    def _call_numpy(self, data: np.ndarray) -> np.ndarray:
        patches = np.lib.stride_tricks.sliding_window_view(
            data, (self.kernel_size, self.kernel_size), axis=(2, 3)
        )
        # (T, C, out_h, out_w, k, k)
        if self.stride > 1:
            patches = patches[:, :, :: self.stride, :: self.stride]
        T, C, out_h, out_w, k, _ = patches.shape
        return (
            patches.transpose(2, 3, 0, 4, 5, 1)
            .reshape(out_h * out_w, T, k * k, C)
            .copy()
        )
