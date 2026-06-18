# Compatibility layer to handle numpy/torch/cupy/jax for computations
# Author: Ammar Mian
# Date: 21/10/2025

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Literal, Union
import numpy as np
import numpy.typing as npt
import torch
from torch.nn import Unfold


# ── Type annotations ──────────────────────────────────────────────────────────

# Runtime-safe aliases: only numpy and torch are guaranteed to be present.
type ArrayLike = npt.ArrayLike | torch.Tensor
type Array = npt.NDArray | torch.Tensor

if TYPE_CHECKING:
    # Extend aliases with optional backend types for static analysis.
    # Wrapped in try/except so type-checkers that have the packages installed
    # get the full union; those that don't fall back to the base aliases.
    try:
        import cupy as _cupy_t
        import jax as _jax_t

        type Array = npt.NDArray | torch.Tensor | _cupy_t.ndarray | _jax_t.Array  # type: ignore[no-redef]
        type ArrayLike = npt.ArrayLike | torch.Tensor | _cupy_t.ndarray | _jax_t.Array  # type: ignore[no-redef]
    except ImportError:
        pass


# ── Backend type registry ─────────────────────────────────────────────────────

BACKEND_TYPES: dict = {"numpy": np.ndarray, "torch": torch.Tensor}
try:
    import cupy as _cupy_reg

    BACKEND_TYPES["cupy"] = _cupy_reg.ndarray
    del _cupy_reg
except ImportError:
    pass
try:
    import jax as _jax_reg

    BACKEND_TYPES["jax"] = _jax_reg.Array
    del _jax_reg
except ImportError:
    pass


# ── Private helpers ───────────────────────────────────────────────────────────


def _is_cupy_array(x) -> bool:
    """True if *x* is a CuPy ndarray (duck-typed; no cupy import needed)."""
    return type(x).__module__.startswith("cupy")


def _is_jax_array(x) -> bool:
    """True if *x* is a JAX array (duck-typed; no jax import needed)."""
    return type(x).__module__.startswith("jax")


def _is_torch_module(m: ModuleType) -> bool:
    """True if *m* is the torch module (i.e. not an array-API-like module)."""
    return m is torch


# ── Backend dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Backend:
    """Hardware/library backend descriptor.

    Decouples *library* (numpy / torch / cupy / jax) from *device*
    (cpu / cuda / mps / metal).

    Parameters
    ----------
    lib : Literal["numpy", "torch", "cupy", "jax"]
        The compute library.
    device : str
        Target device: ``"cpu"``, ``"cuda"``, ``"mps"`` (Apple Silicon /
        torch), or ``"metal"`` (Apple Silicon / jax).  Ignored for numpy
        (always CPU).

    Raises
    ------
    ValueError
        If the lib/device combination is unsupported.
    """

    lib: Literal["numpy", "torch", "cupy", "jax"]
    device: str = "cpu"

    def __post_init__(self):
        valid = {
            "numpy": {"cpu"},
            "torch": {"cpu", "cuda", "mps"},
            "cupy": {"cuda"},
            "jax": {"cpu", "cuda", "metal"},
        }
        if self.lib not in valid:
            raise ValueError(f"Unknown lib {self.lib!r}. Valid choices: {sorted(valid)}")
        if self.device not in valid[self.lib]:
            raise ValueError(
                f"Device {self.device!r} is not valid for lib={self.lib!r}. "
                f"Valid choices: {valid[self.lib]}"
            )
        # Pin the JAX default device at construction time so that all
        # jnp.eye / jnp.ones / jnp.zeros etc. land on the right device
        # immediately, without requiring an explicit get_data_on_device call.
        # On Apple Silicon, JAX defaults to Metal; without this, any jnp call
        # with a complex dtype would crash before get_data_on_device can help.
        if self.lib == "jax":
            try:
                import os
                import jax

                # Prevent XLA from pre-allocating ~75% of GPU memory at init.
                os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
                platform = "gpu" if self.device == "cuda" else self.device
                try:
                    jax.config.update("jax_default_device", jax.devices(platform)[0])
                except RuntimeError as e:
                    available = [d.platform for d in jax.devices()]
                    raise RuntimeError(
                        f"JAX cannot use device '{self.device}' (platform='{platform}'). "
                        f"Available JAX devices: {available}. "
                        f"Install jax[cuda] for GPU support, or use 'jax-cpu' instead."
                    ) from e
            except ImportError:
                pass  # JAX not installed; will fail later when actually used

    # ── Factories ──────────────────────────────────────────────────────────

    @staticmethod
    def numpy() -> "Backend":
        return Backend("numpy", "cpu")

    @staticmethod
    def torch_cpu() -> "Backend":
        return Backend("torch", "cpu")

    @staticmethod
    def torch_cuda() -> "Backend":
        return Backend("torch", "cuda")

    @staticmethod
    def torch_mps() -> "Backend":
        return Backend("torch", "mps")

    @staticmethod
    def cupy_cuda() -> "Backend":
        return Backend("cupy", "cuda")

    @staticmethod
    def jax_cpu() -> "Backend":
        return Backend("jax", "cpu")

    @staticmethod
    def jax_cuda() -> "Backend":
        return Backend("jax", "cuda")

    @staticmethod
    def jax_metal() -> "Backend":
        return Backend("jax", "metal")

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def is_torch(self) -> bool:
        return self.lib == "torch"

    @property
    def is_cupy(self) -> bool:
        return self.lib == "cupy"

    @property
    def is_jax(self) -> bool:
        return self.lib == "jax"

    @property
    def is_cuda(self) -> bool:
        """True for any lib running on a CUDA device."""
        return self.device == "cuda"

    @property
    def is_mps(self) -> bool:
        """True for torch on Apple Silicon GPU."""
        return self.lib == "torch" and self.device == "mps"

    @property
    def is_gpu(self) -> bool:
        """True for any hardware accelerator (cuda, mps, metal)."""
        return self.device in {"cuda", "mps", "metal"}

    @property
    def torch_device(self) -> "torch.device":
        """Corresponding ``torch.device`` — only valid when ``is_torch``."""
        if not self.is_torch:
            raise AttributeError("torch_device is only valid for torch backends")
        return torch.device(self.device)

    @property
    def jax_device(self):
        """Corresponding JAX device object — only valid when ``is_jax``."""
        if not self.is_jax:
            raise AttributeError("jax_device is only valid for jax backends")
        import jax

        # JAX uses "gpu" as the platform name for CUDA devices.
        platform = "gpu" if self.device == "cuda" else self.device
        return jax.devices(platform)[0]

    # ── String round-trip ───────────────────────────────────────────────────

    @classmethod
    def from_str(cls, s: str) -> "Backend":
        """Parse a backend string into a :class:`Backend` object."""
        _map = {
            "numpy": cls.numpy,
            "torch-cpu": cls.torch_cpu,
            "torch-cuda": cls.torch_cuda,
            "torch-mps": cls.torch_mps,
            "cupy": cls.cupy_cuda,
            "cupy-cuda": cls.cupy_cuda,
            "jax-cpu": cls.jax_cpu,
            "jax-cuda": cls.jax_cuda,
            "jax-metal": cls.jax_metal,
        }
        if s not in _map:
            raise ValueError(f"Unknown backend string: {s!r}. Valid choices: {sorted(_map)}")
        return _map[s]()

    def __str__(self) -> str:
        if self.lib == "numpy":
            return "numpy"
        return f"{self.lib}-{self.device}"


# ── Internal normalisation helper ─────────────────────────────────────────────


def _normalize_backend(backend: Union[str, "Backend"]) -> "Backend":
    """Convert string or Backend to Backend object."""
    if isinstance(backend, Backend):
        return backend
    return Backend.from_str(backend)


# ── Module accessor ───────────────────────────────────────────────────────────


def get_backend_module(backend: Union[str, "Backend"]) -> ModuleType:
    """Return the compute module for a given backend.

    Parameters
    ----------
    backend : str or Backend
        Backend specification.

    Returns
    -------
    ModuleType
        ``numpy``, ``torch``, ``cupy``, or ``jax.numpy``.

    Raises
    ------
    ImportError
        If the requested optional library (cupy / jax) is not installed.
    ValueError
        If the backend string is not recognised.
    """
    b = _normalize_backend(backend)
    if b.lib == "numpy":
        return np
    if b.lib == "torch":
        return torch
    if b.lib == "cupy":
        try:
            import cupy

            return cupy
        except ImportError:
            raise ImportError(
                "Backend 'cupy' requires CuPy. Install it with: uv sync --extra cupy"
            ) from None
    if b.lib == "jax":
        try:
            import jax.numpy as jnp

            # Default device already pinned in Backend.__post_init__.
            return jnp
        except ImportError:
            raise ImportError(
                "Backend 'jax' requires JAX. "
                "Install it with: uv sync --extra jax  (CPU/CUDA) or "
                "uv sync --extra jax-metal  (Apple Silicon)"
            ) from None
    raise ValueError(f"backend {backend} not found.")


# ── Data movement ─────────────────────────────────────────────────────────────


def get_data_on_device(data: Array, backend: Union[str, "Backend"]) -> Array:
    """Move *data* to the requested backend and device.

    Handles all pairwise conversions between numpy / torch / cupy / jax,
    including automatic float64→float32 downcasting when the target device
    does not support float64 (torch-mps, jax-metal).

    Parameters
    ----------
    data : Array
        Source array (numpy, torch, cupy, or jax).
    backend : str or Backend
        Target backend.

    Returns
    -------
    Array
        Array on the requested backend/device.
    """
    b = _normalize_backend(backend)

    # ── numpy target ──────────────────────────────────────────────────────
    if b.lib == "numpy":
        return to_numpy(data)

    # ── torch target ──────────────────────────────────────────────────────
    if b.lib == "torch":
        if isinstance(data, torch.Tensor):
            return data.to(device=b.torch_device)
        arr = to_numpy(data)
        tensor = torch.from_numpy(arr)
        # MPS does not support float64; downcast to float32 automatically
        if b.is_mps and tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor.to(device=b.torch_device)

    # ── cupy target ───────────────────────────────────────────────────────
    if b.lib == "cupy":
        try:
            import cupy as cp
        except ImportError:
            raise ImportError(
                "Backend 'cupy' requires CuPy. Install it with: uv sync --extra cupy"
            ) from None
        if _is_cupy_array(data):
            return data  # already a cupy array (single-GPU assumed)
        return cp.asarray(to_numpy(data))

    # ── jax target ────────────────────────────────────────────────────────
    if b.lib == "jax":
        try:
            import jax
        except ImportError:
            raise ImportError(
                "Backend 'jax' requires JAX. "
                "Install it with: uv sync --extra jax or --extra jax-metal"
            ) from None

        if _is_jax_array(data):
            # Already a JAX array — just move to the requested device.
            return jax.device_put(data, b.jax_device)

        # Convert to numpy first so we have a plain buffer to hand to device_put.
        # We deliberately avoid jnp.asarray() here: on Apple Silicon that call
        # stages through the default backend (Metal) even when targeting CPU,
        # which fails for complex dtypes that Metal does not support.
        arr_np = to_numpy(data)

        if b.device == "metal":
            # Metal does not support float64 → downcast
            if arr_np.dtype == np.float64:
                arr_np = arr_np.astype(np.float32)
            # Metal does not support complex numbers at all
            elif np.issubdtype(arr_np.dtype, np.complexfloating):
                raise TypeError(
                    "JAX Metal backend does not support complex-valued arrays "
                    f"(got dtype {arr_np.dtype}). "
                    "Use --backend jax-cpu for complex SAR data."
                )

        # jax.device_put accepts a numpy array directly and places it on the
        # requested device without routing through the default backend.
        return jax.device_put(arr_np, b.jax_device)


# ── Central numpy extraction ──────────────────────────────────────────────────


def to_numpy(x: Array) -> np.ndarray:
    """Extract a plain numpy array from any backend array.

    Parameters
    ----------
    x : Array
        Source array (numpy, torch, cupy, or jax).

    Returns
    -------
    np.ndarray
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if _is_cupy_array(x):
        import cupy as cp

        return cp.asnumpy(x)
    if _is_jax_array(x):
        return np.asarray(x)
    # Fallback: try numpy conversion
    return np.asarray(x)


# ── Random sampling ───────────────────────────────────────────────────────────


def sample_standard_normal(
    n_samples: int,
    data_shape: list[int],
    backend: Union[str, "Backend"],
    seed: Optional[int] = None,
) -> Array:
    """Sample standard normal data on the given backend.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    data_shape : list[int]
        Shape of one data sample.
    backend : str or Backend
        Backend specification.
    seed : int, optional
        Seed for the RNG.  For JAX, ``seed=None`` is equivalent to
        ``seed=0`` (deterministic); two calls with ``seed=None`` will
        return the same array.

    Returns
    -------
    Array
        Sampled data on the requested backend.
    """
    b = _normalize_backend(backend)
    shape = (n_samples,) + tuple(data_shape)

    if b.is_torch:
        if seed is not None:
            gen = torch.Generator(device=b.torch_device)
            gen.manual_seed(seed)
            return torch.randn(*shape, generator=gen, device=b.torch_device)
        return torch.randn(*shape, device=b.torch_device)

    if b.is_cupy:
        import cupy as cp

        rng = cp.random.default_rng(seed)
        return rng.standard_normal(shape)

    if b.is_jax:
        import jax

        key = jax.random.PRNGKey(seed if seed is not None else 0)
        arr = jax.random.normal(key, shape=shape)
        return jax.device_put(arr, b.jax_device)

    # numpy
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


def sample_uniform(
    n_samples: int,
    data_shape: list[int],
    backend: Union[str, "Backend"],
    seed: Optional[int] = None,
    low: float = 0.0,
    high: float = 1.0,
) -> Array:
    """Sample uniform data on the given backend.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    data_shape : list[int]
        Shape of one data sample.
    backend : str or Backend
        Backend specification.
    seed : int, optional
        Seed for the RNG.  For JAX, ``seed=None`` is equivalent to
        ``seed=0`` (deterministic).
    low : float
        Lower bound of uniform distribution.
    high : float
        Upper bound of uniform distribution.

    Returns
    -------
    Array
        Sampled data on the requested backend.
    """
    b = _normalize_backend(backend)
    shape = (n_samples,) + tuple(data_shape)

    if b.is_torch:
        if seed is not None:
            gen = torch.Generator(device=b.torch_device)
            gen.manual_seed(seed)
            return low + (high - low) * torch.rand(*shape, generator=gen, device=b.torch_device)
        return low + (high - low) * torch.rand(*shape, device=b.torch_device)

    if b.is_cupy:
        import cupy as cp

        rng = cp.random.default_rng(seed)
        return rng.uniform(low=low, high=high, size=shape)

    if b.is_jax:
        import jax

        key = jax.random.PRNGKey(seed if seed is not None else 0)
        arr = jax.random.uniform(key, shape=shape, minval=low, maxval=high)
        return jax.device_put(arr, b.jax_device)

    # numpy
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=shape)


# ── Array manipulation helpers ────────────────────────────────────────────────


def expand_dims(backend: Union[str, "Backend"], x: Array, axis: int) -> Array:
    """Add a new axis to an array, works for all backends.

    Parameters
    ----------
    backend : str or Backend
    x : Array
    axis : int

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return x.unsqueeze(dim=axis)
    return bm.expand_dims(x, axis=axis)


def make_writable_copy(backend: Union[str, "Backend"], x: Array) -> Array:
    """Return a writable copy of *x*, works for all backends.

    Parameters
    ----------
    backend : str or Backend
    x : Array

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return x.clone()
    return bm.array(x)


def masked_set(
    x: Array,
    mask: Array,
    values: Array,
    backend: Union[str, "Backend"],
) -> Array:
    """Set ``x[mask] = values`` in a backend-agnostic way.

    JAX arrays are immutable, so ``x[mask] = values`` raises a ``TypeError``
    at runtime.  This helper abstracts the difference:

    * **numpy / torch / cupy** — performs the standard in-place assignment
      ``x[mask] = values`` and returns *x*.
    * **jax** — returns ``x.at[mask].set(values)`` (out-of-place), which must
      be assigned back by the caller (``x = masked_set(x, mask, v, backend)``).

    Parameters
    ----------
    x : Array
        Target array to update.
    mask : Array
        Boolean index array.
    values : Array
        Values to write at the masked positions.
    backend : str or Backend

    Returns
    -------
    Array
        Updated array.  For JAX this is a **new** array; for all other
        backends it is the same object as *x* (mutated in-place).
    """
    b = _normalize_backend(backend)
    if b.lib == "jax":
        return x.at[mask].set(values)
    x[mask] = values
    return x


def permute(backend: Union[str, "Backend"], x: Array, dims: tuple) -> Array:
    """Permute array axes, works for all backends.

    Parameters
    ----------
    backend : str or Backend
    x : Array
    dims : tuple
        New axis order (same semantics as ``np.transpose`` / ``torch.permute``).

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return x.permute(dims)
    return bm.transpose(x, dims)


def concatenate(backend: Union[str, "Backend"], arrays: list[Array], axis: int = 0) -> Array:
    """Concatenate arrays along an axis, works for all backends.

    Parameters
    ----------
    backend : str or Backend
    arrays : list[Array]
    axis : int

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return bm.cat(arrays, dim=axis)
    return bm.concatenate(arrays, axis=axis)


# ── Linear algebra helpers ────────────────────────────────────────────────────


def batched_eigh(
    backend: Union[str, "Backend"], X: Array, max_batch_size: int = 16000
) -> tuple[Array, Array]:
    """Eigenvalue decomposition with automatic batching for large CUDA inputs.

    Device-specific behaviour:

    * **JAX (any device, including Metal)**: uses ``jax.vmap`` over all batch
      dimensions — no manual chunking, no MPS-style fallback needed.
    * **torch-mps**: ``linalg.eigh`` is not implemented on MPS; falls back to
      CPU transparently (2 transfers per call).
    * **CUDA (torch or cupy)**: chunks batches larger than *max_batch_size*
      to stay within cuSOLVER limits.
    * **Everything else**: direct ``linalg.eigh`` call.

    Parameters
    ----------
    backend : str or Backend
    X : Array
        SPD matrices of shape ``(..., n, n)``.
    max_batch_size : int
        Max matrices per CUDA chunk (default 16 000).

    Returns
    -------
    tuple[Array, Array]
        (eigenvalues, eigenvectors)
    """
    b = _normalize_backend(backend)
    bm = get_backend_module(backend)

    # JAX: vmap for efficient batched eigh on any device (Metal supported natively)
    if b.is_jax:
        import jax

        batch_dims = len(X.shape) - 2
        eigh_fn = bm.linalg.eigh
        for _ in range(batch_dims):
            eigh_fn = jax.vmap(eigh_fn)
        return eigh_fn(X)

    # torch-mps: eigh not implemented on MPS; fall back to CPU
    if b.is_mps:
        eigvals, eigvecs = torch.linalg.eigh(X.cpu())
        return eigvals.to(b.torch_device), eigvecs.to(b.torch_device)

    # CUDA (torch or cupy): chunk large batches to avoid cuSOLVER limits
    if b.is_cuda:
        batch_shape = X.shape[:-2]
        n_matrices = int(np.prod(batch_shape))

        if n_matrices > max_batch_size:
            X_flat = X.reshape(-1, X.shape[-2], X.shape[-1])
            eigvals_list, eigvecs_list = [], []
            for i in range(0, X_flat.shape[0], max_batch_size):
                chunk = X_flat[i : i + max_batch_size]
                ev, evec = bm.linalg.eigh(chunk)
                eigvals_list.append(ev)
                eigvecs_list.append(evec)

            if _is_torch_module(bm):
                eigenvalues = bm.cat(eigvals_list, dim=0).reshape(batch_shape + (X.shape[-1],))
                eigenvectors = bm.cat(eigvecs_list, dim=0).reshape(X.shape)
            else:
                eigenvalues = bm.concatenate(eigvals_list, axis=0).reshape(
                    batch_shape + (X.shape[-1],)
                )
                eigenvectors = bm.concatenate(eigvecs_list, axis=0).reshape(X.shape)
            return eigenvalues, eigenvectors

    # Default: numpy / cupy-cpu / torch-cpu
    return bm.linalg.eigh(X)


def get_diagembed(backend: Union[str, "Backend"], x: Array) -> Array:
    """Embed a batch of vectors into diagonal matrices.

    Parameters
    ----------
    backend : str or Backend
    x : Array
        Input of shape ``(..., n)``.

    Returns
    -------
    Array
        Diagonal matrices of shape ``(..., n, n)``.
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return bm.diag_embed(x)
    # numpy / cupy / jax: einsum approach (none have diag_embed)
    eye_matrix = bm.eye(x.shape[-1], dtype=x.dtype)
    target_shape = x.shape + (x.shape[-1],)
    eye_broadcasted = bm.broadcast_to(eye_matrix, target_shape)
    return bm.einsum("...i,...ij->...ij", x, eye_broadcasted)


def batched_trace(backend: Union[str, "Backend"], X: Array) -> Array:
    """Compute trace of batched matrices ``(..., n, n) → (...,)``.

    Parameters
    ----------
    backend : str or Backend
    X : Array

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return bm.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1)
    return bm.diagonal(X, axis1=-2, axis2=-1).sum(axis=-1)


def batched_det(backend: Union[str, "Backend"], X: Array) -> Array:
    """Compute determinant of batched matrices ``(..., n, n) → (...,)``.

    Parameters
    ----------
    backend : str or Backend
    X : Array

    Returns
    -------
    Array
    """
    return get_backend_module(backend).linalg.det(X)


# ── Type/scalar utilities ─────────────────────────────────────────────────────


def is_complex(backend: Union[str, "Backend"], X: Array) -> bool:
    """Return True if *X* has a complex dtype.

    Parameters
    ----------
    backend : str or Backend
    X : Array

    Returns
    -------
    bool
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return torch.is_complex(X) if X.numel() > 0 else False
    # numpy, cupy, jax all expose a numpy-compatible dtype
    return np.issubdtype(X.dtype, np.complexfloating)


def create_scalar_array(value, dtype, backend: Union[str, "Backend"]) -> Array:
    """Create a 0-dimensional array holding *value*.

    Parameters
    ----------
    value : float or bool
    dtype : dtype
    backend : str or Backend

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        result = bm.tensor(value, dtype=dtype)
    else:
        result = bm.array(value, dtype=dtype)
    return get_data_on_device(result, backend)


def cast_like(x: Array, ref: Array, backend: Union[str, "Backend"]) -> Array:
    """Cast *x* to the dtype of *ref*, in a backend-agnostic way.

    Abstracts the ``ndarray.astype()`` (numpy/cupy/jax) vs
    ``Tensor.to(dtype=...)`` (torch) difference so callers need not
    check which API is available.

    Parameters
    ----------
    x : Array
        Array to cast.
    ref : Array
        Reference array whose dtype is used as the target.
    backend : str or Backend

    Returns
    -------
    Array
        *x* with the same dtype as *ref*.
    """
    return to_dtype(x, ref.dtype, backend)


def to_complex(x: Array, backend: Union[str, "Backend"]) -> Array:
    """Promote a real array to its complex counterpart.

    Maps float32 → complex64, float64 → complex128.  If *x* is already
    complex, returns it unchanged.  Useful when real-valued eigenvalues
    need to be cast back to complex before einsum with complex eigenvectors.

    Parameters
    ----------
    x : Array
        Real or complex array.
    backend : str or Backend

    Returns
    -------
    Array
        Complex array with the same element size as *x*.
    """
    bm = get_backend_module(backend)
    # numpy-compatible dtype check works for numpy, cupy, and jax
    if not _is_torch_module(bm):
        if np.issubdtype(x.dtype, np.complexfloating):
            return x
        complex_dtype = np.result_type(x.dtype, np.complex64)
        return x.astype(complex_dtype)
    # torch
    if torch.is_complex(x):
        return x
    complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    return x.to(dtype=complex_dtype)


def to_dtype(X: Array, dtype, backend: Union[str, "Backend"]) -> Array:
    """Convert *X* to *dtype*, handling mixed numpy/torch dtype objects.

    Parameters
    ----------
    X : Array
    dtype : np.dtype or torch.dtype
    backend : str or Backend

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)

    if _is_torch_module(bm):
        # Convert numpy dtypes to torch dtypes if necessary
        if isinstance(dtype, np.dtype):
            _np_to_torch = {
                np.float32: torch.float32,
                np.float64: torch.float64,
                np.float16: torch.float16,
                np.complex64: torch.complex64,
                np.complex128: torch.complex128,
                np.int32: torch.int32,
                np.int64: torch.int64,
            }
            dtype = _np_to_torch.get(dtype, dtype)
        return X.to(dtype=dtype)
    else:
        # Convert torch dtypes to numpy dtypes if necessary
        # (cupy and jax both accept numpy-compatible dtypes)
        if isinstance(dtype, torch.dtype):
            _torch_to_np = {
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.float16: np.float16,
                torch.complex64: np.complex64,
                torch.complex128: np.complex128,
                torch.int32: np.int32,
                torch.int64: np.int64,
            }
            dtype = _torch_to_np.get(dtype, dtype)
        return X.astype(dtype)


def to_scalar(x) -> float:
    """Extract a Python float from a backend scalar.

    Uses ``.item()`` for torch/cupy/jax tensors to avoid unnecessary
    device-to-host synchronisation; falls back to ``float()`` otherwise.
    """
    if isinstance(x, torch.Tensor):
        return x.item()
    if _is_cupy_array(x) or _is_jax_array(x):
        return x.item()
    return float(x)


def dtype_itemsize(dtype) -> int:
    """Return byte size of one element for a numpy or torch dtype.

    Parameters
    ----------
    dtype : np.dtype or torch.dtype

    Returns
    -------
    int
    """
    if isinstance(dtype, torch.dtype):
        return torch.empty(0, dtype=dtype).element_size()
    return np.dtype(dtype).itemsize


def normalize_covariance(
    cov: Array,
    normalization: Optional[str],
    backend: Union[str, "Backend"],
    n_features: int,
) -> Array:
    """Normalize covariance matrices.

    Parameters
    ----------
    cov : Array
        Covariance matrices of shape ``(..., n_features, n_features)``.
    normalization : str or None
        ``None`` / ``"none"`` — no normalization.
        ``"diag"`` — normalize so ``cov[..., 0, 0] == 1``.
        ``"trace"`` — normalize so ``trace(cov) == n_features``.
        ``"det"`` — normalize so ``det(cov) == 1``.
    backend : str or Backend
    n_features : int

    Returns
    -------
    Array
    """
    if normalization is None or normalization == "none":
        return cov

    bm = get_backend_module(backend)

    if normalization == "diag":
        scale = cov[..., 0, 0]
    elif normalization == "trace":
        trace = batched_trace(backend, cov)
        scale = trace / n_features
    elif normalization == "det":
        det = batched_det(backend, cov)
        if _is_torch_module(bm):
            scale = bm.pow(det, 1.0 / n_features)
        else:
            scale = bm.power(det, 1.0 / n_features)
    else:
        raise ValueError(
            f"Unknown normalization method: {normalization!r}. "
            f"Must be one of: None, 'none', 'diag', 'trace', 'det'"
        )

    if _is_torch_module(bm):
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
    else:
        scale_expanded = scale[..., None, None]

    return cov / scale_expanded


# ── Memory / device management helpers ───────────────────────────────────────


def empty_cache(backend: Union[str, "Backend"]) -> None:
    """Release unused GPU memory held by the backend's allocator.

    * **torch-cuda**: calls ``torch.cuda.empty_cache()``.
    * **cupy**: frees all blocks in the default memory pool.
    * **jax**: no-op (JAX manages its own memory).
    * **CPU / numpy**: no-op.

    Parameters
    ----------
    backend : str or Backend
    """
    b = _normalize_backend(backend)
    if b.is_torch and b.is_cuda:
        torch.cuda.empty_cache()
    elif b.is_cupy:
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass


def reset_peak_memory(backend: Union[str, "Backend"]) -> None:
    """Reset peak GPU memory statistics (no-op on non-CUDA / non-CuPy backends).

    Parameters
    ----------
    backend : str or Backend
    """
    b = _normalize_backend(backend)
    if b.is_torch and b.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    # cupy/jax: no direct reset API; just free and move on


def peak_memory_bytes(backend: Union[str, "Backend"]) -> Optional[int]:
    """Return peak GPU memory usage in bytes, or ``None`` if unavailable.

    Parameters
    ----------
    backend : str or Backend

    Returns
    -------
    int or None
    """
    b = _normalize_backend(backend)
    if b.is_torch and b.is_cuda:
        return torch.cuda.max_memory_allocated()
    if b.is_cupy:
        try:
            import cupy

            return cupy.get_default_memory_pool().used_bytes()
        except ImportError:
            pass
    if b.is_jax and b.is_gpu:
        try:
            stats = b.jax_device.memory_stats()
            if stats and "peak_bytes_in_use" in stats:
                return stats["peak_bytes_in_use"]
        except Exception:
            pass
    return None


def oom_errors(backend: Union[str, "Backend"]) -> tuple:
    """Return a tuple of OOM exception types for the given backend.

    Suitable for use in ``except`` clauses:

    .. code-block:: python

        try:
            result = manager.process_all_data()
        except oom_errors(backend):
            handle_oom()

    Returns ``(MemoryError,)`` as a safe fallback for backends that do not
    have a specific OOM type.

    Parameters
    ----------
    backend : str or Backend

    Returns
    -------
    tuple[type, ...]
    """
    b = _normalize_backend(backend)
    errors: list[type] = []
    if b.is_torch:
        errors.append(torch.cuda.OutOfMemoryError)
    if b.is_cupy:
        try:
            from cupy.cuda.memory import OutOfMemoryError as CupyOOM

            errors.append(CupyOOM)
        except ImportError:
            pass
    return tuple(errors) if errors else (MemoryError,)


# ── 2-D sliding-window extractor ─────────────────────────────────────────────


class Unfold2D:
    """Backend-agnostic 2-D sliding-window extractor.

    Extracts local patches from a 4-D array and returns them in a format
    compatible with the detection pipeline.

    Input shape:  ``(n_times, n_channels, height, width)``
    Output shape: ``(n_windows, n_times, kernel_size², n_channels)``

    * **torch**: uses ``torch.nn.Unfold`` (GPU-native).
    * **numpy / cupy**: uses ``sliding_window_view`` (same API; cupy mirrors numpy).
    * **jax**: ``jnp`` lacks ``sliding_window_view``; falls back to the numpy
      path on the host and then calls ``jax.device_put``.  Suitable for v1;
      a native ``jax.lax``-based path can replace it in a follow-up.
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self._torch_unfold = Unfold(kernel_size=kernel_size, stride=stride)

    def __call__(self, data: Array, backend: Union[str, "Backend"]) -> Array:
        b = _normalize_backend(backend)
        if b.is_torch:
            return self._call_torch(data)
        if b.is_jax:
            return self._call_jax(data, b)
        # numpy and cupy share the sliding_window_view API
        return self._call_numpy_like(data, get_backend_module(backend))

    def _call_torch(self, data: torch.Tensor) -> torch.Tensor:
        T, C, _, _ = data.shape
        k = self.kernel_size
        patches = self._torch_unfold(data)  # (T, C*k², L)
        return (
            patches.view(T, C, k * k, -1)  # (T, C, k², L)
            .permute(3, 0, 2, 1)  # (L, T, k², C)
            .contiguous()
        )

    def _call_numpy_like(self, data, bm) -> Array:
        """Shared path for numpy and cupy (both have sliding_window_view)."""
        patches = bm.lib.stride_tricks.sliding_window_view(
            data, (self.kernel_size, self.kernel_size), axis=(2, 3)
        )
        # (T, C, out_h, out_w, k, k)
        if self.stride > 1:
            patches = patches[:, :, :: self.stride, :: self.stride]
        T, C, out_h, out_w, k, _ = patches.shape
        return patches.transpose(2, 3, 0, 4, 5, 1).reshape(out_h * out_w, T, k * k, C).copy()

    def _call_jax(self, data, b: "Backend") -> Array:
        """JAX path: compute on numpy host, then device_put result.

        Uses a numpy host fallback to avoid a dependency on a native
        JAX sliding-window primitive for v1.  A jax.lax-based implementation
        can replace this later.  We use jax.device_put(numpy_array, device)
        directly rather than jnp.asarray() to avoid staging through the
        default JAX backend (which is Metal on Apple Silicon, and Metal does
        not support complex dtypes).
        """
        import jax

        data_np = to_numpy(data)
        patches_np = self._call_numpy_like(data_np, np)
        return jax.device_put(patches_np, b.jax_device)
