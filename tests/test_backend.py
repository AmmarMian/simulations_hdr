# Tests for backend module
# Author: Ammar Mian
# Date: 21/10/2025

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add tests directory so we can import conftest helpers directly
sys.path.insert(0, str(Path(__file__).parent))

from hdrlib.core.backend import (
    Backend,
    get_backend_module,
    get_data_on_device,
    to_numpy,
    sample_standard_normal,
    expand_dims,
    make_writable_copy,
    batched_eigh,
    concatenate,
    get_diagembed,
    batched_trace,
    batched_det,
    normalize_covariance,
)
from conftest import (
    as_numpy_for_compare,
    jax_available,
    cupy_available,
)


# ── Backend dataclass tests ───────────────────────────────────────────────────


class TestBackendDataclass:
    """Tests for the Backend dataclass (device: str refactor + new libs)."""

    def test_numpy_factory(self):
        b = Backend.numpy()
        assert b.lib == "numpy"
        assert b.device == "cpu"

    def test_torch_cpu_factory(self):
        b = Backend.torch_cpu()
        assert b.lib == "torch"
        assert b.device == "cpu"

    def test_torch_cuda_factory(self):
        b = Backend.torch_cuda()
        assert b.lib == "torch"
        assert b.device == "cuda"

    def test_torch_mps_factory(self):
        b = Backend.torch_mps()
        assert b.lib == "torch"
        assert b.device == "mps"

    @pytest.mark.skipif(not cupy_available(), reason="CuPy / CUDA not available")
    def test_cupy_factory(self):
        b = Backend.cupy_cuda()
        assert b.lib == "cupy"
        assert b.device == "cuda"
        assert b.is_cupy
        assert b.is_cuda
        assert b.is_gpu

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_jax_cpu_factory(self):
        b = Backend.jax_cpu()
        assert b.lib == "jax"
        assert b.device == "cpu"
        assert b.is_jax
        assert not b.is_gpu

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_jax_metal_factory(self):
        b = Backend.jax_metal()
        assert b.lib == "jax"
        assert b.device == "metal"
        assert b.is_jax
        assert b.is_gpu

    def test_is_gpu(self):
        assert not Backend.numpy().is_gpu
        assert not Backend.torch_cpu().is_gpu
        assert Backend.torch_cuda().is_gpu
        assert Backend.torch_mps().is_gpu

    def test_is_cuda(self):
        assert not Backend.numpy().is_cuda
        assert Backend.torch_cuda().is_cuda

    def test_torch_device_property(self):
        b = Backend.torch_cpu()
        assert b.torch_device == torch.device("cpu")
        b2 = Backend.torch_cuda()
        assert b2.torch_device == torch.device("cuda")

    def test_torch_device_invalid_lib(self):
        with pytest.raises(AttributeError):
            _ = Backend.numpy().torch_device

    def test_invalid_lib_raises(self):
        with pytest.raises(ValueError):
            Backend("tensorflow", "cpu")

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError):
            Backend("numpy", "cuda")  # numpy must be cpu
        with pytest.raises(ValueError):
            Backend("torch", "metal")  # torch doesn't support metal

    def test_from_str_numpy(self):
        assert Backend.from_str("numpy") == Backend.numpy()

    def test_from_str_torch(self):
        assert Backend.from_str("torch-cpu") == Backend.torch_cpu()
        assert Backend.from_str("torch-cuda") == Backend.torch_cuda()
        assert Backend.from_str("torch-mps") == Backend.torch_mps()

    def test_from_str_cupy(self):
        assert Backend.from_str("cupy") == Backend.cupy_cuda()
        assert Backend.from_str("cupy-cuda") == Backend.cupy_cuda()

    def test_from_str_jax(self):
        assert Backend.from_str("jax-cpu") == Backend.jax_cpu()
        assert Backend.from_str("jax-cuda") == Backend.jax_cuda()
        assert Backend.from_str("jax-metal") == Backend.jax_metal()

    def test_from_str_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown backend string"):
            Backend.from_str("tensorflow")
        with pytest.raises(ValueError, match="Unknown backend string"):
            Backend.from_str("torch")  # bare "torch" is invalid

    def test_str_representation(self):
        assert str(Backend.numpy()) == "numpy"
        assert str(Backend.torch_cpu()) == "torch-cpu"
        assert str(Backend.torch_cuda()) == "torch-cuda"
        assert str(Backend.cupy_cuda()) == "cupy-cuda"
        assert str(Backend.jax_cpu()) == "jax-cpu"
        assert str(Backend.jax_metal()) == "jax-metal"


# ── get_backend_module ────────────────────────────────────────────────────────


class TestGetBackendModule:
    """Tests for get_backend_module function."""

    def test_get_numpy_backend(self):
        backend = get_backend_module("numpy")
        assert backend is np

    def test_get_torch_backend(self):
        backend = get_backend_module("torch-cpu")
        assert backend is torch

    def test_invalid_backend_raises_assertion(self):
        with pytest.raises(ValueError):
            get_backend_module("tensorflow")

    def test_case_sensitive(self):
        with pytest.raises(ValueError):
            get_backend_module("NumPy")

    def test_torch_with_device_suffix(self):
        assert get_backend_module("torch-cpu") is torch
        assert get_backend_module("torch-cuda") is torch
        assert get_backend_module("torch-mps") is torch

    def test_torch_without_device(self):
        with pytest.raises(ValueError):
            get_backend_module("torch")

    @pytest.mark.skipif(not cupy_available(), reason="CuPy / CUDA not available")
    def test_get_cupy_backend(self):
        import cupy

        assert get_backend_module("cupy") is cupy

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_get_jax_backend(self):
        import jax.numpy as jnp

        assert get_backend_module("jax-cpu") is jnp


# ── to_numpy and get_data_on_device ──────────────────────────────────────────


class TestToNumpy:
    """Tests for the to_numpy helper."""

    def test_numpy_passthrough(self, numpy_sample_data):
        result = to_numpy(numpy_sample_data)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, numpy_sample_data)

    def test_torch_cpu(self, torch_sample_data):
        result = to_numpy(torch_sample_data)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, torch_sample_data.numpy())

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_torch_mps(self):
        t = torch.randn(3, 3).to("mps")
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_jax_array(self):
        import jax.numpy as jnp

        x = jnp.array([1.0, 2.0, 3.0])
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not cupy_available(), reason="CuPy not available")
    def test_cupy_array(self):
        import cupy as cp

        x = cp.array([1.0, 2.0, 3.0])
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])


class TestGetDataOnDevice:
    """Tests for get_data_on_device function."""

    def test_numpy_to_numpy(self, numpy_sample_data):
        result = get_data_on_device(numpy_sample_data, "numpy")
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, numpy_sample_data)

    def test_torch_to_numpy(self, torch_sample_data):
        result = get_data_on_device(torch_sample_data, "numpy")
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, torch_sample_data.numpy())

    def test_numpy_to_torch_cpu(self, numpy_sample_data):
        result = get_data_on_device(numpy_sample_data, "torch-cpu")
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.from_numpy(numpy_sample_data))
        assert result.device.type == "cpu"

    def test_torch_to_torch_cpu(self, torch_sample_data):
        result = get_data_on_device(torch_sample_data, "torch-cpu")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_numpy_to_torch_cuda(self, numpy_sample_data):
        result = get_data_on_device(numpy_sample_data, "torch-cuda")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_numpy_to_torch_mps(self, numpy_sample_data):
        result = get_data_on_device(numpy_sample_data, "torch-mps")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "mps"

    def test_invalid_backend_raises_assertion(self, numpy_sample_data):
        with pytest.raises(ValueError):
            get_data_on_device(numpy_sample_data, "tensorflow")

    def test_preserves_shape(self, numpy_sample_data):
        result_numpy = get_data_on_device(numpy_sample_data, "numpy")
        result_torch = get_data_on_device(numpy_sample_data, "torch-cpu")
        assert result_numpy.shape == numpy_sample_data.shape
        assert tuple(result_torch.shape) == numpy_sample_data.shape

    def test_preserves_dtype(self):
        data_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = get_data_on_device(data_float32, "torch-cpu")
        assert result.dtype == torch.float32

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_numpy_to_jax_cpu_roundtrip(self, numpy_sample_data):
        jax_arr = get_data_on_device(numpy_sample_data, "jax-cpu")
        back = to_numpy(jax_arr)
        assert np.allclose(back, numpy_sample_data)

    @pytest.mark.skipif(not cupy_available(), reason="CuPy not available")
    def test_numpy_to_cupy_roundtrip(self, numpy_sample_data):
        import cupy as cp

        cupy_arr = get_data_on_device(numpy_sample_data, "cupy")
        assert isinstance(cupy_arr, cp.ndarray)
        back = to_numpy(cupy_arr)
        assert np.allclose(back, numpy_sample_data)


# ── sample_standard_normal ────────────────────────────────────────────────────


class TestSampleStandardNormal:
    """Tests for sample_standard_normal function."""

    def test_numpy_backend_shape(self, sample_data_shape, random_seed):
        result = sample_standard_normal(5, sample_data_shape, "numpy", random_seed)
        assert result.shape == (5,) + tuple(sample_data_shape)

    def test_torch_cpu_backend_shape(self, sample_data_shape, random_seed):
        result = sample_standard_normal(5, sample_data_shape, "torch-cpu", random_seed)
        assert tuple(result.shape) == (5,) + tuple(sample_data_shape)

    def test_numpy_backend_type(self, sample_data_shape, random_seed):
        result = sample_standard_normal(3, sample_data_shape, "numpy", random_seed)
        assert isinstance(result, np.ndarray)

    def test_torch_backend_type(self, sample_data_shape, random_seed):
        result = sample_standard_normal(3, sample_data_shape, "torch-cpu", random_seed)
        assert isinstance(result, torch.Tensor)

    def test_reproducibility_with_seed(self, sample_data_shape):
        r1 = sample_standard_normal(5, sample_data_shape, "numpy", 42)
        r2 = sample_standard_normal(5, sample_data_shape, "numpy", 42)
        assert np.allclose(r1, r2)

    def test_different_seeds_produce_different_results(self, sample_data_shape):
        r1 = sample_standard_normal(5, sample_data_shape, "numpy", 42)
        r2 = sample_standard_normal(5, sample_data_shape, "numpy", 123)
        assert not np.allclose(r1, r2)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_torch_mps_reproducibility(self, sample_data_shape):
        r1 = sample_standard_normal(5, sample_data_shape, "torch-mps", 42)
        r2 = sample_standard_normal(5, sample_data_shape, "torch-mps", 42)
        assert torch.allclose(r1.cpu(), r2.cpu())

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_jax_cpu_shape(self, sample_data_shape, random_seed):
        result = sample_standard_normal(5, sample_data_shape, "jax-cpu", random_seed)
        assert tuple(result.shape) == (5,) + tuple(sample_data_shape)

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_jax_rng_determinism(self, sample_data_shape):
        """Same seed always yields same values in JAX."""
        r1 = sample_standard_normal(5, sample_data_shape, "jax-cpu", 42)
        r2 = sample_standard_normal(5, sample_data_shape, "jax-cpu", 42)
        assert np.allclose(as_numpy_for_compare(r1), as_numpy_for_compare(r2))

    @pytest.mark.skipif(not jax_available(), reason="JAX not installed")
    def test_jax_no_seed_is_deterministic(self, sample_data_shape):
        """seed=None in JAX behaves like seed=0 (deterministic, documented)."""
        r1 = sample_standard_normal(5, sample_data_shape, "jax-cpu", None)
        r2 = sample_standard_normal(5, sample_data_shape, "jax-cpu", None)
        assert np.allclose(as_numpy_for_compare(r1), as_numpy_for_compare(r2))

    @pytest.mark.skipif(not cupy_available(), reason="CuPy not available")
    def test_cupy_shape_and_reproducibility(self, sample_data_shape):
        r1 = sample_standard_normal(5, sample_data_shape, "cupy", 42)
        r2 = sample_standard_normal(5, sample_data_shape, "cupy", 42)
        assert tuple(r1.shape) == (5,) + tuple(sample_data_shape)
        assert np.allclose(as_numpy_for_compare(r1), as_numpy_for_compare(r2))


# ── expand_dims ───────────────────────────────────────────────────────────────


class TestExpandDims:
    """Tests for expand_dims — uses ALL_BACKEND_PARAMS via backend_name fixture."""

    def test_expand_last_axis(self, backend_name):
        data = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        result = expand_dims(backend_name, data, axis=-1)
        assert tuple(result.shape) == (2, 2, 1)

    def test_expand_first_axis(self, backend_name):
        data = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        result = expand_dims(backend_name, data, axis=0)
        assert tuple(result.shape) == (1, 2, 2)

    def test_expand_middle_axis(self, backend_name):
        data = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        result = expand_dims(backend_name, data, axis=1)
        assert tuple(result.shape) == (2, 1, 2)


# ── make_writable_copy ────────────────────────────────────────────────────────


class TestMakeWritableCopy:
    """Tests for make_writable_copy — uses ALL_BACKEND_PARAMS via backend_name fixture."""

    def test_creates_independent_copy(self, backend_name):
        original = get_data_on_device(np.array([1.0, 2.0, 3.0]), backend_name)
        copy = make_writable_copy(backend_name, original)
        copy_np = as_numpy_for_compare(copy).copy()
        orig_np = as_numpy_for_compare(original).copy()
        assert np.allclose(copy_np, orig_np)

    def test_preserves_values(self, backend_name):
        original = get_data_on_device(np.array([1.0, 2.0, 3.0]), backend_name)
        copy = make_writable_copy(backend_name, original)
        assert np.allclose(as_numpy_for_compare(copy), as_numpy_for_compare(original))


# ── batched_eigh ──────────────────────────────────────────────────────────────


class TestBatchedEigh:
    """Tests for batched_eigh — uses ALL_BACKEND_PARAMS via backend_name fixture."""

    # Reference SPD matrices
    _A = np.array([[2.0, 1.0], [1.0, 2.0]])
    _batch = np.array([[[2.0, 1.0], [1.0, 2.0]], [[3.0, 0.5], [0.5, 3.0]]])

    def test_single_matrix_shape(self, backend_name):
        A = get_data_on_device(self._A, backend_name)
        eigvals, eigvecs = batched_eigh(backend_name, A)
        assert tuple(eigvals.shape) == (2,)
        assert tuple(eigvecs.shape) == (2, 2)

    def test_batched_matrices_shape(self, backend_name):
        batch = get_data_on_device(self._batch, backend_name)
        eigvals, eigvecs = batched_eigh(backend_name, batch)
        assert tuple(eigvals.shape) == (2, 2)
        assert tuple(eigvecs.shape) == (2, 2, 2)

    def test_correctness_vs_numpy(self, backend_name):
        """Eigenvalues must match numpy reference to within float tolerance."""
        ref_vals, _ = np.linalg.eigh(self._A)
        A = get_data_on_device(self._A.astype(np.float64), backend_name)
        eigvals, _ = batched_eigh(backend_name, A)
        assert np.allclose(
            np.sort(as_numpy_for_compare(eigvals).astype(np.float64)),
            np.sort(ref_vals),
            atol=1e-5,
        )


# ── concatenate ───────────────────────────────────────────────────────────────


class TestConcatenate:
    """Tests for concatenate — uses ALL_BACKEND_PARAMS via backend_name fixture."""

    def test_concatenate_along_axis_0(self, backend_name):
        a = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        b = get_data_on_device(np.array([[5.0, 6.0], [7.0, 8.0]]), backend_name)
        result = concatenate(backend_name, [a, b], axis=0)
        assert tuple(result.shape) == (4, 2)
        assert np.allclose(as_numpy_for_compare(result)[:2], [[1, 2], [3, 4]])
        assert np.allclose(as_numpy_for_compare(result)[2:], [[5, 6], [7, 8]])

    def test_concatenate_along_axis_1(self, backend_name):
        a = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        b = get_data_on_device(np.array([[5.0, 6.0], [7.0, 8.0]]), backend_name)
        result = concatenate(backend_name, [a, b], axis=1)
        assert tuple(result.shape) == (2, 4)


# ── get_diagembed ─────────────────────────────────────────────────────────────


class TestGetDiagembed:
    """Tests for get_diagembed — uses ALL_BACKEND_PARAMS via backend_name fixture."""

    def test_single_vector(self, backend_name):
        v = get_data_on_device(np.array([1.0, 2.0, 3.0]), backend_name)
        result = get_diagembed(backend_name, v)
        expected = np.diag([1.0, 2.0, 3.0])
        assert tuple(result.shape) == (3, 3)
        assert np.allclose(as_numpy_for_compare(result), expected, atol=1e-6)

    def test_batched_vectors(self, backend_name):
        v = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        result = get_diagembed(backend_name, v)
        assert tuple(result.shape) == (2, 2, 2)
        out = as_numpy_for_compare(result)
        assert np.isclose(out[0, 0, 0], 1.0)
        assert np.isclose(out[0, 1, 1], 2.0)
        assert np.isclose(out[0, 0, 1], 0.0)


# ── batched_trace ─────────────────────────────────────────────────────────────


class TestBatchedTrace:
    """Tests for batched_trace — uses ALL_BACKEND_PARAMS via backend_name fixture."""

    def test_single_matrix(self, backend_name):
        A = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        result = batched_trace(backend_name, A)
        assert np.isclose(float(as_numpy_for_compare(result)), 5.0)

    def test_batched_matrices(self, backend_name):
        batch = get_data_on_device(
            np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            backend_name,
        )
        result = batched_trace(backend_name, batch)
        assert tuple(result.shape) == (2,)
        assert np.allclose(as_numpy_for_compare(result), [5.0, 13.0])


# ── batched_det ───────────────────────────────────────────────────────────────


class TestBatchedDet:
    """Tests for batched_det — uses ALL_BACKEND_PARAMS via backend_name fixture."""

    def test_single_matrix(self, backend_name):
        A = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)
        result = batched_det(backend_name, A)
        assert np.isclose(float(as_numpy_for_compare(result)), 3.0, atol=1e-5)

    def test_batched_matrices(self, backend_name):
        batch = get_data_on_device(
            np.array([[[2.0, 1.0], [1.0, 2.0]], [[3.0, 0.0], [0.0, 3.0]]]),
            backend_name,
        )
        result = batched_det(backend_name, batch)
        assert tuple(result.shape) == (2,)
        assert np.allclose(as_numpy_for_compare(result), [3.0, 9.0], atol=1e-5)


# ── normalize_covariance ──────────────────────────────────────────────────────


class TestNormalizeCovariance:
    """Tests for normalize_covariance — uses ALL_BACKEND_PARAMS."""

    _cov_np = np.array([[2.0, 1.0], [1.0, 2.0]])

    def test_no_normalization(self, backend_name):
        cov = get_data_on_device(self._cov_np, backend_name)
        result = normalize_covariance(cov, None, backend_name, 2)
        assert np.allclose(as_numpy_for_compare(result), self._cov_np)

    def test_none_string_normalization(self, backend_name):
        cov = get_data_on_device(self._cov_np, backend_name)
        result = normalize_covariance(cov, "none", backend_name, 2)
        assert np.allclose(as_numpy_for_compare(result), self._cov_np)

    def test_diag_normalization(self, backend_name):
        cov = get_data_on_device(self._cov_np, backend_name)
        result = normalize_covariance(cov, "diag", backend_name, 2)
        assert np.isclose(as_numpy_for_compare(result)[0, 0], 1.0)

    def test_trace_normalization(self, backend_name):
        cov = get_data_on_device(self._cov_np, backend_name)
        result = normalize_covariance(cov, "trace", backend_name, 2)
        trace = batched_trace(backend_name, result)
        assert np.isclose(float(as_numpy_for_compare(trace)), 2.0, atol=1e-5)

    def test_det_normalization(self, backend_name):
        cov = get_data_on_device(self._cov_np, backend_name)
        result = normalize_covariance(cov, "det", backend_name, 2)
        det = batched_det(backend_name, result)
        assert np.isclose(float(as_numpy_for_compare(det)), 1.0, atol=1e-5)

    def test_batched_trace_normalization(self, backend_name):
        batch_np = np.array([[[2.0, 1.0], [1.0, 2.0]], [[4.0, 0.0], [0.0, 4.0]]])
        batch = get_data_on_device(batch_np, backend_name)
        result = normalize_covariance(batch, "trace", backend_name, 2)
        traces = batched_trace(backend_name, result)
        assert np.allclose(as_numpy_for_compare(traces), [2.0, 2.0], atol=1e-5)

    def test_invalid_normalization_raises_error(self, backend_name):
        cov = get_data_on_device(self._cov_np, backend_name)
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_covariance(cov, "invalid", backend_name, 2)
