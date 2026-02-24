# Tests for backend module
# Author: Ammar Mian
# Date: 21/10/2025

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend import (
    get_backend_module,
    get_data_on_device,
    sample_standard_normal,
    expand_dims,
    make_writable_copy,
    batched_eigh,
    concatenate,
    get_diagembed,
    batched_trace,
    batched_det,
    normalize_covariance,
    BACKEND_TYPES,
)


class TestGetBackendModule:
    """Tests for get_backend_module function."""

    def test_get_numpy_backend(self):
        """Test getting numpy backend module."""
        backend = get_backend_module("numpy")
        assert backend is np

    def test_get_torch_backend(self):
        """Test getting torch backend module."""
        backend = get_backend_module("torch-cpu")
        assert backend is torch

    def test_invalid_backend_raises_assertion(self):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError):
            get_backend_module("invalid")

    def test_case_sensitive(self):
        """Test that backend names are case sensitive."""
        with pytest.raises(ValueError):
            get_backend_module("NumPy")

    def test_torch_with_device_suffix(self):
        """Test that torch-cpu and torch-cuda return torch module."""
        backend_cpu = get_backend_module("torch-cpu")
        assert backend_cpu is torch

        backend_cuda = get_backend_module("torch-cuda")
        assert backend_cuda is torch

    def test_torch_without_device(self):
        """Test that plain 'torch' (without device suffix) now raises ValueError."""
        with pytest.raises(ValueError):
            get_backend_module("torch")


class TestGetDataOnDevice:
    """Tests for get_data_on_device function."""

    def test_numpy_to_numpy(self, numpy_sample_data):
        """Test numpy array stays as numpy."""
        result = get_data_on_device(numpy_sample_data, "numpy")
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, numpy_sample_data)

    def test_torch_to_numpy(self, torch_sample_data):
        """Test converting torch tensor to numpy."""
        result = get_data_on_device(torch_sample_data, "numpy")
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, torch_sample_data.numpy())

    def test_numpy_to_torch_cpu(self, numpy_sample_data):
        """Test converting numpy array to torch CPU tensor."""
        result = get_data_on_device(numpy_sample_data, "torch-cpu")
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.from_numpy(numpy_sample_data))
        assert result.device.type == "cpu"

    def test_torch_to_torch_cpu(self, torch_sample_data):
        """Test torch tensor stays as torch on CPU."""
        result = get_data_on_device(torch_sample_data, "torch-cpu")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_numpy_to_torch_cuda(self, numpy_sample_data):
        """Test converting numpy array to torch CUDA tensor."""
        result = get_data_on_device(numpy_sample_data, "torch-cuda")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"

    def test_invalid_backend_raises_assertion(self, numpy_sample_data):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError):
            get_data_on_device(numpy_sample_data, "invalid")

    def test_preserves_shape(self, numpy_sample_data):
        """Test that conversion preserves data shape."""
        result_numpy = get_data_on_device(numpy_sample_data, "numpy")
        result_torch = get_data_on_device(numpy_sample_data, "torch-cpu")
        assert result_numpy.shape == numpy_sample_data.shape
        assert tuple(result_torch.shape) == numpy_sample_data.shape

    def test_preserves_dtype(self):
        """Test that conversion preserves data type."""
        data_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = get_data_on_device(data_float32, "torch-cpu")
        assert result.dtype == torch.float32


class TestSampleStandardNormal:
    """Tests for sample_standard_normal function."""

    def test_numpy_backend_shape(self, sample_data_shape, random_seed):
        """Test sampling with numpy backend produces correct shape."""
        result = sample_standard_normal(5, sample_data_shape, "numpy", random_seed)
        expected_shape = (5,) + tuple(sample_data_shape)
        assert result.shape == expected_shape

    def test_torch_cpu_backend_shape(self, sample_data_shape, random_seed):
        """Test sampling with torch-cpu backend produces correct shape."""
        result = sample_standard_normal(5, sample_data_shape, "torch-cpu", random_seed)
        expected_shape = (5,) + tuple(sample_data_shape)
        assert tuple(result.shape) == expected_shape

    def test_numpy_backend_type(self, sample_data_shape, random_seed):
        """Test sampling with numpy backend produces numpy array."""
        result = sample_standard_normal(3, sample_data_shape, "numpy", random_seed)
        assert isinstance(result, np.ndarray)

    def test_torch_backend_type(self, sample_data_shape, random_seed):
        """Test sampling with torch backend produces torch tensor."""
        result = sample_standard_normal(3, sample_data_shape, "torch-cpu", random_seed)
        assert isinstance(result, torch.Tensor)

    def test_reproducibility_with_seed(self, sample_data_shape):
        """Test that same seed produces same results."""
        result1 = sample_standard_normal(5, sample_data_shape, "numpy", 42)
        result2 = sample_standard_normal(5, sample_data_shape, "numpy", 42)
        assert np.allclose(result1, result2)

    def test_different_seeds_produce_different_results(self, sample_data_shape):
        """Test that different seeds produce different results."""
        result1 = sample_standard_normal(5, sample_data_shape, "numpy", 42)
        result2 = sample_standard_normal(5, sample_data_shape, "numpy", 123)
        assert not np.allclose(result1, result2)

    def test_single_sample(self, sample_data_shape, random_seed):
        """Test sampling a single sample."""
        result = sample_standard_normal(1, sample_data_shape, "numpy", random_seed)
        expected_shape = (1,) + tuple(sample_data_shape)
        assert result.shape == expected_shape

    def test_empty_data_shape(self, random_seed):
        """Test sampling with empty data shape."""
        result = sample_standard_normal(5, [], "numpy", random_seed)
        assert result.shape == (5,)

    def test_statistical_properties(self, sample_data_shape):
        """Test that samples have approximately standard normal distribution."""
        n_samples = 1000
        result = sample_standard_normal(n_samples, sample_data_shape, "numpy", 42)

        # Check mean is close to 0
        assert np.abs(np.mean(result)) < 0.1

        # Check std is close to 1
        assert np.abs(np.std(result) - 1.0) < 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_cuda_backend(self, sample_data_shape, random_seed):
        """Test sampling with torch-cuda backend."""
        result = sample_standard_normal(3, sample_data_shape, "torch-cuda", random_seed)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"


class TestExpandDims:
    """Tests for expand_dims function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_expand_last_axis(self, backend_name):
        """Test expanding the last axis."""
        data = get_data_on_device(np.array([[1, 2], [3, 4]]), backend_name)
        result = expand_dims(backend_name, data, axis=-1)

        if backend_name == "numpy":
            assert result.shape == (2, 2, 1)
        else:
            assert tuple(result.shape) == (2, 2, 1)

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_expand_first_axis(self, backend_name):
        """Test expanding the first axis."""
        data = get_data_on_device(np.array([[1, 2], [3, 4]]), backend_name)
        result = expand_dims(backend_name, data, axis=0)

        if backend_name == "numpy":
            assert result.shape == (1, 2, 2)
        else:
            assert tuple(result.shape) == (1, 2, 2)

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_expand_middle_axis(self, backend_name):
        """Test expanding a middle axis."""
        data = get_data_on_device(np.array([[1, 2], [3, 4]]), backend_name)
        result = expand_dims(backend_name, data, axis=1)

        if backend_name == "numpy":
            assert result.shape == (2, 1, 2)
        else:
            assert tuple(result.shape) == (2, 1, 2)


class TestMakeWritableCopy:
    """Tests for make_writable_copy function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_creates_copy(self, backend_name):
        """Test that the function creates a copy."""
        original = get_data_on_device(np.array([1.0, 2.0, 3.0]), backend_name)
        copy = make_writable_copy(backend_name, original)

        # Modify the copy
        if backend_name == "numpy":
            copy[0] = 99.0
            assert original[0] != 99.0  # Original unchanged
        else:
            copy[0] = 99.0
            assert original[0].item() != 99.0  # Original unchanged

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_preserves_values(self, backend_name):
        """Test that values are preserved."""
        original = get_data_on_device(np.array([1.0, 2.0, 3.0]), backend_name)
        copy = make_writable_copy(backend_name, original)

        if backend_name == "numpy":
            assert np.allclose(copy, original)
        else:
            assert torch.allclose(copy, original)


class TestBatchedEigh:
    """Tests for batched_eigh function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_single_matrix(self, backend_name):
        """Test eigenvalue decomposition of single matrix."""
        # Create a simple symmetric positive definite matrix
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        A_device = get_data_on_device(A, backend_name)

        eigvals, eigvecs = batched_eigh(backend_name, A_device)

        # Check shapes
        if backend_name == "numpy":
            assert eigvals.shape == (2,)
            assert eigvecs.shape == (2, 2)
        else:
            assert tuple(eigvals.shape) == (2,)
            assert tuple(eigvecs.shape) == (2, 2)

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_batched_matrices(self, backend_name):
        """Test eigenvalue decomposition of batched matrices."""
        # Create batch of symmetric matrices
        batch = np.array([[[2.0, 1.0], [1.0, 2.0]], [[3.0, 0.5], [0.5, 3.0]]])
        batch_device = get_data_on_device(batch, backend_name)

        eigvals, eigvecs = batched_eigh(backend_name, batch_device)

        # Check shapes
        if backend_name == "numpy":
            assert eigvals.shape == (2, 2)
            assert eigvecs.shape == (2, 2, 2)
        else:
            assert tuple(eigvals.shape) == (2, 2)
            assert tuple(eigvecs.shape) == (2, 2, 2)


class TestConcatenate:
    """Tests for concatenate function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_concatenate_along_axis_0(self, backend_name):
        """Test concatenation along axis 0."""
        a = get_data_on_device(np.array([[1, 2], [3, 4]]), backend_name)
        b = get_data_on_device(np.array([[5, 6], [7, 8]]), backend_name)

        result = concatenate(backend_name, [a, b], axis=0)

        if backend_name == "numpy":
            assert result.shape == (4, 2)
            assert np.allclose(result[0], [1, 2])
            assert np.allclose(result[2], [5, 6])
        else:
            assert tuple(result.shape) == (4, 2)

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_concatenate_along_axis_1(self, backend_name):
        """Test concatenation along axis 1."""
        a = get_data_on_device(np.array([[1, 2], [3, 4]]), backend_name)
        b = get_data_on_device(np.array([[5, 6], [7, 8]]), backend_name)

        result = concatenate(backend_name, [a, b], axis=1)

        if backend_name == "numpy":
            assert result.shape == (2, 4)
        else:
            assert tuple(result.shape) == (2, 4)


class TestGetDiagembed:
    """Tests for get_diagembed function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_single_vector(self, backend_name):
        """Test diagonal embedding of single vector."""
        v = get_data_on_device(np.array([1.0, 2.0, 3.0]), backend_name)
        result = get_diagembed(backend_name, v)

        expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

        if backend_name == "numpy":
            assert result.shape == (3, 3)
            assert np.allclose(result, expected)
        else:
            assert tuple(result.shape) == (3, 3)
            assert torch.allclose(result, torch.from_numpy(expected))

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_batched_vectors(self, backend_name):
        """Test diagonal embedding of batched vectors."""
        v = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        result = get_diagembed(backend_name, v)

        if backend_name == "numpy":
            assert result.shape == (2, 2, 2)
            assert result[0, 0, 0] == 1.0
            assert result[0, 1, 1] == 2.0
        else:
            assert tuple(result.shape) == (2, 2, 2)


class TestBatchedTrace:
    """Tests for batched_trace function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_single_matrix(self, backend_name):
        """Test trace of single matrix."""
        A = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend_name)
        result = batched_trace(backend_name, A)

        if backend_name == "numpy":
            assert np.isclose(result, 5.0)  # 1 + 4
        else:
            assert torch.isclose(result, torch.tensor(5.0, dtype=result.dtype))

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_batched_matrices(self, backend_name):
        """Test trace of batched matrices."""
        batch = get_data_on_device(
            np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), backend_name
        )
        result = batched_trace(backend_name, batch)

        if backend_name == "numpy":
            assert result.shape == (2,)
            assert np.allclose(result, [5.0, 13.0])
        else:
            assert tuple(result.shape) == (2,)
            assert torch.allclose(result, torch.tensor([5.0, 13.0], dtype=result.dtype))


class TestBatchedDet:
    """Tests for batched_det function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_single_matrix(self, backend_name):
        """Test determinant of single matrix."""
        A = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)
        result = batched_det(backend_name, A)

        if backend_name == "numpy":
            assert np.isclose(result, 3.0)  # 2*2 - 1*1
        else:
            assert torch.isclose(result, torch.tensor(3.0, dtype=result.dtype))

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_batched_matrices(self, backend_name):
        """Test determinant of batched matrices."""
        batch = get_data_on_device(
            np.array([[[2.0, 1.0], [1.0, 2.0]], [[3.0, 0.0], [0.0, 3.0]]]), backend_name
        )
        result = batched_det(backend_name, batch)

        if backend_name == "numpy":
            assert result.shape == (2,)
            assert np.allclose(result, [3.0, 9.0])
        else:
            assert tuple(result.shape) == (2,)
            assert torch.allclose(result, torch.tensor([3.0, 9.0], dtype=result.dtype))


class TestNormalizeCovariance:
    """Tests for normalize_covariance function."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_no_normalization(self, backend_name):
        """Test that no normalization leaves matrix unchanged."""
        cov = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)
        result = normalize_covariance(cov, None, backend_name, 2)

        if backend_name == "numpy":
            assert np.allclose(result, cov)
        else:
            assert torch.allclose(result, cov)

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_none_string_normalization(self, backend_name):
        """Test that 'none' string leaves matrix unchanged."""
        cov = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)
        result = normalize_covariance(cov, "none", backend_name, 2)

        if backend_name == "numpy":
            assert np.allclose(result, cov)
        else:
            assert torch.allclose(result, cov)

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_diag_normalization(self, backend_name):
        """Test diagonal normalization."""
        cov = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)
        result = normalize_covariance(cov, "diag", backend_name, 2)

        if backend_name == "numpy":
            assert np.isclose(result[0, 0], 1.0)
        else:
            assert torch.isclose(result[0, 0], torch.tensor(1.0, dtype=result.dtype))

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_trace_normalization(self, backend_name):
        """Test trace normalization."""
        cov = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)
        result = normalize_covariance(cov, "trace", backend_name, 2)

        trace = batched_trace(backend_name, result)

        if backend_name == "numpy":
            assert np.isclose(trace, 2.0)
        else:
            assert torch.isclose(trace, torch.tensor(2.0, dtype=trace.dtype))

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_det_normalization(self, backend_name):
        """Test determinant normalization."""
        cov = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)
        result = normalize_covariance(cov, "det", backend_name, 2)

        det = batched_det(backend_name, result)

        if backend_name == "numpy":
            assert np.isclose(det, 1.0)
        else:
            assert torch.isclose(det, torch.tensor(1.0, dtype=det.dtype))

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_batched_trace_normalization(self, backend_name):
        """Test trace normalization on batch of matrices."""
        batch = get_data_on_device(
            np.array([[[2.0, 1.0], [1.0, 2.0]], [[4.0, 0.0], [0.0, 4.0]]]), backend_name
        )
        result = normalize_covariance(batch, "trace", backend_name, 2)

        traces = batched_trace(backend_name, result)

        if backend_name == "numpy":
            assert np.allclose(traces, [2.0, 2.0])
        else:
            assert torch.allclose(traces, torch.tensor([2.0, 2.0], dtype=traces.dtype))

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_invalid_normalization_raises_error(self, backend_name):
        """Test that invalid normalization method raises ValueError."""
        cov = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend_name)

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_covariance(cov, "invalid", backend_name, 2)
