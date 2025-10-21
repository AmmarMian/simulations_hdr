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
        backend = get_backend_module("torch")
        assert backend is torch

    def test_invalid_backend_raises_assertion(self):
        """Test that invalid backend name raises AssertionError."""
        with pytest.raises(AssertionError, match="Backend basename .* unknown"):
            get_backend_module("invalid")

    def test_case_sensitive(self):
        """Test that backend names are case sensitive."""
        with pytest.raises(AssertionError):
            get_backend_module("NumPy")


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
        """Test that invalid backend name raises AssertionError."""
        with pytest.raises(AssertionError, match="Backend name .* unknown"):
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
