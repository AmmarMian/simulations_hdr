# Tests for Backend dataclass and backend refactoring
# Author: Refactor tests
# Date: 2026-02-24

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend import (
    Backend,
    get_backend_module,
    get_data_on_device,
    expand_dims,
    make_writable_copy,
    batched_eigh,
    concatenate,
    batched_trace,
    batched_det,
    normalize_covariance,
)


class TestBackendDataclass:
    """Tests for Backend dataclass."""

    def test_backend_numpy_creation(self):
        """Test creating a numpy backend."""
        b = Backend.numpy()
        assert b.lib == "numpy"
        assert b.device.type == "cpu"
        assert b.is_torch is False
        assert b.is_cuda is False

    def test_backend_torch_cpu_creation(self):
        """Test creating a torch-cpu backend."""
        b = Backend.torch_cpu()
        assert b.lib == "torch"
        assert b.device.type == "cpu"
        assert b.is_torch is True
        assert b.is_cuda is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_torch_cuda_creation(self):
        """Test creating a torch-cuda backend."""
        b = Backend.torch_cuda()
        assert b.lib == "torch"
        assert b.device.type == "cuda"
        assert b.is_torch is True
        assert b.is_cuda is True

    def test_backend_from_str_numpy(self):
        """Test parsing 'numpy' string."""
        b = Backend.from_str("numpy")
        assert b.lib == "numpy"
        assert not b.is_torch

    def test_backend_from_str_torch_cpu(self):
        """Test parsing 'torch-cpu' string."""
        b = Backend.from_str("torch-cpu")
        assert b.lib == "torch"
        assert b.device.type == "cpu"
        assert b.is_torch
        assert not b.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_from_str_torch_cuda(self):
        """Test parsing 'torch-cuda' string."""
        b = Backend.from_str("torch-cuda")
        assert b.lib == "torch"
        assert b.device.type == "cuda"
        assert b.is_torch
        assert b.is_cuda

    def test_backend_from_str_invalid_raises_error(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend string"):
            Backend.from_str("invalid")

    def test_backend_str_representation_numpy(self):
        """Test string representation of numpy backend."""
        b = Backend.numpy()
        assert str(b) == "numpy"

    def test_backend_str_representation_torch_cpu(self):
        """Test string representation of torch-cpu backend."""
        b = Backend.torch_cpu()
        assert str(b) == "torch-cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_str_representation_torch_cuda(self):
        """Test string representation of torch-cuda backend."""
        b = Backend.torch_cuda()
        assert str(b) == "torch-cuda"

    def test_backend_frozen_dataclass(self):
        """Test that Backend is immutable."""
        b = Backend.numpy()
        with pytest.raises(Exception):  # FrozenInstanceError
            b.lib = "torch"

    def test_backend_numpy_cuda_raises_error(self):
        """Test that numpy with CUDA device raises error."""
        with pytest.raises(ValueError, match="numpy backend only supports CPU"):
            Backend("numpy", torch.device("cuda"))

    def test_backend_equality(self):
        """Test backend equality."""
        b1 = Backend.numpy()
        b2 = Backend.numpy()
        b3 = Backend.torch_cpu()
        assert b1 == b2
        assert b1 != b3

    def test_backend_hash(self):
        """Test that backends can be hashed (used in sets/dicts)."""
        b1 = Backend.numpy()
        b2 = Backend.torch_cpu()
        backend_set = {b1, b2}
        assert len(backend_set) == 2


class TestGetBackendModuleWithBackendObject:
    """Tests for get_backend_module with Backend objects."""

    def test_get_backend_module_with_backend_object_numpy(self):
        """Test get_backend_module with Backend object for numpy."""
        b = Backend.numpy()
        module = get_backend_module(b)
        assert module is np

    def test_get_backend_module_with_backend_object_torch(self):
        """Test get_backend_module with Backend object for torch."""
        b = Backend.torch_cpu()
        module = get_backend_module(b)
        assert module is torch

    def test_get_backend_module_backward_compatibility_string(self):
        """Test that get_backend_module still works with strings."""
        module = get_backend_module("numpy")
        assert module is np
        module = get_backend_module("torch-cpu")
        assert module is torch


class TestGetDataOnDeviceWithBackendObject:
    """Tests for get_data_on_device with Backend objects."""

    def test_get_data_on_device_numpy_backend_object(self):
        """Test get_data_on_device with numpy Backend object."""
        data = np.array([1.0, 2.0, 3.0])
        b = Backend.numpy()
        result = get_data_on_device(data, b)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, data)

    def test_get_data_on_device_torch_cpu_backend_object(self):
        """Test get_data_on_device with torch-cpu Backend object."""
        data = np.array([1.0, 2.0, 3.0])
        b = Backend.torch_cpu()
        result = get_data_on_device(data, b)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_data_on_device_torch_cuda_backend_object(self):
        """Test get_data_on_device with torch-cuda Backend object."""
        data = np.array([1.0, 2.0, 3.0])
        b = Backend.torch_cuda()
        result = get_data_on_device(data, b)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"

    def test_get_data_on_device_backward_compatibility(self):
        """Test that get_data_on_device still works with strings."""
        data = np.array([1.0, 2.0, 3.0])
        result_with_string = get_data_on_device(data, "torch-cpu")
        result_with_backend = get_data_on_device(data, Backend.torch_cpu())
        assert torch.allclose(result_with_string, result_with_backend)


class TestFunctionsWithBackendObject:
    """Tests that all backend functions work with Backend objects."""

    @pytest.mark.parametrize("backend", [Backend.numpy(), Backend.torch_cpu()])
    def test_expand_dims_with_backend_object(self, backend):
        """Test expand_dims with Backend object."""
        data = get_data_on_device(np.array([[1, 2], [3, 4]]), backend)
        result = expand_dims(backend, data, axis=-1)
        if backend.lib == "numpy":
            assert result.shape == (2, 2, 1)
        else:
            assert tuple(result.shape) == (2, 2, 1)

    @pytest.mark.parametrize("backend", [Backend.numpy(), Backend.torch_cpu()])
    def test_make_writable_copy_with_backend_object(self, backend):
        """Test make_writable_copy with Backend object."""
        data = get_data_on_device(np.array([1.0, 2.0, 3.0]), backend)
        copy = make_writable_copy(backend, data)
        if backend.lib == "numpy":
            copy[0] = 99.0
            assert data[0] != 99.0
        else:
            copy[0] = 99.0
            assert data[0].item() != 99.0

    @pytest.mark.parametrize("backend", [Backend.numpy(), Backend.torch_cpu()])
    def test_batched_eigh_with_backend_object(self, backend):
        """Test batched_eigh with Backend object."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        A_device = get_data_on_device(A, backend)
        eigvals, eigvecs = batched_eigh(backend, A_device)
        if backend.lib == "numpy":
            assert eigvals.shape == (2,)
            assert eigvecs.shape == (2, 2)
        else:
            assert tuple(eigvals.shape) == (2,)
            assert tuple(eigvecs.shape) == (2, 2)

    @pytest.mark.parametrize("backend", [Backend.numpy(), Backend.torch_cpu()])
    def test_concatenate_with_backend_object(self, backend):
        """Test concatenate with Backend object."""
        a = get_data_on_device(np.array([[1, 2], [3, 4]]), backend)
        b = get_data_on_device(np.array([[5, 6], [7, 8]]), backend)
        result = concatenate(backend, [a, b], axis=0)
        if backend.lib == "numpy":
            assert result.shape == (4, 2)
        else:
            assert tuple(result.shape) == (4, 2)

    @pytest.mark.parametrize("backend", [Backend.numpy(), Backend.torch_cpu()])
    def test_batched_trace_with_backend_object(self, backend):
        """Test batched_trace with Backend object."""
        A = get_data_on_device(np.array([[1.0, 2.0], [3.0, 4.0]]), backend)
        result = batched_trace(backend, A)
        if backend.lib == "numpy":
            assert np.isclose(result, 5.0)
        else:
            assert torch.isclose(result, torch.tensor(5.0, dtype=result.dtype))

    @pytest.mark.parametrize("backend", [Backend.numpy(), Backend.torch_cpu()])
    def test_batched_det_with_backend_object(self, backend):
        """Test batched_det with Backend object."""
        A = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend)
        result = batched_det(backend, A)
        if backend.lib == "numpy":
            assert np.isclose(result, 3.0)
        else:
            assert torch.isclose(result, torch.tensor(3.0, dtype=result.dtype))

    @pytest.mark.parametrize("backend", [Backend.numpy(), Backend.torch_cpu()])
    def test_normalize_covariance_with_backend_object(self, backend):
        """Test normalize_covariance with Backend object."""
        cov = get_data_on_device(np.array([[2.0, 1.0], [1.0, 2.0]]), backend)
        result = normalize_covariance(cov, "trace", backend, 2)
        trace = batched_trace(backend, result)
        if backend.lib == "numpy":
            assert np.isclose(trace, 2.0)
        else:
            assert torch.isclose(trace, torch.tensor(2.0, dtype=trace.dtype))


class TestMixedBackendStringAndObject:
    """Tests that mixed string and Backend object usage works."""

    def test_can_mix_string_and_backend_in_pipeline(self):
        """Test that string and Backend objects can be mixed in a processing pipeline."""
        data = np.array([1.0, 2.0, 3.0])

        # Start with string
        b_torch = get_data_on_device(data, "torch-cpu")

        # Use Backend object
        backend = Backend.torch_cpu()
        expanded = expand_dims(backend, b_torch, axis=-1)

        # Back to string
        result = get_data_on_device(expanded, "numpy")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_backend_conversion_idempotent(self):
        """Test that converting Backend to string and back gives same Backend."""
        original = Backend.torch_cpu()
        converted = Backend.from_str(str(original))
        assert original == converted

    def test_properties_match_string_checks(self):
        """Test that Backend properties match old string-based checks."""
        torch_cuda = Backend.torch_cuda()
        torch_cpu = Backend.torch_cpu()
        numpy = Backend.numpy()

        # Old checks: "torch" in backend_name
        assert torch_cuda.is_torch
        assert torch_cpu.is_torch
        assert not numpy.is_torch

        # Old checks: backend_name == "torch-cuda"
        assert torch_cuda.is_cuda
        assert not torch_cpu.is_cuda
        assert not numpy.is_cuda
