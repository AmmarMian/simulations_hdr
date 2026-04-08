# Tests for hardware resource managers with Backend class
# Author: Refactor tests
# Date: 2026-02-24

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend import Backend
from src.hardware_ressources import ImageCPURessourceManager


class DummyProcessor:
    """Dummy processor for testing resource managers."""

    def __call__(self, sliding_windows, *args, **kwargs):
        """Return mean along time and window dimensions.

        Input: (n_windows, n_times, window_size², n_channels)
        Output: (n_windows,) - one scalar per window
        """
        return sliding_windows.mean(axis=(1, 2, 3))


class TestImageCPURessourceManagerWithBackend:
    """Tests for ImageCPURessourceManager with Backend objects."""

    @pytest.fixture
    def sample_data_torch_cpu(self):
        """Create sample torch CPU tensor data."""
        # Shape: (n_times=2, n_channels=3, height=8, width=8)
        np.random.seed(42)
        data_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        return torch.from_numpy(data_np)

    def test_manager_with_backend_object_torch_cpu(self, sample_data_torch_cpu):
        """Test ImageCPURessourceManager with Backend object."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend=backend,
            splitting=(1, 1),
            verbose=0,
        )

        result = manager.process_all_data()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (6, 6)  # (8-3)//1 + 1 for both dims

    def test_manager_with_string_backend(self, sample_data_torch_cpu):
        """Test ImageCPURessourceManager with string backend."""
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend="torch-cpu",
            splitting=(1, 1),
            verbose=0,
        )

        result = manager.process_all_data()
        assert isinstance(result, torch.Tensor)

    def test_manager_splitting_with_backend_object(self, sample_data_torch_cpu):
        """Test ImageCPURessourceManager with splitting and Backend object."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend=backend,
            splitting=(2, 2),
            verbose=0,
        )

        result = manager.process_all_data()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (6, 6)

    def test_manager_backend_stored_as_backend_object(self, sample_data_torch_cpu):
        """Test that manager stores backend as Backend object."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend=backend,
            splitting=(1, 1),
            verbose=0,
        )

        assert isinstance(manager.backend, Backend)
        assert manager.backend.is_torch
        assert not manager.backend.is_cuda

    def test_manager_backend_converted_from_string(self, sample_data_torch_cpu):
        """Test that string backend is converted to Backend object."""
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend="torch-cpu",
            splitting=(1, 1),
            verbose=0,
        )

        assert isinstance(manager.backend, Backend)
        assert manager.backend == Backend.torch_cpu()

    def test_manager_different_window_sizes(self, sample_data_torch_cpu):
        """Test manager with different window sizes."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        for window_size in [3, 5, 7]:
            manager = ImageCPURessourceManager(
                sample_data_torch_cpu,
                window_size=window_size,
                stride=1,
                process_one_split=processor,
                backend=backend,
                splitting=(1, 1),
                verbose=0,
            )

            result = manager.process_all_data()
            expected_size = (8 - window_size) // 1 + 1
            assert result.shape == (expected_size, expected_size)

    def test_manager_different_strides(self, sample_data_torch_cpu):
        """Test manager with different stride values."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        for stride in [1, 2]:
            manager = ImageCPURessourceManager(
                sample_data_torch_cpu,
                window_size=3,
                stride=stride,
                process_one_split=processor,
                backend=backend,
                splitting=(1, 1),
                verbose=0,
            )

            result = manager.process_all_data()
            expected_size = (8 - 3) // stride + 1
            assert result.shape == (expected_size, expected_size)


class TestBackendPropertiesInResourceManager:
    """Test that Backend properties work correctly in resource managers."""

    @pytest.fixture
    def sample_data_torch_cpu(self):
        """Create sample torch CPU tensor data."""
        np.random.seed(42)
        data_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        return torch.from_numpy(data_np)

    def test_is_torch_property_used_in_unfold(self, sample_data_torch_cpu):
        """Test that is_torch property is correctly used for unfolding."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend=backend,
            splitting=(1, 1),
            verbose=0,
        )

        # The _unfold method uses backend.is_torch internally
        # If it works without error, the property is working
        result = manager.process_all_data()
        assert result is not None

    def test_is_cuda_property_in_delete_temp(self, sample_data_torch_cpu):
        """Test that is_cuda property is correctly used in _delete_temp."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend=backend,
            splitting=(1, 1),
            verbose=0,
        )

        # Process should work without CUDA empty_cache being called
        result = manager.process_all_data()
        assert isinstance(result, torch.Tensor)


class TestBackendConsistency:
    """Test that Backend usage is consistent across different components."""

    @pytest.fixture
    def sample_data_torch_cpu(self):
        """Create sample torch CPU tensor data."""
        np.random.seed(42)
        data_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        return torch.from_numpy(data_np)

    def test_backend_object_immutability(self, sample_data_torch_cpu):
        """Test that Backend objects are immutable when stored in manager."""
        backend = Backend.torch_cpu()
        processor = DummyProcessor()

        manager = ImageCPURessourceManager(
            sample_data_torch_cpu,
            window_size=3,
            stride=1,
            process_one_split=processor,
            backend=backend,
            splitting=(1, 1),
            verbose=0,
        )

        # Store original backend
        original_backend = manager.backend

        # Try to process
        _ = manager.process_all_data()

        # Verify backend hasn't changed
        assert manager.backend == original_backend
