# Pytest configuration and fixtures for 2-detection tests
# Author: Ammar Mian
# Date: 21/10/2025

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_path))


@pytest.fixture(params=["numpy", "torch-cpu"])
def backend_name(request):
    """Parametrized fixture for different backend names."""
    return request.param


@pytest.fixture
def numpy_backend():
    """Fixture providing numpy backend name."""
    return "numpy"


@pytest.fixture
def torch_cpu_backend():
    """Fixture providing torch-cpu backend name."""
    return "torch-cpu"


@pytest.fixture
def sample_data_shape():
    """Fixture providing a sample data shape for testing."""
    return [10, 5, 3]  # (n_times, n_samples, n_features)


@pytest.fixture
def numpy_sample_data():
    """Fixture providing sample numpy data."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 5, 3))


@pytest.fixture
def torch_sample_data():
    """Fixture providing sample torch data."""
    torch.manual_seed(42)
    return torch.randn((10, 5, 3))


@pytest.fixture
def small_n_trials():
    """Fixture providing small number of trials for Monte Carlo tests."""
    return 10


@pytest.fixture
def test_pfa_values():
    """Fixture providing common PFA values for testing."""
    return [0.01, 0.05, 0.1]


@pytest.fixture
def random_seed():
    """Fixture providing a fixed random seed for reproducibility."""
    return 42
