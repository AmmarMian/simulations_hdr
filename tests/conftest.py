# Pytest configuration and fixtures for 2-detection tests
# Author: Ammar Mian
# Date: 21/10/2025

import pytest
import numpy as np
import torch

from hdrlib.core.backend import to_numpy  # noqa: E402


# ── Backend availability predicates ───────────────────────────────────────────

def cupy_available() -> bool:
    """True if CuPy is installed and a CUDA device is present."""
    try:
        import cupy
        return cupy.cuda.is_available()
    except (ImportError, Exception):
        return False


def jax_available() -> bool:
    """True if JAX (CPU) is installed."""
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


def jax_cuda_available() -> bool:
    """True if JAX is installed and a CUDA GPU is present."""
    try:
        import jax
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def jax_metal_available() -> bool:
    """True if jax-metal is installed and a Metal device is present."""
    try:
        import jax
        return len(jax.devices("metal")) > 0
    except Exception:
        return False


# ── Shared parametrize list ───────────────────────────────────────────────────
#
# Use ALL_BACKEND_PARAMS in @pytest.mark.parametrize("backend_name", ALL_BACKEND_PARAMS)
# so new backends are exercised everywhere without touching individual tests.
# Unavailable backends are automatically skipped.

ALL_BACKEND_PARAMS = [
    "numpy",
    "torch-cpu",
    pytest.param(
        "torch-mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS not available"
        ),
    ),
    pytest.param(
        "cupy",
        marks=pytest.mark.skipif(
            not cupy_available(), reason="CuPy / CUDA not available"
        ),
    ),
    pytest.param(
        "jax-cpu",
        marks=pytest.mark.skipif(not jax_available(), reason="JAX not installed"),
    ),
    pytest.param(
        "jax-cuda",
        marks=pytest.mark.skipif(
            not jax_cuda_available(), reason="JAX GPU (CUDA) not available"
        ),
    ),
    pytest.param(
        "jax-metal",
        marks=pytest.mark.skipif(
            not jax_metal_available(), reason="jax-metal not available"
        ),
    ),
]

# Subset for tests that only make sense on CPU backends
CPU_BACKEND_PARAMS = [
    "numpy",
    "torch-cpu",
    pytest.param(
        "jax-cpu",
        marks=pytest.mark.skipif(not jax_available(), reason="JAX not installed"),
    ),
]


# ── Helper for cross-backend comparison ───────────────────────────────────────

def as_numpy_for_compare(x) -> np.ndarray:
    """Convert any backend array to numpy for comparison in tests."""
    return to_numpy(x)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(params=ALL_BACKEND_PARAMS)
def backend_name(request):
    """Parametrized fixture covering all available backends."""
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
