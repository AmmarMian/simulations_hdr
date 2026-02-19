#!/usr/bin/env python3
"""Test script for M-estimators with configurable parameters
Author: Ammar Mian
Date: 22/10/2025
"""

import sys
from pathlib import Path
import os

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
from src.estimation import (
    fixed_point_m_estimation_centered,
    _tyler_m_estimator_function,
    _student_t_m_estimator_function,
    _huber_m_estimator_function,
    TylerEstimator,
    StudentTEstimator,
    HuberEstimator,
)
from src.backend import get_backend_module, get_data_on_device, sample_standard_normal

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# API Style
# Options: "functional", "class-based"
API_STYLE = "class-based"

# Backend configuration
# Options: "numpy", "torch-cpu", "torch-cuda"
BACKEND_NAME = "torch-cuda"

# Data dimensions
N_SAMPLES = 100  # Number of samples
N_FEATURES = 5  # Number of features
BATCH_DIMENSIONS = (
    400,
    400,
)  # Batch dimensions (e.g., (5, 8, 3) creates 5x8x3 batches)

# M-estimator function
# Options: _tyler_m_estimator_function, _student_t_m_estimator_function, _huber_m_estimator_function
M_ESTIMATOR_FUNCTION = _tyler_m_estimator_function

# M-estimator function parameters (passed as **kwargs)
# For Tyler: {'n_features': N_FEATURES}
# For Student-t: {'df': 3, 'n_features': N_FEATURES}
# For Huber: {'lbda': 2.0, 'beta': 1.0}
M_ESTIMATOR_KWARGS = {"n_features": N_FEATURES}

# Algorithm parameters
TOLERANCE = 1e-4
MAX_ITERATIONS = 50
VERBOSITY = True

# Memory optimization (for CUDA with large batches)
# Set to None to disable chunking, or e.g. 2000 to process in chunks
ITERATION_CHUNK_SIZE = 10000 if BACKEND_NAME == "torch-cuda" else None

# Random seed (set to None for random behavior)
RANDOM_SEED = 42

# ============================================================================
# MAIN SCRIPT
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("M-Estimator Test Script")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  API Style: {API_STYLE}")
    print(f"  Backend: {BACKEND_NAME}")
    print(f"  Batch dimensions: {BATCH_DIMENSIONS}")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  N features: {N_FEATURES}")
    if API_STYLE == "functional":
        print(f"  M-estimator: {M_ESTIMATOR_FUNCTION.__name__}")
        print(f"  M-estimator kwargs: {M_ESTIMATOR_KWARGS}")
    print(f"  Tolerance: {TOLERANCE}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Iteration chunk size: {ITERATION_CHUNK_SIZE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    # Generate synthetic data with batch dimensions
    print("Generating synthetic data...")
    data_shape = list(BATCH_DIMENSIONS) + [N_SAMPLES, N_FEATURES]
    X = sample_standard_normal(
        n_samples=1, data_shape=data_shape, backend_name=BACKEND_NAME, seed=RANDOM_SEED
    )[0]  # Remove the extra sample dimension

    print(f"Data shape: {X.shape}")
    print(f"Data type: {type(X)}")
    print(f"Data dtype: {X.dtype}")
    print()

    # Run the estimation
    print("Running M-estimator...")
    print("-" * 70)

    if API_STYLE == "functional":
        # Functional API
        print("Using functional API: fixed_point_m_estimation_centered()")
        covariances = fixed_point_m_estimation_centered(
            X=X,
            m_estimator_function=M_ESTIMATOR_FUNCTION,
            init=None,
            tol=TOLERANCE,
            iter_max=MAX_ITERATIONS,
            verbosity=VERBOSITY,
            backend_name=BACKEND_NAME,
            iteration_chunk_size=ITERATION_CHUNK_SIZE,
            normalization="trace",  # Add normalization for Tyler
            **M_ESTIMATOR_KWARGS,
        )
    else:
        # Class-based API
        print("Using class-based API:")

        # Determine which estimator to use based on M_ESTIMATOR_FUNCTION
        if M_ESTIMATOR_FUNCTION == _tyler_m_estimator_function:
            print("  Creating TylerEstimator...")
            estimator = TylerEstimator(
                normalization="trace",
                backend_name=BACKEND_NAME,
                tol=TOLERANCE,
                iter_max=MAX_ITERATIONS,
                verbosity=VERBOSITY,
                iteration_chunk_size=ITERATION_CHUNK_SIZE,
            )
        elif M_ESTIMATOR_FUNCTION == _student_t_m_estimator_function:
            df = M_ESTIMATOR_KWARGS.get("df", 3)
            print(f"  Creating StudentTEstimator with df={df}...")
            estimator = StudentTEstimator(
                df=df,
                backend_name=BACKEND_NAME,
                tol=TOLERANCE,
                iter_max=MAX_ITERATIONS,
                verbosity=VERBOSITY,
                iteration_chunk_size=ITERATION_CHUNK_SIZE,
            )
        elif M_ESTIMATOR_FUNCTION == _huber_m_estimator_function:
            lbda = M_ESTIMATOR_KWARGS.get("lbda", float("inf"))
            beta = M_ESTIMATOR_KWARGS.get("beta", 1.0)
            print(f"  Creating HuberEstimator with lbda={lbda}, beta={beta}...")
            estimator = HuberEstimator(
                lbda=lbda,
                beta=beta,
                backend_name=BACKEND_NAME,
                tol=TOLERANCE,
                iter_max=MAX_ITERATIONS,
                verbosity=VERBOSITY,
                iteration_chunk_size=ITERATION_CHUNK_SIZE,
            )
        else:
            raise ValueError(f"Unknown M-estimator function: {M_ESTIMATOR_FUNCTION}")

        print(f"  Computing covariance with {type(estimator).__name__}...")
        covariances = estimator.compute(X)

    print("-" * 70)
    print("\nEstimation completed!")
    print(f"Output shape: {covariances.shape}")
    print(f"Expected shape: {BATCH_DIMENSIONS + (N_FEATURES, N_FEATURES)}")

    # Display some results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    # Show first covariance matrix in batch
    backend_basename = BACKEND_NAME.split("-")[0]
    if backend_basename == "torch":
        first_cov = covariances[0, 0].detach().cpu().numpy()
    else:
        first_cov = covariances[0, 0]

    print(f"\nFirst covariance matrix [0, 0]:")
    print(first_cov)

    # Compute some statistics
    if backend_basename == "torch":
        cov_array = covariances.detach().cpu().numpy()
    else:
        cov_array = covariances

    # Check if matrices are positive definite
    eigenvalues = np.linalg.eigvalsh(cov_array)
    min_eig = eigenvalues.min()
    max_eig = eigenvalues.max()

    print(f"\nEigenvalue statistics:")
    print(f"  Min eigenvalue: {min_eig:.6f}")
    print(f"  Max eigenvalue: {max_eig:.6f}")
    print(f"  All positive definite: {min_eig > 0}")

    # Compute Frobenius norms
    frob_norms = np.linalg.norm(cov_array, axis=(-2, -1))
    print(f"\nFrobenius norms statistics:")
    print(f"  Mean: {frob_norms.mean():.6f}")
    print(f"  Std: {frob_norms.std():.6f}")
    print(f"  Min: {frob_norms.min():.6f}")
    print(f"  Max: {frob_norms.max():.6f}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
