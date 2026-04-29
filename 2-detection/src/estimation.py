# Common estimation utilities
# Author: Ammar Mian
# Date: 22/10/2025

from types import ModuleType
from typing import Callable, Optional, Union

from .backend import (
    Backend,
    Array,
    get_backend_module,
    get_diagembed,
    is_complex,
    make_writable_copy,
    expand_dims,
    get_data_on_device,
    batched_eigh,
    concatenate,
    normalize_covariance,
    create_scalar_array,
    to_dtype,
)
from .manifolds import invsqrtm_psd
from abc import ABC, abstractmethod
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)


# ------------------------------------------------------------------------
# Abstract Estimator class
# ------------------------------------------------------------------------
class Estimator(ABC):
    """Abstract class for estimators over data."""

    backend_name: Union[str, Backend]

    @abstractmethod
    def compute(self, X: Array) -> Array:
        """Compute the estimator on the data.

        Parameters
        ----------
        X: Array (torch or numpy)
            data to compute the estimator on. Usually would be of shape
            (..., n_samples, n_features), where ... are batch dimensions.
            Data is assumed to be centered (zero mean).

        Returns
        -------
        Array (torch or numpy)
            estimated covariance matrices of shape (..., n_features, n_features)

        """
        pass


# -----------------------------------------------------------------------
# SCM-estimators
# -----------------------------------------------------------------------
class SCMEstimator(Estimator):
    def __init__(self, assume_centered: bool = True, backend_name: Union[str, Backend] = "numpy"):
        self.assume_centered = assume_centered
        self.backend_name = backend_name
        self.backend_module = get_backend_module(self.backend_name)

    def compute(self, X: Array) -> Array:
        if not self.assume_centered:
            X = X - X.mean(axis=-1, keepdim=True)
        return (1 / X.shape[-2]) * self.backend_module.swapaxes(X, -1, -2).conj() @ X


# -----------------------------------------------------------------------
# M-estimators
# -----------------------------------------------------------------------
def _student_t_m_estimator_function(
    x: Array, df: float = 3, n_features: int = 1, **kwargs
):
    """Student-t mle m-estimator function

    Parameters
    ----------
    x : Array
        input data
    df : int, optional
        degrees of freedom of Student-t law, by default 3
    n_features : int, optional
        optional for compatibility with kwargs but nut so much
        optional, by default 1

    Returns
    -------
    Array
        result of m-estimator function
    """
    return (n_features + df / 2) / (x + df / 2)


def _huber_m_estimator_function(
    x: Array,
    lbda: float = float("inf"),
    beta: float = 1,
    backend_name: Union[str, Backend] = "numpy",
    **kwargs,
):
    """Huber M-estimator function as defined for example in
    > Statistiques des estimateurs robustes pour le traitement du signal et des images
    > p 16, Ph.d Thesis, Gordana Draskovic

    It consists of the function defined for a threshold $\\lambda$ and a real $\\beta$
    by the equation :
    $$
    u(x)=\\frac{1}{\beta}\\min\\left(1, \\lambda/x\\right)
    $$

    Parameters
    ----------
    x : Array
        input data
    lbda : float, optional
        value at which we shift from no ponderation to inverse ponderation.
        By default inf.
    beta : float, optional
        a tuning parameter. By default, 1.
    backend_name : str or Backend, optional
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object. By default "numpy"

    Returns
    -------
    Array
        array of the same shape as input with values that have been scaled depending
        on the tuning parameters

    """
    if lbda <= 0 or beta <= 0:
        raise AssertionError(
            f"Error, lambda or beta can't be negative or null : lambda={lbda}, beta={beta}"
        )
    backend_module = get_backend_module(backend_name)
    # Use ones_like to create a tensor/array of 1s with same shape, dtype, device as x
    ones = backend_module.ones_like(x)
    return (1 / beta) * backend_module.minimum(ones, lbda / x)


def _tyler_m_estimator_function(x, n_features=1, **kwargs):
    """Tyler M-estimator function

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        quadratic form
    n_features : int, optional
        optional for compatibility with kwargs but nut so much
        optional, by default 1

    Returns
    -------
    array-like of shape (n_samples,)
        scaled quadratic form to have Tyler's estimator
    """
    return n_features / x


# Alias for backward compatibility: use invsqrtm_psd from manifolds
invsqrtm = invsqrtm_psd


def _compute_covariance_update(
    X_batch: Array,
    cov_batch: Array,
    m_estimator_function: Callable,
    backend_name: Union[str, Backend],
    debug: bool = False,
    **kwargs,
) -> tuple[Array, Array]:
    """Compute covariance update for a batch of matrices.

    Parameters
    ----------
    X_batch : Array
        Data batch of shape (n_batch, n_samples, n_features)
    cov_batch : Array
        Current covariance estimates of shape (n_batch, n_features, n_features)
    m_estimator_function : Callable
        M-estimator function
    backend_name : str or Backend
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object.
    debug : bool, optional
        Print debug information, by default False
    **kwargs : dict
        Additional arguments for m_estimator_function

    Returns
    -------
    tuple[Array, Array]
        New covariance estimates and errors
    """
    backend_module = get_backend_module(backend_name)

    # Compute new covariance estimates
    if debug:
        print(
            f"    cov_batch eigenvalues (sample 0): {backend_module.linalg.eigvalsh(cov_batch[0])}"
        )

    inv_sqrt_cov = invsqrtm(cov_batch, backend_name)
    if debug:
        print(f"    invsqrtm any NaN: {backend_module.isnan(inv_sqrt_cov).any()}")

    temp = inv_sqrt_cov @ backend_module.swapaxes(X_batch, -1, -2)
    if debug:
        print(
            f"    temp shape: {temp.shape}, any NaN: {backend_module.isnan(temp).any()}"
        )

    quadratic = backend_module.einsum(
        "...ij,...ji->...i", backend_module.swapaxes(temp, -1, -2).conj(), temp
    )
    if debug:
        print(
            f"    quadratic shape: {quadratic.shape}, any NaN: {backend_module.isnan(quadratic).any()}"
        )
        print(
            f"    quadratic range: min={backend_module.real(quadratic).min():.4e}, max={backend_module.real(quadratic).max():.4e}"
        )

    # Add backend_name to kwargs for m-estimator functions that need it (e.g., Huber)
    kwargs_with_backend = {**kwargs, "backend_name": backend_name}
    weights = m_estimator_function(
        backend_module.real(quadratic), **kwargs_with_backend
    )
    if debug:
        print(f"    weights any NaN: {backend_module.isnan(weights).any()}")
        print(f"    weights range: min={weights.min():.4e}, max={weights.max():.4e}")

    # Expand weights dimensions for broadcasting: (..., n_samples) -> (..., 1, n_samples)
    weights_expanded = expand_dims(backend_name, backend_module.sqrt(weights), axis=-2)
    temp = backend_module.swapaxes(X_batch, -1, -2) * weights_expanded
    cov_new_batch = (
        (1 / X_batch.shape[-2]) * temp @ backend_module.swapaxes(temp, -1, -2).conj()
    )
    if debug:
        print(f"    cov_new_batch any NaN: {backend_module.isnan(cov_new_batch).any()}")

    # Condition for stopping
    err_batch = backend_module.linalg.norm(
        cov_new_batch - cov_batch, axis=(-2, -1)
    ) / backend_module.linalg.norm(cov_batch, axis=(-2, -1))

    return cov_new_batch, err_batch


def fixed_point_m_estimation_centered(
    X: Array,
    m_estimator_function: Callable = _tyler_m_estimator_function,
    init: Optional[Array] = None,
    tol: float = 1e-4,
    iter_max: int = 10,
    verbosity: bool = False,
    backend_name: Union[str, Backend] = "numpy",
    debug: bool = False,
    iteration_chunk_size: Optional[int] = None,
    normalization: Optional[str] = None,
    **kwargs,
) -> Array:
    """Fixed-point algorithm for M-estimators of covariance matrix as defined in:
    >Ricardo Antonio Maronna.
    >"Robust $M$-Estimators of Multivariate Location and Scatter." The Annals of Statistics, 4(1) 51-67 January, 1976.
    >https://doi.org/10.1214/aos/1176343347

    Data is assumed to be centered.

    Parameters
    ----------
    X : Array of shape (..., n_samples, n_features)
        data, where n_samples is the number of samples and
          n_features is the number of features and ... are batch dimensions.
    m_estimator_function : function
        function to compute the scaling to apply to the quadratic form.
        it has to be vectorized. By default Tyler's function.
    init : Array, optional
        initial point of algorithm, by default Identity matrix.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 10.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    backend_name : str or Backend, optional
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object. By default "numpy"
    debug : bool, optional
        Print debug information during iterations, by default False
    iteration_chunk_size : int, optional
        Process matrices in chunks of this size within each iteration to reduce memory usage.
        Useful for large batches on GPU. By default None (no chunking).
    normalization : str or None, optional
        Normalization to apply at each iteration. Required for Tyler's estimator.
        Options are:
        - None or 'none': no normalization (default)
        - 'diag': normalize so first diagonal element = 1
        - 'trace': normalize so trace(Sigma) = n_features
        - 'det': normalize so det(Sigma) = 1
    **kwargs :
        Arguments to m_estimator_function

    Returns
    -------
    Array
        estimated covariances matrix
    """

    # Get backend module
    backend_module = get_backend_module(backend_name)

    # Initialisation
    if init is None:
        cov_shape = X.shape[:-2] + (X.shape[-1], X.shape[-1])
        eye_matrix = backend_module.eye(X.shape[-1], dtype=X.dtype)
        # Ensure eye matrix is on the same device as X
        eye_matrix = get_data_on_device(eye_matrix, backend_name)
        eye_broadcasted = backend_module.broadcast_to(eye_matrix, cov_shape)
        covariances = make_writable_copy(backend_name, eye_broadcasted)
        del eye_matrix, eye_broadcasted
    else:
        if init.ndim == 2:
            cov_shape = X.shape[:-2] + (X.shape[-1], X.shape[-1])
            init_broadcasted = backend_module.broadcast_to(init, cov_shape)
            covariances = make_writable_copy(backend_name, init_broadcasted)
        else:
            covariances = init

    # Fixed-point loop
    pbar = None
    if verbosity:
        pbar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        pbar.start()
        task_id = pbar.add_task("Estimating covariance...", total=iter_max)

    # To keep track of converged matrices in batch and not update them anymore
    batch_shape = covariances.shape[:-2]
    # Ensure err is at least a 0-d array, not a scalar
    if len(batch_shape) == 0:
        err = create_scalar_array(backend_module.inf, float, backend_name)
    else:
        err = backend_module.inf * backend_module.ones(batch_shape)
        err = get_data_on_device(err, backend_name)

    iteration = 0
    # Ensure mask_notconverged is at least a 0-d array, not a scalar
    if len(batch_shape) == 0:
        mask_notconverged = create_scalar_array(True, backend_module.bool, backend_name)
    else:
        mask_notconverged = backend_module.ones(batch_shape, dtype=backend_module.bool)
        mask_notconverged = get_data_on_device(mask_notconverged, backend_name)

    try:
        while (backend_module.any(err > tol)) and (iteration < iter_max):
            if debug:
                print(f"\nIteration {iteration}")
                print(f"  mask_notconverged: {mask_notconverged}")
                print(f"  err: {err}")
                print(
                    f"  n_batches to process: {backend_module.sum(mask_notconverged)}"
                )

            # Apply mask to avoid updating converged matrices
            X_batch = X[mask_notconverged]
            cov_batch = covariances[mask_notconverged]
            n_matrices_to_process = X_batch.shape[0]

            if debug:
                print(f"  X_batch.shape: {X_batch.shape}")
                print(f"  cov_batch.shape: {cov_batch.shape}")

            # Check if we need to chunk the iteration for memory efficiency
            if (
                iteration_chunk_size is not None
                and n_matrices_to_process > iteration_chunk_size
            ):
                if debug:
                    print(f"  Chunking into batches of {iteration_chunk_size}")

                # Process in chunks
                cov_new_batch_list = []
                err_batch_list = []

                for chunk_start in range(
                    0, n_matrices_to_process, iteration_chunk_size
                ):
                    chunk_end = min(
                        chunk_start + iteration_chunk_size, n_matrices_to_process
                    )
                    X_chunk = X_batch[chunk_start:chunk_end]
                    cov_chunk = cov_batch[chunk_start:chunk_end]

                    cov_new_chunk, err_chunk = _compute_covariance_update(
                        X_chunk,
                        cov_chunk,
                        m_estimator_function,
                        backend_name,
                        debug=debug,
                        **kwargs,
                    )

                    cov_new_batch_list.append(cov_new_chunk)
                    err_batch_list.append(err_chunk)

                # Concatenate results
                cov_new_batch = concatenate(backend_name, cov_new_batch_list, axis=0)
                err_batch = concatenate(backend_name, err_batch_list, axis=0)
            else:
                # Process all at once (no chunking)
                cov_new_batch, err_batch = _compute_covariance_update(
                    X_batch,
                    cov_batch,
                    m_estimator_function,
                    backend_name,
                    debug=debug,
                    **kwargs,
                )

            iteration += 1

            if debug:
                print(f"  err_batch: {err_batch}")
                print(
                    f"  cov_new_batch[0,0,0]: {cov_new_batch[0, 0, 0] if cov_new_batch.shape[0] > 0 else 'N/A'}"
                )

            # Updating covariances and err
            covariances[mask_notconverged] = cov_new_batch
            err[mask_notconverged] = to_dtype(err_batch, err.dtype, backend_name)

            # Apply normalization if specified
            if normalization is not None:
                covariances = normalize_covariance(
                    covariances, normalization, backend_name, X.shape[-1]
                )

            # Update mask to exclude converged matrices
            mask_notconverged = err > tol

            if debug:
                print(f"  After update - err: {err}")
                print(f"  After update - mask_notconverged: {mask_notconverged}")

            if verbosity:
                pbar.update(
                    task_id,
                    advance=1,
                    description=f"Iteration {iteration}, max err {backend_module.max(err):.2e}",
                )
    finally:
        # Ensure progress bar is stopped even on keyboard interrupt
        if pbar is not None:
            pbar.stop()

    return covariances


# -----------------------------------------------------------------------
# Concrete M-Estimator Classes
# -----------------------------------------------------------------------
class TylerEstimator(Estimator):
    """Tyler's M-estimator of scatter matrix.

    Tyler's estimator is a robust, affine equivariant estimator that is insensitive
    to the scale of the data. It requires normalization at each iteration since it
    only estimates shape, not scale.

    Reference:
    > David E. Tyler.
    > "A Distribution-Free M-Estimator of Multivariate Scatter."
    > The Annals of Statistics, 15(1) 234-251 March, 1987.
    > https://doi.org/10.1214/aos/1176350263

    Parameters
    ----------
    normalization : str, optional
        Normalization method to apply at each iteration. Required for Tyler's estimator.
        Options:
        - 'trace': normalize so trace(Sigma) = n_features (recommended)
        - 'det': normalize so det(Sigma) = 1
        - 'diag': normalize so Sigma[0, 0] = 1
        By default 'trace'.

    backend_name : str, optional
        Backend to use. Choices are: 'numpy', 'torch-cpu', 'torch-cuda'.
        By default 'numpy'.

    tol : float, optional
        Convergence tolerance for the fixed-point algorithm. By default 1e-4.

    iter_max : int, optional
        Maximum number of iterations. By default 50.

    verbosity : bool, optional
        Show progress bar during iterations. By default False.

    debug : bool, optional
        Print debug information during iterations. By default False.

    iteration_chunk_size : int or None, optional
        Process matrices in chunks of this size within each iteration to reduce
        memory usage. Useful for large batches on GPU. By default None (no chunking).

    init : Array or None, optional
        Initial covariance estimate. If None, uses identity matrix. By default None.

    Examples
    --------
    >>> import numpy as np
    >>> from src.estimation import TylerEstimator
    >>> # Generate centered data
    >>> X = np.random.randn(100, 5)  # 100 samples, 5 features
    >>> X = X - X.mean(axis=0)  # Center the data
    >>> # Create estimator
    >>> estimator = TylerEstimator(normalization='trace', tol=1e-4, iter_max=50)
    >>> # Compute covariance
    >>> cov = estimator.compute(X)
    >>> print(cov.shape)  # (5, 5)

    """

    def __init__(
        self,
        normalization: str = "trace",
        backend_name: Union[str, Backend] = "numpy",
        tol: float = 1e-4,
        iter_max: int = 50,
        verbosity: bool = False,
        debug: bool = False,
        iteration_chunk_size: Optional[int] = None,
        init: Optional[Array] = None,
    ):
        self.normalization = normalization
        self.backend_name = backend_name
        self.tol = tol
        self.iter_max = iter_max
        self.verbosity = verbosity
        self.debug = debug
        self.iteration_chunk_size = iteration_chunk_size
        self.init = init

    def compute(self, X: Array) -> Array:
        """Compute Tyler's M-estimator of scatter matrix.

        Parameters
        ----------
        X : Array of shape (..., n_samples, n_features)
            Centered data, where n_samples is the number of samples,
            n_features is the number of features, and ... are batch dimensions.
            Data must be centered (zero mean).

        Returns
        -------
        Array of shape (..., n_features, n_features)
            Estimated scatter matrices.
        """
        n_features = X.shape[-1]

        return fixed_point_m_estimation_centered(
            X=X,
            m_estimator_function=_tyler_m_estimator_function,
            init=self.init,
            tol=self.tol,
            iter_max=self.iter_max,
            verbosity=self.verbosity,
            backend_name=self.backend_name,
            debug=self.debug,
            iteration_chunk_size=self.iteration_chunk_size,
            normalization=self.normalization,
            n_features=n_features,
        )


class StudentTEstimator(Estimator):
    """Student-t M-estimator of covariance matrix.

    The Student-t estimator assumes data follows a multivariate Student-t distribution.
    It provides robustness to outliers, with the degree of robustness controlled by
    the degrees of freedom parameter df. Low df (e.g., 3) gives strong robustness
    and shrinkage, while high df (e.g., 300) approaches Gaussian MLE behavior.

    Reference:
    > Ricardo Antonio Maronna.
    > "Robust M-Estimators of Multivariate Location and Scatter."
    > The Annals of Statistics, 4(1) 51-67 January, 1976.
    > https://doi.org/10.1214/aos/1176343347

    Parameters
    ----------
    df : float, optional
        Degrees of freedom of the Student-t distribution. Lower values (e.g., 3)
        provide stronger robustness to outliers but more shrinkage. Higher values
        (e.g., 300) approach Gaussian behavior. By default 3.

    normalization : str or None, optional
        Normalization method to apply at each iteration. Unlike Tyler, normalization
        is optional for Student-t. Options: 'trace', 'det', 'diag', or None.
        By default None.

    backend_name : str or Backend, optional
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object. By default 'numpy'.

    tol : float, optional
        Convergence tolerance for the fixed-point algorithm. By default 1e-4.

    iter_max : int, optional
        Maximum number of iterations. By default 50.

    verbosity : bool, optional
        Show progress bar during iterations. By default False.

    debug : bool, optional
        Print debug information during iterations. By default False.

    iteration_chunk_size : int or None, optional
        Process matrices in chunks of this size within each iteration to reduce
        memory usage. Useful for large batches on GPU. By default None (no chunking).

    init : Array or None, optional
        Initial covariance estimate. If None, uses identity matrix. By default None.

    Examples
    --------
    >>> import numpy as np
    >>> from src.estimation import StudentTEstimator
    >>> # Generate centered data
    >>> X = np.random.randn(100, 5)  # 100 samples, 5 features
    >>> X = X - X.mean(axis=0)  # Center the data
    >>> # Create estimator with df=3 for robustness
    >>> estimator = StudentTEstimator(df=3, tol=1e-4, iter_max=50)
    >>> # Compute covariance
    >>> cov = estimator.compute(X)
    >>> print(cov.shape)  # (5, 5)

    """

    def __init__(
        self,
        df: float = 3,
        normalization: Optional[str] = None,
        backend_name: Union[str, Backend] = "numpy",
        tol: float = 1e-4,
        iter_max: int = 50,
        verbosity: bool = False,
        debug: bool = False,
        iteration_chunk_size: Optional[int] = None,
        init: Optional[Array] = None,
    ):
        self.df = df
        self.normalization = normalization
        self.backend_name = backend_name
        self.tol = tol
        self.iter_max = iter_max
        self.verbosity = verbosity
        self.debug = debug
        self.iteration_chunk_size = iteration_chunk_size
        self.init = init

    def compute(self, X: Array) -> Array:
        """Compute Student-t M-estimator of covariance matrix.

        Parameters
        ----------
        X : Array of shape (..., n_samples, n_features)
            Centered data, where n_samples is the number of samples,
            n_features is the number of features, and ... are batch dimensions.
            Data must be centered (zero mean).

        Returns
        -------
        Array of shape (..., n_features, n_features)
            Estimated covariance matrices.
        """
        n_features = X.shape[-1]

        return fixed_point_m_estimation_centered(
            X=X,
            m_estimator_function=_student_t_m_estimator_function,
            init=self.init,
            tol=self.tol,
            iter_max=self.iter_max,
            verbosity=self.verbosity,
            backend_name=self.backend_name,
            debug=self.debug,
            iteration_chunk_size=self.iteration_chunk_size,
            normalization=self.normalization,
            df=self.df,
            n_features=n_features,
        )


class HuberEstimator(Estimator):
    """Huber M-estimator of covariance matrix.

    The Huber estimator provides a compromise between efficiency and robustness by
    using a threshold-based weighting scheme. It behaves like the sample covariance
    for small deviations and downweights large outliers beyond a threshold lambda.

    Reference:
    > Statistiques des estimateurs robustes pour le traitement du signal et des images
    > p 16, Ph.d Thesis, Gordana Draskovic

    The weighting function is:
    u(x) = (1/beta) * min(1, lambda/x)

    Parameters
    ----------
    lbda : float, optional
        Threshold parameter. Values below lambda are not downweighted, values above
        are inversely weighted. By default inf (no downweighting, equivalent to sample cov).

    beta : float, optional
        Tuning parameter that controls the overall scaling of weights. By default 1.0.

    normalization : str or None, optional
        Normalization method to apply at each iteration. Options: 'trace', 'det',
        'diag', or None. By default None.

    backend_name : str or Backend, optional
        Backend specification. Can be a string ('numpy', 'torch-cpu', 'torch-cuda')
        or a Backend object. By default 'numpy'.

    tol : float, optional
        Convergence tolerance for the fixed-point algorithm. By default 1e-4.

    iter_max : int, optional
        Maximum number of iterations. By default 50.

    verbosity : bool, optional
        Show progress bar during iterations. By default False.

    debug : bool, optional
        Print debug information during iterations. By default False.

    iteration_chunk_size : int or None, optional
        Process matrices in chunks of this size within each iteration to reduce
        memory usage. Useful for large batches on GPU. By default None (no chunking).

    init : Array or None, optional
        Initial covariance estimate. If None, uses identity matrix. By default None.

    Examples
    --------
    >>> import numpy as np
    >>> from src.estimation import HuberEstimator
    >>> # Generate centered data
    >>> X = np.random.randn(100, 5)  # 100 samples, 5 features
    >>> X = X - X.mean(axis=0)  # Center the data
    >>> # Create estimator with threshold
    >>> estimator = HuberEstimator(lbda=2.0, beta=1.0, tol=1e-4, iter_max=50)
    >>> # Compute covariance
    >>> cov = estimator.compute(X)
    >>> print(cov.shape)  # (5, 5)

    """

    def __init__(
        self,
        lbda: float = float("inf"),
        beta: float = 1.0,
        normalization: Optional[str] = None,
        backend_name: Union[str, Backend] = "numpy",
        tol: float = 1e-4,
        iter_max: int = 50,
        verbosity: bool = False,
        debug: bool = False,
        iteration_chunk_size: Optional[int] = None,
        init: Optional[Array] = None,
    ):
        self.lbda = lbda
        self.beta = beta
        self.normalization = normalization
        self.backend_name = backend_name
        self.tol = tol
        self.iter_max = iter_max
        self.verbosity = verbosity
        self.debug = debug
        self.iteration_chunk_size = iteration_chunk_size
        self.init = init

    def compute(self, X: Array) -> Array:
        """Compute Huber M-estimator of covariance matrix.

        Parameters
        ----------
        X : Array of shape (..., n_samples, n_features)
            Centered data, where n_samples is the number of samples,
            n_features is the number of features, and ... are batch dimensions.
            Data must be centered (zero mean).

        Returns
        -------
        Array of shape (..., n_features, n_features)
            Estimated covariance matrices.
        """
        n_features = X.shape[-1]

        return fixed_point_m_estimation_centered(
            X=X,
            m_estimator_function=_huber_m_estimator_function,
            init=self.init,
            tol=self.tol,
            iter_max=self.iter_max,
            verbosity=self.verbosity,
            backend_name=self.backend_name,
            debug=self.debug,
            iteration_chunk_size=self.iteration_chunk_size,
            normalization=self.normalization,
            lbda=self.lbda,
            beta=self.beta,
        )
