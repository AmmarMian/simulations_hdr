# SAR detectors

from .backend import Array, get_backend_module, get_data_on_device
from .detection import Detector
from .estimation import StudentTEstimator


# Gaussian GLRT
# -------------
class GaussianGLRT(Detector):
    def __init__(self, backend_name: str = "numpy") -> None:
        """Gaussian GLRT on multiple covariance equality testing

        Parameters
        -----------
        backend_name: str
            backend to use. Choices are : numpy, torch-cpu, torch-cuda
        """
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)

    def compute(self, X: Array) -> Array:
        """Compute the Gaussian GLRT statistic.

        Parameters
        ----------
        X: Array (torch or numpy)
            data to compute statistic on. Shape (..., n_times, n_samples, n_features),
            where ... are batches dimensions.

        Returns
        -------
        Array (torch or numpy)
            result of test statistic over all batches dimensions.

        """
        X = get_data_on_device(X, self.backend_name)
        covariances = (
            self.be.einsum("...ab,...cd->...ad", self.be.swapaxes(X, -1, -2).conj(), X)
            / X.shape[-2]
        )
        cov_mean = covariances.mean(axis=-3)
        log_det_cov_mean = self.be.linalg.slogdet(cov_mean)[1]
        res = log_det_cov_mean * X.shape[-3]
        for t in range(X.shape[-3]):
            cov_t = covariances[..., t, :, :]
            log_det_cov_t = self.be.linalg.slogdet(cov_t)[1]
            res -= log_det_cov_t
        return res


# 2-step Detectors
# ----------------
class TwoStepStudentGaussianGLRT(Detector):
    def __init__(self, backend_name: str = "numpy", df: float = 3) -> None:
        """Gaussian GLRT on multiple covariance equality testing but with a Student-t
        estimator of covariances.

        Parameters
        -----------
        backend_name: str
            backend to use. Choices are : numpy, torch-cpu, torch-cuda

        df: float
            Number of degress of freedom of Student-t distribution. By default 3 (Heacy tailed).

        """
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.df = df
        # TODO: add all args
        self.estimator = StudentTEstimator(
            self.df,
            backend_name=self.backend_name,
            iteration_chunk_size=2000,
            iter_max=5,
        )

    def compute(self, X: Array) -> Array:
        """Compute the Gaussian GLRT statistic.

        Parameters
        ----------
        X: Array (torch or numpy)
            data to compute statistic on. Shape (..., n_times, n_samples, n_features),
            where ... are batches dimensions.

        Returns
        -------
        Array (torch or numpy)
            result of test statistic over all batches dimensions.

        """
        X = get_data_on_device(X, self.backend_name)
        covariances = self.estimator.compute(X)
        cov_mean = self.be.mean(covariances, axis=-3)
        log_det_cov_mean = self.be.linalg.slogdet(cov_mean)[1]
        res = log_det_cov_mean * X.shape[-3]
        # TODO: covariance under H0 should be estimated using all time data by reshaping
        for t in range(X.shape[-3]):
            cov_t = covariances[..., t, :, :]
            log_det_cov_t = self.be.linalg.slogdet(cov_t)[1]
            res -= log_det_cov_t
        return res
