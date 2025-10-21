# SAR detectors

from .backend import Array, get_backend_module, get_data_on_device
from .detection import Detector


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
        self.be = get_backend_module(backend_name.split("-")[0])

    def compute(self, X: Array, *args, **kwargs) -> Array:
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

