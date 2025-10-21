# Common detection utilities
# Author : Ammar Mian
# Date: 21/10/2025


from abc import ABC, abstractmethod
from .backend import Array


class Detector(ABC):
    """Abstract class for detectors over data that implement a
    thresholding w.r.t to false alarm probabilitiy."""

    backend_name: str

    @abstractmethod
    def compute(self, X, *args, **kwargs) -> Array:
        """Compute the test statistic without thresholding.

        Parameters
        ----------
        X: Array (torch or numpy)
            data to compute statistic on. Usually would be of shape
            (..., n_times, n_samples, n_features), where ... are batches dimensions.

        *args:
            Additional positional arguments.

        **kwargs:
            Additional keyword arguments

        Returns
        -------
        Array (torch or numpy)
            result of test statistic over all batches dimensions.

        """
        pass

    def get_threshold(self, pfa: float) -> float:
        """Get thresholding value for a given pfa.

        Parameters
        ----------
        pfa: float
            pfa to attain

        Returns
        -------
        float
            thresholding to apply to get desired pfa.
        """
        raise NotImplementedError()
