# Common detection utilities
# Author : Ammar Mian
# Date: 21/10/2025


from abc import ABC, abstractmethod
from typing import Any, Tuple, Union
from .backend import Array, Backend


class Detector(ABC):
    """Abstract class for detectors over data that implement a
    thresholding w.r.t to false alarm probabilitiy."""

    backend_name: Union[str, Backend]

    @abstractmethod
    def compute(self, X: Array, *args, **kwargs) -> Array:
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


class OnlineDetector(ABC):
    """Abstract class for online detectors that are detectors saving some state and iterating
    over time to compute the result."""

    backend_name: Union[str, Backend]

    def save_state(self, state: Any) -> None:
        """Save state after computation of one time"""
        self.state = state

    def reset_state(self) -> None:
        """Reset detector state. Call before processing a new sequence."""
        self.state = None

    @abstractmethod
    def compute(
        self, past_value: Array, X: Array, state: Any, *args, **kwargs
    ) -> Tuple[Array, Any]:
        pass

    @abstractmethod
    def initialize(self, X: Array) -> Array:
        """Initialize detector state and compute initial result for a new sequence."""
        pass

    def update(self, past_value: Array, X: Array, *args, **kwargs) -> Array:
        """Compute the statistic with new data and current state"""
        res, new_state = self.compute(past_value, X, self.state)
        self.save_state(new_state)
        return res
