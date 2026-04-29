# Online SAR detection tools

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Any, Tuple

from src.backend import (
    Backend,
    get_backend_module,
    Array,
)
from src.detection import OnlineDetector

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# Gaussian GLRT State
# -------------------
@dataclass
class GaussianGLRTOnlineState:
    """State for OnlineGaussianGLRT detector.

    Attributes
    ----------
    sum_St : Array
        Accumulated sum of sample covariance matrices
    logdet_sum_St : Array
        Log determinant of sum_St
    n_times : int
        Number of time steps processed
    """
    sum_St: Array
    logdet_sum_St: Array
    n_times: int


# Gaussian GLRT Online
# --------------------
class OnlineGaussianGLRT(OnlineDetector):
    def __init__(self, backend_name: Union[str, Backend] = "numpy") -> None:
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.state = None

    def initialize(self, X: Array) -> Array:
        """Initialize detector state with two first dates.

        Parameters
        ----------
        X : Array
            Shape (..., 2, n_samples, n_features)

        Returns
        -------
        Array
            Initial detection statistic
        """
        assert X.shape[-3] == 2, "Only two dates to initialize state"
        S_matrices = self.be.einsum(
            "...ab,...bc->...ac", self.be.swapaxes(X, -1, -2).conj(), X
        )
        sum_St = self.be.sum(S_matrices, axis=-3)
        logdet_sum_St = self.be.linalg.slogdet(sum_St)[1]
        res = (
            2 * logdet_sum_St
            - self.be.linalg.slogdet(S_matrices[..., 0, :, :])[1]
            - self.be.linalg.slogdet(S_matrices[..., 1, :, :])[1]
        )

        state = GaussianGLRTOnlineState(
            sum_St=sum_St,
            logdet_sum_St=logdet_sum_St,
            n_times=2,
        )

        self.state = state

        return res

    def compute(
        self, past_value: Array, X: Array, state: Any, *args, **kwargs
    ) -> Tuple[Array, Any]:
        """Compute next statistic given current state.

        Parameters
        ----------
        past_value: Array
            past values of test statistic of shape (...), where ... are batches dimensions

        X: Array (torch or numpy)
            data to compute statistic on. Shape (..., n_samples, n_features),
            where ... are batches dimensions. This is the data for 1 time only

        state: GaussianGLRTOnlineState
            Current detector state

        Returns
        -------
        Tuple[Array, GaussianGLRTOnlineState]
            result of test statistic and updated state over all batches dimensions.
        """
        S_Tplusone = self.be.einsum(
            "...ab,...bc->...ac", self.be.swapaxes(X, -1, -2).conj(), X
        )
        sum_STplusone = state.sum_St + S_Tplusone
        logdet_sum_STplusone = self.be.linalg.slogdet(sum_STplusone)[1]
        T = state.n_times
        new_T = T + 1

        new_state = GaussianGLRTOnlineState(
            sum_St=sum_STplusone,
            logdet_sum_St=logdet_sum_STplusone,
            n_times=new_T,
        )

        return (
            past_value
            + new_T * logdet_sum_STplusone
            - T * state.logdet_sum_St
            - self.be.linalg.slogdet(S_Tplusone)[1]
        ), new_state
