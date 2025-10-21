# 2-detection package
from .backend import Array, ArrayLike, get_backend_module, get_data_on_device, sample_standard_normal
from .detection import Detector

__all__ = [
    "Array",
    "ArrayLike",
    "get_backend_module",
    "get_data_on_device",
    "sample_standard_normal",
    "Detector",
]
