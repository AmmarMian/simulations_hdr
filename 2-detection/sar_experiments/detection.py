# SAR Specific detection tools
# Re-exports all detectors from offline and online modules

import sys
from pathlib import Path

# Add project root to path for src imports
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    # When imported as package member
    from .detection_offline import (
        GaussianGLRT,
        TwoStepStudentGaussianGLRT,
        TwoStepTylerGaussianGLRT,
        DeterministicCompoundGaussianGLRT,
        ScaleAndShapeKroneckerGLRT,
        _tyler_matandtext_fixed_point,
    )
    from .detection_online import (
        OnlineGaussianGLRT,
        GaussianGLRTOnlineState,
    )
except ImportError:
    # When imported as top-level module
    from detection_offline import (
        GaussianGLRT,
        TwoStepStudentGaussianGLRT,
        TwoStepTylerGaussianGLRT,
        DeterministicCompoundGaussianGLRT,
        ScaleAndShapeKroneckerGLRT,
        _tyler_matandtext_fixed_point,
    )
    from detection_online import (
        OnlineGaussianGLRT,
        GaussianGLRTOnlineState,
    )

__all__ = [
    # Offline detectors
    "GaussianGLRT",
    "TwoStepStudentGaussianGLRT",
    "TwoStepTylerGaussianGLRT",
    "DeterministicCompoundGaussianGLRT",
    "ScaleAndShapeKroneckerGLRT",
    "_tyler_matandtext_fixed_point",
    # Online detectors
    "OnlineGaussianGLRT",
    "GaussianGLRTOnlineState",
]
