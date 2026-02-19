# SAR Specific detection tools

import sys
from pathlib import Path
import os

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.backend import get_backend_module, get_data_on_device
from src.detection import Detector
from src.estimation import SCMEstimator, TylerEstimator
