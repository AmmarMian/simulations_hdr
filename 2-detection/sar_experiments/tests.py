import sys
from pathlib import Path
import os

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
from src.sar import GaussianGLRT, TwoStepStudentGaussianGLRT

import numpy as np

print("Testing Gaussian GLRT detector...")

print("Generating random data...")
data = np.random.standard_normal((200, 200, 10, 50, 20))

print("Computing GLRT statistic...")
detector = GaussianGLRT(backend_name="torch-cuda")
print(detector.compute(data))

print("Computing 2-step Student-t GLRT statistic...")
detector = TwoStepStudentGaussianGLRT(backend_name="torch-cuda", df=3)
print(detector.compute(data))
