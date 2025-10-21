import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.sar import GaussianGLRT

import numpy as np

print("Testing Gaussian GLRT detector...")

print("Generating random data...")
data = np.random.standard_normal((200, 200, 10, 50, 20))

print("Computing GLRT statistic...")
detector = GaussianGLRT(backend_name="torch-cuda")
print(detector.compute(data))
