from dataclasses import dataclass
import numpy as np

@dataclass
class PointSource:
		position: np.ndarray
		intensity: float
		