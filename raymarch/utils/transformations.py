import numpy as np


class Transformation:
    def __init__(self):
        self.R = np.eye(3).astype(np.float32)
        self.T = np.zeros(3).astype(np.float32)

    def __call__(self, x):
        self.R = self.R @ x.R
        self.T = self.T + self.R @ x.T
        return self


class Translate(Transformation):
    def __init__(self, vec):
        super().__init__()
        self.T = np.array(vec).astype(np.float32)


class Rotate(Transformation):
    def __init__(self, x, y, z):
        super().__init__()
        self.R = np.array(
            [
                [
                    np.cos(z) * np.cos(y),
                    np.cos(z) * np.sin(y) * np.sin(x) - np.sin(z) * np.cos(x),
                    np.cos(z) * np.sin(y) * np.cos(x) + np.sin(z) * np.sin(x),
                ],
                [
                    np.sin(z) * np.cos(y),
                    np.sin(z) * np.sin(y) * np.sin(x) + np.cos(z) * np.cos(x),
                    np.sin(z) * np.sin(y) * np.cos(x) - np.cos(z) * np.sin(x),
                ],
                [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x)],
            ]
        ).astype(np.float32)
