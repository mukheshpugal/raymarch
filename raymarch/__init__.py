from pathlib import Path

import pycuda.autoinit
from pycuda.compiler import SourceModule

with open(Path(__file__).parents[0] / "src/src.cu", "r") as f:
    source = f.read()
mod = SourceModule(
    source,
    options=[
        "-allow-unsupported-compiler",
    ],
)

from .modules import objects
from .modules.camera import Camera
from .modules.scene import Scene
