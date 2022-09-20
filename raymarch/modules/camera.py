import numpy as np
import pycuda.driver as cuda

from raymarch import mod


class Camera:
    def __init__(self, aspect_ratio=4 / 3, fov=np.pi / 3, near=1, far=20):
        self.aspect_ratio = aspect_ratio
        self.fov = fov
        self.near = near
        self.far = np.float32(far)

        xlim = near * np.tan(fov / 2)
        ylim = -xlim / aspect_ratio
        x = np.linspace(-xlim, xlim, 640)
        y = np.linspace(-ylim, ylim, 480)
        pixelgrid = np.dstack(np.meshgrid(x, y, near)).astype(np.float32)

        self.pixelgrid_gpu = cuda.mem_alloc(pixelgrid.nbytes)
        cuda.memcpy_htod(self.pixelgrid_gpu, pixelgrid)

        self.surface = np.zeros((480, 640, 3)).astype(np.float32)
        self.surface_gpu = cuda.mem_alloc(self.surface.nbytes)

        self.clearMatrix()
        self.R_gpu = cuda.mem_alloc(9 * 4)
        self.T_gpu = cuda.mem_alloc(3 * 4)

        self._render = mod.get_function("render")

    def setMatrix(self, transformation):
        self.R = transformation.R
        self.T = transformation.T

    def clearMatrix(self):
        self.R = np.eye(3).astype(np.float32)
        self.T = np.zeros((3, 1)).astype(np.float32)

    def render(self, objects_buffer, s, p, light_sources, l):
        cuda.memcpy_htod(self.R_gpu, self.R)
        cuda.memcpy_htod(self.T_gpu, self.T)
        self._render(
            self.pixelgrid_gpu,
            self.surface_gpu,
            self.R_gpu,
            self.T_gpu,
            objects_buffer,
            np.int32(s),
            np.int32(p),
            self.far,
            np.int32(1),
            light_sources,
            np.int32(l),
            block=(32, 32, 1),
            grid=(20, 15, 1),
        )
        self.clearMatrix()
        cuda.memcpy_dtoh(self.surface, self.surface_gpu)
        return self.surface
