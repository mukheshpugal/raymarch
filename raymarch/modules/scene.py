from typing import Dict

import pycuda.driver as cuda

from .objects import Object


class Scene:
    def __init__(self, cameras: Dict, objects: Dict):
        self.cameras = cameras
        self.objects = objects
        self.objects_buffer = None
        self.sources_buffer = None

    def pack_objects(self):
        array, self.s, self.p = Object.pack_objects()
        if self.objects_buffer:
            self.objects_buffer.free()
        self.objects_buffer = cuda.mem_alloc(array.nbytes)
        cuda.memcpy_htod(self.objects_buffer, array)

    def pack_sources(self):
        array, self.l = Object.pack_sources()
        if self.sources_buffer:
            self.sources_buffer.free()
        self.sources_buffer = cuda.mem_alloc(array.nbytes)
        cuda.memcpy_htod(self.sources_buffer, array)

    def render(self):
        if Object.oflag:
            self.pack_objects()
            Object.oflag = False
        if Object.lflag:
            self.pack_sources()
            Object.lflag = False
        return {
            name: camera.render(self.objects_buffer, self.s, self.p, self.sources_buffer, self.l)
            for name, camera in self.cameras.items()
        }
