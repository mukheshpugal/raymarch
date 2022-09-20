import numpy as np


class Object:
    s = []
    p = []
    l = []
    oflag = True
    lflag = True

    @classmethod
    def pack_objects(self):
        return np.array(self.s + self.p).astype(np.float32), len(self.s), len(self.p)

    @classmethod
    def pack_sources(self):
        return np.array(self.l).astype(np.float32), len(self.l)


class Sphere(Object):
    def __init__(self, center, radius, color):
        self.data = np.r_[center, radius, color]
        self.index = len(self.s)
        self.s.append(self.data)

    def modify(self, center, radius, color):
        self.data = np.r_[center, radius, color]
        self.s[self.index] = self.data
        Object.oflag = True


class Plane(Object):
    def __init__(self, n, h, color):
        self.data = np.r_[n, h, color]
        self.index = len(self.p)
        self.p.append(self.data)

    def modify(self, n, h, color):
        self.data = np.r_[n, h, color]
        self.p[self.index] = self.data
        Object.oflag = True


class PointLight(Object):
    def __init__(self, pos):
        self.data = np.array(pos)
        self.index = len(self.l)
        self.l.append(self.data)

    def modify(self, pos):
        self.data = np.array(pos)
        self.l[self.index] = self.data
        Object.lflag = True
