import numpy as np
GPU_SUPPORT = True
try:
	import pycuda.driver as cuda
	import pycuda.autoinit
	from pycuda.compiler import SourceModule
except ModuleNotFoundError:
	print('Cuda unavailable')
	GPU_SUPPORT = False


class Renderer():
	def __init__(
			self,
			ASPECT_RATIO,
			FIELD_OF_VIEW,
			NEAR_CLIPPING_PLANE,
			FAR_CLIPPING_PLANE,
			Width:int,
			Height:int,
			epsilon,
			useGPU=False
			):
		self.ASPECT_RATIO = ASPECT_RATIO;
		self.FIELD_OF_VIEW = FIELD_OF_VIEW;
		self.NEAR_CLIPPING_PLANE = NEAR_CLIPPING_PLANE;
		self.FAR_CLIPPING_PLANE = FAR_CLIPPING_PLANE;
		self.epsilon = epsilon
		self.useGPU = useGPU and GPU_SUPPORT
		self.elementsSet = False
		xlim = float(NEAR_CLIPPING_PLANE) * np.tan(FIELD_OF_VIEW / 2.)
		ylim = xlim / ASPECT_RATIO
		x = np.linspace(-xlim, xlim, Width)
		y = np.linspace(-ylim, ylim, Height)
		self.MESH = np.dstack(np.meshgrid(x, y, NEAR_CLIPPING_PLANE))
		if self.useGPU:
			self.MESH_gpu = cuda.mem_alloc(self.MESH.nbytes)
			self.subjectMask_gpu = cuda.mem_alloc(self.MESH.nbytes//12)
			self.projectedPoints_gpu = cuda.mem_alloc(self.MESH.nbytes)
			self.display_buffer = np.zeros((480, 640)).astype(np.float32)


	def projectCPU(self, point):
		intensity = 0.
		direction = point / np.linalg.norm(point)
		distance = 0.
		while distance < self.FAR_CLIPPING_PLANE:
			currentPoint = distance * direction
			distances = [sub.maxDistance(currentPoint) for sub in self.subjects]
			maxArg = np.argmin(distances)
			marchDist = distances[maxArg]
			if marchDist < self.epsilon:
				return self.subjects[maxArg].lambertian(currentPoint, self.sources)
			distance += marchDist
		return intensity

	def setElements(
			self,
			subjects,
			sources,
			minFunction = """
				__device__ float min(float a, float b) {
					if (a < b) return a;
					return b;
				}
			"""):
		self.elementsSet = True
		if self.useGPU:
			distanceFunc_gpu = minFunction + '__device__ float getDistance(float x, float y, float z) {float minDist = '+str(self.FAR_CLIPPING_PLANE)+';'
			for subject in subjects:
				distanceFunc_gpu += 'minDist = min(minDist, '+subject.getFuncCpp()+');'
			distanceFunc_gpu += 'return minDist;}'
			self.distanceFunc_gpu = distanceFunc_gpu

		else:
			self.subjects = subjects
			self.sources = sources

	def render(self):
		if not self.elementsSet:
			raise Exception('Call Renderer.setElements() atleast once before calling render')
		if self.useGPU:
			pass
		else:
			return np.apply_along_axis(self.projectCPU, -1, self.MESH)
