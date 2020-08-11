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
		self.MESH = np.dstack(np.meshgrid(x, y, NEAR_CLIPPING_PLANE)).astype(np.float32)
		if self.useGPU:
			self.MESH_gpu = cuda.mem_alloc(self.MESH.nbytes)
			cuda.memcpy_htod(self.MESH_gpu, self.MESH)
			self.subjectMask_gpu = cuda.mem_alloc(self.MESH.nbytes//12)
			self.projectedPoints_gpu = cuda.mem_alloc(self.MESH.nbytes)
			self.display_buffer = np.zeros((480, 640)).astype(np.float32)
			self.buffer_gpu = cuda.mem_alloc(self.display_buffer.nbytes)


	def marchCPU(self, point):
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
				__device__ float dmin(float a, float b) {
					if (a < b) return a;
					return b;
				}
			"""):
		self.elementsSet = True
		self.distanceFunc_gpu = """
			__device__ float getDistance(Vec3 &p) {
				float d = (p - Vec3(0.0, 0.0, 7.0)).mag();
				return (d - 1);
			}"""
		if self.useGPU:
			distanceFunc_gpu = minFunction + '__device__ float getDistance(Vec3 p) {float minDist = '+str(self.FAR_CLIPPING_PLANE)+';'
			for subject in subjects:
				distanceFunc_gpu += 'minDist = dmin(minDist, '+subject.getCudaFunc()+');'
			distanceFunc_gpu += 'return minDist;}'
			self.distanceFunc_gpu = distanceFunc_gpu

		else:
			self.subjects = subjects
			self.sources = sources

	def compile(self):
		with open('vec3.cu', 'r') as f:
			vec3Cuda = f.read()
		mod = SourceModule(vec3Cuda + self.distanceFunc_gpu + """
			__global__ void march(float *mesh, bool *isSubject, float *points)
			{
				int subject_id = threadIdx.x + 32*blockIdx.x + 640*(threadIdx.y + 32*blockIdx.y);
				int mesh_id = 3*subject_id;
				Vec3 currentPoint(mesh[mesh_id], mesh[mesh_id+1], mesh[mesh_id+2]);
				Vec3 direction = currentPoint.normalize();
				float distance = 0.0;
				while (distance < 10) {
					float marchDistance = getDistance(currentPoint);
					if (marchDistance < 0.001) {
						isSubject[subject_id] = true;
						points[mesh_id] = currentPoint.x;
						points[mesh_id+1] = currentPoint.y;
						points[mesh_id+2] = currentPoint.z;
						break;
					}
					distance += marchDistance;
					currentPoint = direction * distance;
				}
			}
			__global__ void shade(float *points, bool *isSubject, float *display, float sourcex, float sourcey, float sourcez) {
				int subject_id = threadIdx.x + 32*blockIdx.x + 640*(threadIdx.y + 32*blockIdx.y);
				int mesh_id = 3*subject_id;
				if (isSubject[subject_id]) {
					float mainx = points[mesh_id];
					float mainy = points[mesh_id+1];
					float mainz = points[mesh_id+2];
					float maindist = getDistance(Vec3(mainx, mainy, mainz));
					float normx = getDistance(Vec3(mainx+0.001, mainy, mainz)) - maindist;
					float normy = getDistance(Vec3(mainx, mainy+0.001, mainz)) - maindist;
					float normz = getDistance(Vec3(mainx, mainy, mainz+0.001)) - maindist;

					Vec3 unitNorm = Vec3(normx, normy, normz).normalize();
					Vec3 unitSource = Vec3(sourcex, sourcey, sourcez).normalize();

					float intensity = Vec3::dot(unitNorm, unitSource);

					if (intensity < 0)
						intensity = 0;
					display[subject_id] = intensity;
				}
				else
					display[subject_id] = 0;
			}
			""")
		self.marchGPU = mod.get_function('march')
		self.shadeGPU = mod.get_function('shade')

	def render(self):
		if not self.elementsSet:
			raise Exception('Call Renderer.setElements() atleast once before calling render')
		if self.useGPU:
			self.marchGPU(self.MESH_gpu, self.subjectMask_gpu, self.projectedPoints_gpu, block=(32, 32, 1), grid=(20, 15, 1))
			self.shadeGPU(self.projectedPoints_gpu, self.subjectMask_gpu, self.buffer_gpu, np.float32(10.), np.float32(-10.), np.float32(-10.), block=(32, 32, 1), grid=(20, 15, 1))
			cuda.memcpy_dtoh(self.display_buffer, self.buffer_gpu)
			return self.display_buffer.copy()
		return np.apply_along_axis(self.marchCPU, -1, self.MESH)
