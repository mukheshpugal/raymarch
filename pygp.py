import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

with open('vec3.cu', 'r') as f:
	vec3Source = f.read()
kernelString = """
	__device__ float sdPlane(Vec3 &p, Vec3 &n, float h)
	{
		// n must be normalized
		return Vec3::dot(p, n) + h;
	}
	__device__ float getDistance(Vec3 &p, float maxDist=10) {
		maxDist = min(maxDist, ((p - Vec3(0.0, 0.0, 8.0)).mag() - 1.5));
		maxDist = min(maxDist, ((p - Vec3(2.0, 2.0, 8.0)).mag() - 1.0));
		maxDist = min(maxDist, ((p - Vec3(-2.0, 2.0, 8.0)).mag() - 1.0));
		maxDist = min(maxDist, ((p - Vec3(2.0, -2.0, 8.0)).mag() - 1.0));
		maxDist = min(maxDist, ((p - Vec3(-2.0, -2.0, 8.0)).mag() - 1.0));

		//maxDist = min(maxDist, sdPlane(p, Vec3(0, 0, -1), 12));
		//maxDist = min(maxDist, sdPlane(p, Vec3(0, -1, 0), 3));
		//maxDist = min(maxDist, sdPlane(p, Vec3(0, 1, 0), 3));
		//maxDist = min(maxDist, sdPlane(p, Vec3(1, 0, 0), 4));
		//maxDist = min(maxDist, sdPlane(p, Vec3(-1, 0, 0), 4));

		return maxDist;
	}

	__device__ Vec3 march(Vec3 &from, Vec3 &to, float maxDistance, bool *hit, float epsilon=0.00001) {
		Vec3 direction = (to - from).normalize();
		float distance = 0.0;
		Vec3 currentPoint = from + direction*distance;
		while (distance < maxDistance) {
			float marchDistance = getDistance(currentPoint);
			if (marchDistance < epsilon) {
				*hit = true;
				return (currentPoint);
			}
			distance += marchDistance;
			currentPoint = from + direction*distance;
		}
		*hit = false;
		return (from + direction*maxDistance);
	}

	__global__ void render(float *mesh, float *display, float sourcex, float sourcey, float sourcez)
	{
		int pixel_id = threadIdx.x + 32*blockIdx.x + 640*(threadIdx.y + 32*blockIdx.y);
		int mesh_id = 3*pixel_id;

		bool hit;
		float ambient = 0.1;
		Vec3 point = march(Vec3(0, 0, 0), Vec3(mesh[mesh_id], mesh[mesh_id+1], mesh[mesh_id+2]), 15, &hit);

		display[pixel_id] = 0;
		if (hit) {
			display[pixel_id] = ambient;
			// compute surface normal
			float maindist = getDistance(point);
			float normx = getDistance(point + Vec3(0.0001, 0, 0)) - maindist;
			float normy = getDistance(point + Vec3(0, 0.0001, 0)) - maindist;
			float normz = getDistance(point + Vec3(0, 0, 0.0001)) - maindist;
			Vec3 normal = Vec3(normx, normy, normz).normalize();

			// for each source
			bool isShadow;
			Vec3 source = Vec3(sourcex, sourcey, sourcez);
			Vec3 startP = (point + normal * 0.001);
			float maxMarchDistance = (source - startP).mag();
			march(startP, source, maxMarchDistance, &isShadow);
			if (!isShadow) {
				float intensityLambertian = Vec3::dot(normal, (source - point).normalize());
				//float intensityBlinnPhong = powf(Vec3::dot(normal, ((point*-1).normalize() + (source - point).normalize()) / 2), 10);
				//float intensity = 0.9*(0.2*intensityLambertian + 0.8*intensityBlinnPhong);
				float intensity = intensityLambertian;
				if (intensity < 0) intensity = 0;
				intensity += ambient;
				if (intensity > 1) intensity = 1;
				display[pixel_id] = intensity;
			}
		}
	}
	"""
mod = SourceModule(vec3Source+kernelString)
render = mod.get_function('render')
buffer_ref = np.zeros((480, 640)).astype(np.float32)

ASPECT_RATIO = 4/3.;
FIELD_OF_VIEW = np.pi/3;
xlim = 1. * np.tan(FIELD_OF_VIEW / 2.)
ylim = -xlim / ASPECT_RATIO
x = np.linspace(-xlim, xlim, 640)
y = np.linspace(-ylim, ylim, 480)
MESH = np.dstack(np.meshgrid(x, y, 1.)).astype(np.float32)

mesh_gpu = cuda.mem_alloc(MESH.nbytes)
display_gpu = cuda.mem_alloc(buffer_ref.nbytes)

cuda.memcpy_htod(mesh_gpu, MESH)

def getFrame(x, y, z):
	render(mesh_gpu, display_gpu, np.float32(x), np.float32(y), np.float32(z), block=(32, 32, 1), grid=(20, 15, 1))
	cuda.memcpy_dtoh(buffer_ref, display_gpu)

	B = (173*buffer_ref).astype(np.uint8)
	G = (52*buffer_ref).astype(np.uint8)
	R = (64*buffer_ref).astype(np.uint8)
	# R = np.zeros_like(B)

	frame = cv2.merge((B, G, R))
	return frame

import cv2
import time

angle1 = 0.0
angle2 = 0.0

while True:
	t1 = time.time()
	# frame = getFrame(10, 0, 4)
	frame = getFrame(np.sin(angle1)*np.cos(angle2), np.cos(angle1), 6 + np.sin(angle1)*np.sin(angle2))
	cv2.imshow('disp', frame)
	angle1 += 0.005
	angle2 += 0.023
	if angle1 >= 2*np.pi:
		angle1 -= 2*np.pi
	if angle2 >= 2*np.pi:
		angle2 -= 2*np.pi
	if cv2.waitKey(1) & 0xff == ord('q'): break
	print((time.time() - t1)**-1)
