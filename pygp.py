import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

mod = SourceModule("""
	class Vec3 {
		public:
		float x, y, z;
		__device__ Vec3(float xin, float yin, float zin) {
			x = xin;
			y = yin;
			z = zin;
		}
		__device__ float mag() {
			return (sqrtf(x*x + y*y + z*z));
		}
		__device__ static float dot(Vec3 v1, Vec3 v2) {
			return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
		}
		__device__ Vec3 normalize() {
			return (Vec3(x, y, z) / mag());
		}
		__device__ Vec3 operator + (Vec3 &vec) {
			return (Vec3(x+vec.x, y+vec.y, z+vec.z));
		}
		__device__ Vec3 operator - (Vec3 &vec) {
			return (Vec3(x-vec.x, y-vec.y, z-vec.z));
		}
		__device__ Vec3 operator / (float s) {
			return (Vec3(x/s, y/s, z/s));
		}
		__device__ Vec3 operator * (float s) {
			return (Vec3(s*x, s*y, s*z));
		}
	};

	__device__ float getDistance(Vec3 &p) {
		float d = (p - Vec3(0.0, 0.0, 7.0)).mag();
		return (d - 1);
	}

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

march = mod.get_function('march')
shade = mod.get_function('shade')
isSubject_ref = np.zeros((480, 640)).astype(np.bool)
buffer_ref = np.zeros((480, 640)).astype(np.float32)

ASPECT_RATIO = 4/3.;
FIELD_OF_VIEW = np.pi/3;
xlim = 1. * np.tan(FIELD_OF_VIEW / 2.)
ylim = xlim / ASPECT_RATIO
x = np.linspace(-xlim, xlim, 640)
y = np.linspace(-ylim, ylim, 480)
MESH = np.dstack(np.meshgrid(x, y, 1.)).astype(np.float32)

mesh_gpu = cuda.mem_alloc(MESH.nbytes)
isSubject_gpu = cuda.mem_alloc(MESH.nbytes//12)
points_gpu = cuda.mem_alloc(MESH.nbytes)
display_gpu = cuda.mem_alloc(buffer_ref.nbytes)

cuda.memcpy_htod(mesh_gpu, MESH)
buffer_final = buffer_ref

def getFrame(x, y, z):
	march(mesh_gpu, isSubject_gpu, points_gpu, block=(32, 32, 1), grid=(20, 15, 1))
	shade(points_gpu, isSubject_gpu, display_gpu, np.float32(x), np.float32(y), np.float32(z), block=(32, 32, 1), grid=(20, 15, 1))
	cuda.memcpy_dtoh(buffer_final, display_gpu)
	frame = (255*buffer_final).astype(np.uint8)
	return frame

import cv2
import time

angle1 = 0.
angle2 = 0.
while True:
	t1 = time.time()
	frame = getFrame(10*np.sin(angle1), 10*np.sin(angle2), 10*np.cos(angle1)*np.cos(angle2))
	cv2.imshow('disp', frame)
	angle1 += 0.01
	angle2 += 0.00834
	if angle1 > 2*np.pi:
		angle1 -= 2*np.pi
	if angle2 > 2*np.pi:
		angle2 -= 2*np.pi
	if cv2.waitKey(1) & 0xff == ord('q'): break
	print((time.time() - t1)**-1)