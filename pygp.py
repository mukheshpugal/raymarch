import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

mod = SourceModule("""
	__device__ float getDistance(float x, float y, float z) {
		float d = sqrtf((x-3)*(x-3) + (y-2)*(y-2) + (z-7)*(z-7));
		return (d - 1);
	}

	__global__ void march(float *mesh, bool *isSubject, float *points)
	{
		int subject_id = threadIdx.x + 32*blockIdx.x + 640*(threadIdx.y + 32*blockIdx.y);
		int mesh_id = 3*subject_id;
		float currentx = mesh[mesh_id];
		float currenty = mesh[mesh_id+1];
		float currentz = mesh[mesh_id+2];
		float mag = sqrtf(currentx*currentx + currenty*currenty + currentz*currentz);
		float dirx = currentx/mag;
		float diry = currenty/mag;
		float dirz = currentz/mag;
		float distance = 0.0;
		while (distance < 10) {
			float marchDistance = getDistance(currentx, currenty, currentz);
			if (marchDistance < 0.001) {
				isSubject[subject_id] = true;
				points[mesh_id] = currentx;
				points[mesh_id+1] = currenty;
				points[mesh_id+2] = currentz;
				break;
			}
			distance += marchDistance;
			currentx = distance*dirx;
			currenty = distance*diry;
			currentz = distance*dirz;
		}
	}
	__global__ void shade(float *points, bool *isSubject, float *display, float sourcex, float sourcey, float sourcez) {
		int subject_id = threadIdx.x + 32*blockIdx.x + 640*(threadIdx.y + 32*blockIdx.y);
		int mesh_id = 3*subject_id;
		if (isSubject[subject_id]) {
			float mainx = points[mesh_id];
			float mainy = points[mesh_id+1];
			float mainz = points[mesh_id+2];
			float maindist = getDistance(mainx, mainy, mainz);
			float normx = getDistance(mainx+0.001, mainy, mainz) - maindist;
			float normy = getDistance(mainx, mainy+0.001, mainz) - maindist;
			float normz = getDistance(mainx, mainy, mainz+0.001) - maindist;

			float normmag = sqrtf(normx*normx + normy*normy + normz*normz);
			float sourcemag = sqrtf(sourcex*sourcex + sourcey*sourcey + sourcez*sourcez);

			float intensity = (normx*sourcex + normy*sourcey + normz*sourcez)/(sourcemag*normmag);
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