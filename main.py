import cv2
import numpy as np

from renderer import Renderer
from subjects import Sphere, Plane
from source import PointSource

subjects = [
	Sphere(np.array([-0.5, -0.5, 7]), 1., 1.),
	Sphere(np.array([0, 0, 7]), 1., 1.),
	Sphere(np.array([1.732/2, -0.2, 7]), 1., 1.)
	# Plane(np.array([0, -1, 0]), np.array([0, 0, 7]))
	]
sources = [
	PointSource(np.array([10., -10., -10.]), 0.75),
	PointSource(np.array([-10., -10., -5.]), 1.)
	]
renderer = Renderer(
	ASPECT_RATIO=4/3.,
	FIELD_OF_VIEW=np.pi/3,
	NEAR_CLIPPING_PLANE=1.,
	FAR_CLIPPING_PLANE=10.,
	Width=640,
	Height=480,
	epsilon=1e-3,
	useGPU=True
	)
renderer.setElements(
	subjects,
	sources,
	minFunction="""
		__device__ float dmin(float a, float b, float k=0.2) {
		    float h = max(k-abs(a-b), 0.0)/k;
    		return (min(a, b) - h*h*k*(1.0/4.0));
		}
	"""
	)
renderer.compile()

import time

while True:
	t1 = time.time()
	frame = (255*renderer.render()).astype(np.uint8)
	cv2.imshow('disp', frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	print((time.time() - t1))
