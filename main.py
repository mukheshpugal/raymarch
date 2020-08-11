import cv2
import numpy as np

from renderer import Renderer
from subjects import Sphere, Plane
from source import PointSource

subjects = [
	Sphere(np.array([0, -1, 7]), 1., 1.),
	Plane(np.array([0, -1, 0]), np.array([0, 0, 7]))

	]
sources = [
	PointSource(np.array([10., -10., -10.]), 0.75),
	PointSource(np.array([-10., -10., -5.]), 1.)
	]
renderer = Renderer(
	ASPECT_RATIO=1.5,
	FIELD_OF_VIEW=np.pi/3,
	NEAR_CLIPPING_PLANE=1.,
	FAR_CLIPPING_PLANE=10.,
	Width=600,
	Height=400,
	epsilon=1e-3,
	useGPU=True
	)
renderer.setElements(subjects, sources)

while True:
	frame = (255*renderer.render()).astype(np.uint8)
	cv2.imshow('disp', frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
