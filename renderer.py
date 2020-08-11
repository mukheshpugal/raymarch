import numpy as np

class Renderer():
	def __init__(self, ASPECT_RATIO, FIELD_OF_VIEW, NEAR_CLIPPING_PLANE, FAR_CLIPPING_PLANE, Width:int, Height:int, epsilon):
		self.ASPECT_RATIO = ASPECT_RATIO;
		self.FIELD_OF_VIEW = FIELD_OF_VIEW;
		self.NEAR_CLIPPING_PLANE = NEAR_CLIPPING_PLANE;
		self.FAR_CLIPPING_PLANE = FAR_CLIPPING_PLANE;
		self.epsilon = epsilon
		xlim = float(NEAR_CLIPPING_PLANE) * np.tan(FIELD_OF_VIEW / 2.)
		ylim = xlim / ASPECT_RATIO
		x = np.linspace(-xlim, xlim, Width)
		y = np.linspace(-ylim, ylim, Height)
		self.MESH = np.dstack(np.meshgrid(x, y, NEAR_CLIPPING_PLANE))

	def project(self, point):
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

	def setElements(self, subjects, sources):
		self.subjects = subjects
		self.sources = sources

	def render(self):
		return np.apply_along_axis(self.project, -1, self.MESH)
