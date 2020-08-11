import numpy as np

class Sphere():

	def __init__(self, center, radius, albedo):
		self.center = center
		self.radius = radius
		self.albedo = albedo

	def lambertian(self, point, sources):
		vec1 = (point - self.center) / np.linalg.norm(point - self.center)
		radiance = 0.
		for source in sources:
			vec2 = (source.position - point) / np.linalg.norm(source.position - point)
			illuminance = max(np.dot(vec1, vec2) / np.pi, 0.)
			radiance += illuminance * source.intensity

		return min(radiance, 1.)

	def maxDistance(self, point: np.ndarray):
		return np.linalg.norm(point - self.center) - self.radius


# Error
# Error
# Error
class Plane():

	def __init__(self, normal, point):
		self.normal = normal / np.linalg.norm(normal)
		self.point = point

	def lambertian(self, point, sources):
		vec1 = self.normal
		radiance = 0.
		for source in sources:
			vec2 = (source.position - point) / np.linalg.norm(source.position - point)
			illuminance = max(np.dot(vec1, vec2) / np.pi, 0.)
			radiance += illuminance * source.intensity

		return min(radiance, 1.)

	def maxDistance(self, point: np.ndarray):
		return np.dot(point - self.point, self.normal)
		