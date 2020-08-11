import pygame
import sys
import numpy as np
from pygame.gfxdraw import pixel
from subject import Sphere
from tqdm import tqdm

WIDTH = 600
HEIGHT = 400

FIELD_OF_VIEW = np.pi / 2
ASPECT_RATIO = WIDTH / float(HEIGHT)
NEAR_CLIPPING_PLANE = 3.0
FAR_CLIPPING_PLANE = 10.0
CAMERA_CENTER = np.array([0, 0, 0])
EPSILON = 1e-1
MESH = np.array([[[(i - WIDTH / 2) / 100.0, (j - HEIGHT / 2) / 100.0, NEAR_CLIPPING_PLANE] for j in range(HEIGHT)] for i in range(WIDTH)])
MESH = np.array([[[(i - WIDTH / 2) * 2 * NEAR_CLIPPING_PLANE * np.tan(FIELD_OF_VIEW / 2) / WIDTH, (j - HEIGHT / 2) * 2 * NEAR_CLIPPING_PLANE * np.tan(FIELD_OF_VIEW / 2) / (HEIGHT * ASPECT_RATIO), NEAR_CLIPPING_PLANE] for j in range(HEIGHT)] for i in range(WIDTH)])

subject = Sphere(np.array([0, 0, 5]), 1.0)
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
done = False
clock = pygame.time.Clock()
FRAME_RATE = 60

def setPixel(color, i, j):
	pixel(SCREEN, i, j, (color, color, color))

def march(p1, p2):
	slopes = (p2 - p1) / np.linalg.norm(p2 - p1)
	rayPath = NEAR_CLIPPING_PLANE
	currentPoint = p2
	pixelIntensity = 0.25
	prevDist = FAR_CLIPPING_PLANE
	while rayPath < FAR_CLIPPING_PLANE:
		maxDist = subject.maxDistance(currentPoint)
		rayPath += maxDist
		currentPoint = p2 + rayPath * slopes
		# if maxDist > prevDist:
		# 	break
		if maxDist < EPSILON:
			pixelIntensity = 1.0
			break
		# prevDist = maxDist
	return pixelIntensity

def render():
	for i in tqdm(range(WIDTH)):
		for j in range(HEIGHT):
			color = 255 * march(CAMERA_CENTER, MESH[i, j])
			# color = 255 * i * j / (WIDTH * HEIGHT)
			setPixel(color, i, j)

while not done:
	render()
	pygame.draw.circle(SCREEN, (255, 0, 0), pygame.mouse.get_pos(), 5)
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True
	pygame.display.flip()
	clock.tick(FRAME_RATE)
