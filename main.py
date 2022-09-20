from time import time

import cv2
import numpy as np

from raymarch import Camera, Scene, objects
from raymarch.utils import transformations as tf

main = Camera()
l1 = objects.PointLight(None)
l2 = objects.PointLight([1, 2, -2])
scene = Scene(
    cameras={
        "main": main,
    },
    objects={
        "s": objects.Sphere([0, 0, 0], 1, [1, 1, 0]),
        "pl": objects.Plane([1, 0, 0], 4, [1, 0, 0]),
        "pr": objects.Plane([-1, 0, 0], 4, [0, 1, 0]),
        "pu": objects.Plane([0, -1, 0], 4, [0, 0, 1]),
        "pd": objects.Plane([0, 1, 0], 4, [0, 1, 1]),
        "pb": objects.Plane([0, 0, -1], 8, [1, 0, 1]),
        "l1": l1,
        "l2": l2,
    },
)

i = 0
while True:
    t1 = time()
    main.setMatrix(tf.Translate([0, 0, -8]))

    l1.modify([-1, 2, -2 + np.sin(i / 100) * 2])
    l2.modify([1, 2, -2 - np.sin(i / 100) * 2])

    frame = (255 * scene.render()["main"]).astype(np.uint8)
    cv2.imshow("disp", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    i += 1
    print(1 / (1e-6 + time() - t1))
