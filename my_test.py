import numpy as np

import cv2

img_name = 'test_green.jpg'
img1 = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), 1) ############
img2 = cv2.imread(img_name)
print(img1)
print(img2)
