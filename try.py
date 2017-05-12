import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('tree.jpeg', 0)

blur1 = cv2.GaussianBlur(img, (3, 3), 1)
blur2 = cv2.GaussianBlur(img, (3, 3), 256)

plt.imshow(blur1, cmap='gray')
plt.show()
plt.imshow(blur2, cmap='gray')
plt.show()
