import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('home.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(
    gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg', img)

plt.imshow(img)
plt.show()
