import cv2
import numpy as np
import argparse
import math
from matplotlib import pyplot as plt
import time


#To get path of image file
''''
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True, help = "Path to the image")
args = vars(ap.parse_args())

img_orig = cv2.imread(args["image"])
'''
start_time = time.time()

img_orig = cv2.imread('cat.jpg',0)
base = cv2.resize(img_orig,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)

#  sift parameters
nScales = 3
nOctaves = 3
k = math.pow(2,1./nScales) # CHANGED FROM SQRT(2) TO 2^1/NSCALES
sigma = 1.6
sigma_init = 0.5
sigma_base = math.sqrt(sigma*sigma - sigma_init*sigma_init*4)
contrastThreshold = 0.04
curvatureThreshold = 10

sig = [0]
sig[0] = sigma_base
sig.append(math.sqrt(k*k - 1)*sigma)

for i in range(2,nScales+3):
    sig_next = k*sig[i-1]
    sig.append(sig_next)

#  building gaussian pyramid
gaussian_pyramid = [[0 for x in range(nScales + 3)]
                    for y in range(nOctaves)]

for i in range(nOctaves):
    for j in range(nScales+3):
        if i==0 and j==0:
            gaussian_pyramid[i][j] = cv2.GaussianBlur(base,(0,0),sig[j])
        elif i>0 and j==0:
            gaussian_pyramid[i][j] = cv2.resize(gaussian_pyramid[i-1][nScales], None, fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
        elif j>0:
            gaussian_pyramid[i][j] = cv2.GaussianBlur(gaussian_pyramid[i][j-1],(0,0),sig[j])

# The difference of gaussians image pyramid
dog_pyramid = [[0 for x in range(nScales + 2)]
               for y in range(nOctaves)]
# Create the DOG pyramid by taking successive diff of the gaussian pyramid
for i in range(nOctaves):
    for j in range(nScales + 2):
        dog_pyramid[i][j] = cv2.subtract(gaussian_pyramid[i][j+1], gaussian_pyramid[i][j])

# Checks the 26 neighbours of the point (i,j,x,y)
# Returns 1 if it is a maxima , 0 otherwise

def check_max(i, j, x, y):

    threshold = 0.5 * 0.04 * 255 / nScales

    if np.abs(dog_pyramid[i][j][x][y])<=threshold or dog_pyramid[i][j][x][y]<=0:
        return 0

    for k in range(-1, 2):
        for a in range(-1, 2):
            for b in range(-1, 2):
                if (not (k == 0 and a == 0 and b == 0)):
                    if (dog_pyramid[i][j][x][y] <= dog_pyramid[i][j + k][x + a][y + b]):
                        return 0
    return 1


# Checks the 26 neighbours of the point (i,j,x,y)
# Returns 1 if it is a minima , 0 otherwise
def check_min(i, j, x, y):
    threshold = 255*(1 - (0.5 * 0.04 / nScales)) #  CHANGED

    for k in range(-1, 2):
        for a in range(-1, 2):
            for b in range(-1, 2):
                if not (k==0 and a==0 and b==0):
                    if (dog_pyramid[i][j][x][y] >= dog_pyramid[i][j + k][x + a][y + b]):
                        return 0

    if np.abs(dog_pyramid[i][j][x][y])>=threshold or dog_pyramid[i][j][x][y]<=0: #  CHANGED
        return 0

    return 1



# Create a list that stores 3D gradients for each octave
# Hence the first dimension corresponds to an octave
#grad_voxel = [[0 for i in range(nScales)] for j in range(nOctaves)]
grad_voxel_img = [[0 for i in range(nScales)] for j in range(nOctaves)]

# Magnitude and angle for the 3D gradient, stored for each octave
grad_mag = [0 for i in range(nOctaves)]
grad_angle = [0 for i in range(nOctaves)]

# Create the 3D gradients
for i in range(nOctaves):  # 0 is along sigma, 1 for y and 2 for x
    #grad_voxel[i][2], grad_voxel[i][1], grad_voxel[i][0] = np.gradient(dog_pyramid[i])
    # Now 0 has x, 1 has y, and 2 has sigma
    grad_voxel_img[i][2], grad_voxel_img[i][0], grad_voxel_img[i][1] = np.gradient(gaussian_pyramid[i])
    # Now 0 has x, 1 has y, and 2 has sigma
    grad_mag[i] = np.sqrt(np.square(grad_voxel_img[i][0]) + np.square(grad_voxel_img[i][1]))
    grad_angle[i] = np.arctan2(grad_voxel_img[i][1], grad_voxel_img[i][0])
    grad_angle[i] *= 180. / np.pi  # Radians to degrees
    grad_angle[i] = grad_angle[i] + 180.0# [-180, 180] -> [0, 360]


# Lists to store extrema
# Each entry is (i,j,x,y,v). Here v is the DOG value at that point
extrema = []
refined_extrema = []

# Contains refined_extrema along with the angle
keypoints = []

no_of_bins = 10
div = 360 / no_of_bins
window = 2


# Checks if the extrema (i,j,x,y) is to be retained or not
# Appends it to refined_extrema if found to be acceptable
# Returns without any modifications otherwise
def refine_extrema(i, j, x, y):
    if (x < window or x > gaussian_pyramid[i][j].shape[0] - window - 1):
        return
    if (y < window or y > gaussian_pyramid[i][j].shape[1] - window - 1):
        return
    # Gradient at (i,j,x,y)
    grad = [0 for p in range(3)]
    # Hessian at (i,j,x,y)
    hessian = [[0 for p in range(3)] for q in range(3)]

    grad[0] = (1 / 255.0) * (float(dog_pyramid[i][j][x + 1][y]) - float(dog_pyramid[i][j][x - 1][y])) * 0.5
    grad[1] = (1 / 255.0) * (float(dog_pyramid[i][j][x][y + 1]) - float(dog_pyramid[i][j][x][y - 1])) * 0.5
    grad[2] = (1 / 255.0) * (float(dog_pyramid[i][j + 1][x][y]) - float(dog_pyramid[i][j - 1][x][y])) * 0.5
    hessian[0][0] = (1 / 255.0) * (float(dog_pyramid[i][j][x + 1][y]) +
                                   float(dog_pyramid[i][j][x - 1][y]) - 2 * float(dog_pyramid[i][j][x][y]))
    hessian[1][1] = (1 / 255.0) * (float(dog_pyramid[i][j][x][y + 1]) +
                                   float(dog_pyramid[i][j][x][y - 1]) - 2 * float(dog_pyramid[i][j][x][y]))
    hessian[2][2] = (1 / 255.0) * (float(dog_pyramid[i][j + 1][x][y]) +
                                   float(dog_pyramid[i][j - 1][x][y]) - 2 * float(dog_pyramid[i][j][x][y]))
    hessian[0][1] = (1 / 255.0) * (float(dog_pyramid[i][j][x + 1][y + 1]) + float(dog_pyramid[i][j][x - 1]
                                                                                  [y - 1]) - float(
        dog_pyramid[i][j][x + 1][y - 1]) - float(dog_pyramid[i][j][x - 1][y + 1])) * 0.25
    hessian[1][0] = hessian[0][1]
    hessian[0][2] = (1 / 255.0) * (float(dog_pyramid[i][j + 1][x + 1][y]) + float(dog_pyramid[i][j - 1]
                                                                                  [x - 1][y]) - float(
        dog_pyramid[i][j - 1][x + 1][y]) - float(dog_pyramid[i][j + 1][x - 1][y])) * 0.25
    hessian[2][0] = hessian[0][2]
    hessian[1][2] = (1 / 255.0) * (float(dog_pyramid[i][j + 1][x][y + 1]) + float(dog_pyramid[i][j - 1]
                                                                                  [x][y - 1]) - float(
        dog_pyramid[i][j - 1][x][y + 1]) - float(dog_pyramid[i][j + 1][x][y - 1])) * 0.25
    hessian[2][1] = hessian[1][2]

    det_check = np.linalg.det(hessian)
    if (det_check == 0):  # Numeric stability check
        return

    # Equations assume pixel values in 0 to 1
    # Hence each grad and hessian should both be divided by 255.
    # But delta has to again be upscaled as x,y etc are stored in [0, 255]
    # These scalings are equivalent to the following
    delta = -1.0 * np.dot(np.linalg.inv(hessian), grad)
    #CHANGED 10 TO 0.5
    if delta[0]>0.5 or delta[1]>0.5 or delta[2]>0.5:  # Numeric stability check
        return

    newval = dog_pyramid[i][j][x][y] / 255.0 + 0.5 * np.dot(grad, delta)

    if (np.abs(newval) < contrastThreshold/nScales):
        return

    trace = hessian[0][0] + hessian[1][1]
    det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0]
    if (det <= 0):
        return

    thr = 10.0  # Curvature threshold
    if (trace * trace * thr >= (thr + 1) * (thr + 1) * det):
        return

    ret = [i, j + delta[2], x + delta[0], y + delta[1], newval]

    #if (np.abs(delta[2]) >= 1):  # Numerical stability check (already checked)
     #   return

    refined_extrema.append(ret)


# Iterate over all pixels, in every image, over all scales
# in every octave and store all the interest points
for i in range(nOctaves):
    for j in range(1, nScales + 1):
        for x in range(1, dog_pyramid[i][j].shape[0] - 1):
            for y in range(1, dog_pyramid[i][j].shape[1] - 1):
                is_max = check_max(i, j, x, y)
                if is_max:
                    #extrema.append([i, j, x, y, dog_pyramid[i][j][x][y]])
                    refine_extrema(i, j, x, y)
                else:
                    is_min = check_min(i, j, x, y)
                    if is_min:
                        #extrema.append([i, j, x, y, dog_pyramid[i][j][x][y]])
                        refine_extrema(i, j, x, y)

# e^(-(x^2 + y^2)/(2*var^2))
def gaussian(i, j, var):
    return math.exp(math.pow((i * 1.0 / var),2) / (-2.0) + math.pow((j * 1.0 / var),2) / (-2.0))

for i in range(len(refined_extrema)):
    w = refined_extrema[i]
    point = (np.rint(w)).astype(int)  # (i,j,x,y) rounded to closest int
    # var = 1.5 that of curr scale
    var = 1.5 * sigma * (2 ** point[0]) * (k ** point[1])
    histogram = np.zeros(no_of_bins)
    for j in range(-window, window + 1):
        for k in range(-window, window + 1):
            t = np.int(np.floor(grad_angle[point[0]][point[1]][point[2] + j][point[3] + k] / div))
            histogram[t] += gaussian(j, k, var) * grad_mag[point[0]][point[1]][point[2] + j][point[3] + k]

    # If only one keypoint is desired at a point
    r = np.argmax(histogram)
    angle = r * div + div / 2.0
    w = refined_extrema[i]
    w.append(angle)
    keypoints.append(w)

    # For multiple keypoints at the same point
    m = np.amax(histogram)
    p = m * 0.8
    for r in range(len(histogram)):
        if (histogram[r] >= p):
            angle = r * div + div / 2.0

            # Parabolic interpolation
            # Replace above expression of angle for more accuracy
            # x = np.array([r-1,r,r+1])
            # x = x*div + div/2.0
            # y = np.array([histogram[r-1], histogram[r], histogram[r+1]])
            # coeff = np.polyfit(x,y,2)
            # angle = point[2] - 0.25*np.square(coeff[1])/coeff[0] # c - b^2/4a
            # keypoints.append(refined_extrema[i].append(angle))
            w = refined_extrema[i]
            w.append(angle)
            keypoints.append(w)

block_dim = 2  # => 4 x 4 grid with each element being a 4 x 4 pixel grid
no_of_ang = 8  # Angles are binned into 8 directions
bins = 360 / no_of_ang
sift_descriptors = []


def calc_descriptor(index):
    point = (np.rint(keypoints[index])).astype(int)
    i = point[0]
    j = point[1]
    x = point[2]
    y = point[3]
    mag = point[4]
    ang = point[5]
    if (x < 8 or x > gaussian_pyramid[i][j].shape[0] - 8):
        return
    if (y < 8 or y > gaussian_pyramid[i][j].shape[1] - 8):
        return

    var = block_dim * 4
    descriptor = [i, j, x, y]
    for p in range(-block_dim, block_dim):
        for q in range(-block_dim, block_dim):
            histogram = np.zeros(no_of_ang)
            for r in range(4):
                for s in range(4):
                    a = ((grad_angle[i][j][x + p * 4 + r][y + q * 4 + s] - ang) % 360) / bins
                    t = np.int(np.floor(a))
                    histogram[t] += gaussian(j, k, var) * grad_mag[i][j][x + p * 4 + r][y + q * 4 + s]
            descriptor = np.concatenate((descriptor, histogram))
    sift_descriptors.append(descriptor)


for index in range(len(keypoints)):
    calc_descriptor(index)
print("--- %s seconds ---" % (time.time() - start_time))
# Display keypoint locations on the original image

plt.imshow(base, cmap='gray')
for index in range(len(refined_extrema)):
    i = refined_extrema[index][0]
    x = refined_extrema[index][2] * (2 ** i)
    y = refined_extrema[index][3] * (2 ** i)
    plt.scatter(x=y, y=x, c='r', s=10)  # x and y are interchanged in plotting
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
print str(len(keypoints))
gray = img_orig
sift = cv2.SIFT()
kp = sift.detect(gray, None)
img1 = cv2.drawKeypoints(
    gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img1)
plt.show()
print str(len(kp))


