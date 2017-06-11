import cv2
import numpy as np
import math
import time
from matplotlib import pyplot as plt

nScales = 3
nOctaves = 3
k = math.pow(2,1./nScales)
sigma = 1.6
sigma_init = 0.5
gaussian_pyramid = [[0 for x in range(nScales + 3)] for y in range(nOctaves)]
dog_pyramid = [[0 for x in range(nScales + 2)]for y in range(nOctaves)]
img_orig = cv2.imread('image.png',0)
base = cv2.resize(img_orig,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)

def findSigma(x):
    global sigma
    if not x==0:
        return math.sqrt(k*k-1)*sigma*(k**(x-1))
    return sigma

def buildGaussianPyramid():
    global nOctaves,nScales,sigma,gaussian_pyramid,dog_pyramid

    for i in range(nOctaves):
        for j in range(nScales + 3):
            if i == 0 and j == 0:
                gaussian_pyramid[i][j] = cv2.GaussianBlur(base, (0, 0), sig[j])
            elif i > 0 and j == 0:
                gaussian_pyramid[i][j] = cv2.resize(gaussian_pyramid[i - 1][nScales], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            elif j > 0:
                gaussian_pyramid[i][j] = cv2.GaussianBlur(gaussian_pyramid[i][j - 1], (0, 0), sig[j])

def buildDoGPyramid():
    global nOctaves,nScales,gaussian_pyramid,dog_pyramid
    for i in xrange(nOctaves):
        for j in  xrange(nScales+2):
            dog_pyramid[i][j] = cv2.subtract(gaussian_pyramid[i][j+1],gaussian_pyramid[i][j])

def check_extrema(i,j):
    iterTable = [[u,v,w] for u in [0,1,-1] for v in [0,1,-1] for w in [0,1,-1]][1:]
    threshold = 0.5 * 0.04 * 255 / nScales
    
    global dog_pyramid
    #here i was @chinmay
    maxmask = np.greater(dog_pyramid[ i][ j][ 1 : (dog_pyramid[i][j].shape[0] - 1), 1 : (dog_pyramid[i][j].shape[1] - 1)],threshold) 
    minmask = np.copy(maxmask)
    
    for state in iterTable :
        maxmask[:,:] = np.logical_and(maxmask,np.greater(dog_pyramid[ i][ j][ 1 : (dog_pyramid[i][j].shape[0] - 1), 1 : (dog_pyramid[i][j].shape[1] - 1)], dog_pyramid[ i][ (j + state[0]) ][ (1 + state[1]) : (dog_pyramid[i][j].shape[0] - 1 + state[1]), (1 + state[2]) : (dog_pyramid[i][j].shape[1] - 1 + state[2])]))
        minmask[:,:] = np.logical_and(minmask,np.less(dog_pyramid[ i][ j][ 1 : (dog_pyramid[i][j].shape[0] - 1), 1 : (dog_pyramid[i][j].shape[1] - 1)], dog_pyramid[ i][ (j + state[0]) ][ (1 + state[1]) : (dog_pyramid[i][j].shape[0] - 1 + state[1]), (1 + state[2]) : (dog_pyramid[i][j].shape[1] - 1 + state[2])]))

    return (np.argwhere(maxmask) + 1, np.argwhere(maxmask) + 1)

# refine extremas using taylor expansion of the DoG pyramid
def refine_extrema(i, j, x, y):
    global dog_pyramid

    window = 2
    ret = []
    if x<window or x>dog_pyramid[i][j].shape[0]-1-window :
        return False, ret
    if y<window or y>dog_pyramid[i][j].shape[1]-1-window :
        return False, ret

    # Gradient at (i,j,x,y), dD/dx
    grad = [0 for m in xrange(3)]
    # Hessian at (i,j,x,y), d^2D/dx^2
    hessian = [[0 for p in range(3)] for q in range(3)]

    grad[0] = (1.0 / 255.0) * (float(dog_pyramid[i][j][x + 1][y]) - float(dog_pyramid[i][j][x - 1][y])) * 0.5
    grad[1] = (1.0 / 255.0) * (float(dog_pyramid[i][j][x][y + 1]) - float(dog_pyramid[i][j][x][y - 1])) * 0.5
    grad[2] = (1.0 / 255.0) * (float(dog_pyramid[i][j + 1][x][y]) - float(dog_pyramid[i][j - 1][x][y])) * 0.5

    hessian[0][0] = (1.0 / 255.0) * (float(dog_pyramid[i][j][x + 1][y]) +
                                   float(dog_pyramid[i][j][x - 1][y]) - 2 * float(dog_pyramid[i][j][x][y]))
    hessian[1][1] = (1.0 / 255.0) * (float(dog_pyramid[i][j][x][y + 1]) +
                                   float(dog_pyramid[i][j][x][y - 1]) - 2 * float(dog_pyramid[i][j][x][y]))
    hessian[2][2] = (1.0 / 255.0) * (float(dog_pyramid[i][j + 1][x][y]) +
                                   float(dog_pyramid[i][j - 1][x][y]) - 2 * float(dog_pyramid[i][j][x][y]))
    hessian[0][1] = (1.0 / 255.0) * (float(dog_pyramid[i][j][x + 1][y + 1]) + float(dog_pyramid[i][j][x - 1][y - 1]) -
                                     float(dog_pyramid[i][j][x + 1][y - 1]) - float(dog_pyramid[i][j][x - 1][y + 1])) * 0.25
    hessian[1][0] = hessian[0][1]
    hessian[0][2] = (1.0 / 255.0) * (float(dog_pyramid[i][j + 1][x + 1][y]) + float(dog_pyramid[i][j - 1][x - 1][y]) -
                                     float(dog_pyramid[i][j - 1][x + 1][y]) - float(dog_pyramid[i][j + 1][x - 1][y])) * 0.25
    hessian[2][0] = hessian[0][2]
    hessian[1][2] = (1.0 / 255.0) * (float(dog_pyramid[i][j + 1][x][y + 1]) + float(dog_pyramid[i][j - 1][x][y - 1]) -
                                     float(dog_pyramid[i][j - 1][x][y + 1]) - float(dog_pyramid[i][j + 1][x][y - 1])) * 0.25
    hessian[2][1] = hessian[1][2]

    det_check = np.linalg.det(hessian)
    if (det_check == 0):  # Numeric stability check
        return False, ret

    delta = -1.0*np.dot(np.linalg.inv(hessian),grad)

    if math.fabs(delta[0])>0.5 or math.fabs(delta[1])>0.5 or math.fabs(delta[2])>0.5:  # Numeric stability check
        return False, ret

    newval = dog_pyramid[i][j][x][y] / 255.0 + 0.5 * np.dot(grad, delta)

    if (np.abs(newval) < contrastThreshold/nScales):
        return False,ret

    trace = hessian[0][0] + hessian[1][1]
    det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0]
    if (det <= 0):
        return False,ret

    thr = 10.0  # Curvature threshold
    if (trace * trace * thr >= (thr + 1) * (thr + 1) * det):
        return False,ret

    ret = [i, j + delta[2], x + delta[0], y + delta[1], newval] # Why store newval?

    return True,ret


def gaussianKernel(ksize,sig):
    k = cv2.getGaussianKernel(ksize,sig)
    kern = np.outer(k,k)
    kern = kern*(1./kern[int(round(ksize/2))][int(round(ksize/2))])
    return kern


def calcKeypointOrientation(kp):
    global k,sigma,gaussian_pyramid
    no_of_bins = 36
    div = 360/no_of_bins

    #radius = int(round((2**kp[0])*(k**kp[1])*sigma))
    #window = 2*radius + 1
    radius = 2
    window = 5
    sig = ((2**kp[0])*(k**kp[1])*sigma)*1.5

    point = np.rint(kp).astype(int)
    gkern = gaussianKernel(window,sig)
    histogram = np.zeros(no_of_bins)
    grad_voxel_img = [0,0]
    grad_voxel_img[0],grad_voxel_img[1] = np.gradient(gaussian_pyramid[point[0]][point[1]][point[2]-radius:point[2]+radius+1,point[3]-radius:point[3]+radius+1])
    grad_mag = np.sqrt(np.square(grad_voxel_img[0])+np.square(grad_voxel_img[1]))
    grad_angle = np.arctan2(grad_voxel_img[1], grad_voxel_img[0])
    grad_angle *= 180.0 / np.pi  # Radians to degrees
    grad_angle = grad_angle%360.0  # [-180, 180] -> [0, 360]

    grad_angle = grad_angle*0.1
    grad_angle = grad_angle.astype(int)

    for i in range(window):
        for j in range(window):
            histogram[grad_angle[i][j]] += grad_mag[i][j]*gkern[i][j]

    r = np.argmax(histogram)
    angle = []
    angle.append(r * div + div / 2.0)

    m = histogram[r]
    p = m * 0.8
    for i in range(len(histogram)):
        if (histogram[i] >= p and i!=r):
            angle.append(i * div + div / 2.0)

            # Parabolic interpolation
            # Replace above expression of angle for more accuracy
            # x = np.array([r-1,r,r+1])
            # x = x*div + div/2.0
            # y = np.array([histogram[r-1], histogram[r], histogram[r+1]])
            # coeff = np.polyfit(x,y,2)
            # angle = point[2] - 0.25*np.square(coeff[1])/coeff[0] # c - b^2/4a
            # keypoints.append(refined_extrema[i].append(angle))

    return angle



if __name__ == '__main__':
    start_time = time.time()
    # Input image
    #img_orig = cv2.imread('cat.jpg',0)
    #base = cv2.resize(img_orig,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)

    sigma_base = math.sqrt(sigma*sigma - sigma_init*sigma_init*4)
    contrastThreshold = 0.04
    curvatureThreshold = 10

    #find value of sigma for each image
    sig = [i for i in xrange(nScales+3)]
    sig = map(findSigma,sig)
    sig[0] = sigma_base

    #build gaussian pyramid
    buildGaussianPyramid()

    #build DoG pyramid
    buildDoGPyramid()

    print("1)--- %s seconds ---" % (time.time() - start_time))
    extrema = []

    keypoints = []

    #find extrema
    for i in xrange(nOctaves):
        for j in xrange(1,nScales+1):
            maxI, minI = check_extrema(i,j)
            for u in maxI:
                is_ext, ext = refine_extrema(i, j, u[0], u[1])

                if is_ext:
                    extrema.append(ext)
            for u in minI:
                is_ext, ext = refine_extrema(i, j, u[0], u[1])
                
                if is_ext :
                    extrema.append(ext)
    print("2)--- %s seconds ---" % (time.time() - start_time))
    # Create a list that stores 3D gradients for each octave
    # Hence the first dimension corresponds to an octave

    # Magnitude and angle for the 3D gradient, stored for each octave
    # Create the 3D gradients

    for kp in extrema:
        angle = calcKeypointOrientation(kp)
        kp.append(angle)
        keypoints.append(kp)

    print("3)--- %s seconds ---" % (time.time() - start_time))
    plt.imshow(base, cmap='gray')
    for index in range(len(extrema)):
        i = extrema[index][0]
        x = extrema[index][2] * (2 ** i)
        y = extrema[index][3] * (2 ** i)
        plt.scatter(x=y, y=x, c='r', s=10)  # x and y are interchanged in plotting
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()
    print str(len(keypoints))
    gray = img_orig
    sift = cv2.SIFT()
    kp_sift = sift.detect(gray, None)
    img1 = cv2.drawKeypoints(
        gray, kp_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img1)
    plt.show()
    print str(len(kp_sift))
    print keypoints


