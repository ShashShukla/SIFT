import cv2
import numpy as np
import math
#from matplotlib import pyplot as plt

img = cv2.imread('cat.jpg',0)

def resize(image, width):
    r = float(width) / image.shape[1]
    dim = (int(image.shape[0] * r),width)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image

img = resize(img,400)

h = np.zeros((3,3))
h[2][2] = 1.0

trans = 100.0
theta = 30.0

h[1][1] = h[0][0] = math.cos(theta*(math.pi/180.0))
h[0][1] = (-1)*math.sin(theta*(math.pi/180.0))
h[1][0] = math.sin(theta*(math.pi/180.0))

h[0][2]+=trans
h[1][2]+=trans
h[2][0]+=0.001
h[2][1]+=0

shape = np.array([img.shape[0]-1,img.shape[1]-1])
shape = np.append(shape,1)

edge = np.array([[0,0,1],[0,img.shape[1]-1,1],[img.shape[0]-1,0,1],[img.shape[0]-1,img.shape[1]-1,1]])
for i in range(4):
    edge[i] = np.dot(h,edge[i])


max_row = np.amax(edge[:,0])
max_col = np.amax(edge[:,1])

min_row = np.amin(edge[:,0])
min_col = np.amin(edge[:,1])

if min_row < 0:
    max_row += (-1) * min_row
    h[0][2] += (-1) * min_row
    edge[:,0] += (-1) * min_row
    min_row = 0

if min_col < 0:
    max_col += (-1) * min_col
    h[1][2] += (-1) * min_col
    edge[:,1] += (-1) * min_col
    min_col = 0

h_inv = np.linalg.inv(h)
print h
print h_inv

img2 = np.zeros((int(max_row+1),int(max_col+1)))

for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        x1 = np.dot(h_inv,[i,j,1])
        x = np.array([x1[0]/x1[2], x1[1]/x1[2]])
        # check whether the pixel belongs to the original image
        if x[0] >= 0 and x[0]<img.shape[0]-1 and x[1] >= 0 and x[1] < img.shape[1]-1:
            # applying bilinear interpolation
            x0 = [int(x[0]),int(x[1])]

            A = np.array([[1, x0[0], x0[1], x0[0]*x0[1]],
                          [1, x0[0]+1, x0[1], (x0[0]+1) * x0[1]],
                          [1, x0[0], x0[1]+1, x0[0] * (x0[1]+1)],
                          [1, x0[0]+1, x0[1]+1, (x0[0]+1) * (x0[1]+1)]])
            b = np.dot(np.linalg.inv(A).T,[1, x[0], x[1], x[0]*x[1]])

            img2[i][j] = (img[x0[0],x0[1]]*b[0] + img[x0[0]+1,x0[1]]*b[1] + img[x0[0],x0[1]+1]*b[2] + img[x0[0]+1,x0[1]+1]*b[3])/255.0

        else:
            continue


cv2.imshow('image',img)

cv2.imshow('translated image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
