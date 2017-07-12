# -*- coding: utf-8 -*-
from stitch import kd_tree
import numpy as np
import cv2     
import os

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,ratio = 0.70):
    
    mtree = kd_tree.tree(kpsA, featuresA)
    tempM0 = np.apply_along_axis(mtree.knn_search,1,featuresB)
    # get this gay loop straight, dunno why but this pussy complains on vectorization
    matches = np.argwhere(np.array([ tempM0[:,1][i][0]/tempM0[:,1][i][1] for i in np.arange(tempM0.shape[0])]) < ratio)
    matches = np.transpose(matches)[0]
    
    ptsA = []
    ptsB = []

    for i in matches:
        ptsA.append(tempM0[i][0][0].data[0])
        ptsB.append(kpsB[i])
    
    return np.array(ptsA), np.array(ptsB)

if __name__ == '__main__':
    # figuring out how to get to the img directory
    path = os.path.realpath(__file__)
    path = path[:path.rindex('/')]
    #importing 2 placeholder images
    imA = cv2.imread(path + '/img/tree.jpeg')
    imB = cv2.imread(path + '/img/home.jpeg')
    # getting the grayscale
    imAG = cv2.cvtColor(imA,cv2.COLOR_BGR2GRAY)
    imBG = cv2.cvtColor(imB,cv2.COLOR_BGR2GRAY)
    # get the kp and descriptors
    kpA, desA = cv2.SIFT().detectAndCompute(imAG,None)
    kpB, desB = cv2.SIFT().detectAndCompute(imBG,None)
    # match the keypoints and kaboom
    mptsA, mptsB = matchKeypoints(kpA,kpB,desA,desB)