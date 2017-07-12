# -*- coding: utf-8 -*-
from stitch import kd_tree
import numpy as np
import imutils
import cv2     

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,ratio = 0.70):
    mtree = kd_tree.tree(kpsA, featuresA)
    tempM0 = np.apply_along_axis(mtree.knn_search,1,featuresB)
    # get this gay loop straight, dunno why but this pussy complains on vectorization
    matches = np.argwhere(np.array([ tempM0[:,1][i][0]/tempM0[:,1][i][1] for i in np.arange(tempM0.shape[0])]) < ratio)
    matches = np.transpose(matches)[0]
    
    ptsA = []
    ptsB = []

    for i in matches:
        ptsA.append(kpsA[i])
        ptsB.append(kpsB[i])
    
    return np.array(ptsA), np.array(ptsB)