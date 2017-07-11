#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np

class node:
    '''i : dimension with max range VARIANCE METHOD IS COMPUTATIONALLY EXPENSIVE
    m : median of the data ROLLED with -i "REMEMBER" 
    m_orig : the original median'''
    def __init__(self,ltree,rtree,dimension,med_rolled,feature = []):
        self.L = ltree
        self.R = rtree
        self.i = dimension
        self.m = med_rolled
        if (self.L == False) and (self.R == False):
            self.data = feature
        
    def is_leaf(self):
        return (self.L == False) and (self.R == False)
    
    def dist_from_edge(self,q):
        '''used only when it is a node'''
        return np.abs(self.m[0] - q[self.i])
    
    def dist(self,q):
        '''Used only after confirming that it is a leaf'''
        t = self.data[1] - q
        return np.dot(t,t)
    

class tree:
    '''ind = range(128) (np.arrange(number of dimensions ))
    bt = number of (leaves ? / nodes ?) to check during backtracking'''
    def __init__(self,keypoints,features,bt_nodes = 2):
        self.feat_vec = features
        self.kp = keypoints
        # stored values to avoid unnecessary recalculation
        self.ind = np.arange(self.feat_vec.shape[1])
        # properties explicit to the tree
        self.root = self.__make_node(self.init_lookup())
        self.bt = bt_nodes
        
    def init_lookup(self):
        # vectorized alternatives provide barely any speedup (will try memoisation)
        # AT : Half expecting it to slow down the program :P
        temp = np.flipud(np.transpose(self.feat_vec))
        res = [np.lexsort(np.roll(temp, ctr, axis = 0)) for ctr in np.arange(temp.shape[0])]
        return np.array(res,dtype = np.uint64)

    def __make_node(self,lookup):
        if(lookup.shape[1] == 1):
            return node(False,False,False,False, (self.kp[lookup[0,0]],self.feat_vec[lookup[0,0]]))
        if(lookup.shape[1] == 0): 
            return False
        max_dim = np.argmax(self.feat_vec[lookup[:,lookup.shape[1]-1],self.ind]- self.feat_vec[lookup[:,0],self.ind])
        #alternate var method for a more balanced tree, but about ~ 1.5X more construction time
        #max_dim = np.argmax(np.var(self.feat_vec),axis =0)
        med = lookup.shape[1] / 2  
        maskL = np.zeros(self.feat_vec.shape[0],dtype = bool)
        maskL[lookup[max_dim, : med]] = True
        maskR = np.zeros(self.feat_vec.shape[0],dtype = bool)
        maskR[lookup[max_dim, med :]] = True
        lookL = np.extract(maskL[lookup],lookup).reshape([self.feat_vec.shape[1],lookup[max_dim, : med].shape[0]])
        lookR = np.extract(maskR[lookup],lookup).reshape([self.feat_vec.shape[1],lookup[max_dim, med :].shape[0]])
        #delete unnecesary data before proceeding into recursion
        del maskL
        del maskR
        
        med = self.feat_vec[lookup[max_dim,med]] if lookup.shape[1] % 2 == 1 else (self.feat_vec[lookup[max_dim,med]] + self.feat_vec[lookup[max_dim,med - 1]]) /2.0
        return node(self.__make_node(lookL),self.__make_node(lookR),max_dim,np.roll(med, - max_dim, axis = 0).tolist())
        
    def drop_down(self, node, q, queue):
        # make this iterative, maybe
        if node.is_leaf():
            return (node.data[1],queue,q)
         
        if  np.roll(q, -node.i).tolist() < node.m :
            return self.drop_down(node.L,queue,q)
        else:
            return self.drop_down(node.R,queue,q)