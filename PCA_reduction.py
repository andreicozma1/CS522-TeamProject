#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

import matplotlib.image as mpimg
class PCA:
    def __init__(self,n_components):
        self.n_components=n_components
        self.component=None
        self.mean=None
    def fit(self,X):
        # mean
        self.mean=np.mean(X,axis=0)
        X=X-self.mean
        # covariance
        cov=np.cov(X.T)
        
        #eigenvectors and eigen values
        eigenvalues,eigenvectors=np.linalg.eig(cov)
        #v[:,i] column vector is eigenvector
        #sort eigenvectors
        eigenvectors=eigenvectors.T
        idxs=np.argsort(eigenvalues)[::-1]
        eigenvalues=eigenvalues[idxs]
        eigenvectors=eigenvectors[idxs]
        #store first n eigenvectors
        self.components=eigenvectors[0:self.n_components]
    def transform(self,X):
        #project data
        X=X-self.mean
        self.data_reduced=np.dot(X,self.components.T)
        return self.data_reduced
    def inverse_transform(self):
        data_inverse=np.dot(self.data_reduced, self.components)+self.mean
        return data_inverse

