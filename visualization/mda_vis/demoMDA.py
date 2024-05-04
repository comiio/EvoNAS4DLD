# Load all necessary python packages needed for the reported analyses
# in our manuscript
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

%matplotlib inline

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import sklearn
import umap
import pandas as pd
from umap.parametric_umap import ParametricUMAP
import numpy as np
from mda import *

# Font size for all the MDA visualizations shown below   
FS = 16

import tensorflow as tf
tf.test.is_gpu_available()

from sklearn.decomposition import PCA
neighborNum = 5
testDataFeatures = np.load('tstfeature_sub3_6_pca.npy')
# Load data labels (lung diseases including COVID) corresponding to input test lung x-ray images
Y = np.load('test_label.npy')
Y = Y.reshape(Y.shape[0],-1)

# Load predicted labels by the ResNet50
Y_pred = np.load('test_pred_label.npy')
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)
# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)

# Use the outline of the output manifold to generate the MDA visualization of the ResNet50 features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=Y, cmap='jet', s=5)
plt.xlabel("MDA1",  fontsize=20) 
plt.ylabel("MDA2", fontsize=20)  
#plt.title('MDA visualization of the ResNet50 features for classification task')
plt.xticks(fontsize=16)
plt.yticks( fontsize=16)

# Load feature data extracted by the ResNet50 from test x-ray images at the 3rd residual block's last convolutional
# layer in substructure 4.
from sklearn.decomposition import PCA
neighborNum = 5
testDataFeatures = np.load('tstfeature_sub3_6_pca_pruning.npy')
# Load data labels (lung diseases including COVID) corresponding to input test lung x-ray images
Y = np.load('test_label_pruning.npy')
Y = Y.reshape(Y.shape[0],-1)

# Load predicted labels by the ResNet50
Y_pred = np.load('test_pred_label_pruning.npy')
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)


# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)

# Use the outline of the output manifold to generate the MDA visualization of the ResNet50 features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=Y, cmap='jet', s=5)
plt.xlabel("MDA1",  fontsize=20) 
plt.ylabel("MDA2", fontsize=20)  
#plt.title('MDA visualization of the ResNet50 features for classification task')
plt.xticks(fontsize=16)
plt.yticks( fontsize=16)
