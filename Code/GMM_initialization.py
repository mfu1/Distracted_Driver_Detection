# EECS 545 - Group Project
# Author: Hui(Phoebe) Liang
# Run one iteration of K-means to initialize means for Gaussian Mixture

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm

import os, glob, cv2, math, pickle
from scipy import misc
import scipy.io as spio
import scipy.sparse as sps

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib.pyplot import imshow
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image


def im2col(A, BSZ, stepsize=1):
    '''Rearrange image blocks into columns; sliding; by the strided method
    Used for extracting features '''
    '''Same function as this matlab function:
    https://www.mathworks.com/help/images/ref/im2col.html
    Code is from:
    https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python''' 
    m, n = A.shape
    s0, s1 = A.strides
    nrows = m - BSZ[0] + 1
    ncols = n - BSZ[1] + 1
    shp = BSZ[0], BSZ[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0] * BSZ[1], -1)[:, ::stepsize]


def normalize(x):
    ''' normalize for contrast'''
    xMean = np.broadcast_to(np.mean(x, axis=1), x.shape[::-1]).T
    xSD = np.broadcast_to(np.sqrt(np.var(x, axis=1) + 10), x.shape[::-1]).T # remove the "+10"
    return 1.0 * (x - xMean) / xSD

def standardize(x):
    ''' standardize for linear prediction model (e.g., linear SVM), also add
    the constant term '''
    xMean = np.mean(x, axis=0)
    xSd = np.sqrt(np.var(x, axis=0) + 0.01)
    xs = (x - xMean) / xSd
    return np.concatenate((xs, np.ones((xs.shape[0], 1))), axis=1)


def whiten(x, meanX=None, sigmaToMinusOneHalfX=None):
    ''' whiten. Use the provided mean and Sigma^(-1/2) if given,
    and compute them if not given '''
    assert ((meanX is sigmaToMinusOneHalfX is None) or
            ((meanX is not None) and (sigmaToMinusOneHalfX is not None)))
    if meanX is None:
        meanX = np.mean(x, axis=0)
        covX = np.cov(x.T)
        w, V = np.linalg.eigh(covX)
        wInvMat = np.matrix(np.diag(np.sqrt(1.0 / (w + 0.1))))
        VMat = np.matrix(V)
        sigmaToMinusOneHalfX = VMat * wInvMat * VMat.T
    xWhitened = np.dot((x - meanX), sigmaToMinusOneHalfX.A)
    return xWhitened, meanX, sigmaToMinusOneHalfX


def extrctSubpatches(x, dChannel, IMAGE_DIM, wRFSize):
    ''' extract overlapping sub-patches into rows of 'patches' '''
    imgSz = np.prod(IMAGE_DIM[0:2])
    ptc = list()
    for dCnl in range(dChannel):
        ptc.append(im2col(np.reshape(X[dCnl * imgSz: (dCnl + 1) * imgSz],IMAGE_DIM[0:2], order='F').T, [wRFSize, wRFSize]))
    return np.concatenate(tuple(i for i in ptc), axis=0).T


def reshapeAndPool(patches, kNum, IMAGE_DIM, wRFSize):
    ''' first reshape to K-channel image, and then
     pool over quadrants'''

    # reshape to kNum-channel image
    pRows = IMAGE_DIM[0] - wRFSize + 1
    pCols = IMAGE_DIM[1] - wRFSize + 1
    patches = np.reshape(patches, (pRows, pCols, kNum), order='F')

    # pool over quadrants
    halfR = int(round(pRows / 2.0))
    halfC = int(round(pCols / 2.0))
    q1 = np.sum(patches[:halfR, :halfC, :], axis=(0, 1))
    q2 = np.sum(patches[halfR:, :halfC, :], axis=(0, 1))
    q3 = np.sum(patches[:halfR, halfC:, :], axis=(0, 1))
    q4 = np.sum(patches[halfR:, halfC:, :], axis=(0, 1))
    return np.concatenate((q1, q2, q3, q4))


def cmptTriangleActivation(patches, centroids):
    ''' compute 'triangle' activation function when extracting the features
    when using the method of KMeans'''
    xx = np.sum(patches**2, axis=1)  # X^2; dim is
    cc = np.sum(centroids**2, axis=1)
    xc = np.dot(patches, centroids.T)
    z = np.sqrt(cc + (xx.T - 2 * xc.T).T)  # distances
    mu = np.mean(z, axis=1)  # average distance to centroids for each patch
    return np.maximum((mu - z.T).T, 0)


def extrctRdPtchs(trainX, dChannel, IMAGE_DIM, wRFSize, numRdmPtchs):
    ''' extract random patches'''
    numFtrRF = wRFSize * wRFSize * dChannel
    patches = np.zeros((numRdmPtchs, numFtrRF))
    for i in range(0, numRdmPtchs):
        r = np.random.randint(0, IMAGE_DIM[0] - wRFSize)
        c = np.random.randint(0, IMAGE_DIM[1] - wRFSize)
        patch = np.reshape(trainX[np.mod(i, trainX.shape[0])],IMAGE_DIM, order='F')
#         print np.mod(i, trainX.shape[0])
#         print IMAGE_DIM
#         print patch.shape
        patch = patch[r:r + wRFSize, c:c + wRFSize]
#         print patch.shape
        patches[i] = patch.flatten(order='F')
#         if i == 3:
#             break
    return patches


def load_train_data():
    train_x = []
    train_y = []
    print('reading images')
    for j in range(0, 10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for f in files:
            img = get_im(f)
            train_x.append(img)
            train_y.append(j)
    return np.array(train_x), np.array(train_y)


def get_im(path):
    img = misc.imread(path)
    resized = cv2.resize(img, (64, 48))
    #return resized
    return resized.flatten(order='F')

#Read Data
x_train, y_train = load_train_data()
train_X, test_X, train_Y, test_Y = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

np.save("train_X_phoebe.npy", train_X)
np.save("train_Y_phoebe.npy", train_Y)
np.save("validation_X_phoebe.npy", validation_X)
np.save("validation_Y_phoebe.npy", validation_Y)
np.save("test_X_phoebe.npy", test_X)
np.save("test_Y_phoebe.npy", test_Y)

# Initialize 

# 100 features, 5000 patches 
# 200 features, 10000 random patches
# 500 features, 15000 random patches

# 3*3*2, C (1, 5, 10, 50)

dChannel = 3  # d, the number of channel, 3 for color images
IMAGE_DIM = np.array([48, 64, dChannel])

wRFSize = 6  # w, or the receptive field size
numRdmPtchs = 12000  # number of random patches

whitening = True

numCentroidsKM = 300
numIterationsKM = 30
kNum = numCentroidsKM

numFtrRF = wRFSize * wRFSize * dChannel


# extract random patches and pre-process

print ("patches processing")

train_X = np.load("train_X_phoebe.npy")

patches = extrctRdPtchs(train_X, dChannel, IMAGE_DIM, wRFSize, numRdmPtchs)
np.save("patches_phoebe12000.npy", patches)

print ("patches normalizing")
patches_normalized = normalize(patches)
np.save("patches_normal_phoebe12000.npy", patches_normalized)


print ("patches whitening")
# whiten
if whitening:
    patches_whiten, meanPatches, sigmaToMinusOneHalfPatches = whiten(patches_normalized)
np.save("patches_whiten_phoebe12000.npy", patches_whiten)


print ("running kmeans")
#kmeans = KMeans(n_clusters=numCentroidsKM, random_state=1, n_init=1).fit(patches_whiten)
kmeans = KMeans(n_clusters=numCentroidsKM, random_state=1, n_init=1).fit(patches_normalized)

centers = kmeans.cluster_centers_
np.save("kmeans_center_phoebe300.npy", centers)

labels = kmeans.labels_
np.save("kmeans_labels_phoebe300.npy", labels)





