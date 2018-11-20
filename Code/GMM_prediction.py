# EECS 545 - Group Project
# @ Author: Hui(Phoebe) Liang
# Predict using learned parameters from Gaussian Mixture Model

from numpy import linalg as LA
from scipy.stats import multivariate_normal as mvn
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
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from datetime import datetime


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
    xSD = np.broadcast_to(np.sqrt(np.var(x, axis=1) + 10),
                          x.shape[::-1]).T
    return 1.0 * (x - xMean) / xSD


def standardize(x):
    ''' standardize for linear prediction model (e.g., linear SVM), also add
    the constant term '''
    xMean = np.mean(x, axis=0)
    xSd = np.sqrt(np.var(x, axis=0))
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
        ptc.append(im2col(np.reshape(X[dCnl * imgSz: (dCnl + 1) * imgSz],
                   IMAGE_DIM[0:2], order='F').T, [wRFSize, wRFSize]))
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


def extrctRdPtchs(trainX, dChannel, IMAGE_DIM, wRFSize, numRdmPtchs):
    ''' extract random patches'''
    numFtrRF = wRFSize * wRFSize * dChannel
    patches = np.zeros((numRdmPtchs, numFtrRF))
    for i in range(0, numRdmPtchs):
        r = np.random.randint(0, IMAGE_DIM[0] - wRFSize)
        c = np.random.randint(0, IMAGE_DIM[1] - wRFSize)
        patch = np.reshape(trainX[np.mod(i, trainX.shape[0])],
                           IMAGE_DIM, order='F')
        patch = patch[r:r + wRFSize, c:c + wRFSize]
        patches[i] = patch.flatten(order='F')
    return patches

def cmptTriangleActivation(patches, centroids):
    ''' compute 'triangle' activation function when extracting the features
    when using the method of KMeans'''
    xx = np.sum(patches**2, axis=1)  # X^2; dim is
    cc = np.sum(centroids**2, axis=1)
    xc = np.dot(patches, centroids.T)
    z = np.sqrt(cc + (xx.T - 2 * xc.T).T)  # distances
    mu = np.mean(z, axis=1)  # average distance to centroids for each patch
    return np.maximum((mu - z.T).T, 0)


train_X = np.load("train_X_phoebe.npy")
train_Y = np.load("train_Y_phoebe.npy")
validation_X = np.load("validation_X_phoebe.npy")
validation_Y = np.load("validation_Y_phoebe.npy")
test_X = np.load("test_X_phoebe.npy")
test_Y = np.load("test_Y_phoebe.npy")


from scipy.stats import multivariate_normal as mvn
sigmas = np.load("./GMM/patches5000/GMM_sigmas.npy")
print ("sigmas: ", sigmas.shape)
centroids = np.load("./GMM/patches5000/GMM_mus.npy")
print ("centroids: ", centroids.shape)
mus = np.load("./GMM/patches5000/GMM_mus.npy")
features = centroids.shape[0]

dChannel = 3  # d, the number of channel, 3 for color images
IMAGE_DIM = np.array([48, 64, dChannel])

wRFSize = 6 
numRdmPtchs = 5000  # number of random patches

whitening = True

numCentroidsKM = 100 
numIterationsKM = 30
kNum = numCentroidsKM

numFtrRF = wRFSize * wRFSize * dChannel  # num of features of a receptive field

trainXC = np.full([train_X.shape[0], kNum * 4], np.NaN)
imgSz = np.prod(IMAGE_DIM[0:2])
for i, X in enumerate(train_X):
    # display the progress
    if i % 1000 == 0:
        print('Extracting features:', i + 1, '/', len(train_X))
    # extract overlapping sub-patches into rows of 'patches'
    patches = extrctSubpatches(X, dChannel, IMAGE_DIM, wRFSize)
    # normalize
    patches = normalize(patches)
    # whiten
    if whitening:
        patches = whiten(patches)[0]
    patches_features = np.zeros((patches.shape[0], centroids.shape[0]))

    for k in range(features):
        patches_features[:, k] = mvn.pdf(patches, mus[k], sigmas[k])
    trainXC[i] = reshapeAndPool(patches_features, kNum, IMAGE_DIM, wRFSize)


# get features
valiXC = np.full([validation_X.shape[0], kNum * 4], np.NaN)
imgSz = np.prod(IMAGE_DIM[0:2])
for i, X in enumerate(validation_X):
    # display the progress
    if i % 1000 == 0:
        print('Extracting features:', i + 1, '/', len(validation_X))
    # extract overlapping sub-patches into rows of 'patches'
    patches = extrctSubpatches(X, dChannel, IMAGE_DIM, wRFSize)
    # normalize
    patches = normalize(patches)
    # whiten
    if whitening:
        patches = whiten(patches)[0]

    patches_features = np.zeros((patches.shape[0], centroids.shape[0]))
    for k in range(features):
        patches_features[:, k] = mvn.pdf(patches, mus[k], sigmas[k])
    valiXC[i] = reshapeAndPool(patches_features, kNum, IMAGE_DIM, wRFSize)

# get features
testXC = np.full([test_X.shape[0], kNum * 4], np.NaN)
imgSz = np.prod(IMAGE_DIM[0:2])
for i, X in enumerate(test_X):
    # display the progress
    if i % 1000 == 0:
        print('Extracting features:', i + 1, '/', len(test_X))
    # extract overlapping sub-patches into rows of 'patches'
    patches = extrctSubpatches(X, dChannel, IMAGE_DIM, wRFSize)
    # normalize
    patches = normalize(patches)
    # whiten
    if whitening:
        #patches = whiten(patches, meanPatches, sigmaToMinusOneHalfPatches)[0]
        patches = whiten(patches)[0]
    patches_features = np.zeros((patches.shape[0], centroids.shape[0]))
    for k in range(features):
        patches_features[:, k] = mvn.pdf(patches, mus[k], sigmas[k])
    testXC[i] = reshapeAndPool(patches_features, kNum, IMAGE_DIM, wRFSize)

# Prediction 

trainXCStd = standardize(trainXC)
valiXCStd = standardize(valiXC)
testXCStd = standardize(testXC)

c = [1, 5, 10, 50]

train_scores = []
validation_scores = []
test_scores = []

for C_svm in c: 
    # train using SVM
    clf = LinearSVC(penalty='l2', loss='hinge', C=C_svm, random_state=42)
    clf.fit(trainXCStd, train_Y)
    accrcyTrain = 1.0*sum(clf.predict(trainXCStd) == train_Y) / len(train_Y)
    train_scores.append(accrcyTrain)
    accrcyVali = 1.0*sum(clf.predict(valiXCStd) == validation_Y) / len(validation_Y)
    validation_scores.append(accrcyVali)
    accrcyTest = 1.0*sum(clf.predict(testXCStd) == test_Y) / len(test_Y)
    test_scores.append(accrcyTest)

    print('training accuracy is', accrcyTrain)
    print('validation accuracy is', accrcyVali)
    print('test accuracy is', accrcyTest)
    print('the end time is', datetime.now())

train_scores = np.asarray(train_scores)
validation_scores = np.asarray(validation_scores)
test_scores = np.asarray(test_scores)


np.save("train_scores100.npy", train_scores)
np.save("validation_scores100.npy", validation_scores)
np.save("test_scores100.npy", test_scores)
