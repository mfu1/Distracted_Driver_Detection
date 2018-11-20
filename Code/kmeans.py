
import numpy as np
import scipy.io as spio
import scipy.sparse as sps
from scipy import misc
from sklearn.model_selection import train_test_split
import copy
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, glob, cv2, math, pickle
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
        patch = np.reshape(trainX[np.mod(i, trainX.shape[0])],
                           IMAGE_DIM, order='F')
        patch = patch[r:r + wRFSize, c:c + wRFSize]
        patches[i] = patch.flatten(order='F')
    return patches


def runKmeans(patches, numCentroidsKM, numIterationsKM, numFtrRF):
    ''' run k-means, returns centroids'''
    x2 = np.sum(patches**2, axis=1)
    centroids = np.random.normal(size=(numCentroidsKM, numFtrRF)) * 0.1
    losses = list()
    # for itr in range(numIterationsKM):
    for itr in range(numIterationsKM):
        # print('K-means iteration:', itr + 1, '/', numIterationsKM)

        c2 = 0.5 * np.sum(centroids**2, 1)
        summation = np.full([numCentroidsKM, numFtrRF], np.NaN)
        counts = np.full([numCentroidsKM, 1], np.NaN)
        df = (c2.T - np.dot(centroids, patches.T).T).T
        indx = np.argmin(df, axis=0)
        val = df[indx, np.arange(df.shape[1])]
        losses.append(np.sum(0.5 * x2 + val))
        sIndct = sps.csr_matrix((np.ones(numRdmPtchs),
                                (np.array(range(numRdmPtchs)), indx)),
                                shape=(numRdmPtchs, numCentroidsKM))
        summation = np.dot(sIndct.T.toarray(), patches)
        counts = np.sum(sIndct.toarray(), axis=0)
        centroids = (1.0 * summation.T / counts.T).T
        if len(np.where(counts == 0)[0]) > 0:
            centroids[np.where(counts == 0)[0]] = 0

    # fig, ax = plt.subplots()
    # for axis in [ax.xaxis, ax.yaxis]:
    #     axis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.plot(losses, label='Train')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('losses.png')
    # # plt.show()
    # plt.close()

    return centroids, losses


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
    return resized.flatten(order='F')


print('the start time of this trial is', datetime.now())
# 0. ------------------------  Parameters

dChannel = 3  # d, the number of channel, 3 for color images
IMAGE_DIM = np.array([48, 64, dChannel])

wRFSize = 6  # w, or the receptive field size (20)

whitening = True
numCentroidsKM = 100
numRdmPtchs = 5000

numIterationsKM = 40
kNum = numCentroidsKM
numFtrRF = wRFSize * wRFSize * dChannel  # num of features of a receptive field

# ------------------------ 1. Load training data

trainX = np.load('train_X_phoebe.npy')
trainY = np.load('train_Y_phoebe.npy')
valiX = np.load('validation_X_phoebe.npy')
valiY = np.load('validation_Y_phoebe.npy')
testX = np.load('test_X_phoebe.npy')
testY = np.load('test_Y_phoebe.npy')

# ------------------------ 2. extract random patches and pre-process

# extract random patches
patches = extrctRdPtchs(trainX, dChannel, IMAGE_DIM, wRFSize, numRdmPtchs)

# normalize for contrast
patches = normalize(patches)

# whiten
if whitening:
    patches, meanPatches, sigmaToMinusOneHalfPatches = whiten(patches)

# ------------------------ 3. run kmeans

centroids, losses = runKmeans(patches, numCentroidsKM,
                              numIterationsKM, numFtrRF)

# ------------------------ 4. pre-process and extract features

# get features for all training images
trainXC = np.full([trainX.shape[0], kNum * 4], np.NaN)
imgSz = np.prod(IMAGE_DIM[0:2])
for i, X in enumerate(trainX):
    # display the progress
    if i % 5000 == 0:
        print('Extracting features:', i + 1, '/', len(trainX))
    # extract overlapping sub-patches into rows of 'patches'
    patches = extrctSubpatches(X, dChannel, IMAGE_DIM, wRFSize)
    # normalize
    patches = normalize(patches)
    # whiten
    if whitening:
        # patches = whiten(patches, meanPatches, sigmaToMinusOneHalfPatches)[0]
        patches = whiten(patches)[0]
    # compute the 'triangle activation function'
    patchesTrg = cmptTriangleActivation(patches, centroids)
    # (patches is now the data matrix of activations for each patch)
    # reshape to numK*channel image, and pool over quadrants
    trainXC[i] = reshapeAndPool(patchesTrg, kNum, IMAGE_DIM, wRFSize)

# get features of validation data
valiXC = np.full([valiX.shape[0], kNum * 4], np.NaN)
imgSz = np.prod(IMAGE_DIM[0:2])
for i, X in enumerate(valiX):
    # display the progress
    if i % 5000 == 0:
        print('Extracting features:', i + 1, '/', len(valiX))
    # extract overlapping sub-patches into rows of 'patches'
    patches = extrctSubpatches(X, dChannel, IMAGE_DIM, wRFSize)
    # normalize
    patches = normalize(patches)
    # whiten
    if whitening:
        # patches = whiten(patches, meanPatches, sigmaToMinusOneHalfPatches)[0]
        patches = whiten(patches)[0]
    # compute the 'triangle activation function'
    patchesTrg = cmptTriangleActivation(patches, centroids)
    # (patches is now the data matrix of activations for each patch)
    # reshape to numK*channel image, and pool over quadrants
    valiXC[i] = reshapeAndPool(patchesTrg, kNum, IMAGE_DIM, wRFSize)

# get features of test data
testXC = np.full([testX.shape[0], kNum * 4], np.NaN)
imgSz = np.prod(IMAGE_DIM[0:2])
for i, X in enumerate(testX):
    # display the progress
    if i % 5000 == 0:
        print('Extracting features:', i + 1, '/', len(testX))
    # extract overlapping sub-patches into rows of 'patches'
    patches = extrctSubpatches(X, dChannel, IMAGE_DIM, wRFSize)
    # normalize
    patches = normalize(patches)
    # whiten
    if whitening:
        # patches = whiten(patches, meanPatches, sigmaToMinusOneHalfPatches)[0]
        patches = whiten(patches)[0]
    # compute the 'triangle activation function'
    patchesTrg = cmptTriangleActivation(patches, centroids)
    # (patches is now the data matrix of activations for each patch)
    # reshape to numK*channel image, and pool over quadrants
    testXC[i] = reshapeAndPool(patchesTrg, kNum, IMAGE_DIM, wRFSize)


# ------------------------ 5. train using a linear model

print('print the parameters and the results')
print('whitening is', whitening)
print('numCentroidsKM is', numCentroidsKM)
print('numRdmPtchs is', numRdmPtchs)

# standardize data for the linear model
trainXCStd = standardize(trainXC)
valiXCStd = standardize(valiXC)
testXCStd = standardize(testXC)

trainXCStd[np.nonzero(trainXCStd != trainXCStd)] = 0
valiXCStd[np.nonzero(valiXCStd != valiXCStd)] = 0
testXCStd[np.nonzero(testXCStd != testXCStd)] = 0


C_svm_vals = [1, 5, 10, 50]
for C_svm in C_svm_vals:
    # train using SVM
    clf = LinearSVC(penalty='l2', loss='hinge', C=C_svm, random_state=42)
    clf.fit(trainXCStd, trainY)
    accrcyTrain = sum(clf.predict(trainXCStd) == trainY) / len(trainY)
    accrcyVali = sum(clf.predict(valiXCStd) == valiY) / len(valiY)
    accrcyTest = sum(clf.predict(testXCStd) == testY) / len(testY)
    print('C is', C_svm)
    print('training accuracy is', accrcyTrain)
    print('validation accuracy is', accrcyVali)
    print('test accuracy is', accrcyTest)

print('the end time of this trial is', datetime.now())
