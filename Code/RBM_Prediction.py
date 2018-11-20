"""
Author: Mei Fu
Date: Dec.17, 2017
"""

import numpy as np

from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from PIL import Image
import json, time

def Sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

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
    return xWhitened

def load_data(direct):
    train_X = np.load(direct + "train_X_phoebe.npy")
    train_Y = np.load(direct + "train_Y_phoebe.npy")
    validation_X = np.load(direct + "validation_X_phoebe.npy")
    validation_Y = np.load(direct + "validation_Y_phoebe.npy")
    test_X = np.load(direct + "test_X_phoebe.npy")
    test_Y = np.load(direct + "test_Y_phoebe.npy")
    return train_X, train_Y, validation_X, validation_Y, test_X, test_Y

def load_params(hidSize):
    with open("Params_RBM_{}{}.txt".format(hidSize[0], hidSize[1]), "r") as inFile:
        weights_dic = eval(inFile.read())
        Params = {}
        Params['W'] = np.array(weights_dic['W'])
        Params['hBias'] = np.array(weights_dic['hBias'])
        Params['vBias'] = np.array(weights_dic['vBias'])
    return Params

def getHiddenFeatures(X, hidSize):
    """
    X - a row of visible states
    return a row of corresponding hidden features
    """
    N = X.shape[0]
    Params = load_params(hidSize)    
    nHidden = hidSize[0] * hidSize[1]

    hidden_z = np.dot(X, Params["W"]) + Params["hBias"]
    hidden_a = Sigmoid(hidden_z)

    return hidden_a

def run(direct, hidSize, whitening):

    start_time = time.time()

    # load data
    train_X, train_Y, validation_X, validation_Y, test_X, test_Y = load_data(direct)

    # preprocessing
    if whitening:
        ## grey-scaling & whitening
        Train_Imgs = np.array([whiten(np.asarray(
            Image.fromarray(x.reshape((48,64,3))).convert('L'))).reshape(48*64,) for x in train_X])
        Validation_Imgs = np.array([whiten(np.asarray(
            Image.fromarray(x.reshape((48,64,3))).convert('L'))).reshape(48*64,) for x in validation_X])
        Test_Imgs = np.array([whiten(np.asarray(
            Image.fromarray(x.reshape((48,64,3))).convert('L'))).reshape(48*64,) for x in test_X])

    if not whitening:
        ## grey-scaling
        Train_Imgs = np.array([np.asarray(
            Image.fromarray(x.reshape((48,64,3))).convert('L')).reshape(48*64,) for x in train_X])
        Validation_Imgs = np.array([np.asarray(
            Image.fromarray(x.reshape((48,64,3))).convert('L')).reshape(48*64,) for x in validation_X])
        Test_Imgs = np.array([np.asarray(
            Image.fromarray(x.reshape((48,64,3))).convert('L')).reshape(48*64,) for x in test_X])

    ## 0-1 scaling
    scaler = MinMaxScaler()
    scaler.fit(Train_Imgs)
    Train_Imgs = scaler.transform(Train_Imgs)
    Validation_Imgs = scaler.transform(Validation_Imgs)
    Test_Imgs = scaler.transform(Test_Imgs)

    ## shuffling
    idx = list(range(len(Train_Imgs)))
    np.random.shuffle(idx)
    Train_Imgs = Train_Imgs[idx]
    train_Y = train_Y[idx]

    # get features
    new_train_X = getHiddenFeatures(Train_Imgs, hidSize)
    new_validation_X = getHiddenFeatures(Validation_Imgs, hidSize)
    new_test_X = getHiddenFeatures(Test_Imgs, hidSize)

    # training SVM
    for C in [50]:
        svc = SVC(kernel='linear', random_state = 42, C=C)
        svc.fit(new_train_X, train_Y)

        # predictions
        accrcyTrain = sum(svc.predict(new_train_X) == train_Y) * 1.0 / len(train_Y)
        accrcyVali = sum(svc.predict(new_validation_X) == validation_Y) * 1.0 / len(validation_Y)
        accrcyTest = sum(svc.predict(new_test_X) == test_Y) * 1.0 / len(test_Y)
        print('training accuracy is', accrcyTrain)
        print('validation accuracy is', accrcyVali)
        print('test accuracy is', accrcyTest)
        print(svc.predict(new_test_X))
        print(svc.predict(test_Y))

    print('\n\n--- %s seconds ---' % (time.time() - start_time))


# run
if __name__ == "__main__":

    direct = "data/"

    hidSize = (24, 32)
    whitening = True

    run(direct, hidSize, whitening)



