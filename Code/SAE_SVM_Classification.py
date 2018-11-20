# Junxu Lu
import numpy as np
from sklearn.svm import LinearSVC
from PIL import Image
import json, time
start_time = time.time()

def Sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def normalize(x):
    '''normalize for contrast'''
    xMean = np.broadcast_to(np.mean(x, axis=1), x.shape[::-1]).T
    xSD = np.broadcast_to(np.sqrt(np.var(x, axis=1) + 10),
                          x.shape[::-1]).T
    return 1.0 * (x - xMean) / xSD

def whiten(x, meanX=None, sigmaToMinusOneHalfX=None):
    ''' 
    use the provided mean and Sigma^(-1/2) if given, and compute them if not given
    if whiten_status == False, whiten(x) will return x
    '''
    if whiten_status == False:
        return x
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

def getHiddenFeatures(x, Params):
    z_1 = np.dot(x, Params["W_1"]) + Params["b_1"]
    a_1 = Sigmoid(z_1)
    hidSize = (24, 32)
    hiddenX = a_1.reshape(hidSize)
    return hiddenX

# read data
train_X = np.load("data/train_X_phoebe.npy")
train_Y = np.load("data/train_Y_phoebe.npy")
validation_X = np.load("data/validation_X_phoebe.npy")
validation_Y = np.load("data/validation_Y_phoebe.npy")
test_X = np.load("data/test_X_phoebe.npy")
test_Y = np.load("data/test_Y_phoebe.npy")

# read trained weights (W1 ,b1, W2, b2)
with open("Params.txt", "r") as inFile:
    weights_dic = eval(inFile.read())
    Params = {}
    Params['W_1'] = np.array(weights_dic['W_1'])
    Params['b_1'] = np.array(weights_dic['b_1'])
    Params['W_2'] = np.array(weights_dic['W_2'])
    Params['b_2'] = np.array(weights_dic['b_2'])

# preprocessing
whiten_status = True # if whiten_status == False, whiten(x) will return x
Train_Imgs = np.array([whiten(normalize(np.asarray(
    Image.fromarray(x.reshape((48,64,3))).convert('L')))).reshape(48*64,) for x in train_X])
Test_Imgs = np.array([whiten(normalize(np.asarray(
    Image.fromarray(x.reshape((48,64,3))).convert('L')))).reshape(48*64,) for x in test_X])
print("preprocessing finished !")

# get features
new_train_X = [getHiddenFeatures(x, Params).reshape(24*32,) for x in Train_Imgs]
new_test_X = [getHiddenFeatures(x, Params).reshape(24*32,) for x in Test_Imgs]
print("feature extraction finished !")

# training SVM
C_svm = 5 # best for this model
clf = LinearSVC(penalty = 'l2', loss = 'hinge', C = C_svm, random_state = 42)
clf.fit(new_train_X, train_Y)
print("SVM training finished !")

# predictions
accrcyTrain = sum(clf.predict(new_train_X) == train_Y) * 1.0 / len(train_Y)
accrcyTest = sum(clf.predict(new_test_X) == test_Y) * 1.0 / len(test_Y)
print('training accuracy is', accrcyTrain)
print('test accuracy is', accrcyTest)


print('\n\n--- %s seconds ---' % (time.time() - start_time))


