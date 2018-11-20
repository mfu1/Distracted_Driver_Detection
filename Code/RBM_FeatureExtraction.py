"""
Author: Mei Fu
Date: Dec.17, 2017
"""

import time
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from RBM import RBM

def whiten(x, meanX=None, sigmaToMinusOneHalfX=None):
    # use the provided mean and Sigma^(-1/2) if given,
    # and compute them if not given
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

def run(direct, visSize, hidSize, epochs, alpha, lamda):
	
	start_time = time.time()

	# initalize modelRBM
	modelRBM = RBM(nVisible = visSize[0] * visSize[1], nHidden = hidSize[0] * hidSize[1])

	# read and preprocessing training data
	X_train = np.load(direct + "train_X_phoebe.npy")
	## grey-scale, flatten, whiten
	imgs = np.array([whiten(np.asarray(Image.fromarray(x.reshape((visSize[0],visSize[1],3))).convert('L'))).reshape(visSize[0] * visSize[1],) for x in X_train])
	## 0-1 scaling
	scaler = MinMaxScaler()
	scaler.fit(imgs)
	imgs = scaler.transform(imgs)
	## shuffle
	idx = list(range(len(imgs)))
	np.random.shuffle(idx)
	imgs = imgs[idx]

	# execute feature extraction
	modelRBM.Execute(imgs, epochs=epochs, alpha=alpha, lamda=lamda)
	lastX = imgs[-1]
			
	print('\n\n--- %s seconds ---' % (time.time() - start_time))

	# check
	modelRBM.ShowLoss()
	modelRBM.PrintFeatures(lastX, visSize, hidSize) # last image
	# save params
	with open("Params_RBM_{}{}.txt".format(hidSize[0], hidSize[1]), "w") as output:
		output.write(str(modelRBM.GetParams()))
		output.close()

	modelRBM.Reset() # clear loss

# run
if __name__ == "__main__":

	direct = 'data/'

	visSize = (48, 64)
	hidSize = (24, 32)

	run(direct, visSize, hidSize, epochs=10, alpha=0.02, lamda=0.0002)



