# Junxu Lu, Mei Fu
import time
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.datasets import mnist
from SparseAutoencoder import SparseAutoencoder
start_time = time.time()

def normalize(x):
    # normalize input data
    xMean = np.broadcast_to(np.mean(x, axis=1), x.shape[::-1]).T
    xSD = np.broadcast_to(np.sqrt(np.var(x, axis=1)), x.shape[::-1]).T # remove the "+10"
    return 1.0 * (x - xMean) / xSD

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

def run(direct, visSize, hidSize, epoches, batch_size, alpha, beta, Lambda, rho, activation):
	
	start_time = time.time()

	# initalize AE
	AE = SparseAutoencoder(visSize, hidSize, alpha, beta, Lambda, rho, activation)

	# each epoch
	for epoch in range(epoches):
		# each image
		X_train = np.load(direct + "train_X_phoebe.npy")
		X = X_train
		# X = [batch_size * i : batch_size * (i + 1)]
		imgs = np.array([whiten(normalize(np.asarray(Image.fromarray(x.reshape((48,64,3))).convert('L')))) for x in X])
		idx = list(range(len(imgs)))
		np.random.shuffle(idx)
		AE.Execute(imgs[idx])
		print("Loss after epoch {}: {}".format(epoch+1, AE.GetLoss()))

	print('\n\n--- %s seconds ---' % (time.time() - start_time))
	AE.ShowLoss()

	# save params
	with open("Params.txt", "w") as output:
		output.write(str(AE.GetParams()))
		output.close()

	AE.Reset() # clear loss

if __name__ == "__main__":
	direct = 'data/'
	visSize = (48, 64)
	hidSize = (24, 32)
	run(direct, visSize, hidSize, epoches=10, batch_size=100, 
		alpha=0.1, beta=6, Lambda=0.001, rho=0.05, activation="sigmoid")

