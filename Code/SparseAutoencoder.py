# Junxu Lu, Mei Fu
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn import preprocessing
from sklearn.feature_extraction import image

class SparseAutoencoder():
    def __init__(self, visSize, hidSize, 
                 alpha=0.005, beta=6, Lambda=0.001, rho=0.05, activation="sigmoid"):
        self.rng = np.random.RandomState(545)
        self.D = 0 # we don't have input X yet
        self.Track_Losses = []
        self.Track_MSE = []
        self.preventUnderflow = 100

        # hyperparams
        self.visSize = visSize
        self.hidSize = hidSize
        self.k = self.hidSize[0] * self.hidSize[1] # Number of Nodes in the Hidden Layer (2nd Layer)
        self.alpha = alpha # Learning Rate
        self.beta = beta # Weight of the Sparsity Penalty Term
        self.Lambda = Lambda # Weight Decay Parameter
        self.rho = rho # desired average activation of hidden units
        self.activation = activation

        # functions
        self.Sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
        self.dSigmoid = lambda z: self.Sigmoid(z) * (1 - self.Sigmoid(z))
        self.Relu = lambda z: z * (z > 0)
        self.dRelu = lambda z: 1. * (z > 0)

        assert self.activation in ["sigmoid", "relu"]
        
        if self.activation == "sigmoid":
            # self.scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) # normalize to 0,1
            self.scaler = preprocessing.StandardScaler(with_mean=True, with_std=True) # normalize to -1,1
            self.Activation = self.Sigmoid
            self.dActivation = self.dSigmoid
        
        elif self.activation == "relu":
            self.scaler = preprocessing.StandardScaler(with_mean=True, with_std=True) # normalize to -1,1
            self.Activation = self.Relu
            self.dActivation = self.dRelu

    def InitParams(self):
        self.D = self.visSize[0] * self.visSize[1] # == self.x.shape[1]
        self.Jw = 0

        self.W_1 = np.random.rand(self.D, self.k) / self.preventUnderflow
        self.b_1 = np.zeros([1, self.k])
        self.Z_1 = np.zeros([self.N, self.k])
        self.A_1 = np.zeros([self.N, self.k])
        self.dA_1 = np.zeros([self.N, self.k])
        self.dZ_1 = np.zeros([self.N, self.k])
        self.dW_1 = np.zeros([self.D, self.k])
        self.db_1 = np.zeros([1, self.k])

        self.W_2 = np.random.rand(self.k, self.D) / self.preventUnderflow
        self.b_2 = np.zeros([1, self.D])
        self.Z_2 = np.zeros([self.N, self.D])
        self.A_2 = np.zeros([self.N, self.D])
        self.dA_2 = np.zeros([self.N, self.D])
        self.dZ_2 = np.zeros([self.N, self.D])
        self.dW_2 = np.zeros([self.k, self.D])
        self.db_2 = np.zeros([1, self.D])

    def Execute(self, X):
        # Data Preprocessing          
        self.x = X.reshape(len(X), -1)
        self.x = self.scaler.fit_transform(self.x)
        self.N = len(self.x) # N patches

        # Initialize Parameters
        if self.D == 0:
            self.InitParams()
        assert self.D == self.x.shape[1]

        # Forward Propagation
        self.Z_1 = np.dot(self.x, self.W_1) + self.b_1
        self.A_1 = self.Activation(self.Z_1)
        self.Z_2 = np.dot(self.A_1, self.W_2) + self.b_2
        self.A_2 = self.Activation(self.Z_2)

        # Loss Function (averaged over N patches)
        sum_of_square_error = 0.5 * np.linalg.norm(self.x - self.A_2, ord="fro")**2 / self.N
        
        sum_of_square_weights = np.sum(self.W_1**2) + np.sum(self.W_2**2)
        weight_decay = 0.5 * self.Lambda * sum_of_square_weights

        rho_cap = np.sum(self.A_1, axis=0) / self.N # size: 1 x k
        KL_divergence = np.sum(self.rho * np.log(self.rho / rho_cap) + (1 - self.rho) * np.log((1 - self.rho) / (1 - rho_cap)))

        self.Jw = sum_of_square_error + weight_decay + self.beta * KL_divergence
            
        # Back Propagation (dW, db are averaged over N patches)
        self.dA_2 = -(self.x - self.A_2)
        self.dZ_2 = np.multiply(self.dA_2, self.dActivation(self.Z_2))
        self.dW_2 = np.dot(self.A_1.T, self.dZ_2) / self.N + self.Lambda * self.W_2
        self.db_2 = np.sum(self.dZ_2, axis=0) / self.N
                
        deri_KL_divergence = - (self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap))
        self.dA_1 = np.dot(self.dZ_2, self.W_2.T) + np.matrix(self.beta * deri_KL_divergence)
        self.dZ_1 = np.multiply(self.dA_1, self.dActivation(self.Z_1))
        self.dW_1 = np.dot(self.x.T, self.dZ_1) / self.N + self.Lambda * self.W_1
        self.db_1 = np.sum(self.dZ_1, axis=0) / self.N

        # Update Weights
        self.W_1 -= self.alpha * self.dW_1
        self.W_2 -= self.alpha * self.dW_2
        self.b_1 -= self.alpha * self.db_1
        self.b_2 -= self.alpha * self.db_2

        self.Track_Losses.append(self.Jw)
        self.Track_MSE.append(sum_of_square_error)

    def GetLoss(self):
        return self.Track_Losses[-1]

    def ShowLoss(self):
        plt.plot(self.Track_Losses, label="Loss")
        plt.plot(self.Track_MSE, label="MSE")
        plt.legend()
        plt.show()

    def GetParams(self):
        return {"W_1": self.W_1.tolist(), "W_2": self.W_2.tolist(), "b_1": self.b_1.tolist(), "b_2": self.b_2.tolist()}

    def VisualW1(self):
        figure, axes = plt.subplots(nrows=self.hidSize[1], ncols=self.hidSize[0])
        idx = 0
        for axis in axes.flat:
            image = axis.imshow(self.W_1[:, idx].reshape(self.visSize),
                                cmap=plt.cm.gray, interpolation='nearest')
            axis.set_frame_on(False)
            axis.set_axis_off()
            idx += 1

        plt.show()

    def Transform(self, X):
        # Preprocessing
        x = X.reshape(1, -1)
        x = self.scaler.fit_transform(x)

        # Forward Pass
        Z_1 = np.dot(x, self.W_1) + self.b_1
        A_1 = self.Activation(Z_1)
        Z_2 = np.dot(A_1, self.W_2) + self.b_2
        A_2 = self.Activation(Z_2)

        # Reconstruct Image
        hiddenX = A_1.reshape(self.hidSize)

        A_2 = self.scaler.inverse_transform(A_2)
        newX = A_2.reshape(self.visSize)

        return hiddenX, newX

    def PrintFeatures(self, X):
        hiddenX, newX = self.Transform(X)
        Image.fromarray(X).show()
        # Image.fromarray(hiddenX).show()
        Image.fromarray(newX).show()
        print("\nOriginal X:")
        print(X)
        print("\nHidden Feature:")
        print(hiddenX)
        print("\nTransformed X:")
        print(newX)

    def Reset(self):
        self.Track_Losses = []
        self.D = 0
