import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class RBM:
  
  def __init__(self, nVisible, nHidden):
    self.nHidden = nHidden
    self.nVisible = nVisible

    np_rng = np.random.RandomState(545)
    # uniform distribution, mean 0 and standard deviation 0.1
    self.W = np.asarray(np_rng.uniform(low = -0.1 * np.sqrt(6. / (nHidden + nVisible)),
                                       high = 0.1 * np.sqrt(6. / (nHidden + nVisible)),
                                       size = (nVisible, nHidden)))

    # bias
    self.hBias = np.zeros((1, nHidden))
    self.vBias = np.zeros((1, nVisible))

    self.Track_Losses = []
    self.Sigmoid = lambda z: 1.0 / (1 + np.exp(-z))

  def Execute(self, X, epochs=1000, alpha=0.05, lamda=2e-4):
    """
    X - row matrix of states
    alpha - learning rate
    """
    self.N = X.shape[0]

    for epoch in range(epochs):      
      # positive CD phase: propagate the visible units activation upwards to the hidden units with sigmoid function. 
      # The state of the positive hidden units is 0 or 1, where 1 represents the positive hidden probability is greater than the value of sampling from a uniform distribution over [0,1). 
      pos_hidden_z = np.dot(X, self.W) + self.hBias
      pos_hidden_a = self.Sigmoid(pos_hidden_z)
      pos_hidden_s = np.zeros((self.N, self.nHidden))
      pos_hidden_s[pos_hidden_a > np.random.rand(self.N, self.nHidden)] = 1

      pos_associations = np.dot(X.T, pos_hidden_a)

      # negative CD phase: reconstruct the visible units and sample again from the hidden units
      neg_visible_z = np.dot(pos_hidden_s, self.W.T) + self.vBias
      neg_visible_a = self.Sigmoid(neg_visible_z)
      neg_hidden_z = np.dot(neg_visible_a, self.W) + self.hBias
      neg_hidden_a = self.Sigmoid(neg_hidden_z)
      neg_hidden_s = np.zeros((self.N, self.nHidden))
      neg_hidden_s[neg_hidden_a > np.random.rand(self.N, self.nHidden)] = 1

      neg_associations = np.dot(neg_visible_a.T, neg_hidden_a)

      # update weights
      self.W += alpha * ((pos_associations - neg_associations) / self.N - lamda * self.W)
      self.vBias += alpha / self.N * (np.sum(X, axis=0) - np.sum(neg_visible_a, axis=0))
      self.hBias += alpha / self.N * (np.sum(pos_hidden_a, axis=0) - np.sum(neg_hidden_a, axis=0))

      # update cost
      beliefError = np.sum((X - neg_visible_a) ** 2)
      self.Track_Losses.append(beliefError)
      print("Epoch %s: error is %s" % (epoch, beliefError))

  def GetHiddenFeatures(self, X):
    """
    X - row matrix of visible states
    return a matrix of corresponding hidden features
    """
    N = 1 # used to debug

    hidden_z = np.dot(X, self.W) + self.hBias
    hidden_a = self.Sigmoid(hidden_z)
    hidden_s = np.ones((N, self.nHidden))
    hidden_s[hidden_a > np.random.rand(N, self.nHidden)] = 1

    return hidden_a
    
  def GetVisibleData(self, X):
    """
    X - row matrix of hidden states
    return a matrix of corresponding data
    """
    N = 1 # used to debug

    visible_z = np.dot(X, self.W.T) + self.vBias
    visible_a = self.Sigmoid(visible_z)
    visible_s = np.ones((N, self.nVisible))
    visible_s[visible_a > np.random.rand(N, self.nVisible)] = 1

    return visible_a

  def GetLoss(self):
    return self.Track_Losses[-1]
  
  def ShowLoss(self):
    plt.plot(self.Track_Losses, label="Loss")
    plt.legend()
    plt.show()

  def GetParams(self):
      return {"W": self.W.tolist(), 
              "hBias": self.hBias.tolist(), 
              "vBias": self.vBias.tolist()}

  def PrintFeatures(self, X, visSize, hidSize):
      hiddenX = self.GetHiddenFeatures(X)
      newX = self.GetVisibleData(hiddenX)

      print("\nOriginal X:")
      print(X.reshape(visSize))
      print("\nHidden Feature:")
      print(hiddenX.reshape(hidSize))
      print("\nTransformed X:")
      print(newX.reshape(visSize))

  def Reset(self):
      self.Track_Losses = []

# Test
if __name__ == "__main__":
    
    np.random.seed(545)

    test_1 = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]])
    test_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]])

    AE = RBM(test_2, 6)
    AE.train()
    AE.ShowLoss()
    AE.PrintFeatures()

