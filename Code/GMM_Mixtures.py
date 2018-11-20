# EECS 545 - Group Project
# @ Author: Hui(Phoebe) Liang
# Feature Learning using Gaussian Mixture Model


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


def em_gmm(observations, _pis, _mus, _sigmas, tol=0.01, iterations=1000):
    num_of_obs, dim_of_ob = observations.shape
    old_log_like = 0
    len_of_latent = len(_mus)
    loss = []
    iterations_list = []
    sigmas = np.zeros((len_of_latent, dim_of_ob, dim_of_ob))

    for i in range(iterations):
        # Expectation step
        iterations_list.append(i)
        print ("iteration: ", i)
        if i == 0: 
            posterior_mat = np.zeros((len_of_latent, num_of_obs))
            for k in range(len_of_latent):
                posterior_mat[k] = _pis[k] * mvn(_mus[k], _sigmas[k]).pdf(observations)
            old_posterior_mat = posterior_mat
            posterior_mat = posterior_mat / posterior_mat.sum(0)
            if np.array_equal(old_posterior_mat, posterior_mat):
                print ("equal")
            else:
                print ("not equal")
        else:
            posterior_mat = np.zeros((len_of_latent, num_of_obs))
            for k in range(len_of_latent):
                try: 
                    posterior_mat[k] = pis[k] * mvn.pdf(observations, mus[k], sigmas[k])
                except: 
                    random = np.random.randint(1, 2, size=dim_of_ob)
                    sigmas[k] = np.diag(random)
                    posterior_mat[k] = pis[k] * mvn.pdf(observations, mus[k], sigmas[k])
            old_posterior_mat = posterior_mat
            posterior_mat = posterior_mat / posterior_mat.sum(0)
            if np.array_equal(old_posterior_mat, posterior_mat):
                print ("equal")
            else:
                print ("not equal") 

        # Maximization step
        pis = np.sum(posterior_mat, axis=1) / num_of_obs

        mus = np.matmul(posterior_mat, observations)

        for k in range(len_of_latent):
            mus[k] = mus[k] / posterior_mat[k].sum()

        # input data
        # https://stats.stackexchange.com/questions/219302/singularity-issues-in-gaussian-mixture-model
        
        for k in range(len_of_latent):
            # diff = np.dot(observations.T, observations)
            sigmas[k] = np.dot(((observations-mus[k]).T*posterior_mat[k]), (observations-mus[k]))
            sigmas[k] = sigmas[k] / posterior_mat[k].sum()

        # Update and evaluate log likehood
        new_log_like = 0
        like_sum = np.zeros((num_of_obs, len_of_latent))
        for k in range(len_of_latent):
            try: 
                like_sum[:, k] = mvn.pdf(observations, mus[k], sigmas[k])
            except: 
                random = np.random.randint(1, 2, size=dim_of_ob)
                sigmas[k] = np.diag(random)
                like_sum[:, k] = mvn.pdf(observations, mus[k], sigmas[k])

        new_log_like = np.sum(np.log(np.sum(like_sum, axis=1)))
        loss.append(new_log_like)

        if np.abs(new_log_like - old_log_like) < tol:
            break
        old_log_like = new_log_like

    return new_log_like, pis, mus, sigmas, posterior_mat, iterations_list, loss


patches_whiten = np.load("./GMM/patches12000/patches_whiten_phoebe12000.npy")
#patches_whiten = np.load("./GMM/patches12000/patches_normal_phoebe12000.npy")

patches_labels = np.load("./GMM/patches12000/kmeans_labels_phoebe300.npy")
centers = np.load("./GMM/patches12000/kmeans_center_phoebe300.npy")
#centers = np.random.random((200, 108))

# unique_labels, counts_labels = np.unique(patches_labels, return_counts=True)
# pis = counts_labels
# pis_array = pis.astype(float)/float(pis.sum())

len_of_latent = centers.shape[0]
init_prob = 1.0/float(len_of_latent)

pis = [init_prob]*len_of_latent
pis_array = np.asarray(pis)
print ("pis: ", pis_array.shape)

dim_of_ob = patches_whiten.shape[1]
initial_sigmas = np.array([np.eye(dim_of_ob)] * len_of_latent)
print ("initial_sigmas finish: ", initial_sigmas.shape)

print ("run")

log_like, pis, mus, sigmas, posterior_mat, iterations_list, loss = em_gmm(patches_whiten, pis_array, centers, initial_sigmas)

import matplotlib.pyplot as plt
loss = np.asarray(loss)
iterations_list = np.asarray(iterations_list)
plt.plot(loss, label='Train')
plt.xlabel('Iteration')
plt.ylabel('log likelihood')
plt.legend()
plt.savefig('losses300.png')
plt.show()

np.save("GMM_loglike.npy", log_like)
np.save("GMM_pis.npy", pis)
np.save("GMM_mus.npy", mus)
np.save("GMM_sigmas.npy", sigmas)
np.save("GMM_posterior_mat.npy", posterior_mat)

