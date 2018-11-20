# Distracted_Driver_Detection

## Getting Started


This readme provides detailed instructions on how to read images, split data, extract features, train the features, and evaluate the system.


## Dataset: 
Link to download the dataset https://www.kaggle.com/c/state-farm-distracted-driver-detection


## Prerequisites


In order to run the system, you will need the following packages:


* Python 3 or Python 2 (The codes for Gaussian Mixture models is written and run in Python 2.7)
* Numpy 1.13.1
* Sklearn 0.18
* Scipy 1.0.0
* Open-cv 3.3.0
* matplotlib 1.5.1
* Pillow 4.2.1
* Keras 2.0.9
* Tensorflow 1.0.0
* Theano 0.9.0


## Run the systems


Step 1: download the dataset into the same folder with the python scripts. All images are saved in a folder named “imgs.”  Training Data are separated into 10 folders. Each label is labeled by its class label. For example, all images belonging to class label 0 is in the folder named “c0”, and all images belonging to class label 1 is in the folder named “c1.” 


Step 2: download external libraries needed


Step 3: Save python scripts in the same folder where your imgs folder is. 


Step 4: Initialization.py will be run first to read all images in each class folder, randomly splits images into training (80%), validation (20%), and test (20%). Once completed, data will be saved in .npy formats and shared to other team members. 


### Feature Learning


#### Gaussian Mixtures Method


##### Initialization cluster means by running one iteration of K-means


GMM_initialization.py is used to extract the initial centroids for the number of clusters selected. 


As described in the report, we have extracted three feature sets using randomly selected patches from the images. 
* 100 features, 5,000 random patches
* 200 features, 10,000 random patches
* 300 features, 12, 000 random patches 


Before running the GMM_initialization.py, we need to change the number of features and number of random patches as shown above. 


* numRdmPtchs (# of random patches for feature learning)
* numCentroidsKM (# of centroids/features to learn)


To run GMM_initialization.py, open terminal and type “python GMM_initialization.py”.  


##### Use Gaussian Mixture Model to estimate and update means, covariance matrices in each cluster 


To run Gaussian_Mixtures.py, open terminal and type “python Gaussian_Mixtures.py.” Once completed, the system will also save the centroids and covariance matrices into local folder in .npy formats. 


##### Prediction 


Use parameters learned in the previous step to train all patches extracted for all training images. Then used the trained SVM model to predict on validation and test data and get the accuracies for each. 


To run GMM_prediction.py, first load previous saved cluster means and covariance matrices in the last steps. Then change the variable “numCentroidsKM” (number of features) accordingly.  If the number of centroids is 100, then change the variable to 100 in the script. Next, open terminal and type “python Gaussian_Mixtures.py”.


#### Kmeans Clustering Method
To predict the classes using k-means method to extract the features, simply run kmeans.py. 
# Paramters
To change the number of clusters, modify the values of numCentroidsKM. To change the penalty term in the SVM, modify the values of C_svm. To change whether or not the whiten as a preprocess step, change the value of whitening. 
# Construct the clusters
Section 3 of the code kmeans.py derive the centroids using the standard k-means algorithm. 
# Convolutional extraction
Section 4 of the code kmeans.py perform the convolutional extraction and other pre-processing steps. 
# SVM classification
Section 5 of the code kmeans.py perform the SVM to predict the classes of the images based on 


#### Sparse autoencoders
##### Construction of Sparse Autoencoders
SparseAutoencoder.py contains the definition of a class SparseAutoencoder, it is utilized in other python codes.
##### Feature Extraction
SAE_FeatureExtraction.py is used to extract features from the original data. It takes the npy format data files inside the “data” folder as input, builds a sparse autoencoder, and generates an output file named Params.txt, which contains the weights and biases (W1 ,b1, W2, b2) learned by the sparse autoencoder from the raw data. These weights and biases will be used in extracting the features of test data, and the extracted features will be used as the input of soft-margin SVM classification (next step).
##### Classification (soft-margin SVM)
SAE_SVM_Classification.py is used to make predictions on the test dataset. It takes the Params.txt as input and print the classification accuracies on both train and test set as output. This python file first preprocesses on the raw training and testing dataset. Then, using the weights and biases stored in Params.txt, it extracts features of the raw data. Finally, the extracted features are used in building a soft-margin SVM classifier, and we will make predictions on the testing data using this classifier and the extracted features.




#### Sparse Restricted Boltzmann Machine
##### Construction of Sparse Restricted Boltzmann Machine
RBM.py contains the definition of a class RMB, it is utilized in other python codes.


##### Feature Extraction
RBM_FeatureExtraction.py is used to get the weights that maps the input image data to the hidden features. It reads the Sparse Restricted Boltzmann Machine module from RBM.py.


The user can specify the data directory, the size of the image (width x length), and the size of the hidden feature (width x length). Other hyperparameters including max epochs, learning rate alpha, and weight decay controller lamda.


The extraction pipeline is as follows: 1) Initialize modelRBM by assigning RBM object; 2) Read and preprocess the training data, including grey-scaling, whitening, flattening, 0-1 scaling, and shuffling; 3) Execute RBM algorithm; 4) Check results, including learning curve and the output of the hidden and reconstructed layers; 5) Save trained weights to feed into the classification model. These steps are automatically performed when running RBM_FeatureExtraction.py.


##### Training and Prediction
RBM_Prediction.py is used to train and evaluate the classification model. We uses the trained weights to map the input preprocessed image to its hidden features, and feed the hidden features into the linear SVM classifier. It needs to specify the input data and weights directory. It is recommended to put the parameter txt file and “data/” file in the same level with RBM_Prediction.py, and put training, validation, and test array files in the “data/”. We use one fold cross validation to tune the hyperparameter including C, whiten or non-whiten, and the size of hidden nodes.


In all, the setting variables are in the “if __name__ == "__main__":” section.




#### VGG19
##### Training and Prediction
The VGG19 file mainly includes data and trainer. The module is intended to run in the google cloud, but can also be run locally. The setup.py and requirement.txt are used to let the google cloud load these required models, and the empty __init__.py is also required by google cloud. The user can create a bucket in the google cloud storage, and upload the contents in the VGG19 to the bucket. Here is the sample code that I used to submit the training task:
BUCKET_NAME="eecs545bucket"
JOB_NAME="VGG19_train_$(date +%Y%m%d_%H%M%S)" 
JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME 
REGION=us-east1


gcloud ml-engine jobs submit training $JOB_NAME --job-dir $JOB_DIR --runtime-version 1.0 --module-name trainer.task --package-path trainer --region $REGION -- --train-file gs://$BUCKET_NAME/data/ --TRAIN_SIZE 18000 --BATCH_SIZE 32 --EPOCHS 50
The user can check the log file to see the accuracy after each epoch, and see the final validation score.