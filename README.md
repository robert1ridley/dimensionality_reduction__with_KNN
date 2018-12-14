# PCA, SVD and Isomap Dimensionality Reduction with 1-NN for Classification

This repository contains code for three different dimensionality-reduction techniques, 
PCA (Principal Component Analysis), SVD (Single Value Decomposition) and Isomap.
There is also a 1-Nearest Neighbors algorithm which is used to classify the 
reduced-dimensionality test data. There are two sets of data, `sonar` and `splice`.
These data sets can be found in the `/data` folder.

## Requirements

* Python version: 3.5.1

## Start Developing

After cloning the repository:

* Setting up the environment：
    - `cd dimensionality_reduction__with_KNN`
    - Create a virtual environmnet: `python3 -m venv venv`
    - `source venv/bin/activate`
    - Install the project dependencies：`pip install –r requirements.txt`

* Start the PCA test:
    - Ensure that you are inside `/dimensionality_reduction__with_KNN` and that your virtual environment is running.
    - Enter `python tests/pca_test.py <training dataset filepath> <test dataset filepath> <number of dimentions>`.
    For example, to run a test on the `sonar` dataset with parameter dimensions reduced to 10, enter 
    `python tests/pca_test.py data/sonar-train.txt data/sonar-test.txt 10`. This will output the 1-NN accuracy in the terminal.
    - Deactivate your virtual environment by entering `deactivate`.
    
* Start the SVD test:
    - Ensure that you are inside `/dimensionality_reduction__with_KNN` and that your virtual environment is running.
    - Enter `python tests/svd_test.py <training dataset filepath> <test dataset filepath> <number of dimentions>`.
    For example, to run a test on the `sonar` dataset with parameter dimensions reduced to 10, enter 
    `python tests/svd_test.py data/sonar-train.txt data/sonar-test.txt 10`. This will output the 1-NN accuracy in the terminal.
    - Deactivate your virtual environment by entering `deactivate`.
    
 * Start the Isomap test:
    - Ensure that you are inside `/dimensionality_reduction__with_KNN` and that your virtual environment is running.
    - Enter `python tests/isomap_test.py <training dataset filepath> <test dataset filepath> <number of dimentions> <number of neighbors for distance calculation>`.
    For example, to run a test on the `sonar` dataset with parameter dimensions reduced to 10 and the gensim distance for the 4 nearest neighbors, enter 
    `python tests/isomap_test.py data/sonar-train.txt data/sonar-test.txt 10 4`. This will output the 1-NN accuracy in the terminal.
    - Deactivate your virtual environment by entering `deactivate`.
