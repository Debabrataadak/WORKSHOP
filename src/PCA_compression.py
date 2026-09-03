#!bash/bin

import numpy as np
import pickle
from tqdm import trange
from sklearn.decomposition import IncrementalPCA



class PCA():
    r"""
    Principal Component Analysis of (log)-power spectra    
    """

    def __init__(self,
                 parameters_name,
                 modes,
                 n_pcas,
                 parameters_filenames,
                 features_filenames,
                 verbose=False,
                 ):
        r"""
        Initialiser:
        
        parameters (list):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles(ell) or k-values in the (log)-spectra
        n_pcas (int):
            number of PCA components
        parameters_filenames (list [str]):
            list of .npz filenames for parameters
        features_filenames (list [str]):
            list of .npz filenames for (log)-spectra
        verbose (bool):
            whether to print messages at intermediate steps or not
        """
        # attributes
        self.parameters_name = parameters_name
        self.n_parameters = len(parameters_name)
        self.modes = modes
        self.n_modes = len(self.modes)
        self.n_pcas = n_pcas
        self.parameters_filenames = parameters_filenames
        self.features_filenames = features_filenames
        self.n_batches = len(self.parameters_filenames)

        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)

        # verbose
        self.verbose= verbose

        # print initialization info, if verbose
        if self.verbose:
            print(f"\nInitialized PCA compression with {self.n_pcas} components with given inputs\n")


    #function to sort input parameters
    def organise_params(self,input_dict,):
        r"""
        Sort input parameters

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                input parameters sorted according to `parameters`
        """
        if self.parameters_name is not None:
            return np.stack([input_dict[k] for k in self.parameters_name], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)


    # compute mean and std for (log)-spectra and parameters in batches
    def standardise_features_and_parameters(self):
        r"""
        Compute mean and std for (log)-spectra and parameters
        """
        # mean and std
        self.features_mean = np.zeros(self.n_modes)
        self.features_std = np.zeros(self.n_modes)
        self.parameters_mean = np.zeros(self.n_parameters)
        self.parameters_std = np.zeros(self.n_parameters)

        # loop over training data files, accumulate means and stds
        for i in range(self.n_batches):

            features = np.load(self.features_filenames[i] + ".npz")['features']
            parameters = self.organise_params(np.load(self.parameters_filenames[i] + ".npz"))

            # accumulate
            self.features_mean += np.mean(features, axis=0)/self.n_batches
            self.features_std += np.std(features, axis=0)/self.n_batches
            self.parameters_mean += np.mean(parameters, axis=0)/self.n_batches
            self.parameters_std += np.std(parameters, axis=0)/self.n_batches
        del features
        del parameters
        if self.verbose:
            print(f"\n mean and standard deviation computation is done\n")
    # train PCA incrementally
    def train_pca(self):
        r"""
        Train PCA incrementally
        """
        # loop over training data files, increment PCA
        with trange(self.n_batches) as t:
            for i in t:
                # load (log)-spectra and mean+std
                features = np.load(self.features_filenames[i] + ".npz")['features']
                normalised_features = (features - self.features_mean)/self.features_std

                # partial PCA fit
                self.PCA.partial_fit(normalised_features)
        del features
        # set the PCA transform matrix or eigen vectors
        self.pca_transform_matrix = self.PCA.components_


    # transform the training data set to PCA basis
    def transform_and_stack_training_data(self, 
                                          filename = './tmp', 
                                          retain = True,
                                          ):
        r"""
        Compress the training data set to PCA coefficients

        Parameters:
            filename (str):
                filename tag (no suffix) for PCA coefficients and parameters
            retain (bool):
                whether to retain training data as attributes
        """
        if self.verbose:
            print("starting PCA compression")
        print("Lets initialise the mean, std of features and parameters and compute PCA eigen-vectors")
        self.standardise_features_and_parameters()
        self.train_pca()

        # transform the (log)-spectra to PCA basis
        training_pca = np.concatenate([self.PCA.transform((np.load(self.features_filenames[i] + ".npz")['features'] - self.features_mean)/self.features_std) for i in range(self.n_batches)])

        # stack the input parameters
        training_parameters = np.concatenate([self.organise_params(np.load(self.parameters_filenames[i] + '.npz')) for i in range(self.n_batches)])

        # mean and std of PCA basis
        self.pca_mean = np.mean(training_pca, axis=0)
        self.pca_std = np.std(training_pca, axis=0)

        # save stacked transformed training data
        self.pca_filename = filename
        np.save(self.pca_filename + '_pca.npy', training_pca)
        np.save(self.pca_filename + '_parameters.npy', training_parameters)

        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters
        if self.verbose:
            print("PCA compression done and saved")
            if retain:
                print("parameters and PCA coefficients of training set stored in memory")


    # validate PCA given some validation data
    def validate_pca_basis(self,
                           features_filename,
                           ):
        r"""
        Validate PCA given some validation data

        Parameters:
            features_filename (str):
                filename tag (no suffix) for validation (log)-spectra

        Returns:
            features_pca (numpy.ndarray):
                PCA of validation (log)-spectra
            features_in_basis (numpy.ndarray):
                inverse PCA transform of validation (log)-spectra
        """
        # load (log)-spectra and standardise
        self.standardise_features_and_parameters()
        self.train_pca()
        features = np.load(features_filename + ".npz")['features']
        normalised_features = (features - self.features_mean)/self.features_std

        # transform to PCA basis and back
        features_pca = self.PCA.transform(normalised_features)
        features_in_basis = np.dot(features_pca, self.pca_transform_matrix)*self.features_std + self.features_mean

        # return PCA coefficients and (log)-spectra in basis
        return features_pca, features_in_basis
