#!/usr/bin/env python
# Author: D.Adak

import numpy as np
import tensorflow as tf
import pickle
from tqdm import trange
dtype = tf.float32

@tf.keras.utils.register_keras_serializable(
    package="My_emulator"
)
class CustomActivation(tf.keras.layers.Layer):

    def __init__(self, units, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.units = units
        r"""
        Non-linear activation function
        """
    def build(self, input_shape):

        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True
        )

        self.beta = self.add_weight(
            name="beta",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):

        return (
            self.beta
            + (1.0 - self.beta)
            * tf.sigmoid(self.alpha * x)) * x

    def get_config(self):

        config = super().get_config()

        config.update({
            "units": self.units
        })

        return config

class PCAplusNN(tf.keras.Model):
    def __init__(self,
                 cp_pca=None,
                 n_hidden=[512,512,512],
                 restore=False,
                 restore_filename=None,
                 trainable=True,
                 optimizer=None,
                 verbose=False,
                 ):
        r"""
        Initialiser
        """
        super(PCAplusNN, self).__init__()
        if restore is True:
            self.restore(restore_filename)

        else:
            self.cp_pca = cp_pca

            self.parameters = self.cp_pca.parameters_name
            self.n_parameters = len(self.parameters)

            self.pca_transform_matrix_ = self.cp_pca.pca_transform_matrix

            self.modes = self.cp_pca.modes
            self.n_modes = self.cp_pca.n_modes
            self.n_pcas = self.pca_transform_matrix_.shape[0]

            self.n_hidden = n_hidden

            self.architecture = [self.n_parameters] + self.n_hidden + [self.n_pcas]

            self.n_layers = len(self.architecture) - 1

            self.parameters_mean_ = self.cp_pca.parameters_mean
            self.parameters_std_ = self.cp_pca.parameters_std

            self.pca_mean_ = self.cp_pca.pca_mean
            self.pca_std_ = self.cp_pca.pca_std

            self.features_mean_ = self.cp_pca.features_mean
            self.features_std_ = self.cp_pca.features_std
            
        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean_, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std_, dtype=dtype, name='parameters_std')

        # PCA mean and std
        self.pca_mean = tf.constant(self.pca_mean_, dtype=dtype, name='pca_mean')
        self.pca_std = tf.constant(self.pca_std_, dtype=dtype, name='pca_std')

        # (log)-spectra mean and std
        self.features_mean = tf.constant(self.features_mean_, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std_, dtype=dtype, name='features_std')

        # pca transform matrix
        self.pca_transform_matrix = tf.constant(self.pca_transform_matrix_, dtype=dtype, name='pca_transform_matrix')

        # ==========================================
        # Keras neural network
        # ==========================================
        if restore is False:
            self.network = tf.keras.Sequential(name="emulator_network")

            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=np.sqrt(2.0 / self.n_parameters)
            )

            for i in range(self.n_layers):

                # Dense layer
                self.network.add(
                    tf.keras.layers.Dense(
                        self.architecture[i+1],
                        kernel_initializer=kernel_initializer,
                        bias_initializer="zeros",
                        trainable=trainable,
                        name=f"dense_{i}"
                    )
                )

                # Custom activation weights for hidden layers only
                if i < self.n_layers - 1:

                    self.network.add(
                        CustomActivation(
                            self.architecture[i+1],
                            trainable=trainable,
                            name=f"activation_{i}"
                        )
                    )
        self.optimizer = optimizer or tf.keras.optimizers.legacy.Adam()
        self.verbose = verbose
        # print initialization info, if verbose
        if self.verbose:
            message = "\nInitialized PCAplusNN model, \n" \
                            f"The model maps {self.n_parameters} input parameters to {self.n_pcas} PCA coefficients \n" \
                            f"and then invert the PCA coefficients to obtain power spectrum for {self.n_modes} modes \n" \
                            f"The model uses {len(self.n_hidden)} hidden layers, \n" \
                            f"with {list(self.n_hidden)} neuron, respectively. \n"
            print(message)

    def forward_pass_tf(self, parameters_tensor):
        r"""
        Forward pass through the network to predict the PCA coefficients,
        fully implemented in TensorFlow

        Parameters:
            parameters_tensor (Tensor):
                tensor of shape (number_of_cosmologies, number_of_cosmological_parameters)

        Returns:
            Tensor:
                PCA predictions
        """
        # Standardise input parameters
        x = (parameters_tensor - self.parameters_mean) / self.parameters_std

        # Keras neural network
        x = self.network(x)

        # Rescale predicted PCA coefficients
        x = (
            x * self.pca_std
            + self.pca_mean
        )

        return x
        
    def predictions_tf_(self, parameters_dict):
        r"""
        Predictions of log(spectra) given dict of input parameters,
        

        Parameters:
            parameters_dict (dict):
                input parameters

        Returns:
            Tensor:
                output predictions
        """
        
        
        parameters_arr = self.organise_params(parameters_dict)
        parameters_tensor = tf.convert_to_tensor(
        parameters_arr,
        dtype=dtype)#tf.constant(parameters_arr,dtype=dtype)
        
        

        pca_coefficients = self.forward_pass_tf(
                parameters_tensor
            )

        return (
            tf.matmul(
                pca_coefficients,
                self.pca_transform_matrix
            )
            * self.features_std
            + self.features_mean
        )

    #@tf.function
    def predictions_tf(self, parameters_dict):
        r"""
        Predictions of log(spectra) given dict of input parameters,
        

        Parameters:
            parameters_dict (dict):
                input parameters

        Returns:
            Tensor:
                output predictions
        """
        
        """
        parameters_arr = self.organise_params(parameters_dict)
        parameters_tensor = tf.convert_to_tensor(
        parameters_arr,
        dtype=dtype)#tf.constant(parameters_arr,dtype=dtype)
        """
        if isinstance(parameters_dict, dict):

            parameters_arr = self.organise_params(parameters_dict)

            parameters_tensor = tf.convert_to_tensor(
                parameters_arr,
                dtype=dtype
            )

        else:

            parameters_tensor = parameters_tensor = parameters_dict

        pca_coefficients = self.forward_pass_tf(
                parameters_tensor
            )

        return (
            tf.matmul(
                pca_coefficients,
                self.pca_transform_matrix
            )
            * self.features_std
            + self.features_mean
        )

    #@tf.function
    def ten_to_predictions_tf(self, parameters_dict):
        r"""
        predicted power-spectra  for given dict of input parameters,
        fully implemented in TensorFlow. It raises 10 to the output
        of ``predictions_tf``

        Parameters:
            parameters (dict):
                input parameters

        Returns:
            Tensor:
               10^output predictions
        """
        return tf.pow(
            10.,
            self.predictions_tf(parameters_dict)
        )
    def ten_to_predictions_np(self, parameters_dict):
        r"""
                predicted power-spectra  for given dict input parameters,
        

        Parameters:
            parameters (dict):
                input parameters

        Returns:
            Tensor:
               10^output predictions
        """
        log_ptrdict = self.predictions_tf(parameters_dict).numpy()
        return 10**log_ptrdict
        
        
    def predictions_np(self, parameters_dict):
        r"""
            predicted power-spectra  for given dict input parameters,
    

        Parameters:
            parameters (dict):
                input parameters

        Returns:
            Tensor:
            10^output predictions
        """
        ptrdict = self.predictions_tf_(parameters_dict).numpy()
        return ptrdict
    ## loss function
    @tf.function
    def compute_loss(self,
                     training_parameters,
                     training_pca):
        r"""
        compute loss function 
        """
        return tf.sqrt(
            tf.reduce_mean(
                tf.math.squared_difference(
                    self.forward_pass_tf(training_parameters),
                    training_pca
                )
            )
        )
    # compute loss and gradient
    @tf.function
    def compute_loss_and_gradients(self,
                                   training_parameters,
                                   training_pca):

        with tf.GradientTape() as tape:

            loss = tf.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(
                        self.forward_pass_tf(training_parameters),
                        training_pca
                    )
                )
            )

        gradients = tape.gradient(
            loss,
            self.network.trainable_variables
        )

        return loss, gradients
    # back-propagation from computed loss and gradients
    @tf.function
    def training_step(self,
                  training_parameters,
                  training_pca):
        r"""
        Optimizes loss

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_pca (Tensor):
                true PCA components

        Returns:
            loss (Tensor):
                mean squared difference
        """

        loss, gradients = self.compute_loss_and_gradients(
            training_parameters,
            training_pca
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_variables)
        )

        return loss
    #function to sort input parameters to make the saved parameter_file order and given order of parameter array
    def organise_params(self,
                               input_dict,
                               ):
        r"""
        Sort input parameters

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)
    


    #
    ## restore the trained model and relevent metric related to PCA inverse transformation
    def restore(self, filename):

        # ==========================================
        # Load metadata
        # ==========================================

        with open(filename + ".pkl", "rb") as f:
            metadata = pickle.load(f)

        self.parameters = metadata["parameters"]
        self.n_parameters = metadata["n_parameters"]

        self.modes = metadata["modes"]
        self.n_modes = metadata["n_modes"]

        self.n_pcas = metadata["n_pcas"]

        self.n_hidden = metadata["n_hidden"]
        self.n_layers = metadata["n_layers"]
        self.architecture = metadata["architecture"]


        # ==========================================
        # PCA / normalization quantities
        # ==========================================

        self.parameters_mean_ = metadata["parameters_mean"]
        self.parameters_std_ = metadata["parameters_std"]

        self.pca_mean_ = metadata["pca_mean"]
        self.pca_std_ = metadata["pca_std"]

        self.features_mean_ = metadata["features_mean"]
        self.features_std_ = metadata["features_std"]

        self.pca_transform_matrix_ = \
            metadata["pca_transform_matrix"]


        # ==========================================
        # TensorFlow constants
        # ==========================================

        self.parameters_mean = tf.constant(
            self.parameters_mean_,
            dtype=dtype
        )

        self.parameters_std = tf.constant(
            self.parameters_std_,
            dtype=dtype
        )

        self.pca_mean = tf.constant(
            self.pca_mean_,
            dtype=dtype
        )

        self.pca_std = tf.constant(
            self.pca_std_,
            dtype=dtype
        )

        self.features_mean = tf.constant(
            self.features_mean_,
            dtype=dtype
        )

        self.features_std = tf.constant(
            self.features_std_,
            dtype=dtype
        )

        self.pca_transform_matrix = tf.constant(
            self.pca_transform_matrix_,
            dtype=dtype
        )


#        # ==========================================
#        # Reconstruct Keras network
#        # ==========================================
         #$$$$$$$$$$$$$$$
#        self.network = tf.keras.models.load_model(
#            filename + ".keras",
#            custom_objects={
#                "CustomActivation": CustomActivation
#            })
        self.network = tf.keras.models.load_model(
            filename + ".keras",
            )
        #@@@@@@@@@@@@@@@@@@@@@@@@
#        self.network = tf.keras.Sequential(
#            name="emulator_network"
#        )
#
#        kernel_initializer = tf.keras.initializers.RandomNormal(
#            mean=0.0,
#            stddev=np.sqrt(
#                2.0 / self.n_parameters
#            )
#        )
#
#        for i in range(self.n_layers):
#
#            self.network.add(
#                tf.keras.layers.Dense(
#                    self.architecture[i + 1],
#                    kernel_initializer=kernel_initializer,
#                    bias_initializer="zeros",
#                    name=f"dense_{i}"
#                )
#            )
#
#            if i < self.n_layers - 1:
#
#                self.network.add(
#                    CustomActivation(
#                        self.architecture[i + 1],
#                        name=f"activation_{i}"
#                    )
#                )
#
#
#        # ==========================================
#        # Build network
#        # ==========================================
#
#        dummy_input = tf.zeros(
#            (1, self.n_parameters),
#            dtype=dtype
#        )
#
#        self.network(dummy_input)
#
#
#        # ==========================================
#        # Load trained Keras weights
#        # ==========================================
#
#        self.network.load_weights(
#            filename + ".weights.h5"
#        )
    #save the trained model and relevent metric related to PCA inverse transformation
    def save(self, filename):

        # Save Keras network weights
        self.network.save_weights(
            filename + ".weights.h5"
        )
        ###
        self.network.save(filename + ".keras")
        # Save PCA / emulator metadata
        metadata = {
            "parameters": self.parameters,
            "n_parameters": self.n_parameters,

            "modes": self.modes,
            "n_modes": self.n_modes,

            "n_pcas": self.n_pcas,

            "parameters_mean": self.parameters_mean.numpy(),
            "parameters_std": self.parameters_std.numpy(),

            "pca_mean": self.pca_mean.numpy(),
            "pca_std": self.pca_std.numpy(),

            "features_mean": self.features_mean.numpy(),
            "features_std": self.features_std.numpy(),

            "pca_transform_matrix":
                self.pca_transform_matrix.numpy(),

            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
            "architecture": self.architecture,
        }

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(metadata, f)
    def update_emulator_parameters(self):
        r"""
        Update emulator parameters before saving them
        """


        # put shift and scale parameters to numpy arrays
        self.parameters_mean_ = self.parameters_mean.numpy()
        self.parameters_std_ = self.parameters_std.numpy()
        self.pca_mean_ = self.pca_mean.numpy()
        self.pca_std_ = self.pca_std.numpy()
        self.features_mean_ = self.features_mean.numpy()
        self.features_std_ = self.features_std.numpy()

        # pca transform matrix
        self.pca_transform_matrix_ = self.pca_transform_matrix.numpy()
    # ==========================================
    #         main TRAINING function
    # ==========================================
    def train(self,
              filename_saved_model,
              # cooling schedule
              validation_split=0.1,
              learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              batch_sizes=[1024, 1024, 1024, 1024, 1024],
              gradient_accumulation_steps = [1, 1, 1, 1, 1],
              # early stopping set up
              patience_values = [100,100,100,100,100],
              max_epochs = [1000,1000,1000,1000,1000],
             ):
        r"""
        Train the model

        Parameters:
            filename_saved_model (str):
                filename tag where model will be saved
            validation_split (float):
                percentage of training data used for validation
            learning_rates (list [float]):
                learning rates for each step of learning schedule
            batch_sizes (list [int]):
                batch sizes for each step of learning schedule
            gradient_accumulation_steps (list [int]):
                batches for gradient accumulations for each step of learning schedule
            patience_values (list [int]):
                early stopping patience for each step of learning schedule
            max_epochs (list [int]):
                maximum number of epochs for each step of learning schedule
        """
        # check correct number of steps
        assert len(learning_rates)==len(batch_sizes)\
               ==len(gradient_accumulation_steps)==len(patience_values)==len(max_epochs), \
               'Number of learning rates, batch sizes, gradient accumulation steps, patience values and max epochs are not matching!'

        # training start info, if verbose
        if self.verbose:
            message = "Starting PCAplusNN training, \n" \
                            f"using {int(100*validation_split)} per cent of training samples for validation. \n" \
                            f"Performing training with   {len(learning_rates)} learning steps, with \n" \
                            f"{list(learning_rates)} trailing learning rates \n" \
                            f"{list(batch_sizes)} batch sizes \n" \
                            f"{list(gradient_accumulation_steps)} gradient accumulation steps (this function is not defined ) \n" \
                            f"Recommend to use gradient_accumulation_steps = [1] for all learning steps\n"\
                            f"{list(patience_values)} patience values \n" \
                            f"{list(max_epochs)} max epochs \n"
            print(message)

        # casting
        training_parameters = tf.convert_to_tensor(self.cp_pca.training_parameters, dtype=dtype)
        training_pca = tf.convert_to_tensor(self.cp_pca.training_pca, dtype=dtype)

        # training/validation split
        n_validation = int(training_parameters.shape[0] * validation_split)
        n_training = training_parameters.shape[0] - n_validation
        ### starting traiuning and validation loss###
        training_loss = [np.infty]
        validation_loss = [np.infty]
        best_loss = np.infty
        where_stops=[]
        # train using cooling/heating schedule for lr/batch-size
        for i in range(len(learning_rates)):

            print('learning rate = ' + str(learning_rates[i]) + ', batch size = ' + str(batch_sizes[i]))

            # set learning rate
            self.optimizer.lr = learning_rates[i]

            # split into validation and training sub-sets
            training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

            # create iterable dataset (given batch size)
            training_data = tf.data.Dataset.from_tensor_slices((training_parameters[training_selection], training_pca[training_selection])).shuffle(n_training).batch(batch_sizes[i])

            # set up training loss
#            training_loss = [np.infty]
#            validation_loss = [np.infty]
#            best_loss = np.infty
            early_stopping_counter = 0

            # loop over epochs
            
            with trange(max_epochs[i]) as t:
                # Check weights at START of epoch, start with ((i!=0)) since for i==0 weights are not yet defined unless self.training_step () is called
#                if(i!=0):
#                    weights_start = self.network.get_weights()
#
#                    print(
#                        f"\nEpoch {epoch}: "
#                        f"first weight = {weights_start[0].flat[0]}"
#                    )
                for epoch in t:
                    
                    # loop over batches
                    epoch_training_loss = 0.0
                    n_batches_epoch = 0
                    for theta, pca in training_data:

                        # training step: check whether to accumulate gradients or not (only worth doing this for very large batch sizes)
                        if gradient_accumulation_steps[i] == 1:
                            loss = self.training_step(theta, pca)
                        else:
                            print(f"Recommend to use gradient_accumulation_steps = [1] for all learning steps since\n training_step_with_accumulated_gradients is not defined in this version of code\n" )
                            loss = self.training_step_with_accumulated_gradients(theta, pca, accumulation_steps=gradient_accumulation_steps[i])
                        ## count the loss after each batch and add one to count 'n_batches_epoch' that epoch has used
                        epoch_training_loss += loss.numpy()
                        n_batches_epoch += 1
                    # now append the average loss over batch
                    training_loss.append(epoch_training_loss / n_batches_epoch)
                    # compute validation loss at the end of the epoch
                    validation_loss.append(self.compute_loss(training_parameters[~training_selection], training_pca[~training_selection]).numpy())

                    # update the progressbar
                    t.set_postfix(loss=validation_loss[-1])
                    
                
                    # early stopping condition
                    if validation_loss[-1] < best_loss:
                        best_loss = validation_loss[-1]
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    if early_stopping_counter >= patience_values[i]:
                        self.update_emulator_parameters()
                        self.save(filename_saved_model)
                        #where_stops.append(epoch)###TODO: this might be bug ,,,why saved the epoch twice?
                        #np.savetxt(filename_saved_model + "early_stop_epochs.txt",where_stops)
                        print(f"Reached early stop criteria at epoch: {epoch}."
                         f"Validation loss = {best_loss}")
                        #print('Validation loss = ' + str(best_loss))
                        print('Model saved.')
                        np.savez(filename_saved_model + "_history.npz",training_loss=np.array(training_loss),validation_loss=np.array(validation_loss))
                        print('Training and validation loss are saved.')
                        ##### Check weights at END of epoch
#                        weights_end = self.network.get_weights()
#
#                        print(
#                            f"Epoch {epoch}: "
#                            f"last weight = {weights_end[0].flat[0]}"
#                        )
#
#                        # Check whether weights changed during this epoch
#                        if(i!=0):
#                            changed = any(
#                                not np.array_equal(w_start, w_end)
#                                for w_start, w_end in zip(
#                                    weights_start,
#                                    weights_end
#                                )
#                            )
#
#                            print("Weights changed during epoch:", changed)
                        break
                        
                self.update_emulator_parameters()
                self.save(filename_saved_model)
                print('Reached max number of epochs. Validation loss = ' + str(best_loss))
                print('Model saved.')
                np.savez(filename_saved_model + "_history.npz",training_loss=np.array(training_loss),validation_loss=np.array(validation_loss))
                print('Training and validation loss are saved.')
                where_stops.append(epoch) ###TODO: this might be bug ,,,why saved the epoch twice?
                np.savetxt(filename_saved_model + "early_stop_epochs.txt",where_stops)
