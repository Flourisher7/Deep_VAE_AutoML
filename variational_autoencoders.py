# Import necessary libraries and modules
import pandas as pd  # Import Pandas library and alias it as 'pd'
import numpy as np  # Import NumPy library and alias it as 'np'
import matplotlib.pyplot as plt  # Import Matplotlib's pyplot module and alias it as 'plt'
from pandas import DataFrame  # Import the DataFrame class from Pandas
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler from scikit-learn
import warnings  # Import the warnings module
import os  # Import the os module for operating system functions
import sidetable as stb

# Set TensorFlow logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # Import TensorFlow and alias it as 'tf'
from tensorflow import keras  # Import the Keras API from TensorFlow
from keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D  # Import specific Keras layers
from keras.layers import Flatten, MaxPooling1D, Reshape, Conv1DTranspose  # More Keras layers
from keras.optimizers import Adam, SGD, Adagrad  # Import different optimizers
from keras import backend as k  # Import Keras backend and alias it as 'k'
from keras.layers import RepeatVector, Lambda  # More Keras layers
from math import floor  # Import the 'floor' function from the math module

# Suppress Python warnings for a cleaner output
warnings.filterwarnings('ignore')


class DeepLearning:
    """
    This class encapsulates the neural networks models for time-series data types.
    """
    # Set Random Seed
    tf.random.set_seed(1234)  # random seed value for reproducible result
    # NN model hyperparameters
    STEP_SIZE = 1               # Step size for data sampling
    SPLIT_FRACTION = 0.99       # Fraction for data splitting
    PAST = 240                  # Number of past data points to consider
    FUTURE = 24                 # Number of future data points to predict
    LEARNING_RATE = 0.0001      # Learning rate for training
    BATCH_SIZE = 64             # Batch size for training
    EPOCHS = 100                # Number of training epochs
    VALIDATION_SPLIT = 0.2      # Fraction of data used for validation
    HIDDEN_LAYERS = [32, 16]    # Structure of hidden layers
    NUM_FEATURES = 3            # Number of input features
    ACTIVATION = 'relu'         # Activation function
    OPTIMIZER = 'ADAM'          # Optimizer for training
    FREQUENCY = 1               # Sampling frequency
    DEEP_NN = "cnn1d"           # Type of neural network
    LATENT_DIM = 6  # Dimension of the latent space
    VERBOSE = 0  # Verbosity level

    def __init__(self):
        """
        Initialize the DeepLearning class and perform data preprocessing.

        Reads and loads CSV files, handles missing data, and sets hyperparameters.
        :param model: a neural network model such as LSTM
        """
        # Define the output folder name and file name
        self.folder_results = "Results"
        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(self.folder_results):
            os.makedirs(self.folder_results)

        # read and load CSV file in a DataFrame
        self.raw_process_data = pd.read_csv(r'Data/raw_data.csv', na_values=' ', encoding='unicode_escape')
        self.raw_effluent = pd.read_csv(r'Data/raw_data.csv', na_values=' ', encoding='unicode_escape')
        self.feature_keys = ['Effluent Stage 1 Channel Ammonia_(mg/L)', 'BR5 G2 Zone Flow_(m3/hr)',
                             'BR6 Swing Zone Flow_(m3/hr)', 'BR2 G1 Zone Flow_(m3/hr)', 'BR1 pH Meter',
                             'BR2 pH Meter', 'BR3 pH Meter', 'BR5 pH Meter', 'BR6 pH Meter',
                             'Dig.5 Temp (C)', 'Dig.6 Temp (C)', 'Dig.3 Gas Flow (m3/hr)',
                             'Dig.4  Gas Flow (m3/hr)', 'Inlet flow (L/s)',
                             'Stage 1 Effluent Channel Nitrate_(mg/L)', 'Dig.6 Gas Flow (m3/hr)',
                             'BR5 G1 Zone Flow_(m3/hr)']
        self.features = [self.raw_process_data.columns[0:2], self.feature_keys]
        self.missing_value_data = self.raw_process_data[self.feature_keys]
        self.missing_value_data.to_csv(self.folder_results+'/Missing_Data.csv')
        missing_report = self.missing_value_data.stb.missing()
        missing_report.to_csv(self.folder_results+'/Missing_Data_Report.csv')
        obs_row_index = np.where(np.isfinite(np.sum(self.missing_value_data.values, axis=1)))
        nan_row_index = np.where(np.isnan(np.sum(self.missing_value_data.values, axis=1)))
        self.nan_index = np.where(np.isnan(self.missing_value_data))
        self.data_complete = np.copy(self.missing_value_data.values[obs_row_index[0], :])
        index = self.raw_process_data['Date (day)']  # )[obs_row_index]
        self.data_complete = DataFrame(self.data_complete, columns=self.raw_process_data[self.feature_keys].columns)
        self.data_complete.to_csv(self.folder_results+'/Complete_Data.csv')
        self.data_complete2 = self.raw_process_data.values[obs_row_index[0], :]
        self.data_complete2 = DataFrame(self.data_complete2, columns=self.raw_process_data.columns)
        self.data_complete2.to_csv(self.folder_results+'/Complete_Data2.csv')

        self.data_incomplete = np.copy(self.raw_process_data.values[nan_row_index[0], :])
        DataFrame(self.data_incomplete, columns=self.raw_process_data.columns).\
            to_csv(self.folder_results+'/Incomplete_Data.csv')
        data_summary = DataFrame(self.missing_value_data.describe(percentiles=[.25, .5, .75], include=None,
                                                                  exclude=None, datetime_is_numeric=False))
        data_summary.to_csv(self.folder_results+'/Missing_Data_Summary.csv')
        self.interpolated_data = pd.concat([self.missing_value_data.select_dtypes(include=['object']).
                                           fillna(method='backfill'), self.missing_value_data.
                                           select_dtypes(include=['float']).interpolate()], axis=1)
        self.interpolated_data.to_csv(self.folder_results+'/Interpolated_Data.csv')

        reconstructed_data = self.missing_value_data.fillna(self.interpolated_data)
        DataFrame(reconstructed_data).to_csv(self.folder_results+'/Reconstructed_Data.csv')
        # Raw Data Visualization
        self.titles = ["Inflow Total (ML/d)", "DIG 1 Raw Flow Rate (m3/d)", "RST Thicknd Sludge Volume",
                       "DAF12 TWAS Volume"]
        self.target_key = self.feature_keys
        self.feature_keys_pred = self.feature_keys
        self.colors = ['blue', 'orange', 'green', 'red']
        self.time_key = "Date (day)"
        self.split_fraction = self.SPLIT_FRACTION

    def scale(self, n):
        """
        Scale the data and create sequences for training.

        Args:
            n (int): The interval at which to sample the data.

        Returns:
            X_train (array): Scaled input data.
            y_train (array): Scaled target data.
            scaler_influent: Scaler used for data scaling.
        """
        process_data = self.data_complete2[self.feature_keys][0::n]

        # Data transformation and scaling
        scaler_influent = MinMaxScaler().fit(process_data)
        train_scaled_influent = scaler_influent.transform(process_data)
        train_scaled_effluent = scaler_influent.transform(process_data)
        X_train, y_train = self.create_sequences(train_scaled_influent,
                                                 train_scaled_effluent, self.STEP_SIZE)

        return X_train, y_train, scaler_influent

    def scale2(self, n):
        """
        Scale the data and create sequences for training.

        Args:
            n (int): The interval at which to sample the data.

        Returns:
            X_train (array): Scaled input data.
            y_train (array): Scaled target data.
            scaler_influent: Scaler used for data scaling.
        """
        process_data = self.data_incomplete[self.feature_keys][0::n]
        process_data.to_csv(self.folder_results+'/Process_Data.csv')
        # Data transformation and scaling
        scaler_influent = MinMaxScaler().fit(process_data)
        train_scaled_influent = scaler_influent.transform(process_data)
        train_scaled_effluent = scaler_influent.transform(process_data)
        X_train, y_train = self.create_sequences(train_scaled_influent,
                                                 train_scaled_effluent, self.STEP_SIZE)

        return X_train, y_train, scaler_influent

    def convert_params(self, params):
        """
        Convert a list of parameters into a set of hyperparameters for the neural network model.

        Args:
            params (list): List of float parameters that define the hyperparameters.

        Returns:
            tuple: A tuple containing the following hyperparameters:
                - hidden_layer_sizes (list): List of hidden layer sizes.
                - activation (str): Activation function for the model.
                - optimizer (str): Optimizer used for training.
                - learning_rate (float): Learning rate for training.
                - deep_nn (str): Type of deep neural network.
                - latent_dim (int): Dimension of the latent space.
                - epochs (int): Number of training epochs.
        """
        # transform the layer sizes from float (possibly negative) values into hiddenLayerSizes tuples:
        if round(params[1]) <= 0:
            hidden_layer_sizes = [round(params[0])]
        elif round(params[2]) <= 0:
            hidden_layer_sizes = [round(params[0]), round(params[1])]
        elif round(params[3]) <= 0:
            hidden_layer_sizes = [round(params[0]), round(params[1]), round(params[2])]
        else:
            hidden_layer_sizes = [round(params[0]), round(params[1]), round(params[2]), round(params[3])]

        activation = ['tanh', 'relu', 'sigmoid'][floor(params[4])]
        optimizer = ['SDG', 'ADAM', 'Adagrad'][floor(params[5])]
        learning_rate = params[6]
        deep_nn = ['bi_lstm', 'lstm', 'gru', 'cnn1d'][floor(params[7])]
        latent_dim = round(params[8])
        epochs = round(params[9]) * 50
        return hidden_layer_sizes, activation, optimizer, learning_rate, deep_nn, latent_dim, epochs

    def get_mse(self, params):
        """
        Get the Mean Squared Error (MSE) of the neural network model.

        Args:
            params (list): List of float parameters defining the hyperparameters.

        Returns:
            list: A list containing the calculated MSE.
        """
        # Convert parameters into hyperparameters
        HIDDEN_LAYER_SIZES, ACTIVATION, OPTIMIZER, LEARNING_RATE, DEEP_NN, LATENT_DIM, EPOCHS = \
            self.convert_params(params)

        # Preprocess and scale the data
        X_train, y_train = self.scale(self.FREQUENCY)[0:2]

        # Create a Variational Autoencoder (VAE) based on the selected deep neural network type
        if DEEP_NN == 'bi_lstm':
            vae, enc, dec = self.bi_lstm_vae(X_train, HIDDEN_LAYER_SIZES, ACTIVATION, LATENT_DIM,
                                             LEARNING_RATE, OPTIMIZER)
        elif DEEP_NN == 'lstm':
            vae, enc, dec = self.lstm_vae(X_train, HIDDEN_LAYER_SIZES, ACTIVATION, LATENT_DIM,
                                             LEARNING_RATE, OPTIMIZER)
        elif DEEP_NN == 'gru':
            vae, enc, dec = self.gru_vae(X_train, HIDDEN_LAYER_SIZES, ACTIVATION, LATENT_DIM,
                                             LEARNING_RATE, OPTIMIZER)
        else:
            vae, enc, dec = self.cnn1d_vae(X_train, HIDDEN_LAYER_SIZES, ACTIVATION, LATENT_DIM,
                                             LEARNING_RATE, OPTIMIZER)

        # Fit the model and calculate MSE
        history = self.fit_model(vae, X_train, X_train, EPOCHS)
        mse = vae.history.history['val_mse']

        return [np.abs(history.history['val_mse']).mean()]

    def create_bigru(self, x_train, hidden_layers, activation):
        """
        Create a Bidirectional Gated Recurrent Unit (BiGRU) model.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of units in each hidden layer.
            activation (str): Activation function for the hidden layers.

        Returns:
            tf.keras.Model: A Bidirectional Gated Recurrent Unit (BiGRU) model.
        """
        size = len(hidden_layers)
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

        # First hidden layer with BiDirectional GRU
        layer1 = Bidirectional(GRU(units=hidden_layers[size - 2], activation=activation))
        x = layer1(inputs)
        print(x)

        if size == 2:
            # Second hidden layer
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 3:
            # Second hidden layer
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
            # Third layer
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 4:
            # Second hidden layer
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
            # Third layer
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
            # Fourth layer
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)

        outputs = (Dense(1))(x)

        # Create a model by specifying its inputs and outputs in the graph layers
        model = keras.Model(inputs=inputs, outputs=outputs, name="create_bigru")
        model.summary()

        # Compile the model
        if self.OPTIMIZER == 'ADAM':
            opt = Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'SDG':
            opt = SGD(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'Adagrad':
            opt = Adagrad(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])

        return model

    def create_bilstm(self, x_train, hidden_layers, activation):
        """
        Create a Bidirectional Long Short-Term Memory (BiLSTM) model.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of units in each hidden layer.
            activation (str): Activation function for the hidden layers.

        Returns:
            tf.keras.Model: A Bidirectional Long Short-Term Memory (BiLSTM) model.
        """
        size = len(hidden_layers)
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # first hidden layer
        layer1 = Bidirectional(LSTM(units=hidden_layers[size - 2], activation=activation))
        x = layer1(inputs)
        print(x)

        if size == 2:
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 3:
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 4:
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        outputs = (Dense(1))(x)

        # create model by specifying its inputs and outputs in the graph layers
        model = keras.Model(inputs=inputs, outputs=outputs, name="create_blstm")
        model.summary()

        # Compile model
        if self.OPTIMIZER == 'ADAM':
            opt = Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'SDG':
            opt = SGD(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'Adagrad':
            opt = Adagrad(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=opt, loss='mse', metrics='mse')

        return model

    def create_lstm(self, x_train, hidden_layers, activation):
        """
        Create a Long Short-Term Memory (LSTM) model.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of units in each hidden layer.
            activation (str): Activation function for the hidden layers.

        Returns:
            tf.keras.Model: A Long Short-Term Memory (LSTM) model.
        """
        size = len(hidden_layers)
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # first hidden layer
        x = LSTM(units=hidden_layers[size - 2], activation=activation)(inputs)

        if size == 2:

            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 3:

            x = Dense(units=hidden_layers[size - 2], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 4:
            x = Dense(units=hidden_layers[size - 3], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 2], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)

        outputs = Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        if self.OPTIMIZER == 'ADAM':
            opt = Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'SDG':
            opt = SGD(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'Adagrad':
            opt = Adagrad(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=opt, loss='mse', metrics='mse')

        return model

    def create_gru(self, x_train, hidden_layers, activation):
        """
        Create a Gated Recurrent Unit (GRU) model.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of units in each hidden layer.
            activation (str): Activation function for the hidden layers.

        Returns:
            tf.keras.Model: A Gated Recurrent Unit (GRU) model.
        """
        size = len(hidden_layers)
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # first hidden layer
        x = GRU(units=hidden_layers[0], activation=activation)(inputs)

        if size == 2:
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 3:
            x = Dense(units=hidden_layers[size - 2], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 4:
            x = Dense(units=hidden_layers[size - 3], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 2], activation=activation)(x)
            x = Dense(units=hidden_layers[size - 1], activation=activation)(x)

        outputs = Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        if self.OPTIMIZER == 'ADAM':
            opt = Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'SDG':
            opt = SGD(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'Adagrad':
            opt = Adagrad(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=opt, loss='mse', metrics='mse')

        return model

    def cnn1d_vae(self, x_train, hidden_layers, activation, latent_dim, learning_rate, optimizer):
        """
        Create a Convolutional Variational Autoencoder (VAE) model for 1D data.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of filters in each convolutional layer.
            activation (str): Activation function for the layers.
            latent_dim (int): Dimensionality of the latent space.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimizer for training.

        Returns:
            tuple: A tuple containing the CNN-VAE model, encoder, and decoder.
        """
        size = len(hidden_layers)
        encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        x = Conv1D(filters=hidden_layers[0], kernel_size=1, activation=activation)(encoder_inputs)
        if size == 2:
            x = Conv1D(hidden_layers[size - 1], kernel_size=1, activation=activation)(x)
            x = MaxPooling1D(pool_size=1)(x)
            x = Flatten()(x)
        if size == 3:
            x = Conv1D(filters=hidden_layers[size - 2], kernel_size=1, activation=activation)(x)
            x = Conv1D(filters=hidden_layers[size - 1], kernel_size=1, activation=activation)(x)
            x = MaxPooling1D(pool_size=1)(x)
            x = Flatten()(x)
        if size == 4:
            x = Conv1D(filters=hidden_layers[size - 3], kernel_size=1, activation=activation)(x)
            x = Conv1D(filters=hidden_layers[size - 2], kernel_size=1, activation=activation)(x)
            x = Conv1D(filters=hidden_layers[size - 1], kernel_size=1, activation=activation)(x)
            x = MaxPooling1D(pool_size=1)(x)
            x = Flatten()(x)

        # Latent Space
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling().call([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="cnn1d_encoder")

        # Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = Dense(1 * 32, activation=activation)(latent_inputs)
        x = Reshape((1, 32))(x)
        x = Conv1DTranspose(hidden_layers[size - 1], kernel_size=1, activation=activation)(x)

        if size == 2:
            x = Conv1DTranspose(hidden_layers[size - 2], kernel_size=1, activation=activation)(x)
        if size == 3:
            x = Conv1DTranspose(hidden_layers[size - 2], kernel_size=1, activation=activation)(x)
            x = Conv1DTranspose(hidden_layers[size - 3], kernel_size=1, activation=activation)(x)
        if size == 4:
            x = Conv1DTranspose(hidden_layers[size - 2], kernel_size=1, activation=activation)(x)
            x = Conv1DTranspose(hidden_layers[size - 3], kernel_size=1, activation=activation)(x)
            x = Conv1DTranspose(hidden_layers[size - 4], kernel_size=1, activation=activation)(x)

        decoder_outputs = Conv1DTranspose(x_train.shape[2], kernel_size=1, activation=activation)(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="cnn1d_decoder")

        # Latent Space
        z_mean, z_log_var, z = encoder(encoder_inputs)
        reconstruction = decoder(z)

        cnn1d_vae = keras.Model(encoder_inputs, reconstruction, name="cnn1d_vae")

        # Compile model
        if optimizer == 'ADAM':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'SDG':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer == 'Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        cnn1d_vae.add_loss(self.vae_loss(encoder_inputs, reconstruction, z_mean, z_log_var))
        cnn1d_vae.add_metric(self.kl_loss(z_mean, z_log_var), name="kl", aggregation="mean")
        cnn1d_vae.compile(optimizer=opt, metrics='mse')

        return cnn1d_vae, encoder, decoder

    def cnn1d_lstm_vae(self, x_train, hidden_layers, activation, latent_dim, learning_rate, optimizer):
        """
        Create a Convolutional Variational Autoencoder (VAE) model for 1D data with LSTM layers.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of filters in each convolutional layer.
            activation (str): Activation function for the layers.
            latent_dim (int): Dimensionality of the latent space.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimizer for training.

        Returns:
            tuple: A tuple containing the CNN-VAE model, encoder, and decoder.
        """
        size = len(hidden_layers)
        encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        x = Conv1D(filters=hidden_layers[0], kernel_size=1, activation=activation, padding='same')(encoder_inputs)

        if size == 2:
            x = Conv1D(hidden_layers[size - 1], kernel_size=1, activation=activation, padding='same')(x)
            x = MaxPooling1D(pool_size=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(filters=hidden_layers[size - 1], activation=activation)(x)
        if size == 3:
            x = Conv1D(filters=hidden_layers[size - 2], kernel_size=1, activation=activation, padding='same')(x)
            x = Conv1D(filters=hidden_layers[size - 1], kernel_size=1, activation=activation, padding='same')(x)
            x = MaxPooling1D(pool_size=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(filters=hidden_layers[size - 1], activation=activation)(x)
        if size == 4:
            x = Conv1D(filters=hidden_layers[size - 3], kernel_size=1, activation=activation, padding='same')(x)
            x = Conv1D(filters=hidden_layers[size - 2], kernel_size=1, activation=activation, padding='same')(x)
            x = Conv1D(filters=hidden_layers[size - 1], kernel_size=1, activation=activation, padding='same')(x)
            x = MaxPooling1D(pool_size=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(filters=hidden_layers[size - 1], activation=activation)(x)

        # Latent Space
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling().call([z_mean, z_log_var])
        z1 = Lambda(self.sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z1], name="cnn1d_encoder")

        # Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = Dense(1 * 32, activation=activation)(latent_inputs)
        x = Reshape((1, 32))(x)
        x = Conv1DTranspose(hidden_layers[size - 1], kernel_size=1, activation=activation, padding='same')(x)

        if size == 2:
            x = Conv1DTranspose(hidden_layers[size - 2], kernel_size=1, activation=activation, padding='same')(x)
        if size == 3:
            x = Conv1DTranspose(hidden_layers[size - 2], kernel_size=1, activation=activation, padding='same')(x)
            x = Conv1DTranspose(hidden_layers[size - 3], kernel_size=1, activation=activation, padding='same')(x)
        if size == 4:
            x = Conv1DTranspose(hidden_layers[size - 2], kernel_size=1, activation=activation, padding='same')(x)
            x = Conv1DTranspose(hidden_layers[size - 3], kernel_size=1, activation=activation, padding='same')(x)
            x = Conv1DTranspose(hidden_layers[size - 4], kernel_size=1, activation=activation, padding='same')(x)

        decoder_outputs = Conv1DTranspose(x_train.shape[2], kernel_size=1, activation=activation, padding='same')(x)

        decoder = keras.Model(latent_inputs, decoder_outputs, name="cnn1d_decoder")

        # Encoder + decoder
        z_mean, z_log_var, z = encoder(encoder_inputs)
        reconstruction = decoder(z)

        cnn1d_vae = keras.Model(encoder_inputs, reconstruction, name="cnn1d_vae")

        # Compile model
        if optimizer == 'ADAM':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'SDG':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer == 'Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        cnn1d_vae.add_loss(self.vae_loss(encoder_inputs, reconstruction, z_mean, z_log_var))
        cnn1d_vae.add_metric(self.kl_loss(z_mean, z_log_var), name="kl", aggregation="mean")
        cnn1d_vae.compile(optimizer=opt, metrics='mse')

        return cnn1d_vae, encoder, decoder

    def lstm_vae(self, x_train, hidden_layers, activation, latent_dim, learning_rate, optimizer):
        """
        Create a Variational Autoencoder (VAE) model with LSTM layers.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of LSTM units in each layer.
            activation (str): Activation function for the LSTM layers.
            latent_dim (int): Dimensionality of the latent space.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimizer for training.

        Returns:
            tuple: A tuple containing the VAE model, encoder, and decoder.
        """
        size = len(hidden_layers)
        encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # first hidden layer
        x = LSTM(units=hidden_layers[0], activation=activation, return_sequences=True)(encoder_inputs)

        if size == 2:
            x = LSTM(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 3:
            x = LSTM(units=hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = LSTM(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 4:
            x = LSTM(units=hidden_layers[size - 3], activation=activation, return_sequences=True)(x)
            x = LSTM(units=hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = LSTM(units=hidden_layers[size - 1], activation=activation)(x)

        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling().call([z_mean, z_log_var])
        lstm_encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="lstm_encoder")

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = RepeatVector(x_train.shape[1])(latent_inputs)

        x = LSTM(hidden_layers[size - 1], activation=activation, return_sequences=True)(x)
        if size == 2:
            x = LSTM(hidden_layers[size - 2], activation=activation)(x)
        if size == 3:
            x = LSTM(hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = LSTM(hidden_layers[size - 3], activation=activation)(x)
        if size == 4:
            x = LSTM(hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = LSTM(hidden_layers[size - 3], activation=activation, return_sequences=True)(x)
            x = LSTM(hidden_layers[size - 4], activation=activation)(x)

        decoder_outputs = Dense(x_train.shape[2], activation=activation)(x)
        lstm_decoder = keras.Model(latent_inputs, decoder_outputs, name="lstm_decoder")

        # Encoder + decoder
        z_mean, z_log_var, z = lstm_encoder(encoder_inputs)
        reconstruction = lstm_decoder(z)
        vae_lstm = keras.Model(encoder_inputs, reconstruction, name="vae")

        # Compile model
        if optimizer == 'ADAM':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'SDG':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer == 'Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        vae_lstm.add_loss(self.vae_loss(encoder_inputs, reconstruction, z_mean, z_log_var))
        vae_lstm.add_metric(self.kl_loss(z_mean, z_log_var), name="kl", aggregation="mean")
        vae_lstm.compile(optimizer=opt, metrics='mse')

        return vae_lstm, lstm_encoder, lstm_decoder

    def bi_gru_vae(self, x_train, hidden_layers, activation, latent_dim, learning_rate, optimizer):
        """
        Create a Variational Autoencoder (VAE) model with Bidirectional GRU (bi-GRU) layers.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of bi-GRU units in each layer.
            activation (str): Activation function for the bi-GRU layers.
            latent_dim (int): Dimensionality of the latent space.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimizer for training.

        Returns:
            tuple: A tuple containing the bi-GRU VAE model, encoder, and decoder.
        """
        size = len(hidden_layers)
        encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # first hidden layer
        x = Bidirectional(GRU(units=hidden_layers[0], activation=activation, return_sequences=True))(encoder_inputs)

        if size == 2:

            x = Bidirectional(GRU(units=hidden_layers[size - 1], activation=activation))(x)
        elif size == 3:

            x = Bidirectional(GRU(units=hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(GRU(units=hidden_layers[size - 1], activation=activation))(x)
        elif size == 4:
            x = Bidirectional(GRU(units=hidden_layers[size - 3], activation=activation, return_sequences=True))(x)
            x = Bidirectional(GRU(units=hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(GRU(units=hidden_layers[size - 1], activation=activation))(x)
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling().call([z_mean, z_log_var])
        bi_gru_encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="bi_gru_encoder")
        # bi_lstm_encoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = RepeatVector(x_train.shape[1])(latent_inputs)

        x = Bidirectional(GRU(hidden_layers[size - 1], activation=activation, return_sequences=True))(x)
        if size == 2:
            x = Bidirectional(GRU(hidden_layers[size - 2], activation=activation))(x)
        if size == 3:
            x = Bidirectional(GRU(hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(GRU(hidden_layers[size - 3], activation=activation))(x)
        if size == 4:
            x = Bidirectional(GRU(hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(GRU(hidden_layers[size - 3], activation=activation, return_sequences=True))(x)
            x = Bidirectional(GRU(hidden_layers[size - 4], activation=activation))(x)

        decoder_outputs = Dense(x_train.shape[2], activation=activation)(x)
        bi_gru_decoder = keras.Model(latent_inputs, decoder_outputs, name="bi_gru_decoder")

        # Encoder + decoder
        z_mean, z_log_var, z = bi_gru_encoder(encoder_inputs)
        reconstruction = bi_gru_decoder(z)
        bi_gru_vae = keras.Model(encoder_inputs, reconstruction, name="bi_gru_vae")

        # Compile model
        if optimizer == 'ADAM':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'SDG':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer == 'Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        bi_gru_vae.add_loss(self.vae_loss(encoder_inputs, reconstruction, z_mean, z_log_var))
        bi_gru_vae.add_metric(self.kl_loss(z_mean, z_log_var), name="kl", aggregation="mean")
        bi_gru_vae.compile(optimizer=opt, metrics='mse')

        return bi_gru_vae, bi_gru_encoder, bi_gru_decoder

    def bi_lstm_vae(self, x_train, hidden_layers, activation, latent_dim, learning_rate, optimizer):
        """
        Create a Variational Autoencoder (VAE) model with Bidirectional LSTM (bi-LSTM) layers.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of bi-LSTM units in each layer.
            activation (str): Activation function for the bi-LSTM layers.
            latent_dim (int): Dimensionality of the latent space.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimizer for training.

        Returns:
            tuple: A tuple containing the bi-LSTM VAE model, encoder, and decoder.
        """
        size = len(hidden_layers)
        encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # first hidden layer
        x = Bidirectional(LSTM(units=hidden_layers[0], activation=activation, return_sequences=True))(encoder_inputs)

        if size == 2:

            x = Bidirectional(LSTM(units=hidden_layers[size - 1], activation=activation))(x)
        elif size == 3:

            x = Bidirectional(LSTM(units=hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(LSTM(units=hidden_layers[size - 1], activation=activation))(x)
        elif size == 4:
            x = Bidirectional(LSTM(units=hidden_layers[size - 3], activation=activation, return_sequences=True))(x)
            x = Bidirectional(LSTM(units=hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(LSTM(units=hidden_layers[size - 1], activation=activation))(x)
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling().call([z_mean, z_log_var])
        bi_lstm_encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="bi_lstm_encoder")
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = RepeatVector(x_train.shape[1])(latent_inputs)

        x = Bidirectional(LSTM(hidden_layers[size - 1], activation=activation, return_sequences=True))(x)
        if size == 2:
            x = Bidirectional(LSTM(hidden_layers[size - 2], activation=activation))(x)
        if size == 3:
            x = Bidirectional(LSTM(hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(LSTM(hidden_layers[size - 3], activation=activation))(x)
        if size == 4:
            x = Bidirectional(LSTM(hidden_layers[size - 2], activation=activation, return_sequences=True))(x)
            x = Bidirectional(LSTM(hidden_layers[size - 3], activation=activation, return_sequences=True))(x)
            x = Bidirectional(LSTM(hidden_layers[size - 4], activation=activation))(x)

        decoder_outputs = Dense(x_train.shape[2], activation=activation)(x)
        bi_lstm_decoder = keras.Model(latent_inputs, decoder_outputs, name="bi_lstm_decoder")

        # Encoder + decoder
        z_mean, z_log_var, z = bi_lstm_encoder(encoder_inputs)
        reconstruction = bi_lstm_decoder(z)
        bi_lstm_vae = keras.Model(encoder_inputs, reconstruction, name="bi_lstm_vae")

        # Compile model
        if optimizer == 'ADAM':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'SDG':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer == 'Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        bi_lstm_vae.add_loss(self.vae_loss(encoder_inputs, reconstruction, z_mean, z_log_var))
        bi_lstm_vae.add_metric(self.kl_loss(z_mean, z_log_var), name="kl", aggregation="mean")
        bi_lstm_vae.compile(optimizer=opt, metrics='mse')

        return bi_lstm_vae, bi_lstm_encoder, bi_lstm_decoder

    def gru_vae(self, x_train, hidden_layers, activation, latent_dim, learning_rate, optimizer):
        """
        Create a Variational Autoencoder (VAE) model with GRU layers.

        Args:
            x_train (ndarray): Training data with shape (number of samples, time steps, number of features).
            hidden_layers (list): List of integers representing the number of GRU units in each layer.
            activation (str): Activation function for the GRU layers.
            latent_dim (int): Dimensionality of the latent space.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimizer for training.

        Returns:
            tuple: A tuple containing the GRU VAE model, encoder, and decoder.
        """
        size = len(hidden_layers)
        encoder_inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # first hidden layer
        x = GRU(units=hidden_layers[0], activation=activation, return_sequences=True)(encoder_inputs)

        if size == 2:

            x = GRU(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 3:

            x = GRU(units=hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = GRU(units=hidden_layers[size - 1], activation=activation)(x)
        elif size == 4:
            x = GRU(units=hidden_layers[size - 3], activation=activation, return_sequences=True)(x)
            x = GRU(units=hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = GRU(units=hidden_layers[size - 1], activation=activation)(x)
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling().call([z_mean, z_log_var])
        gru_encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="gru_encoder")

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = RepeatVector(x_train.shape[1])(latent_inputs)

        x = GRU(hidden_layers[size - 1], activation=activation, return_sequences=True)(x)
        if size == 2:
            x = GRU(hidden_layers[size - 2], activation=activation)(x)
        if size == 3:
            x = GRU(hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = GRU(hidden_layers[size - 3], activation=activation)(x)
        if size == 4:
            x = GRU(hidden_layers[size - 2], activation=activation, return_sequences=True)(x)
            x = GRU(hidden_layers[size - 3], activation=activation, return_sequences=True)(x)
            x = GRU(hidden_layers[size - 4], activation=activation)(x)

        decoder_outputs = Dense(x_train.shape[2], activation=activation)(x)
        gru_decoder = keras.Model(latent_inputs, decoder_outputs, name="gru_decoder")

        # Encoder + decoder
        z_mean, z_log_var, z = gru_encoder(encoder_inputs)
        reconstruction = gru_decoder(z)
        gru_vae = keras.Model(encoder_inputs, reconstruction, name="vae")

        # Compile model
        if optimizer == 'ADAM':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'SDG':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer == 'Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        gru_vae.add_loss(self.vae_loss(encoder_inputs, reconstruction, z_mean, z_log_var))
        gru_vae.add_metric(self.kl_loss(z_mean, z_log_var), name="kl", aggregation="mean")
        gru_vae.add_metric(self.reconstruction_loss(encoder_inputs, reconstruction), name="rl", aggregation="mean")
        gru_vae.compile(optimizer=opt, metrics='mse')

        return gru_vae, gru_encoder, gru_decoder

    def vae_loss(self, original, out_decoded, z_mean, z_log_sigma):
        """
        Calculate the VAE loss, which combines the reconstruction loss and KL divergence loss.

        Args:
            original (tensor): The original input data.
            out_decoded (tensor): The reconstructed output from the VAE.
            z_mean (tensor): The mean of the latent space distribution.
            z_log_sigma (tensor): The logarithm of the variance of the latent space distribution.

        Returns:
            tensor: The computed VAE loss.
        """
        # Compute the mean squared error (MSE) for the reconstruction loss
        reconstruction_loss = keras.losses.MeanSquaredError()(original, out_decoded)
        reconstruction_loss *= self.STEP_SIZE

        # Compute the KL divergence loss
        kl_loss = 1 + z_log_sigma - k.square(z_mean) - k.exp(z_log_sigma)
        kl_loss = k.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Combine the reconstruction loss and KL divergence loss
        vae_loss = k.mean(reconstruction_loss + kl_loss)

        return vae_loss

    def reconstruction_loss(self, original, out_decoded):
        """
        Calculate the reconstruction loss, which is the mean squared error (MSE) between
        the original input data and the reconstructed output from the VAE.

        Args:
            original (tensor): The original input data.
            out_decoded (tensor): The reconstructed output from the VAE.

        Returns:
            tensor: The computed reconstruction loss (MSE).
        """
        reconstruction_loss = keras.losses.MeanSquaredError()(original, out_decoded)
        reconstruction_loss *= self.STEP_SIZE
        return reconstruction_loss

    def kl_loss(self, z_mean, z_log_sigma):
        """
        Calculate the KL divergence loss between the learned latent space and a prior distribution.

        Args:
            z_mean (tensor): Mean of the learned latent space.
            z_log_sigma (tensor): Logarithm of the standard deviation of the learned latent space.

        Returns:
            tensor: The computed KL divergence loss.
        """
        kl_loss = 1 + z_log_sigma - k.square(z_mean) - k.exp(z_log_sigma)
        kl_loss = k.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return k.mean(kl_loss)

    def create_cnn1d(self, hidden_layers, activation):
        """
        Create a 1D Convolutional Neural Network (CNN) model with variable hidden layers.

        Args:
            hidden_layers (list): List of integers specifying the number of filters for each convolutional layer.
            activation (str): Activation function for the model layers.

        Returns:
            keras.Model: Compiled CNN model.

        This function constructs a CNN model with different configurations based on the size of the hidden_layers list.

        The model architecture consists of convolutional layers, max-pooling layers, and dense layers. The number and type
        of these layers are determined by the size of the hidden_layers list.

        """
        size = len(hidden_layers)
        latent_inputs = keras.Input(shape=(self.LATENT_DIM,))
        x = Dense(1 * 32, activation=activation)(latent_inputs)
        x = Reshape((1, 32))(x)
        x = Conv1D(filters=hidden_layers[0], kernel_size=1, activation=activation, padding='same')(x)
        if size == 2:
            x = MaxPooling1D(pool_size=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hidden_layers[size - 1], activation=activation)(x)
        if size == 3:
            x = Conv1D(filters=hidden_layers[size - 2], kernel_size=1, activation=activation, padding='same')
            x = MaxPooling1D(pool_size=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hidden_layers[size - 1], activation=activation)(x)

        if size == 4:
            x = Conv1D(filters=hidden_layers[size - 3], kernel_size=1, activation=activation, padding='same')(x)
            x = Conv1D(filters=hidden_layers[size - 2], kernel_size=1, activation=activation, padding='same')(x)
            x = MaxPooling1D(pool_size=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hidden_layers[size - 1], activation=activation)(x)

        outputs = Dense(units=1)(x)
        model = keras.Model(inputs=latent_inputs, outputs=outputs)

        # Compile model
        if self.OPTIMIZER == 'ADAM':
            opt = Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'SDG':
            opt = SGD(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'Adagrad':
            opt = Adagrad(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=opt, loss='mse', metrics='mse')
        return model

    def format_params(self, params):
        """
        Format a set of parameters and their values as a string.

        Args:
            params (dict): A dictionary containing various parameter values.

        Returns:
            str: A formatted string containing parameter names and their corresponding values.

        This function takes a dictionary of parameters and formats them as a string for display or logging purposes.

        Example:
        If params = {'hidden_layer_sizes': [64, 32], 'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.001},
        the function would return a string like:
        "'hidden_layer_sizes'=[64, 32]\n 'activation'='relu'\n 'optimizer'='adam'\n 'learning_rate'=0.001"

        """
        hidden_layer_sizes, activation, optimizer, learning_rate, deep_nn, latent_dim, epoch \
            = self.convert_params(params)  # hidden_layer_sizes,
        return "'hidden_layer_sizes'={}\n " \
               "'activation'={}\n " \
               "'optimizer'='{}'\n " \
               "'learning_rate'={}\n " \
               "'deep_nn'={}\n" \
               "'latent_dim'={}\n" \
               "'epoch'='{}'".format(hidden_layer_sizes, activation, optimizer,
                                     learning_rate, deep_nn, latent_dim, epoch)

    def reconstruction(self, model_enc, model_dec, data):
        """
        Reconstruct data using encoder and decoder models.

        Args:
            model_enc (keras.Model): The encoder model.
            model_dec (keras.Model): The decoder model.
            data (numpy.ndarray): The input data for reconstruction.

        Returns:
            numpy.ndarray: Reconstructed predictions.

        This function takes an encoder model, a decoder model, and input data and returns the reconstructed predictions
        based on the input data.

        Example:
        If model_enc and model_dec are trained models for encoding and decoding, and data is a set of input data:
        reconstructed_data = reconstruction(model_enc, model_dec, data)
        The function returns the reconstructed data based on the models.

        """
        # Get the latent space representation (z_mean) and make predictions
        z_mean, z_log_var, z = model_enc.predict(data)
        predictions = model_dec.predict(z_mean)
        print('Prediction shape', predictions.shape)

        return predictions

    def impute(self, model_enc, model_dec, data_corrupt, max_iter=10):
        """
        Use a VAE to impute missing values in data_corrupt. Missing values are indicated by NaN.

        Args:
            model_enc (keras.Model): The encoder model.
            model_dec (keras.Model): The decoder model.
            data_corrupt (pd.DataFrame): The input data with missing values (NaN).
            max_iter (int): Maximum number of imputation iterations (default is 10).

        Returns:
            pd.DataFrame: The imputed data.

        This function takes an encoder model, a decoder model, and data with missing values. It iteratively imputes the
        missing values using a Variational Autoencoder (VAE).

        Example:
        If model_enc and model_dec are trained models for encoding and decoding, and data_corrupt is a DataFrame with
        missing values, you can impute the missing values as follows:
        imputed_data = impute(model_enc, model_dec, data_corrupt, max_iter=10)
        The function returns the imputed data with missing values replaced.

        """
        data_corrupt = data_corrupt.values
        missing_row_ind = np.where(np.isnan(np.sum(data_corrupt, axis=1)))
        DataFrame(missing_row_ind).to_csv(self.folder_results +'/missing_row_ind.csv')
        data_miss_val = data_corrupt[missing_row_ind[0], :]
        DataFrame(data_miss_val).to_csv(self.folder_results+'/data_miss_val.csv')
        na_ind = np.where(np.isnan(data_miss_val))
        DataFrame(na_ind).to_csv(self.folder_results+'/na_ind.csv')
        data_miss_val[na_ind] = 0
        DataFrame(data_miss_val).to_csv(self.folder_results+'/data_miss_val2.csv')

        data_miss_val2, data_miss_y, scaler = self.scale(self.FREQUENCY)

        for i in range(max_iter):
            # Predict the latent space
            z_mean, z_log_var, z = model_enc.predict(data_miss_val2)

            # Generate new samples from latent space
            epsilon = np.random.normal(size=(len(data_miss_val2), self.LATENT_DIM))
            new_z = z_mean + np.exp(0.5 * z_log_var) * epsilon

            # Decode the new samples
            new_data = model_dec.predict(new_z)
            new_data = new_data.reshape(new_data.shape[0], new_data.shape[2])
            new_data = scaler.inverse_transform(new_data)
            data_miss_val[na_ind] = new_data[na_ind]

        data_corrupt[missing_row_ind, :] = data_miss_val
        data_imputed = data_corrupt

        return data_imputed

    def show_raw_visualization(self, data):
        """
        Create a raw data visualization for multiple features over time.

        Args:
            data (pd.DataFrame): The raw data containing time and feature columns.

        This method visualizes multiple features over time from the raw data using Matplotlib.

        Example:
        To visualize multiple features over time from a DataFrame 'data', you can use this method as follows:
        obj.show_raw_visualization(data)
        """
        self.time_data = data[self.time_key]
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
        )
        for i in range(len(self.feature_keys)):
            key = self.feature_keys[i]
            c = self.colors[i % (len(self.colors))]
            t_data = data[key]
            t_data.index = self.time_data
            t_data.head()
            ax = t_data.plot(
                ax=axes[i // 2, i % 2],
                color=c,
                title="{} - {}".format(self.titles[i], key),
                rot=25,
            )
            ax.legend([self.titles[i]])
        plt.tight_layout()

    def show_heatmap(self, data):
        """
        Create a heatmap to visualize feature correlations in the data.

        Args:
            data (pd.DataFrame): The data for which feature correlations are to be visualized.

        This method generates a heatmap to visualize the correlations between different features in the data.

        Example:
        To create a heatmap to visualize feature correlations for a DataFrame 'data', you can use this method as follows:
        obj.show_heatmap(data)
        """
        plt.matshow(data.corr())
        plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
        plt.gca().xaxis.tick_bottom()
        plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title("Feature Correlation Heatmap", fontsize=14)
        plt.show()

    def create_sequences(self, x, y, step_size=1):
        """
        Create sequences from input data for time series forecasting.

        Args:
            x (array-like): Input time series data.
            y (array-like): Target time series data.
            step_size (int, optional): The size of the sequence window. Default is 1.

        Returns:
            tuple: A tuple containing the input sequences (Xs) and corresponding target values (ys).

        This method creates input-output sequences for time series forecasting by sliding a fixed-size
        sequence window over the input data.

        Example:
        To create sequences from input data 'x' and target data 'y' with a step size of 3, you can use
        this method as follows:
        Xs, ys = obj.create_sequences(x, y, step_size=3)
        """
        Xs, ys = [], []
        for i in range(len(x) - step_size):
            v = x[i:i + step_size]
            Xs.append(v)
            ys.append(y[i + step_size])
        return np.array(Xs), np.array(ys)

    def coeff_determination(self, y_true, y_pred):
        """
        Calculate the coefficient of determination (R-squared) for a regression model.

        Args:
            y_true (tensor-like): True target values.
            y_pred (tensor-like): Predicted target values.

        Returns:
            float: The coefficient of determination (R-squared) value.

        The coefficient of determination (R-squared) is a statistical measure that represents the
        proportion of the variance in the dependent variable that is predictable from the independent
        variables in a regression model. It provides insights into the goodness of fit of the model.

        Example:
        To calculate the R-squared value for true target values 'y_true' and predicted values 'y_pred',
        you can use this method as follows:
        r_squared = obj.coeff_determination(y_true, y_pred)
        """
        SS_res = k.sum(k.square(y_true - y_pred))
        SS_tot = k.sum(k.square(y_true - k.mean(y_true)))
        return 1 - SS_res / (SS_tot + k.epsilon())

    def fit_model(self, model, xtrain, ytrain, epochs):
        """
        Fit a Keras model to the training data.

        Args:
            model (keras.Model): The Keras model to be trained.
            xtrain (numpy.ndarray): Training input data.
            ytrain (numpy.ndarray): Training target data.
            epochs (int): Number of training epochs.

        Returns:
            keras.callbacks.History: A history object containing training metrics.

        This method fits a Keras model to the provided training data using the specified number of epochs.
        It also incorporates early stopping to prevent overfitting by monitoring the validation loss.

        Example:
        To train a Keras model 'my_model' with training data 'x_train' and 'y_train' for 100 epochs, you
        can use this method as follows:
        history = obj.fit_model(my_model, x_train, y_train, epochs=100)
        """
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(xtrain, ytrain, epochs=epochs, validation_split=self.VALIDATION_SPLIT,
                            batch_size=self.BATCH_SIZE, shuffle=False, callbacks=[early_stop], verbose=self.VERBOSE)

        return history

    def get_metrics(self, history):
        """
        Get the Mean Squared Error (MSE) from the validation history.

        Args:
            history (keras.callbacks.History): History object containing training metrics.

        Returns:
            list: List of Mean Squared Error (MSE) values from the validation history.

        This method extracts and returns the Mean Squared Error (MSE) values from the validation history.

        Example:
        To obtain MSE values from a 'history' object, you can use this method as follows:
        mse_values = obj.get_metrics(history)
        """
        mse = history.history['val_mse']
        for val_mse in mse:
            return val_mse

    def plot_loss(self, history, model_name):
        """
        Plot and save the KL divergence loss for training and validation data.

        Args:
            history (keras.callbacks.History): History object containing training metrics.
            model_name (str): Name of the model for labeling the plot.

        This method plots and saves the KL divergence loss for training and validation data, as well as saving the summary data to CSV files.

        Example:
        To plot KL divergence loss for a model 'my_model' with its history, you can use this method as follows:
        plot_loss(my_model_history, 'my_model')
        """
        # Save the summary data to a CSV file
        df_summary = DataFrame(history.history)
        df_summary.to_csv(self.folder_results+'/summary.csv', mode='a')

        # Create DataFrames for KL divergence loss and validation KL loss
        df_data1 = DataFrame(history.history['kl'], columns=['KL_' + model_name])
        df_data2 = DataFrame(history.history['val_kl'], columns=['val_KL_' + model_name])
        data_final = pd.concat([df_data1, df_data2], axis=1)
        # Save the KL divergence loss data to a CSV file (change the file name accordingly)
        data_final.to_csv(self.folder_results+'/loss_pH_Reactor1.csv', mode='a')  # change the file name accordingly

        # Plot KL divergence loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Training vs Validation KL for ' + model_name)
        plt.ylabel('KL Divergence')
        plt.xlabel('epoch')
        plt.legend(['Training KL', 'Validation KL'], loc='upper right')
        plt.show()

    # Make prediction
    def prediction(self, model, x_test, scaler_effluent2):
        """
        Make predictions using a trained model and perform inverse scaling.

        Args:
            model (keras.Model): The trained Keras model for making predictions.
            x_test (numpy.ndarray): Input data for making predictions.
            scaler_effluent2: Scaler used to inverse-transform the predictions.

        Returns:
            numpy.ndarray: Predicted values after inverse scaling.

        This method takes a trained Keras model, input data, and a scaler used for inverse scaling and returns the predictions. It performs inverse scaling to obtain the actual predicted values.

        Example:
        To make predictions using a model 'my_model' on input data 'x_test' with a scaler 'my_scaler', you can use this method as follows:
        predictions = prediction(my_model, x_test, my_scaler)
        """
        # Make predictions using the model
        prediction = model.predict(x_test)
        print("Prediction shape:", prediction.shape)

        # Perform inverse scaling to get actual predicted values
        prediction = scaler_effluent2.inverse_transform(prediction)

        return prediction

    def prediction_cnn(self, model, x_test, scaler_effluent2):
        """
        Make predictions using a trained CNN model and perform inverse scaling.

        Args:
            model (keras.Model): The trained Keras CNN model for making predictions.
            x_test (numpy.ndarray): Input data for making predictions.
            scaler_effluent2: Scaler used to inverse-transform the predictions.

        Returns:
            numpy.ndarray: Predicted values after inverse scaling.

        This method takes a trained Keras CNN model, input data, and a scaler used for inverse scaling and returns the predictions. It performs inverse scaling to obtain the actual predicted values.

        Example:
        To make predictions using a CNN model 'my_cnn_model' on input data 'x_test' with a scaler 'my_scaler', you can use this method as follows:
        predictions = prediction_cnn(my_cnn_model, x_test, my_scaler)
        """
        # Make predictions using the CNN model
        prediction = model.predict(x_test)
        print("Prediction shape:", prediction.shape)

        # Reshape the predictions if needed (remove dimensions with size 1)
        prediction = prediction.reshape(prediction.shape[0], prediction.shape[2])

        # Perform inverse scaling to get actual predicted values
        prediction = scaler_effluent2.inverse_transform(prediction)
        return prediction

    def plot_future(self, prediction, model_name, y_test):
        """
        Plot the test data and model predictions for visualization.

        Args:
            prediction (numpy.ndarray): Model predictions.
            model_name (str): Name of the model used for predictions.
            y_test (numpy.ndarray): Actual test data for comparison.

        This method generates a plot to visualize the test data and model predictions for a given model. It also saves the data and the plot in the 'Results' directory.

        Example:
        To plot the test data and predictions for a model named 'my_model' with predictions 'my_predictions' and actual test data 'my_test_data', you can use this method as follows:
        plot_future(my_predictions, 'my_model', my_test_data)
        """
        plt.figure(figsize=(10, 6))

        time = self.raw_process_data[self.time_key][0::self.FREQUENCY]
        df_data = DataFrame(np.array(time), columns=['Date (day)'])
        df_data1 = DataFrame(y_test, columns=self.feature_keys)
        df_data2 = DataFrame(prediction, columns=self.feature_keys_pred)
        data_final = pd.concat([df_data, df_data1, df_data2], axis=1)
        data_final.to_csv(self.folder_results+'/Prediction_test_' + model_name + '.csv')

        plt.plot(np.array(time), np.array(y_test), label='Test data')
        plt.plot(np.array(time), np.array(prediction), label='Prediction')

        plt.title('Test data vs prediction for ' + model_name)
        plt.legend(loc='upper left')
        plt.xlabel('Date (day)')
        plt.ylabel('pH Reactor 1')
        plt.savefig('../Data/pH_reactor1.png')
        plt.show()

    def plot_future_train(self, prediction, model_name, y_test):
        """
        Plot the training and prediction data for visualization.

        Args:
            prediction (numpy.ndarray): Model predictions.
            model_name (str): Name of the model used for predictions.
            y_test (numpy.ndarray): Actual test data for comparison.

        This method generates a plot to visualize the training and prediction data for a given model. It also saves the data and the plot in the 'Results' directory.

        Example:
        To plot the training data and predictions for a model named 'my_model' with predictions 'my_predictions' and actual test data 'my_test_data', you can use this method as follows:
        plot_future_train(my_predictions, 'my_model', my_test_data)
        """
        plt.figure(figsize=(20, 12))
        range_future = len(prediction)
        time_raw = self.data_complete2[self.time_key][:-1][0::self.FREQUENCY]
        train_size = int(len(time_raw) * self.SPLIT_FRACTION)
        print('Print prediction shape: ', prediction.shape)
        print('Print y test shape: ', y_test.shape)
        print('Time size: ', len(time_raw))
        time = time_raw.iloc[train_size:][self.STEP_SIZE:]
        df_data = DataFrame(np.array(time_raw), columns=['Date (day)'])  # , 'Test data', 'Predicted data'])
        df_test_data = DataFrame(y_test, columns=self.feature_keys)  # , 'Test data', 'Predicted data'])
        df_test_data.to_csv(self.folder_results+'/Y_TRAIN.csv')
        df_prediction = DataFrame(prediction, columns=self.feature_keys_pred)  # , 'Test data', 'Predicted data'])
        df_prediction.to_csv(self.folder_results+'/PREDICTION.csv')
        data_final = pd.concat([df_data, df_test_data, df_prediction], axis=1)
        data_final.to_csv(self.folder_results+'/Testing_' + model_name + '.csv')
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(np.array(time_raw), np.array(y_test[:, 0]), label='Testing data')
        axs[0, 0].plot(np.array(time_raw), np.array(prediction[:, 0]), label='Prediction')
        axs[0, 1].plot(np.array(time_raw), np.array(y_test[:, 1]), label='Testing data')
        axs[0, 1].plot(np.array(time_raw), np.array(prediction[:, 1]), label='Prediction')
        axs[1, 0].plot(np.array(time_raw), np.array(y_test[:, 2]), label='Testing data')
        axs[1, 0].plot(np.array(time_raw), np.array(prediction[:, 2]), label='Prediction')
        axs[1, 1].plot(np.array(time_raw), np.array(y_test[:, 3]), label='Testing data')
        axs[1, 1].plot(np.array(time_raw), np.array(prediction[:, 3]), label='Prediction')
        axs[2, 0].plot(np.array(time_raw), np.array(y_test[:, 4]), label='Testing data')
        axs[2, 0].plot(np.array(time_raw), np.array(prediction[:, 4]), label='Prediction')
        axs[2, 1].plot(np.array(time_raw), np.array(y_test[:, 5]), label='Testing data')
        axs[2, 1].plot(np.array(time_raw), np.array(prediction[:, 5]), label='Prediction')

        reconstructed_data = self.missing_value_data.fillna(df_prediction)
        reconstructed_data.to_csv(self.folder_results+'/Reconstructed_Data.csv')

        plt.title('Test data vs prediction for ' + model_name)
        plt.legend(loc='upper left')
        plt.xlabel('Date (day)')
        plt.ylabel('NH4')
        plt.savefig(self.folder_results+'/NH4_vae.png')
        plt.show()

    def evaluate_prediction(self, predictions, actual, model_name):
        """
         Evaluate the performance of a model's predictions.

         Args:
             predictions (numpy.ndarray): Model predictions.
             actual (numpy.ndarray): Actual target values.
             model_name (str): Name of the model used for predictions.

         This method computes and prints metrics to evaluate the performance of a model's predictions, including Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Coefficient of Determination (R^2).

         Example:
         To evaluate predictions for a model named 'my_model' with predictions 'my_predictions' and actual target values 'my_actual_data', you can use this method as follows:
         evaluate_prediction(my_predictions, my_actual_data, 'my_model')
         """
        errors = predictions - actual
        print(errors)
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        SS_res = np.sum(np.square(actual - predictions))
        SS_tot = np.sum(np.square(actual - np.mean(predictions)))
        coefficient_R2 = (1 - SS_res / SS_tot)

        print(model_name + ':')
        print('Mean Absolute Error: {:.4f}'.format(mae))
        print('Root Mean Square Error: {:.4f}'.format(rmse))
        print('Coefficient of Determination: {:.4f}'.format(coefficient_R2))
        print('')

    def model_evaluate(self, model, x_test, y_test):
        """
        Evaluate a model's performance on a test dataset.

        Args:
            model (keras.Model): The model to be evaluated.
            x_test (numpy.ndarray): Input test data.
            y_test (numpy.ndarray): Target test data.

        This method evaluates the provided model using the test data and prints the test score and accuracy.

        Returns:
            float: The test accuracy.

        Example:
        To evaluate a model 'my_model' on the test data 'x_test' and 'y_test', you can use this method as follows:
        accuracy = model_evaluate(my_model, x_test, y_test)
        """
        score, accuracy = model.evaluate(x_test, y_test)
        print('Test score:', score)
        print('Test accuracy:', accuracy)
        return accuracy

    def sampling(self, args):
        """
        Perform sampling from a Gaussian distribution with the given mean and log variance.

        Args:
            args (tuple): A tuple containing the mean and log variance of the distribution.

        This function generates random samples from a Gaussian distribution with a mean and log variance.
        It's used in the VAE's latent space sampling.

        Returns:
            keras.Tensor: A sample from the Gaussian distribution.

        Example:
        To perform sampling with mean and log variance from 'z_mean' and 'z_log_var', you can use this function as follows:
        sampled = sampling([z_mean, z_log_var])
        """
        z_mean, z_log_sigma = args
        epsilon = k.random_normal(shape=(k.shape(z_mean)[0], k.shape(z_mean)[1]), mean=0.0, stddev=1.)
        return z_mean + k.exp(0.5 * z_log_sigma) * epsilon


class Sampling():
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    This class is used for sampling the latent space vector 'z' in a Variational Autoencoder (VAE)
    based on the provided mean ('z_mean') and log variance ('z_log_var').

    Attributes:
        None

    Methods:
        call(inputs): Takes the mean and log variance of the latent space and samples 'z' from it.

    Example:
    To use this class for sampling 'z' in your VAE model:

    ```python
    # Instantiate the Sampling class
    sampler = Sampling()

    # Provide 'z_mean' and 'z_log_var'
    z_mean, z_log_var = ...  # Get the mean and log variance from your model

    # Sample 'z'
    z = sampler.call([z_mean, z_log_var])
    ```

    """
    def call(self, inputs):
        """
        Samples 'z' from (z_mean, z_log_var) using the reparameterization trick.

        Args:
            inputs (list): A list containing 'z_mean' and 'z_log_var' tensors.

        Returns:
            z (tensor): The sampled latent space vector 'z'.
        """
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def main():
    """
    The main function for training, evaluating, and visualizing VAE models for imputing data.

    This function sets up and trains Variational Autoencoder (VAE) models for imputing missing data in the input dataset.
    It supports different neural network architectures such as BiLSTM, LSTM, GRU, BiGRU, and CNN1D.

    After training, the function evaluates the models, generates imputed data, and creates visualizations.

    Args:
        None

    Returns:
        None
    """

    nn = DeepLearning()
    X_train, y_train, scaler_influent = nn.scale(nn.FREQUENCY)
    print("Training dataset shape: ", X_train.shape, X_train.shape[0], X_train.shape[1], X_train.shape[2])
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape)
    X_train2, y_train2, scaler_influent2 = nn.scale(nn.FREQUENCY)
    X_train_raw = scaler_influent.inverse_transform(X_train.reshape(X_train.shape[0], X_train.shape[2]))
    X_train_raw2 = scaler_influent.inverse_transform(X_train2.reshape(X_train2.shape[0], X_train2.shape[2]))
    DataFrame(X_train_raw).to_csv(nn.folder_results+'/X_train_raw.csv')
    print('X_train_raw: ', X_train_raw.shape)

    deep_nn = nn.DEEP_NN
    if deep_nn == "bilstm":
        bi_lstm_vae, bi_lstm_enc, bi_lstm_dec = nn.bi_lstm_vae(X_train, nn.HIDDEN_LAYERS, nn.ACTIVATION,
                                                               nn.LATENT_DIM, nn.LEARNING_RATE, nn.OPTIMIZER)
        bi_lstm_history = nn.fit_model(bi_lstm_vae, X_train, X_train, nn.EPOCHS)
        nn.plot_loss(bi_lstm_history, 'BiLSTM - VAE')
        prediction_bilstm_vae = nn.prediction(bi_lstm_vae, X_train, scaler_influent)
        nn.plot_future_train(prediction_bilstm_vae, 'CNN1D-VAE', X_train_raw)
        nn.evaluate_prediction(prediction_bilstm_vae, X_train_raw, 'CNN1D-VAE')
    elif deep_nn == "lstm":
        lstm_vae, lstm_enc, lstm_dec = nn.lstm_vae(X_train, nn.HIDDEN_LAYERS, nn.ACTIVATION, nn.LATENT_DIM,
                                                   nn.LEARNING_RATE, nn.OPTIMIZER)
        lstm_history = nn.fit_model(lstm_vae, X_train, X_train, nn.EPOCHS)
        nn.plot_loss(lstm_history, 'LSTM - VAE')
        prediction_lstm_vae = nn.prediction(lstm_vae, X_train, scaler_influent)
        print("prediction", prediction_lstm_vae)
        nn.plot_future_train(prediction_lstm_vae, 'LSTM-VAE', X_train_raw)
        nn.evaluate_prediction(prediction_lstm_vae, X_train_raw, 'LSTM-VAE')
    elif deep_nn == "gru":
        gru_vae, gru_enc, gru_dec = nn.gru_vae(X_train, nn.HIDDEN_LAYERS, nn.ACTIVATION, nn.LATENT_DIM,
                                               nn.LEARNING_RATE, nn.OPTIMIZER)
        gru_history_vae = nn.fit_model(gru_vae, X_train, X_train, nn.EPOCHS)
        nn.plot_loss(gru_history_vae, 'GRU - VAE')
        prediction_gru_vae = nn.prediction(gru_vae, X_train, scaler_influent)
        print("prediction", prediction_gru_vae)
        nn.plot_future_train(prediction_gru_vae, 'GRU-VAE', X_train_raw)
        nn.evaluate_prediction(prediction_gru_vae, X_train_raw, 'GRU-VAE')
    elif deep_nn == "bigru":
        bi_gru_vae, bi_gru_enc, bi_gru_dec = nn.bi_gru_vae(X_train, nn.HIDDEN_LAYERS, nn.ACTIVATION, nn.LATENT_DIM,
                                               nn.LEARNING_RATE, nn.OPTIMIZER)
        bi_gru_history_vae = nn.fit_model(bi_gru_vae, X_train, X_train, nn.EPOCHS)
        nn.plot_loss(bi_gru_history_vae, 'BI_GRU - VAE')
        prediction_bi_gru_vae = nn.prediction(bi_gru_vae, X_train, scaler_influent)
        print("prediction", prediction_bi_gru_vae)
        nn.plot_future_train(prediction_bi_gru_vae, 'BI_GRU-VAE', X_train_raw)
        nn.evaluate_prediction(prediction_bi_gru_vae, X_train_raw, 'BI_GRU-VAE')

    else:
        cnn1d_vae, cnn1d_enc, cnn1d_dec = \
        nn.cnn1d_vae(X_train, nn.HIDDEN_LAYERS, nn.ACTIVATION, nn.LATENT_DIM,
                                            nn.LEARNING_RATE, nn.OPTIMIZER)
        cnn1d_history = nn.fit_model(cnn1d_vae, X_train, X_train, nn.EPOCHS)
        nn.plot_loss(cnn1d_history, 'CNN1D - VAE')
        prediction_cnn1d_vae = nn.prediction_cnn(cnn1d_vae, X_train, scaler_influent)
        print("prediction", prediction_cnn1d_vae)

        nn.evaluate_prediction(prediction_cnn1d_vae, X_train_raw, 'CNN1D-VAE')
        x_train_pred = cnn1d_vae.predict(X_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train_pred), axis=1)
        plt.hist(train_mae_loss, bins=50)
        plt.xlabel("Train MAE loss")
        plt.ylabel("No of samples")
        z_mean, z_log_var, z = cnn1d_enc.predict(X_train)
        x_decoded2 = cnn1d_dec.predict(z_mean)
        x_decoded = cnn1d_vae(X_train)
        print('x_decoded - z dec shape', x_decoded2.shape)
        print('x_decoded - vae shape', x_decoded.shape)

        r_data = nn.reconstruction(cnn1d_enc, cnn1d_dec, X_train2)

        r_data = r_data.reshape(r_data.shape[0], r_data.shape[2])
        r_data = scaler_influent.inverse_transform(r_data)
        DataFrame(r_data).to_csv(nn.folder_results+'/reconstructed_data3.csv')

        # scaler_effluent = MinMaxScaler().fit(self.data_complete)

        imputed_data = nn.impute(cnn1d_enc, cnn1d_dec, nn.raw_process_data[nn.feature_keys])
        imputed_data = DataFrame(imputed_data, columns=nn.feature_keys)
        imputed_data.to_csv(nn.folder_results+'/Imputed_data2.csv')


if __name__ == "__main__":
    main()
