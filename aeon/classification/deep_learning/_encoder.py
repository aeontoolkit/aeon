"""Encoder Classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["EncoderClassifier"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks import EncoderNetwork


class EncoderClassifier(BaseDeepClassifier):
    """
    Establish the network structure for an Encoder.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    kernel_size : array of int, default = [5, 11, 21]
        Specifying the length of the 1D convolution windows.
    n_filters : array of int, default = [128, 256, 512]
        Specifying the number of 1D convolution filters used for each layer,
        the shape of this array should be the same as kernel_size.
    max_pool_size : int, default = 2
        Size of the max pooling windows.
    activation : string, default = sigmoid
        Keras activation function.
    dropout_proba : float, default = 0.2
        Specifying the dropout layer probability.
    padding : string, default = same
        Specifying the type of padding used for the 1D convolution.
    strides : int, default = 1
        Specifying the sliding rate of the 1D convolution filter.
    fc_units : int, default = 256
        Specifying the number of units in the hidden fully
        connected layer used in the EncoderNetwork.
    file_path : str, default = "./"
        File path when saving model_Checkpoint callback.
    save_best_model : bool, default = False
        Whether or not to save the best model, if the
        modelcheckpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name.
    save_last_model : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file.
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if
        save_last_model is set to False, this parameter
        is discarded.
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    .. [1] Serrà et al. Towards a Universal Neural Network Encoder for Time Series
    In proceedings International Conference of the Catalan Association
    for Artificial Intelligence, 120--129 2018.


    """

    _tags = {
        "python_dependencies": ["tensorflow"],
    }

    def __init__(
        self,
        n_epochs=100,
        batch_size=12,
        kernel_size=None,
        n_filters=None,
        dropout_proba=0.2,
        activation="sigmoid",
        max_pool_size=2,
        padding="same",
        strides=1,
        fc_units=256,
        callbacks=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        random_state=None,
        use_bias=True,
        optimizer=None,
    ):
        self.n_filters = n_filters
        self.max_pool_size = max_pool_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.dropout_proba = dropout_proba
        self.fc_units = fc_units

        self.callbacks = callbacks
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.use_bias = use_bias
        self.optimizer = optimizer

        self.history = None

        super().__init__(
            batch_size=batch_size,
            random_state=random_state,
            last_file_name=last_file_name,
        )

        self._network = EncoderNetwork(
            kernel_size=self.kernel_size,
            max_pool_size=self.max_pool_size,
            n_filters=self.n_filters,
            fc_units=self.fc_units,
            strides=self.strides,
            padding=self.padding,
            dropout_proba=self.dropout_proba,
            activation=self.activation,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d, m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m, d). This method also assumes (m, d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m, d).
        n_classes : int
            The number of classes, which becomes the size of the output layer.

        Returns
        -------
        output : a compiled Keras Model
        """
        import numpy as np
        import tensorflow as tf

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(
            units=n_classes, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam(learning_rate=0.00001)
            if self.optimizer is None
            else self.optimizer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints)
        y : np.ndarray
            The training data class labels of shape (n_cases,).

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.save_init_model:
            self.training_model_.save(self.file_path + self.init_file_name + ".keras")

        if self.verbose:
            self.training_model_.summary()

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        if self.callbacks is None:
            self.callbacks_ = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.file_path + self.file_name_ + ".keras",
                    monitor="loss",
                    save_best_only=True,
                ),
            ]
        else:
            self.callbacks_ = self._get_model_checkpoint_callback(
                callbacks=self.callbacks,
                file_path=self.file_path,
                file_name=self.file_name_,
            )

        self.history = self.training_model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        try:
            self.model_ = tf.keras.models.load_model(
                self.file_path + self.file_name_ + ".keras",
                compile=False,
            )
            if not self.save_best_model:
                os.remove(self.file_path + self.file_name_ + ".keras")
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        gc.collect()
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default = "default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return "default" set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param1 = {
            "n_epochs": 8,
            "batch_size": 4,
            "use_bias": False,
            "n_filters": [2],
            "kernel_size": [2],
            "fc_units": 8,
            "strides": 2,
            "dropout_proba": 0,
        }

        test_params = [param1]

        return test_params
