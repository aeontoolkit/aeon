"""Encoder Regressor."""

__author__ = ["AnonymousCodes911"]
__all__ = ["EncoderRegressor"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.networks import EncoderNetwork
from aeon.regression.deep_learning.base import BaseDeepRegressor


class EncoderRegressor(BaseDeepRegressor):
    """
    Establishing the network structure for an Encoder.

    Adapted from the implementation used in classification.deeplearning

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
    output_activation   : str, default = "linear",
        the output activation of the regressor
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
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if
        save_last_model is set to False, this parameter
        is discarded.
    n_epochs:
        The number of times the entire training dataset
        will be passed forward and backward
        through the neural network.
    random_state : int or None, default=None
        Seed for random number generation.
    loss:
        The loss function to use for training.
    metrics: str or list of str, default="accuracy"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.
    use_bias:
        Whether to use bias in the dense layers.
    optimizer:
        The optimizer to use for training.
    verbose:
        Whether to print progress messages during training.

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    ..[1] Serrà et al. Towards a Universal Neural Network Encoder for Time Series
    In proceedings International Conference of the Catalan Association
    for Artificial Intelligence, 120--129 2018.

    """

    _tags = {
        "python_dependencies": ["tensorflow", "tensorflow_addons"],
    }

    def __init__(
        self,
        n_epochs=100,
        batch_size=12,
        kernel_size=None,
        n_filters=None,
        dropout_proba=0.2,
        activation="sigmoid",
        output_activation="linear",
        max_pool_size=2,
        padding="same",
        strides=1,
        fc_units=256,
        callbacks=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        verbose=False,
        loss="mean_squared_error",
        metrics="accuracy",
        use_bias=True,
        optimizer=None,
        random_state=None,
    ):
        self.n_filters = n_filters
        self.max_pool_size = max_pool_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.output_activation = output_activation
        self.padding = padding
        self.dropout_proba = dropout_proba
        self.fc_units = fc_units
        self.random_state = random_state
        self.callbacks = callbacks
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.best_file_name = best_file_name
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.use_bias = use_bias
        self.optimizer = optimizer

        self.history = None

        super().__init__(
            batch_size=batch_size,
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

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d, m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m, d). This method also assumes (m, d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
        The shape of the data fed into the input layer, should be (m, d).
        Gives
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        tf.random.set_seed(self.random_state)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(
            units=1, activation=self.output_activation, use_bias=self.use_bias
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
            metrics=self._metrics,
        )

        return model

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The training input samples.
        y : np.ndarray of shape n
            The training data Target Values.

        Gives
        -------
        self : object
        """
        import tensorflow as tf

        # Transpose X to conform to Keras input style
        X = X.transpose(0, 2, 1)
        check_random_state(self.random_state)
        if isinstance(self.metrics, str):
            self._metrics = [self.metrics]
        else:
            self._metrics = self.metrics
        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape)

        if self.verbose:
            self.training_model_.summary()

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        self.callbacks_ = (
            [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.file_path + self.file_name_ + ".keras",
                    monitor="loss",
                    save_best_only=True,
                ),
            ]
            if self.callbacks is None
            else self.callbacks
        )

        self.history = self.training_model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        try:
            self.model_ = tf.keras.models.load_model(
                self.file_path + self.file_name_ + ".keras", compile=False
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
            For regressors, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            predictions to compare against.

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
            "fc_units": 8,
            "strides": 2,
            "dropout_proba": 0,
        }

        test_params = [param1]

        return test_params
