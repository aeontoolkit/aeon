"""Fully Convolutional Network (FCN) regressor."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["FCNRegressor"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.networks import FCNNetwork
from aeon.regression.deep_learning.base import BaseDeepRegressor


class FCNRegressor(BaseDeepRegressor):
    """Fully Convolutional Network (FCN).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers        : int, default = 3
        number of convolution layers
    n_filters       : int or list of int, default = [128,256,128]
        number of filters used in convolution layers
    kernel_size     : int or list of int, default = [8,5,3]
        size of convolution kernel
    dilation_rate   : int or list of int, default = 1
        the dilation rate for convolution
    strides         : int or list of int, default = 1
        the strides of the convolution filter
    padding         : str or list of str, default = "same"
        the type of padding used for convolution
    activation      : str or list of str, default = "relu"
        activation used after the convolution
    use_bias        : bool or list of bool, default = True
        whether or not ot use bias in convolution
    n_epochs        : int, default = 2000
        the number of epochs to train the model
    batch_size      : int, default = 16
        the number of samples per gradient update.
    use_mini_batch_size : bool, default = False,
        whether or not to use the mini batch size formula
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose         : boolean, default = False
        whether to output extra information
    output_activation   : str, default = "linear",
        the output activation of the regressor
    loss            : string, default="mean_squared_error"
        fit parameter for the keras model
    metrics         : list of strings, default="mean_squared_error",
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.
    optimizer       : keras.optimizers object, default = Adam(lr=0.01)
        specify the optimizer and the learning rate to be used.
    file_path       : str, default = "./"
        file path to save best model
    save_best_model     : bool, default = False
        Whether or not to save the best model, if the
        modelcheckpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name
    save_last_model     : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name      : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded
    last_file_name      : str, default = "last_model"
        The name of the file of the last model, if
        save_last_model is set to False, this parameter
        is discarded
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.
    callbacks       : keras.callbacks, default = None

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    References
    ----------
    .. [1] Zhao et. al, Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from aeon.regression.deep_learning import FCNRegressor
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> rgs = FCNRegressor(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> rgs.fit(X, y)  # doctest: +SKIP
    FCNRegressor(...)
    """

    def __init__(
        self,
        n_layers=3,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        n_epochs=2000,
        batch_size=16,
        use_mini_batch_size=False,
        callbacks=None,
        verbose=False,
        output_activation="linear",
        loss="mse",
        metrics="mean_squared_error",
        random_state=None,
        use_bias=True,
        optimizer=None,
    ):
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.use_bias = use_bias
        self.output_activation = output_activation
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.use_mini_batch_size = use_mini_batch_size
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.optimizer = optimizer
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name

        self.history = None

        super().__init__(batch_size=batch_size, last_file_name=last_file_name)

        self._network = FCNNetwork(
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
        )

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)

        Returns
        -------
        output : a compiled Keras Model
        """
        import numpy as np
        import tensorflow as tf

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(
            units=1,
            activation=self.output_activation,
        )(output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=self._metrics,
        )

        return model

    def _fit(self, X, y):
        """Fit the regressor on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints).
        y : np.ndarray
            The training data target values of shape (n_cases,).

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        if isinstance(self.metrics, str):
            self._metrics = [self.metrics]
        else:
            self._metrics = self.metrics

        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape)

        if self.save_init_model:
            self.training_model_.save(self.file_path + self.init_file_name + ".keras")

        if self.verbose:
            self.training_model_.summary()

        if self.use_mini_batch_size:
            mini_batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            mini_batch_size = self.batch_size

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        if self.callbacks is None:
            self.callbacks_ = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
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
            y,
            batch_size=mini_batch_size,
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
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For Regressors, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param = {
            "n_epochs": 2,
            "batch_size": 4,
            "use_bias": False,
            "n_layers": 1,
            "n_filters": 4,
            "kernel_size": 3,
            "padding": "valid",
            "strides": 2,
        }

        return [param]
