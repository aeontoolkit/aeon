"""Multi Layer Perceptron Network (MLP) regressor."""

__author__ = ["Aadya-Chinubhai", "hadifawaz1999"]
__all__ = ["MLPRegressor"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.networks import MLPNetwork
from aeon.regression.deep_learning.base import BaseDeepRegressor


class MLPRegressor(BaseDeepRegressor):
    """Multi Layer Perceptron Network (MLP).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_epochs : int, default = 2000
        the number of epochs to train the model
    batch_size : int, default = 16
        the number of samples per gradient update.
    callbacks : callable or None, default
    verbose : boolean, default = False
        whether to output extra information
    loss : string, default="mean_squared_error"
        fit parameter for the keras model
    metrics : list of strings, default="mean_squared_error"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.
    file_path : str, default = "./"
        file_path when saving model_Checkpoint callback
    save_best_model : bool, default = False
        Whether or not to save the best model, if the
        modelcheckpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name
    save_last_model : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if
        save_last_model is set to False, this parameter
        is discarded
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
    activation : string or a tf callable, default="relu"
        Activation function used in the output linear layer.
        List of available activation functions:
        https://keras.io/api/layers/activations/
    output_activation : str = "linear"
        Activation for the last layer in a Regressor
    use_bias : boolean, default = True
        whether the layer uses a bias vector.
    optimizer : keras.optimizer, default=keras.optimizers.Adadelta()


    References
    ----------
    .. [1] Wang et. al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.

    Examples
    --------
    >>> from aeon.regression.deep_learning import MLPRegressor
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> mlp = MLPRegressor(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> mlp.fit(X_train, y_train)  # doctest: +SKIP
    MLPRegressor(...)
    """

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        callbacks=None,
        verbose=False,
        loss="mse",
        metrics="mean_squared_error",
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        random_state=None,
        activation="relu",
        output_activation="linear",
        use_bias=True,
        optimizer=None,
    ):
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.use_bias = use_bias
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.optimizer = optimizer
        self.random_state = random_state
        self.output_activation = output_activation

        self.history = None

        super().__init__(
            batch_size=batch_size,
            last_file_name=last_file_name,
        )

        self._network = MLPNetwork()

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
        from tensorflow import keras

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=1, activation=self.output_activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.Adadelta() if self.optimizer is None else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=self._metrics,
        )
        return model

    def _fit(self, X, y):
        """Fit the Regressor on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The training input samples.
        y : np.ndarray of shape n
            The training data target values.

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

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        if self.callbacks is None:
            self.callbacks_ = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=200, min_lr=0.1
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
        }

        return [param]
