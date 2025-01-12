"""Multi Layer Perceptron Network (MLP) classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["MLPClassifier"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks import MLPNetwork


class MLPClassifier(BaseDeepClassifier):
    """Multi Layer Perceptron Network (MLP).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, optional (default=3)
        The number of dense layers in the MLP.
    n_units : Union[int, List[int]], optional (default=200)
        Number of units in each dense layer.
    activation : Union[str, List[str]], optional (default='relu')
        Activation function(s) for each dense layer.
    dropout_rate : Union[float, List[Union[int, float]]], optional (default=None)
        Dropout rate(s) for each dense layer. If None, a default rate of 0.2 is used.
        Dropout rate(s) are typically a number in the interval [0, 1].
    dropout_last : float, default = 0.3
        The dropout rate of the last layer.
    use_bias : bool, default = True
        Condition on whether or not to use bias values for dense layers.
    n_epochs : int, default = 2000
        the number of epochs to train the model
    batch_size : int, default = 16
        the number of samples per gradient update.
    use_mini_batch_size : boolean, default = False
        Condition on using the mini batch size formula
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to
        ModelCheckpoint and ReduceLROnPlateau.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose : boolean, default = False
        whether to output extra information
    loss : str, default = "categorical_crossentropy"
        The name of the keras training loss.
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
    optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
        The keras optimizer used for training.
    metrics : str or list[str], default="accuracy"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    References
    ----------
    .. [1] Wang et. al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.

    Examples
    --------
    >>> from aeon.classification.deep_learning import MLPClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> mlp = MLPClassifier(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> mlp.fit(X_train, y_train)  # doctest: +SKIP
    MLPClassifier(...)
    """

    def __init__(
        self,
        n_layers=3,
        n_units=200,
        activation="relu",
        dropout_rate=None,
        dropout_last=None,
        use_bias=True,
        n_epochs=2000,
        batch_size=16,
        use_mini_batch_size=False,
        callbacks=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics="accuracy",
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        random_state=None,
        optimizer=None,
    ):
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_last = dropout_last
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.use_mini_batch_size = use_mini_batch_size
        self.use_bias = use_bias
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.optimizer = optimizer

        self.history = None

        super().__init__(
            batch_size=batch_size,
            random_state=random_state,
            last_file_name=last_file_name,
        )

        self._network = MLPNetwork(
            n_layers=self.n_layers,
            n_units=self.n_units,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            dropout_last=self.dropout_last,
            use_bias=self.use_bias,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)
        n_classes: int
            The number of classes, which becomes the size of the output layer

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

        output_layer = keras.layers.Dense(units=n_classes, activation="softmax")(
            output_layer
        )

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

        if isinstance(self.metrics, list):
            self._metrics = self.metrics
        elif isinstance(self.metrics, str):
            self._metrics = [self.metrics]

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

        if self.use_mini_batch_size:
            mini_batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            mini_batch_size = self.batch_size

        self.history = self.training_model_.fit(
            X,
            y_onehot,
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
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
            "use_bias": False,
        }

        test_params = [param1]

        return test_params
