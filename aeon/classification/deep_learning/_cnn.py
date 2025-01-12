"""Time Convolutional Neural Network (CNN) classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["TimeCNNClassifier"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks import TimeCNNNetwork


class TimeCNNClassifier(BaseDeepClassifier):
    """
    Time Convolutional Neural Network (CNN).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, default = 2
        The number of convolution layers in the network.
    kernel_size : int or list of int, default = 7
        Kernel size of convolution layers, if not a list, the same kernel size
        is used for all layer, len(list) should be n_layers.
    n_filters : int or list of int, default = [6, 12]
        Number of filters for each convolution layer, if not a list, the same n_filters
        is used in all layers.
    avg_pool_size : int or list of int, default = 3
        The size of the average pooling layer, if not a list, the same
        max pooling size is used for all convolution layer.
    activation : str or list of str, default = "sigmoid"
        Keras activation function used in the model for each layer, if not a list,
        the same activation is used for all layers.
    padding : str or list of str, default = 'valid'
        The method of padding in convolution layers, if not a list, the same padding
        used for all convolution layers.
    strides : int or list of int, default = 1
        The strides of kernels in the convolution and max pooling layers, if not a
        list, the same strides are used for all layers.
    strides_pooling : int or list of int, default = None
        Strides for the pooling layers. If None, defaults to pool_size.
        If not a list, the same strides are used for all pooling layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers, if not a list, the same dilation
        rate is used all over the network.
    use_bias : bool or list of bool, default = True
        Condition on whether to use bias values for convolution layers,
        if not a list, the same condition is used for all layers.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    n_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    verbose : boolean, default = False
        Whether to output extra information.
    loss : str, default = "mean_squared_error"
        The name of the keras training loss.
    optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
        The keras optimizer used for training.
    metrics : str or list[str], default="accuracy"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to
        ModelCheckpoint.
    file_path : file_path for the best model
        Only used if checkpoint is used as callback.
    save_best_model : bool, default = False
        Whether to save the best model, if the modelcheckpoint callback is used by
        default, this condition, if True, will prevent the automatic deletion of the
        best saved model from file and the user can choose the file name.
    save_last_model : bool, default = False
        Whether to save the last model, last epoch trained, using the base class method
        save_last_model_to_file.
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if save_best_model is set to False,
        this parameter is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if save_last_model is set to False,
        this parameter is discarded.
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    References
    ----------
    .. [1] Zhao et. al, Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from aeon.classification.deep_learning import TimeCNNClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> cnn = TimeCNNClassifier(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> cnn.fit(X_train, y_train)  # doctest: +SKIP
    TimeCNNClassifier(...)
    """

    def __init__(
        self,
        n_layers=2,
        kernel_size=7,
        n_filters=None,
        avg_pool_size=3,
        activation="sigmoid",
        padding="valid",
        strides=1,
        strides_pooling=None,
        dilation_rate=1,
        n_epochs=2000,
        batch_size=16,
        callbacks=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        verbose=False,
        loss="mean_squared_error",
        metrics="accuracy",
        random_state=None,
        use_bias=True,
        optimizer=None,
    ):
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.padding = padding
        self.strides = strides
        self.strides_pooling = strides_pooling
        self.dilation_rate = dilation_rate
        self.avg_pool_size = avg_pool_size
        self.activation = activation
        self.use_bias = use_bias

        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.history = None

        super().__init__(
            batch_size=batch_size,
            random_state=random_state,
            last_file_name=last_file_name,
        )

        self._network = TimeCNNNetwork(
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
            avg_pool_size=self.avg_pool_size,
            activation=self.activation,
            padding=self.padding,
            strides=self.strides,
            strides_pooling=self.strides_pooling,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
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
            The shape of the data fed into the input layer, should be (m, d)
        n_classes : int
            The number of classes, which becomes the size of the output layer

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
            units=n_classes, activation=self.activation
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
        """
        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
            "avg_pool_size": 4,
        }

        test_params = [param1]

        return test_params
