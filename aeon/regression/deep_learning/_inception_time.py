"""InceptionTime and Inception regressors."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["InceptionTimeRegressor"]

import gc
import os
import time
from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state

from aeon.networks import InceptionNetwork
from aeon.regression.base import BaseRegressor
from aeon.regression.deep_learning.base import BaseDeepRegressor


class InceptionTimeRegressor(BaseRegressor):
    """InceptionTime ensemble regressor.

    Ensemble of IndividualInceptionRegressor, as described in [1]_.
    This ensemble regressor is adapted from the classier InceptionTime

    Parameters
    ----------
        n_regressors       : int, default = 5,
            the number of Inception models used for the
            Ensemble in order to create
            InceptionTime.
        depth               : int, default = 6,
            the number of inception modules used
        n_filters          : int or list of int32, default = 32,
            the number of filters used in one inception
            module, if not a list,
            the same number of filters is used in
            all inception modules
        n_conv_per_layer   : int or list of int, default = 3,
            the number of convolution layers in each inception
            module, if not a list,
            the same number of convolution layers is used
            in all inception modules
        kernel_size         : int or list of int, default = 40,
            the head kernel size used for each inception
            module, if not a list,
            the same is used in all inception modules
        use_max_pooling     : bool or list of bool, default = True,
            conditioning whether or not to use max pooling layer
            in inception modules,if not a list,
            the same is used in all inception modules
        max_pool_size       : int or list of int, default = 3,
            the size of the max pooling layer, if not a list,
            the same is used in all inception modules
        strides             : int or list of int, default = 1,
            the strides of kernels in convolution layers for each
            inception module, if not a list,
            the same is used in all inception modules
        dilation_rate       : int or list of int, default = 1,
            the dilation rate of convolutions in each inception
            module, if not a list,
            the same is used in all inception modules
        padding             : str or list of str, default = "same",
            the type of padding used for convoltuon for each
            inception module, if not a list,
            the same is used in all inception modules
        activation          : str or list of str, default = "relu",
            the activation function used in each inception
            module, if not a list,
            the same is used in all inception modules
        use_bias            : bool or list of bool, default = False,
            condition whether or not convolutions should
            use bias values in each inception
            module, if not a list,
            the same is used in all inception modules
        use_residual        : bool, default = True,
            condition whether or not to use residual
            connections all over Inception
        use_bottleneck      : bool, default = True,
            condition whether or not to use bottlenecks
            all over Inception
        bottleneck_size     : int, default = 32,
            the bottleneck size in case use_bottleneck = True
        use_custom_filters  : bool, default = False,
            condition on whether or not to use custom
            filters in the first inception module
        output_activation   : str, default = "linear",
            the output activation for the regressor
        batch_size          : int, default = 64
            the number of samples per gradient update.
        use_mini_batch_size : bool, default = False
            condition on using the mini batch size
            formula Wang et al.
        n_epochs           : int, default = 1500
            the number of epochs to train the model.
        callbacks           : callable or None, default
        ReduceOnPlateau and ModelCheckpoint
            list of tf.keras.callbacks.Callback objects.
        file_path           : str, default = './'
            file_path when saving model_Checkpoint callback
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
        random_state : int, RandomState instance or None, default=None
            If `int`, random_state is the seed used by the random number generator;
            If `RandomState` instance, random_state is the random number generator;
            If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.
            Seeded random number generation can only be guaranteed on CPU processing,
            GPU processing will be non-deterministic.
        verbose             : boolean, default = False
            whether to output extra information
        optimizer           : keras optimizer, default = Adam
        loss                : keras loss,
                              default = mean_squared_error
        will be set to accuracy as default if None

    Notes
    -----
    ..[1] Fawaz et al. InceptionTime: Finding AlexNet for Time Series
    regression, Data Mining and Knowledge Discovery, 34, 2020

    ..[2] Ismail-Fawaz et al. Deep Learning For Time Series
    regression Using New
    Hand-Crafted Convolution Filters, 2022 IEEE International
    Conference on Big Data.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/regressors/inception.py

    and Ismail-Fawaz et al.
    https://github.com/MSD-IRIMAS/CF-4-TSC

    Examples
    --------
    >>> from aeon.regression.deep_learning import InceptionTimeRegressor
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> inctime = InceptionTimeRegressor(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> inctime.fit(X, y)  # doctest: +SKIP
    InceptionTimeRegressor(...)
    """

    _tags = {
        "python_dependencies": "tensorflow",
        "capability:multivariate": True,
        "non-deterministic": True,
        "cant-pickle": True,
        "algorithm_type": "deeplearning",
    }

    def __init__(
        self,
        n_regressors=5,
        n_filters=32,
        n_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        use_custom_filters=False,
        output_activation="linear",
        file_path="./",
        save_last_model=False,
        save_best_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="mse",
        optimizer=None,
    ):
        self.n_regressors = n_regressors

        self.n_filters = n_filters
        self.n_conv_per_layer = n_conv_per_layer
        self.use_max_pooling = use_max_pooling
        self.max_pool_size = max_pool_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.use_custom_filters = use_custom_filters
        self.output_activation = output_activation

        self.file_path = file_path

        self.save_last_model = save_last_model
        self.save_best_model = save_best_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.last_file_name = last_file_name
        self.init_file_name = init_file_name

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.use_mini_batch_size = use_mini_batch_size
        self.loss = loss
        self.optimizer = optimizer

        self.regressors_ = []

        super().__init__()

    def _fit(self, X, y):
        """Fit each of the Individual Inception models.

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints).
        y : np.ndarray
            The training data target values of shape (n_cases,).

        Returns
        -------
        self : object
            fitted estimator
        """
        self.regressors_ = []
        rng = check_random_state(self.random_state)

        for n in range(0, self.n_regressors):
            rgs = IndividualInceptionRegressor(
                n_filters=self.n_filters,
                n_conv_per_layer=self.n_conv_per_layer,
                kernel_size=self.kernel_size,
                use_max_pooling=self.use_max_pooling,
                max_pool_size=self.max_pool_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding=self.padding,
                activation=self.activation,
                use_bias=self.use_bias,
                use_residual=self.use_residual,
                use_bottleneck=self.use_bottleneck,
                depth=self.depth,
                use_custom_filters=self.use_custom_filters,
                output_activation=self.output_activation,
                file_path=self.file_path,
                save_best_model=self.save_best_model,
                save_last_model=self.save_last_model,
                save_init_model=self.save_init_model,
                best_file_name=self.best_file_name + str(n),
                last_file_name=self.last_file_name + str(n),
                init_file_name=self.init_file_name + str(n),
                batch_size=self.batch_size,
                use_mini_batch_size=self.use_mini_batch_size,
                n_epochs=self.n_epochs,
                callbacks=self.callbacks,
                loss=self.loss,
                optimizer=self.optimizer,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                verbose=self.verbose,
            )
            rgs.fit(X, y)
            self.regressors_.append(rgs)
            gc.collect()

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict the values of the test set using InceptionTime.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The testing input samples.

        Returns
        -------
        Y : np.ndarray of shape = (n_cases)
            The predicted values

        """
        ypreds = np.zeros(shape=(X.shape[0]))

        for rgs in self.regressors_:
            ypreds = ypreds + rgs._predict(X)

        ypreds = ypreds / self.n_regressors

        return ypreds

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For regressors, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default=[None]
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param1 = {
            "n_regressors": 2,
            "n_epochs": 2,
            "batch_size": 4,
            "depth": 1,
            "kernel_size": 2,
            "n_filters": 1,
            "n_conv_per_layer": 1,
            "use_residual": False,
            "use_bottleneck": False,
            "use_max_pooling": False,
            "depth": 1,
            "use_custom_filters": False,
        }

        return [param1]


class IndividualInceptionRegressor(BaseDeepRegressor):
    """Single Inception regressor.

    Parameters
    ----------
        depth               : int, default = 6,
            the number of inception modules used
        n_filters          : int or list of int32, default = 32,
            the number of filters used in one inception module, if not a list,
            the same number of filters is used in all inception modules
        n_conv_per_layer   : int or list of int, default = 3,
            the number of convolution layers in each inception module, if not a list,
            the same number of convolution layers is used in all inception modules
        kernel_size         : int or list of int, default = 40,
            the head kernel size used for each inception module, if not a list,
            the same is used in all inception modules
        use_max_pooling     : bool or list of bool, default = True,
            condition whether or not to use max pooling layer
            in inception modules,if not a list,
            the same is used in all inception modules
        max_pool_size       : int or list of int, default = 3,
            the size of the max pooling layer, if not a list,
            the same is used in all inception modules
        strides             : int or list of int, default = 1,
            the strides of kernels in convolution layers for
            each inception module, if not a list,
            the same is used in all inception modules
        dilation_rate       : int or list of int, default = 1,
            the dilation rate of convolutions in each inception module, if not a list,
            the same is used in all inception modules
        padding             : str or list of str, default = "same",
            the type of padding used for convoltuon for each
            inception module, if not a list,
            the same is used in all inception modules
        activation          : str or list of str, default = "relu",
            the activation function used in each inception module, if not a list,
            the same is used in all inception modules
        use_bias            : bool or list of bool, default = False,
            condition whether or not convolutions should
            use bias values in each inception
            module, if not a list,
            the same is used in all inception modules
        use_residual        : bool, default = True,
            condition whether or not to use residual connections all over Inception
        use_bottleneck      : bool, default = True,
            condition whether or not to use bottlesnecks all over Inception
        bottleneck_size     : int, default = 32,
            the bottleneck size in case use_bottleneck = True
        use_custom_filters  : bool, default = False,
            condition on whether or not to use custom filters
            in the first inception module
        output_activation   : str, default = "linear",
            the output activation of the regressor
        batch_size          : int, default = 64
            the number of samples per gradient update.
        use_mini_batch_size : bool, default = False
            condition on using the mini batch size formula Wang et al.
        n_epochs           : int, default = 1500
            the number of epochs to train the model.
        callbacks           : callable or None, default
        ReduceOnPlateau and ModelCheckpoint
            list of tf.keras.callbacks.Callback objects.
        file_path           : str, default = './'
            file_path when saving model_Checkpoint callback
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
        random_state : int, RandomState instance or None, default=None
            If `int`, random_state is the seed used by the random number generator;
            If `RandomState` instance, random_state is the random number generator;
            If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.
            Seeded random number generation can only be guaranteed on CPU processing,
            GPU processing will be non-deterministic.
        verbose             : boolean, default = False
            whether to output extra information
        optimizer           : keras optimizer, default = Adam
        loss                : keras loss, default = mean_squared_error
        to accuracy as default if None

    Notes
    -----
    ..[1] Fawaz et al. InceptionTime: Finding AlexNet for Time Series
    regression, Data Mining and Knowledge Discovery, 34, 2020

    ..[2] Ismail-Fawaz et al. Deep Learning For Time Series regression Using New
    Hand-Crafted Convolution Filters, 2022 IEEE International Conference on Big Data.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/regressors/inception.py

    and Ismail-Fawaz et al.
    https://github.com/MSD-IRIMAS/CF-4-TSC

    Examples
    --------
    >>> from aeon.regression.deep_learning import IndividualInceptionRegressor
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> inc = IndividualInceptionRegressor(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> inc.fit(X, y)  # doctest: +SKIP
    IndividualInceptionRegressor(...)
    """

    def __init__(
        self,
        n_filters=32,
        n_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        use_custom_filters=False,
        output_activation="linear",
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="mse",
        optimizer=None,
    ):
        # predefined
        self.n_filters = n_filters
        self.n_conv_per_layer = n_conv_per_layer
        self.use_max_pooling = use_max_pooling
        self.max_pool_size = max_pool_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.n_epochs = n_epochs
        self.use_custom_filters = use_custom_filters
        self.output_activation = output_activation

        self.file_path = file_path

        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.use_mini_batch_size = use_mini_batch_size
        self.loss = loss
        self.optimizer = optimizer

        super().__init__(batch_size=batch_size, last_file_name=last_file_name)

        self._network = InceptionNetwork(
            n_filters=self.n_filters,
            n_conv_per_layer=self.n_conv_per_layer,
            kernel_size=self.kernel_size,
            use_max_pooling=self.use_max_pooling,
            max_pool_size=self.max_pool_size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            use_residual=self.use_residual,
            use_bottleneck=self.use_bottleneck,
            bottleneck_size=self.bottleneck_size,
            depth=self.depth,
            use_custom_filters=self.use_custom_filters,
        )

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        tf.keras.models.Model
            A compiled Keras Model
        """
        import numpy as np
        import tensorflow as tf

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(1, activation=self.output_activation)(
            output_layer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
        )

        return model

    def _fit(self, X, y):
        """
        Fit the regressor on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of,
            shape (n_cases, n_channels, n_timepoints).
            If a 2D array-like is passed, n_channels is assumed to be 1.
        y : np.ndarray
            The training data target values of shape (n_cases,).

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape_ = X.shape[1:]

        if self.use_mini_batch_size:
            mini_batch_size = int(min(X.shape[0] // 10, self.batch_size))
        else:
            mini_batch_size = self.batch_size
        self.training_model_ = self.build_model(self.input_shape_)

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
            For regressors, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default=[None]
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param1 = {
            "n_epochs": 2,
            "batch_size": 4,
            "depth": 1,
            "kernel_size": 2,
            "n_filters": 1,
            "n_conv_per_layer": 1,
            "use_residual": False,
            "use_bottleneck": False,
            "use_max_pooling": False,
            "depth": 1,
            "use_custom_filters": False,
        }

        return [param1]
