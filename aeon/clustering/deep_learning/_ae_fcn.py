"""Deep Learning Auto-Encoder using FCN Network."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["AEFCNClusterer"]

import gc
import os
import sys
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.clustering._dummy import DummyClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.networks import AEFCNNetwork


class AEFCNClusterer(BaseDeepClusterer):
    """Auto-Encoder based Fully Convolutional Network (FCN), as described in [1]_.

    Parameters
    ----------
    estimator : aeon clusterer, default=None
        An aeon estimator to be built using the transformed data.
        Defaults to aeon TimeSeriesKMeans() with euclidean distance
        and mean averaging method and n_clusters set to 2.
    latent_space_dim : int, default=128
        Dimension of the latent space of the auto-encoder.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    n_layers : int, default = 3
        Number of convolution layers.
    n_filters : int or list of int, default = [128,256,128]
        Number of filters used in convolution layers.
    kernel_size : int or list of int, default = [8,5,3]
        Size of convolution kernel.
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution.
    strides : int or list of int, default = 1
        The strides of the convolution filter.
    padding : str or list of str, default = "same"
        The type of padding used for convolution.
    activation : str or list of str, default = "relu"
        Activation used after the convolution.
    use_bias : bool or list of bool, default = True
        Whether or not ot use bias in convolution.
    n_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    use_mini_batch_size : bool, default = True,
        Whether or not to use the mini batch size formula.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose : boolean, default = False
        Whether to output extra information.
    loss : string, default="mean_squared_error"
        Fit parameter for the keras model. "multi_rec" for multiple mse loss.
        Multiple mse loss computes mean squared error between all embeddings
        of encoder layers with the corresponding reconstructions of the
        decoder layers.
        Fit parameter for the keras model.
    metrics : keras metrics, default = ["mean_squared_error"]
        will be set to mean_squared_error as default if None
    optimizer : keras.optimizers object, default = Adam(lr=0.01)
        Specify the optimizer and the learning rate to be used.
    file_path : str, default = "./"
        File path to save best model.
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
        The name of the file of the init model, if
        save_init_model is set to False,
        this parameter is discarded.
    callbacks : keras.callbacks, default = None
        List of keras callbacks.

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
    >>> from aeon.clustering.deep_learning import AEFCNClusterer
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> aefcn = AEFCNClusterer(n_epochs=5,batch_size=4)  # doctest: +SKIP
    >>> aefcn.fit(X_train)  # doctest: +SKIP
    AEFCNClusterer(...)
    """

    def __init__(
        self,
        estimator=None,
        latent_space_dim=128,
        temporal_latent_space=False,
        n_layers=3,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=True,
        n_epochs=2000,
        batch_size=32,
        use_mini_batch_size=False,
        random_state=None,
        verbose=False,
        loss="mse",
        metrics=None,
        optimizer="Adam",
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        callbacks=None,
    ):
        self.latent_space_dim = latent_space_dim
        self.temporal_latent_space = temporal_latent_space
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.use_mini_batch_size = use_mini_batch_size
        self.callbacks = callbacks
        self.file_path = file_path
        self.n_epochs = n_epochs
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.random_state = random_state

        super().__init__(
            estimator=estimator,
            batch_size=batch_size,
            last_file_name=last_file_name,
        )

        self._network = AEFCNNetwork(
            latent_space_dim=self.latent_space_dim,
            temporal_latent_space=self.temporal_latent_space,
            n_layers=self.n_layers,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
        )

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape
        (n_channels,n_timepoints). Keras/tensorflow assume
        data is in shape (n_timepoints,n_channels). This method also assumes
        (n_timepoints,n_channels). Transpose should happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be
            (n_timepoints,n_channels).

        Returns
        -------
        output : a compiled Keras Model.
        """
        import numpy as np
        import tensorflow as tf

        if self.metrics is None:
            self._metrics = ["mean_squared_error"]
        elif isinstance(self.metrics, list):
            self._metrics = self.metrics
        elif isinstance(self.metrics, str):
            self._metrics = [self.metrics]
        else:
            raise ValueError("Metrics should be a list, string, or None.")

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        encoder, decoder = self._network.build_network(input_shape, **kwargs)

        input_layer = tf.keras.layers.Input(input_shape, name="input layer")
        encoder_output = encoder(input_layer)
        decoder_output = decoder(encoder_output)
        output_layer = decoder_output

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model.compile(
            optimizer=self.optimizer_,
            loss=self.loss,
            metrics=self._metrics,
        )
        return model

    def _fit(self, X):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases (n), n_channels (d), n_timepoints (m))
            The training input samples.

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

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

        if not self.loss == "multi_rec":
            self.history = self.training_model_.fit(
                X,
                X,
                batch_size=mini_batch_size,
                epochs=self.n_epochs,
                verbose=self.verbose,
                callbacks=self.callbacks_,
            )

        elif self.loss == "multi_rec":
            self.history = self._fit_multi_rec_model(
                autoencoder=self.training_model_,
                inputs=X,
                outputs=X,
                batch_size=mini_batch_size,
                epochs=self.n_epochs,
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

        self._fit_clustering(X=X)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        gc.collect()

        return self

    def _fit_multi_rec_model(
        self,
        autoencoder,
        inputs,
        outputs,
        batch_size,
        epochs,
    ):
        import tensorflow as tf

        train_dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        if isinstance(self.optimizer_, str):
            self.optimizer_ = tf.keras.optimizers.get(self.optimizer_)

        history = {"loss": []}

        def layerwise_mse_loss(autoencoder, inputs, outputs):
            def loss(y_true, y_pred):
                # Calculate MSE for each layer in the encoder and decoder
                mse = 0

                _encoder_intermediate_outputs = (
                    []
                )  # Store embeddings of each layer in the Encoder
                _decoder_intermediate_outputs = (
                    []
                )  # Store embeddings of each layer in the Decoder

                encoder = autoencoder.layers[1]  # Returns Functional API Models.
                decoder = autoencoder.layers[2]  # Returns Functional API Models.

                # Run the models since the below given loop misses the latent space
                # layer which doesn't contribute to the loss.
                logits = encoder(inputs)
                __dec_outputs = decoder(logits)

                # Encoder
                for i in range(self.n_layers):
                    _activation_layer = encoder.get_layer(f"__act_encoder_block{i}")
                    _model = tf.keras.models.Model(
                        inputs=encoder.input, outputs=_activation_layer.output
                    )
                    __output = _model(inputs, training=True)
                    _encoder_intermediate_outputs.append(__output)

                # Decoder
                for i in range(self.n_layers):
                    _activation_layer = decoder.get_layer(f"__act_decoder_block{i}")
                    _model = tf.keras.models.Model(
                        inputs=decoder.input, outputs=_activation_layer.output
                    )
                    __output = _model(logits, training=True)
                    _decoder_intermediate_outputs.append(__output)

                if not (
                    len(_encoder_intermediate_outputs)
                    == len(_decoder_intermediate_outputs)
                ):
                    raise ValueError("The Auto-Encoder must be symmetric in nature.")

                # # Append normal mean_squared_error

                for enc_output, dec_output in zip(
                    _encoder_intermediate_outputs, _decoder_intermediate_outputs
                ):
                    mse += tf.keras.backend.mean(
                        tf.keras.backend.square(enc_output - dec_output)
                    )

                inputs_casted = tf.cast(inputs, dtype=tf.float64)
                __dec_outputs_casted = tf.cast(__dec_outputs, dtype=tf.float64)
                return tf.cast(mse, dtype=tf.float64) + tf.cast(
                    tf.reduce_mean(tf.square(inputs_casted - __dec_outputs_casted)),
                    dtype=tf.float64,
                )

            return loss

        # Initialize callbacks
        for callback in self.callbacks_:
            callback.set_model(autoencoder)
            callback.on_train_begin()

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # Calculate the actual loss by calling the loss function
                    loss_func = layerwise_mse_loss(
                        autoencoder=autoencoder,
                        inputs=x_batch_train,
                        outputs=y_batch_train,
                    )
                    loss_value = loss_func(y_batch_train, autoencoder(x_batch_train))

                grads = tape.gradient(loss_value, autoencoder.trainable_weights)
                self.optimizer_.apply_gradients(
                    zip(grads, autoencoder.trainable_weights)
                )

                epoch_loss += float(loss_value)
                num_batches += 1

                # Update callbacks on batch end
                for callback in self.callbacks_:
                    callback.on_batch_end(step, {"loss": float(loss_value)})

            epoch_loss /= num_batches
            history["loss"].append(epoch_loss)

            sys.stdout.write(
                "Training loss at epoch %d: %.4f\n" % (epoch, float(epoch_loss))
            )

            for callback in self.callbacks_:
                callback.on_epoch_end(epoch, {"loss": float(epoch_loss)})

        # Finalize callbacks
        for callback in self.callbacks_:
            callback.on_train_end()

        return history

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
        param = {
            "n_epochs": 2,
            "batch_size": 4,
            "use_bias": False,
            "n_layers": 2,
            "n_filters": [2, 2],
            "kernel_size": [2, 2],
            "padding": "same",
            "strides": 1,
            "latent_space_dim": 2,
            "estimator": DummyClusterer(n_clusters=2),
        }

        return [param]
