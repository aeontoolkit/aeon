"""Residual Network (ResNet) for clustering."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["AEResNetClusterer"]

import gc
import os
import sys
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.clustering._dummy import DummyClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.networks import AEResNetNetwork


class AEResNetClusterer(BaseDeepClusterer):
    """
    Auto-Encoder with Residual Network backbone for clustering.

    Adapted from the implementation used in [1]_.

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
    n_residual_blocks : int, default = 3
        The number of residual blocks of ResNet's model.
    n_conv_per_residual_block : int, default = 3
        The number of convolution blocks in each residual block.
    n_filters : int or list of int, default = [128, 64, 64]
        The number of convolution filters for all the convolution layers in the same
        residual block, if not a list, the same number of filters is used in all
        convolutions of all residual blocks.
    kernel_size : int or list of int, default = [8, 5, 3]
        The kernel size of all the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
    strides : int or list of int, default = 1
        The strides of convolution kernels in each of the convolution layers in
        one residual block, if not a list, the same kernel size is used in all
        convolution layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
    padding : str or list of str, default = 'padding'
        The type of padding used in the convolution layers in one residual block, if
        not a list, the same kernel size is used in all convolution layers.
    activation : str or list of str, default = 'relu'
        keras activation used in the convolution layers in one residual block,
        if not a list, the same kernel size is used in all convolution layers.
    use_bia : bool or list of bool, default = True
        Condition on whether or not to use bias values in the convolution layers
        in one residual block, if not a list, the same kernel size is used in all
        convolution layers.
    n_epochs : int, default = 1500
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    use_mini_batch_size : bool, default = False
        Condition on using the mini batch size formula Wang et al.
    callbacks : callable or None, default ReduceOnPlateau and ModelCheckpoint
        List of tf.keras.callbacks.Callback objects.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    file_path : str, default = './'
        file_path when saving model_Checkpoint callback.
    save_best_model : bool, default = False
        Whether or not to save the best model, if the modelcheckpoint callback is
        used by default, this condition, if True, will prevent the automatic
        deletion of the best saved model from file and the user can choose the
        file name.
    save_last_model : bool, default = False
        Whether or not to save the last model, last epoch trained, using the base
        class method save_last_model_to_file.
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if save_best_model is set to
        False, this parameter is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if save_last_model is set to
        False, this parameter is discarded.
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if
        save_init_model is set to False,
        this parameter is discarded.
    verbose : boolean, default = False
        whether to output extra information
    loss : string, default = "mean_squared_error"
        fit parameter for the keras model. "multi_rec" for multiple mse loss.
        Multiple mse loss computes mean squared error between all embeddings
        of encoder layers with the corresponding reconstructions of the
        decoder layers.
    optimizer : keras.optimizer, default = keras.optimizers.Adam()
    metrics : list of strings, default = ["mean_squared_error"]
        will be set to mean_squared_error as default if None

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    References
    ----------
    .. [1] Wang et. al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.

    Examples
    --------
    >>> from aeon.clustering.deep_learning import AEResNetClusterer
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> ae_resnet = AEResNetClusterer(n_epochs=20) # doctest: +SKIP
    >>> ae_resnet.fit(X_train, y_train) # doctest: +SKIP
    AEResNetClusterer(...)
    """

    def __init__(
        self,
        estimator=None,
        n_residual_blocks=3,
        n_conv_per_residual_block=3,
        n_filters=None,
        kernel_size=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
        n_epochs=1500,
        callbacks=None,
        verbose=False,
        loss="mse",
        metrics=None,
        batch_size=32,
        use_mini_batch_size=False,
        random_state=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        optimizer="Adam",
    ):
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.use_mini_batch_size = use_mini_batch_size
        self.random_state = random_state
        self.activation = activation
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
            estimator=estimator,
            batch_size=batch_size,
            last_file_name=last_file_name,
        )

        self._network = AEResNetNetwork(
            n_residual_blocks=self.n_residual_blocks,
            n_conv_per_residual_block=self.n_conv_per_residual_block,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            use_bias=self.use_bias,
            activation=self.activation,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
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

        self.optimizer_ = (
            tf.keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        encoder, decoder = self._network.build_network(input_shape, **kwargs)

        input_layer = tf.keras.layers.Input(input_shape, name="input layer")
        encoder_output = encoder(input_layer)
        decoder_output = decoder(encoder_output)
        output_layer = decoder_output

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=self._metrics,
        )

        return model

    def _fit(self, X):
        """Fit the Clusterer on the training set X.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases (n), n_dimensions (d), n_timepoints (m))
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

        if self.use_mini_batch_size:
            mini_batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            mini_batch_size = self.batch_size

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
                self.file_path + self.file_name_ + ".keras", compile=False
            )
            if not self.save_best_model:
                os.remove(self.file_path + self.file_name_ + ".keras")
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

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
                for i in range(self.n_residual_blocks):
                    _activation_layer = encoder.get_layer(f"__act_encoder_block{i}")
                    _model = tf.keras.models.Model(
                        inputs=encoder.input, outputs=_activation_layer.output
                    )
                    __output = _model(inputs, training=True)
                    _encoder_intermediate_outputs.append(__output)

                # Decoder
                for i in range(self.n_residual_blocks):
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

                for enc_output, dec_output in zip(
                    _encoder_intermediate_outputs, _decoder_intermediate_outputs
                ):
                    mse += tf.keras.backend.mean(
                        tf.keras.backend.square(enc_output - dec_output)
                    )

                inputs_casted = tf.cast(inputs, tf.float64)
                __dec_outputs_casted = tf.cast(__dec_outputs, tf.float64)
                return tf.cast(mse, tf.float64) + tf.cast(
                    tf.reduce_mean(tf.square(inputs_casted - __dec_outputs_casted)),
                    tf.float64,
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
            "n_residual_blocks": 2,
            "n_conv_per_residual_block": 1,
            "n_filters": [2, 2],
            "kernel_size": 2,
            "use_bias": False,
            "estimator": DummyClusterer(n_clusters=2),
        }

        test_params = [param]

        return test_params
