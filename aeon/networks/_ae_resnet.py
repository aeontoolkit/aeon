"""Auto-Encoder using Residual Network (AEResNetNetwork)."""

__maintainer__ = ["hadifawaz1999"]


import numpy as np

from aeon.networks.base import BaseDeepLearningNetwork


class AEResNetNetwork(BaseDeepLearningNetwork):
    """
    Establish the network structure for a AE-ResNet.

    Adapted from the implementations used in [1]_.

    Parameters
    ----------
    latent_space_dim : int, default = 128
        Dimension of the auto-encoder's latent space.
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
        The kernel size of all the convolution layers in one residual block, if not a
        list, the same kernel size is used in all convolution layers.
    strides : int or list of int, default = 1
        The strides of convolution kernels in each of the convolution layers in one
        residual block, if not a list, the same kernel size is used in all
        convolution layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers in one residual block, if not a
        list, the same kernel size is used in all convolution layers.
    padding : str or list of str, default = 'padding'
        The type of padding used in the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
    activation : str or list of str, default = 'relu'
        Keras activation used in the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values in the convolution layers in
        one residual block, if not a list, the same kernel size is used in all
        convolution layers.

    Notes
    -----
    Adpated from the implementation source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    References
    ----------
    .. [1] H. Fawaz, G. B. Lanckriet, F. Petitjean, and L. Idoumghar,

    Network originally defined in:

    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }

    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.12",
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        latent_space_dim=128,
        temporal_latent_space=False,
        n_residual_blocks=3,
        n_conv_per_residual_block=3,
        n_filters=None,
        kernel_size=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
    ):
        self.latent_space_dim = latent_space_dim
        self.temporal_latent_space = temporal_latent_space
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block

        super().__init__()

    def _shortcut_layer(
        self, input_tensor, output_tensor, padding="same", use_bias=True
    ):
        import tensorflow as tf

        n_out_filters = int(output_tensor.shape[-1])

        shortcut_layer = tf.keras.layers.Conv1D(
            filters=n_out_filters, kernel_size=1, padding=padding, use_bias=use_bias
        )(input_tensor)
        shortcut_layer = tf.keras.layers.BatchNormalization()(shortcut_layer)

        return tf.keras.layers.Add()([output_tensor, shortcut_layer])

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_dimensions (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        input_layer : keras.layers.Input
            The input layer of the network.
        output_layer : keras.layers.Layer
            The output layer of the network.
        """
        import tensorflow as tf

        self._n_filters_ = [64, 128, 128] if self.n_filters is None else self.n_filters
        self._kernel_size_ = [8, 5, 3] if self.kernel_size is None else self.kernel_size

        if isinstance(self._n_filters_, list):
            assert len(self._n_filters_) == self.n_residual_blocks, (
                f"Number of filters {len(self._n_filters_)} should be"
                f" the same as number of residual blocks but is"
                f" not: {self.n_residual_blocks}."
            )
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_residual_blocks

        if isinstance(self._kernel_size_, list):
            assert len(self._kernel_size_) == self.n_conv_per_residual_block, (
                f"Number of kernel sizes {len(self._kernel_size_)} should be"
                f" the same as number of convolution layers per block but is"
                f" not: {self.n_conv_per_residual_block}."
            )
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_conv_per_residual_block

        if isinstance(self.strides, list):
            assert len(self.strides) == self.n_conv_per_residual_block, (
                f"Number of strides {len(self.strides)} should be"
                f" the same as number of convolution layers per block but is"
                f" not: {self.n_conv_per_residual_block}."
            )
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_conv_per_residual_block

        if isinstance(self.dilation_rate, list):
            assert len(self.dilation_rate) == self.n_conv_per_residual_block, (
                f"Number of dilation rates {len(self.dilation_rate)} should be"
                f" the same as number of convolution layers per block but is"
                f" not: {self.n_conv_per_residual_block}."
            )
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_conv_per_residual_block

        if isinstance(self.padding, list):
            assert len(self.padding) == self.n_conv_per_residual_block, (
                f"Number of paddings {len(self.padding)} should be"
                f" the same as number of convolution layers per block but is"
                f" not: {self.n_conv_per_residual_block}."
            )
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_conv_per_residual_block

        if isinstance(self.activation, list):
            assert len(self.activation) == self.n_conv_per_residual_block, (
                f"Number of activations {len(self.activation)} should be"
                f" the same as number of convolution layers per block but is"
                f" not: {self.n_conv_per_residual_block}."
            )
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_conv_per_residual_block

        if isinstance(self.use_bias, list):
            assert len(self.use_bias) == self.n_conv_per_residual_block, (
                f"Number of use biases {len(self.use_bias)} should be"
                f" the same as number of convolution layers per block but is"
                f" not: {self.n_conv_per_residual_block}."
            )
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_conv_per_residual_block

        input_layer_encoder = tf.keras.layers.Input(input_shape)

        x = input_layer_encoder

        for d in range(self.n_residual_blocks):
            input_block_tensor = x

            for c in range(self.n_conv_per_residual_block):
                conv = tf.keras.layers.Conv1D(
                    filters=self._n_filters[d],
                    kernel_size=self._kernel_size[c],
                    strides=self._strides[d],
                    padding=self._padding[c],
                    dilation_rate=self._dilation_rate[c],
                )(x)
                conv = tf.keras.layers.BatchNormalization()(conv)

                if c == self.n_conv_per_residual_block - 1:
                    conv = self._shortcut_layer(
                        input_tensor=input_block_tensor, output_tensor=conv
                    )

                conv = tf.keras.layers.Activation(activation=self._activation[c])(conv)

                x = conv
        if not self.temporal_latent_space:
            shape_before_flattent = x.shape[1:]

            flatten_layer = tf.keras.layers.Flatten()(x)
            latent_space = tf.keras.layers.Dense(units=self.latent_space_dim)(
                flatten_layer
            )
        else:
            latent_space = tf.keras.layers.Conv1D(
                filters=self.latent_space_dim,
                kernel_size=1,
                strides=self._strides[-1],
                padding=self._padding[-1],
                dilation_rate=self._dilation_rate[-1],
                use_bias=self._use_bias[-1],
            )(x)

        encoder = tf.keras.models.Model(
            inputs=input_layer_encoder, outputs=latent_space, name="encoder"
        )

        if not self.temporal_latent_space:
            input_layer_decoder = tf.keras.layers.Input((self.latent_space_dim,))

            dense_layer = tf.keras.layers.Dense(units=np.prod(shape_before_flattent))(
                input_layer_decoder
            )

            reshape_layer = tf.keras.layers.Reshape(target_shape=shape_before_flattent)(
                dense_layer
            )
            x = reshape_layer
        else:
            input_layer_decoder = tf.keras.layers.Input(latent_space.shape[1:])

            x = input_layer_decoder

        for d in range(self.n_residual_blocks):
            input_block_tensor = x

            for c in range(self.n_conv_per_residual_block)[::-1]:
                conv = tf.keras.layers.Conv1DTranspose(
                    filters=self._n_filters[d],
                    kernel_size=self._kernel_size[c],
                    strides=self._strides[d],
                    padding=self._padding[c],
                    dilation_rate=self._dilation_rate[c],
                )(x)
                conv = tf.keras.layers.BatchNormalization()(conv)

                if c == self.n_conv_per_residual_block - 1:
                    conv = self._shortcut_layer(
                        input_tensor=input_block_tensor, output_tensor=conv
                    )

                conv = tf.keras.layers.Activation(activation=self._activation[c])(conv)

                x = conv

        last_projection_layer = tf.keras.layers.Conv1DTranspose(
            filters=input_shape[-1],
            kernel_size=1,
            padding=self._padding[0],
            strides=self._strides[0],
            dilation_rate=self._dilation_rate[0],
            use_bias=self._use_bias[0],
        )(x)

        decoder = tf.keras.models.Model(
            inputs=input_layer_decoder, outputs=last_projection_layer, name="decoder"
        )

        return encoder, decoder
