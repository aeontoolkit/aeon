"""Time Convolutional Neural Network (TimeCNNNetwork)."""

__maintainer__ = ["hadifawaz1999"]

from aeon.networks.base import BaseDeepLearningNetwork


class TimeCNNNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a CNN.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, default = 2
        The number of convolution layers in the network.
    kernel_size : int or list of int, default = 7
        Kernel size of convolution layers, if not a list, the same kernel size is
        used for all layer, len(list) should be n_layers.
    n_filters : int or list of int, default = [6, 12]
        Number of filters for each convolution layer, if not a list, the same
        `n_filters` is used in all layers.
    avg_pool_size : int or list of int, default = 3
        The size of the average pooling layer, if not a list, the same max pooling
        size is used for all convolution layer.
    activation : str or list of str, default = "sigmoid"
        Keras activation function used in the model for each layer, if not a list,
        the same activation is used for all layers.
    padding : str or list of str, default = "valid"
        The method of padding in convolution layers, if not a list, the same padding
        used for all convolution layers.
    strides : int or list of int, default = 1
        The strides of kernels in the convolution and max pooling layers, if not a list,
        the same strides are used for all layers.
    strides_pooling : int or list of int, default = None
        Strides for the pooling layers. If None, defaults to pool_size.
        If not a list, the same strides are used for all pooling layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers, if not a list, the same dilation
        rate is used all over the network.
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values for convolution layers, if not
        a list, the same condition is used for all layers.

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

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
        use_bias=True,
    ):
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.strides_pooling = strides_pooling
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer.

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        self._n_filters_ = [6, 12] if self.n_filters is None else self.n_filters

        if isinstance(self.kernel_size, list):
            if len(self.kernel_size) != self.n_layers:
                raise ValueError(
                    f"Number of kernels {len(self.kernel_size)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._kernel_size = self.kernel_size
        else:
            self._kernel_size = [self.kernel_size] * self.n_layers

        if isinstance(self._n_filters_, list):
            if len(self._n_filters_) != self.n_layers:
                raise ValueError(
                    f"Number of filters {len(self._n_filters_)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_layers

        if isinstance(self.avg_pool_size, list):
            if len(self.avg_pool_size) != self.n_layers:
                raise ValueError(
                    f"Number of average pools {len(self.avg_pool_size)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._avg_pool_size = self.avg_pool_size
        else:
            self._avg_pool_size = [self.avg_pool_size] * self.n_layers

        if self.strides_pooling is None:
            self._strides_pooling = self._avg_pool_size
        elif isinstance(self.strides_pooling, list):
            if len(self.strides_pooling) != self.n_layers:
                raise ValueError(
                    f"Number of strides for pooling {len(self.strides_pooling)}"
                    f" should be the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._strides_pooling = self.strides_pooling
        else:
            self._strides_pooling = [self.strides_pooling] * self.n_layers

        if isinstance(self.activation, list):
            if len(self.activation) != self.n_layers:
                raise ValueError(
                    f"Number of activations {len(self.activation)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_layers

        if isinstance(self.padding, list):
            if len(self.padding) != self.n_layers:
                raise ValueError(
                    f"Number of paddings {len(self.padding)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_layers

        if isinstance(self.strides, list):
            if len(self.strides) != self.n_layers:
                raise ValueError(
                    f"Number of strides {len(self.strides)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_layers

        if isinstance(self.dilation_rate, list):
            if len(self.dilation_rate) != self.n_layers:
                raise ValueError(
                    f"Number of dilation rates {len(self.dilation_rate)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_layers

        if isinstance(self.use_bias, list):
            if len(self.use_bias) != self.n_layers:
                raise ValueError(
                    f"Number of biases {len(self.use_bias)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)

        if input_shape[0] < 60:
            self._padding = ["same"] * self.n_layers

        x = input_layer

        for i in range(self.n_layers):
            conv = tf.keras.layers.Conv1D(
                filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                strides=self._strides[i],
                padding=self._padding[i],
                dilation_rate=self._dilation_rate[i],
                activation=self._activation[i],
                use_bias=self._use_bias[i],
            )(x)

            conv = tf.keras.layers.AveragePooling1D(
                pool_size=self._avg_pool_size[i],
                strides=self._strides_pooling[i],
            )(conv)

            x = conv

        flatten_layer = tf.keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
