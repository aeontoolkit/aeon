"""Inception Network (InceptionNetwork)."""

__maintainer__ = ["hadifawaz1999"]

from aeon.networks.base import BaseDeepLearningNetwork


class InceptionNetwork(BaseDeepLearningNetwork):
    """Inception Network.

    Parameters
    ----------
    depth : int, default = 6,
            The number of inception modules used
    n_filters : int or list of int32, default = 32
        The number of filters used in one inception module, if not a list, the same
        number of filters is used in all inception modules.
    n_conv_per_layer : int or list of int, default = 3
        The number of convolution layers in each inception module, if not a list,
        the same number of convolution layers is used in all inception modules.
    kernel_size : int or list of int, default = 40
        The head kernel size used for each inception module, if not a list, the same
        is used in all inception modules.
    use_max_pooling : bool or list of bool, default = True
        Conditioning whether or not to use max pooling layer in inception modules,
        if not a list,the same is used in all inception modules.
    max_pool_size : int or list of int, default = 3
        The size of the max pooling layer, if not a list, the same is used in all
        inception modules.
    strides : int or list of int, default = 1
        The strides of kernels in convolution layers for each inception module,
        if not a list, the same is used in all inception modules.
    dilation_rate : int or list of int, default = 1
        The dilation rate of convolutions in each inception module, if not a list,
        the same is used in all inception modules.
    padding : str or list of str, default = 'same'
        The type of padding used for convoltuon for each inception module, if not a
        list, the same is used in all inception modules.
    activation : str or list of str, default = 'relu'
        The activation function used in each inception module, if not a list,
        the same is used in all inception modules.
    use_bias : bool or list of bool, default = False
        Conditioning whether or not convolutions should use bias values in each
        inception module, if not a list, the same is used in all inception modules.
    use_residual : bool, default = True
        Condition whether or not to use residual connections all over Inception.
    use_bottleneck : bool, default = True
        Condition whether or not to use bottlenecks all over Inception.
    bottleneck_size : int, default = 32
        The bottleneck size in case ``use_bottleneck = True``.
    use_custom_filters : bool, default = False
        Condition on whether or not to use custom filters in the first inception module.

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/
    inception.py

    and

    https://github.com/MSD-IRIMAS/CF-4-TSC/blob/main/classifiers/H_Inception.py
    for the custom filters

    Network originally defined in:

    @article{IsmailFawaz2019inceptionTime, Title                    = {
    InceptionTime: Finding AlexNet for Time Series Classification}, Author
                    = {Ismail Fawaz, Hassan and Lucas, Benjamin and
                    Forestier, Germain and Pelletier, Charlotte and Schmidt,
                    Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and
                    Idoumghar, Lhassane and Muller, Pierre-Alain and
                    Petitjean, François}, journal                  = {
                    ArXiv}, Year                     = {2019} }

    Custom filters defined in:

    @inproceedings{ismail-fawaz2022hccf,
    author = {Ismail-Fawaz, Ali and Devanne, Maxime and Weber,
    Jonathan and Forestier, Germain},
    title = {Deep Learning For Time Series Classification
    Using New Hand-Crafted Convolution Filters},
    booktitle = {2022 IEEE International Conference on
    Big Data (IEEE BigData 2022)},
    city = {Osaka},
    country = {Japan},
    pages = {972-981},
    url = {doi.org/10.1109/BigData55660.2022.10020496},
    year = {2022},
    organization = {IEEE}
    }
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
    ):
        self.n_filters = n_filters
        self.n_conv_per_layer = n_conv_per_layer
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.use_max_pooling = use_max_pooling
        self.max_pool_size = max_pool_size
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.bottleneck_size = bottleneck_size
        self.use_custom_filters = use_custom_filters

        super().__init__()

    def hybrid_layer(self, input_tensor, input_channels, kernel_sizes=None):
        """Construct the hybrid layer to compute features of cutom filters.

        Parameters
        ----------
        input_tensor : tensorflow tensor, usually the input layer of the model.
        input_channels : int, the number of input channels in case of multivariate.
        kernel_sizes : list of int, default = [2,4,8,16,32,64],
        the size of the hand-crafted filters.

        Returns
        -------
        hybrid_layer : tensorflow tensor containing the concatenation
        of the output features extracted form hand-crafted convolution filters.

        """
        import numpy as np
        import tensorflow as tf

        kernel_sizes = [2, 4, 8, 16, 32, 64] if kernel_sizes is None else kernel_sizes

        self.keep_track = 0

        """
        Function to create the hybrid layer consisting of non
        trainable Conv1D layers with custom filters.

        Args:

            input_tensor: input tensor
            input_channels : number of input channels, 1 in case of UCR Archive
        """

        conv_list = []

        # for increasing detection filters

        for kernel_size in kernel_sizes:
            filter_ = np.ones(
                shape=(kernel_size, input_channels, 1)
            )  # define the filter weights with the shape corresponding
            # the Conv1D layer in keras (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 == 0] *= -1  # formula of increasing detection filter

            # Create a Conv1D layer with non trainable option and no
            # biases and set the filter weights that were calculated in the
            # line above as the initialization

            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
                name="hybrid-increasse-"
                + str(self.keep_track)
                + "-"
                + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        # for decreasing detection filters

        for kernel_size in kernel_sizes:
            filter_ = np.ones(
                shape=(kernel_size, input_channels, 1)
            )  # define the filter weights with the shape
            # corresponding the Conv1D layer in keras
            # (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 > 0] *= -1  # formula of decreasing detection filter

            # Create a Conv1D layer with non trainable option
            # and no biases and set the filter weights that were
            # calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
                name="hybrid-decrease-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        # for peak detection filters

        for kernel_size in kernel_sizes[1:]:
            filter_ = np.zeros(
                shape=(kernel_size + kernel_size // 2, input_channels, 1)
            )

            xmesh = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape(
                (-1, 1, 1)
            )

            # see utils.custom_filters.py to understand the formulas below

            filter_left = xmesh**2
            filter_right = filter_left[::-1]

            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right

            # Create a Conv1D layer with non trainable option and
            # no biases and set the filter weights that were
            # calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size + kernel_size // 2,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
                name="hybrid-peeks-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        hybrid_layer = tf.keras.layers.Concatenate(axis=2)(
            conv_list
        )  # concantenate all convolution layers
        hybrid_layer = tf.keras.layers.Activation(activation="relu")(
            hybrid_layer
        )  # apply activation ReLU

        return hybrid_layer

    def _inception_module(
        self,
        input_tensor,
        n_filters=32,
        dilation_rate=1,
        padding="same",
        strides=1,
        activation="relu",
        use_bias=False,
        kernel_size=40,
        n_conv_per_layer=3,
        use_max_pooling=True,
        max_pool_size=3,
        use_custom_filters=False,
    ):
        import tensorflow as tf

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding=padding,
                activation="linear",
                use_bias=use_bias,
            )(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [kernel_size // (2**i) for i in range(n_conv_per_layer)]
        self.kernel_size_s = kernel_size_s

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=strides,
                    dilation_rate=dilation_rate,
                    padding=padding,
                    activation="linear",
                    use_bias=use_bias,
                )(input_inception)
            )

        if use_max_pooling:
            max_pool_1 = tf.keras.layers.MaxPool1D(
                pool_size=max_pool_size, strides=strides, padding=padding
            )(input_tensor)

            conv_max_pool = tf.keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=1,
                padding=padding,
                activation="linear",
                use_bias=use_bias,
            )(max_pool_1)

            conv_list.append(conv_max_pool)

        if use_custom_filters:
            hybrid_layer = self.hybrid_layer(
                input_tensor=input_tensor, input_channels=int(input_tensor.shape[-1])
            )
            conv_list.append(hybrid_layer)

        if len(conv_list) > 1:
            x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        else:
            x = conv_list[0]
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def _shortcut_layer(self, input_tensor, out_tensor, padding="same", use_bias=False):
        import tensorflow as tf

        shortcut_y = tf.keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding=padding,
            use_bias=use_bias,
        )(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        # not sure of the whole padding thing

        import tensorflow as tf

        if isinstance(self.n_filters, list):
            assert len(self.n_filters) == self.depth
            self._nb_filters = self.n_filters
        else:
            self._nb_filters = [self.n_filters] * self.depth

        if isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == self.depth
            self._kernel_size = self.kernel_size
        else:
            self._kernel_size = [self.kernel_size] * self.depth

        if isinstance(self.n_conv_per_layer, list):
            assert len(self.n_conv_per_layer) == self.depth
            self._nb_conv_per_layer = self.n_conv_per_layer
        else:
            self._nb_conv_per_layer = [self.n_conv_per_layer] * self.depth

        if isinstance(self.strides, list):
            assert len(self.strides) == self.depth
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.depth

        if isinstance(self.dilation_rate, list):
            assert len(self.dilation_rate) == self.depth
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.depth

        if isinstance(self.padding, list):
            assert len(self.padding) == self.depth
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.depth

        if isinstance(self.activation, list):
            assert len(self.activation) == self.depth
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.depth

        if isinstance(self.use_max_pooling, list):
            assert len(self.use_max_pooling) == self.depth
            self._use_max_pooling = self.use_max_pooling
        else:
            self._use_max_pooling = [self.use_max_pooling] * self.depth

        if isinstance(self.max_pool_size, list):
            assert len(self.max_pool_size) == self.depth
            self._max_pool_size = self.max_pool_size
        else:
            self._max_pool_size = [self.max_pool_size] * self.depth

        if isinstance(self.use_bias, list):
            assert len(self.use_bias) == self.depth
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.depth

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        _use_custom_filters = False

        for d in range(self.depth):
            if d == 0 and self.use_custom_filters:
                _use_custom_filters = True
            else:
                _use_custom_filters = False

            x = self._inception_module(
                x,
                n_filters=self._nb_filters[d],
                dilation_rate=self._dilation_rate[d],
                kernel_size=self._kernel_size[d],
                padding=self._padding[d],
                strides=self._strides[d],
                activation=self._activation[d],
                use_bias=self._use_bias[d],
                use_max_pooling=self._use_max_pooling[d],
                max_pool_size=self._max_pool_size[d],
                n_conv_per_layer=self._nb_conv_per_layer[d],
                use_custom_filters=_use_custom_filters,
            )

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, padding=self._padding[d])
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap_layer
