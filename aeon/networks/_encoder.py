"""Encoder Network (EncoderNetwork)."""

__maintainer__ = ["hadifawaz1999"]

from aeon.networks.base import BaseDeepLearningNetwork


class EncoderNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for an Encoder.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_size : array of int, default = [5, 11, 21]
        Specifies the length of the 1D convolution windows.
    n_filters : array of int, default = [128, 256, 512]
        Specifying the number of 1D convolution filters used for each layer,
        the shape of this array should be the same as kernel_size.
    max_pool_size : int, default = 2
        Size of the max pooling windows.
    activation : string, default = sigmoid
        Keras activation function.
    dropout_proba : float, default = 0.2
        specifying the dropout layer probability.
    padding : string, default = "same"
        Specifying the type of padding used for the 1D convolution.
    strides : int, default = 1
        Specifying the sliding rate of the 1D convolution filter.
    fc_units : int, default = 256
        Specifying the number of units in the hiddent fully connected layer used in
        the EncoderNetwork.

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    .. [1] Serrà et al. Towards a Universal Neural Network Encoder for Time Series
    In proceedings International Conference of the Catalan Association
    for Artificial Intelligence, 120--129 2018.


    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.12",
        "structure": "encoder",
    }

    def __init__(
        self,
        kernel_size=None,
        n_filters=None,
        dropout_proba=0.2,
        max_pool_size=2,
        activation="sigmoid",
        padding="same",
        strides=1,
        fc_units=256,
    ):
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.padding = padding
        self.strides = strides
        self.max_pool_size = max_pool_size
        self.activation = activation
        self.dropout_proba = dropout_proba
        self.fc_units = fc_units

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

        self._kernel_size = (
            [5, 11, 21] if self.kernel_size is None else self.kernel_size
        )
        self._n_filters = [128, 256, 512] if self.n_filters is None else self.n_filters

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

        for i in range(len(self._kernel_size)):
            conv = tf.keras.layers.Conv1D(
                filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                padding=self.padding,
                strides=self.strides,
            )(x)

            conv = tf.keras.layers.GroupNormalization(groups=-1)(conv)
            conv = tf.keras.layers.PReLU(shared_axes=[1])(conv)
            conv = tf.keras.layers.Dropout(self.dropout_proba)(conv)

            if i < len(self._kernel_size) - 1:
                conv = tf.keras.layers.MaxPool1D(pool_size=self.max_pool_size)(conv)

            x = conv

        attention = tf.keras.layers.Attention()([conv, conv, conv])

        # add fully connected hidden layer

        hidden_fc_layer = tf.keras.layers.Dense(
            units=self.fc_units, activation=self.activation
        )(attention)
        hidden_fc_layer = tf.keras.layers.GroupNormalization(groups=-1)(hidden_fc_layer)

        # output layer before classification layer

        flatten_layer = tf.keras.layers.Flatten()(hidden_fc_layer)

        return input_layer, flatten_layer
