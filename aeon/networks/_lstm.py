"""Long Short Term Memory Network (LSTMNetwork)."""

from aeon.networks.base import BaseDeepLearningNetwork


class LSTMNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for an LSTM.

    Inspired by _[1].

    References
    ----------
    .. [1] Malhotra Pankaj, Lovekesh Vig, Gautam Shroff, and Puneet Agarwal.
    Long Short Term Memory Networks for Anomaly Detection in Time Series. In Proceedings
    of the European Symposium on Artificial Neural Networks, Computational Intelligence
    and Machine Learning (ESANN), Vol. 23, 2015.
    https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf
    """

    def __init__(
        self,
        n_nodes=64,
        n_layers=2,
        prediction_horizon=1,
    ):
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.prediction_horizon = prediction_horizon
        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct an LSTM network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (window_size (w), n_channels (d))
            The shape of the data fed into the input layer
        n_nodes : int, optional (default=64)
            The number of LSTM units in each layer
        n_layers : int, optional (default=2)
            The number of LSTM layers

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        # Input layer for the LSTM model
        input_layer = tf.keras.layers.Input(shape=input_shape)

        # Build the LSTM layers
        x = input_layer
        for _ in range(self.n_layers - 1):
            x = tf.keras.layers.LSTM(self.n_nodes, return_sequences=True)(x)

        # Last LSTM layer with return_sequences=False to output final representation
        x = tf.keras.layers.LSTM(self.n_nodes, return_sequences=False)(x)

        # Output Dense layer
        output_layer = tf.keras.layers.Dense(input_shape[1] * self.prediction_horizon)(
            x
        )

        return input_layer, output_layer
