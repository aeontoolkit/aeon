"""Implements an Auto-Encoder based on Attention Bidirectional GRUs."""

import tensorflow as tf

from aeon.networks.base import BaseDeepNetwork


class _AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, n_units, att_size, **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.att_size = att_size

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(stddev=0.1)
        self.W_omega = self.add_weight(
            shape=(self.n_units, self.att_size), initializer=initializer, name="W_omega"
        )
        self.b_omega = self.add_weight(
            shape=(self.att_size,), initializer=initializer, name="b_omega"
        )
        self.u_omega = self.add_weight(
            shape=(self.att_size,), initializer=initializer, name="u_omega"
        )
        super().build(input_shape)

    def call(self, inputs):
        v = tf.tanh(tf.tensordot(inputs, self.W_omega, axes=1) + self.b_omega)
        vu = tf.tensordot(v, self.u_omega, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return output


class AttentionBiGRUNetwork(BaseDeepNetwork):
    """
    A class to implement an Auto-Encoder based on Attention Bidirectional GRUs.

    Parameters
    ----------
        latent_space_dim : int, default=128
            Dimension of the latent space.
        n_layers_encoder : int, default=None
            Number of Attention BiGRU layers in the encoder.
            If None, one layer will be used.
        n_layers_decoder : int, default=None
            Number of Attention BiGRU layers in the decoder.
            If None, one layer will be used.
        activation_encoder : Union[list, str], default="relu"
            Activation function(s) to use in each layer of the encoder.
            Can be a single string or a list.
        activation_decoder : Union[list, str], default="relu"
            Activation function(s) to use in each layer of the decoder.
            Can be a single string or a list.
    """

    def __init__(
        self,
        latent_space_dim=128,
        n_layers_encoder=1,
        n_layers_decoder=1,
        activation_encoder="relu",
        activation_decoder="relu",
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        encoder : a keras Model.
        decoder : a keras Model.
        """
        if isinstance(self.activation_encoder, str):
            self._activation_encoder = [
                self.activation_encoder for _ in range(self.n_layers_encoder)
            ]
        else:
            self._activation_encoder = self.activation_encoder
            assert isinstance(self.activation_encoder, list)
            assert len(self.activation_encoder) == self.n_layers_encoder

        if isinstance(self.activation_decoder, str):
            self._activation_decoder = [
                self.activation_decoder for _ in range(self.n_layers_decoder)
            ]
        else:
            self._activation_decoder = self.activation_decoder
            assert isinstance(self.activation_decoder, list)
            assert len(self.activation_decoder) == self.n_layers_decoder

        if not isinstance(self.n_layers_encoder, int):
            raise ValueError("Number of layers of the encoder must be an integer.")

        if not isinstance(self.n_layers_decoder, int):
            raise ValueError("Number of layers of the decoder must be an integer.")

        input_layer = tf.keras.layers.Input(input_shape)
        x = input_layer

        if self.latent_space_dim is None:
            self.n_filters_RNN = 64
            if x.shape[0] > 250:
                self.n_filters_RNN = 512
        else:
            self.n_filters_RNN = self.latent_space_dim

        self._gate = tf.keras.layers.Dense(self.n_filters_RNN, activation="sigmoid")

        for i in range(self.n_layers_encoder):
            forward_layer = tf.keras.layers.GRU(
                self.n_filters_RNN,
                activation=self._activation_encoder[i],
                return_sequences=True,
            )(x)
            backward_layer = tf.keras.layers.GRU(
                self.n_filters_RNN,
                activation=self._activation_encoder[i],
                return_sequences=True,
                go_backwards=True,
            )(x)
            h_att_fw = _AttentionLayer(self.n_filters_RNN, self.n_filters_RNN)(
                forward_layer
            )
            h_att_bw = _AttentionLayer(self.n_filters_RNN, self.n_filters_RNN)(
                backward_layer
            )
            x = self._gate(h_att_fw) * h_att_fw + self._gate(h_att_bw) * h_att_bw

        encoder = tf.keras.models.Model(inputs=input_layer, outputs=x, name="encoder")

        decoder_inputs = tf.keras.layers.Input(
            shape=(self.latent_space_dim,), name="decoder_input"
        )
        x = tf.keras.layers.RepeatVector(input_shape[0], name="repeat_vector")(
            decoder_inputs
        )

        for i in range(self.n_layers_decoder - 1, -1, -1):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=self.n_filters_RNN // 2,
                    activation=self._activation_decoder[i],
                    return_sequences=True,
                ),
                name=f"decoder_bgru_{i+1}",
            )(x)

        decoder_outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(input_shape[1]), name="decoder_output"
        )(x)
        decoder = tf.keras.models.Model(
            inputs=decoder_inputs, outputs=decoder_outputs, name="decoder"
        )

        return encoder, decoder
