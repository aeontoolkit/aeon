"""LSTM-AD Anomaly Detector."""

__all__ = ["LSTM_AD"]

import gc
import os
import time
from copy import deepcopy

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from aeon.anomaly_detection.deep_learning.base import BaseDeepAnomalyDetector
from aeon.networks import LSTMNetwork


class LSTM_AD(BaseDeepAnomalyDetector):
    """LSTM-AD anomaly detector.

    The LSTM-AD uses stacked LSTM network for anomaly detection in time series. A
    network is trained over non-anomalous data and used as a predictor over a
    number of time steps. The resulting prediction errors are modeled as a
    multivariate Gaussian distribution, which is used to assess the likelihood of
    anomalous behavior.

    ``LSTMAD`` supports univariate and multivariate time series. It can also be
    fitted on a clean reference time series and used to detect anomalies in a different
    target time series with the same number of dimensions.

    .. list-table:: Capabilities
       :stub-columns: 1

       * - Input data format
         - univariate and multivariate
       * - Output data format
         - binary classification
       * - Learning Type
         - supervised

    Parameters
    ----------
    n_layers : int, default=2
        The number of LSTM layers to be stacked.

    n_nodes : int, default=64
        The number of LSTM units in each layer.

    window_size : int, default=20
        The size of the sliding window used to split the time series into windows. The
        bigger the window size, the bigger the anomaly context is. If it is too big,
        however, the detector marks points anomalous that are not. If it is too small,
        the detector might not detect larger anomalies or contextual anomalies at all.
        If ``window_size`` is smaller than the anomaly, the detector might detect only
        the transitions between normal data and the anomalous subsequence.

    prediction_horizon : int, default=1
        The prediction horizon is the number of time steps in the future predicted by
        the LSTM. default value is ``1``, which means the the LSTM will take
        ``window_size`` time steps as input and predict ``1`` time step in the future.

    batch_size : int, default=32
        The number of time steps per gradient update.

    optimizer : keras.optimizer, default=keras.optimizers.Adadelta()

    n_epochs: int, default = 1500
        The number of epochs to train the model.

    patience: int, default = 5
        The number of epochs to watch before early stopping.

    verbose : boolean, default = False
        whether to output extra information

    file_path : str, default = "./"
        file_path when saving model_Checkpoint callback

    save_best_model : bool, default = False
        Whether or not to save the best model, if the
        modelcheckpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name

    save_last_model : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file

    save_init_model : bool, default = False
        Whether to save the initialization of the  model.

    best_file_name : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded

    last_file_name : str, default = "last_model"
        The name of the file of the last model, if
        save_last_model is set to False, this parameter
        is discarded

    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.

    Notes
    -----
    This implementation is inspired by [1]_.

    References
    ----------
    .. [1] Malhotra Pankaj, Lovekesh Vig, Gautam Shroff, and Puneet Agarwal.
    Long Short Term Memory Networks for Anomaly Detection in Time Series. In Proceedings
    of the European Symposium on Artificial Neural Networks, Computational Intelligence
    and Machine Learning (ESANN), Vol. 23, 2015.
    https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.datasets import load_anomaly_detection
    >>> from aeon.anomaly_detection.deep_learning import LSTM_AD
    >>> X, y = load_anomaly_detection(
    ...     name=("KDD-TSAD", "001_UCR_Anomaly_DISTORTED1sddb40")
    ... )
    >>> detector = LSTM_AD(
    ...     n_layers=4, n_nodes=64, window_size=10, prediction_horizon=2
    ... )  # doctest: +SKIP
    >>> detector.fit(X, axis=0)  # doctest: +SKIP
    LSTM_AD(...)
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "requires_y": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        n_layers: int = 2,
        n_nodes: int = 64,
        window_size: int = 20,
        prediction_horizon: int = 1,
        batch_size: int = 32,
        n_epochs: int = 1500,
        patience: int = 5,
        verbose: bool = False,
        loss="mse",
        optimizer=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
    ):
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.verbose = verbose
        self.loss = loss
        self.optimizer = optimizer
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.last_file_name = last_file_name
        self.init_file_name = init_file_name

        self.history = None

        super().__init__()

        self._network = LSTMNetwork(
            self.n_nodes, self.n_layers, self.prediction_horizon
        )

    def build_model(self, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        input_layer, output_layer = self._network.build_network(
            (self.window_size, self.n_channels), **kwargs
        )

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer=self.optimizer_, loss=self.loss)

        return model

    def _fit(self, X: np.array, y: np.array):
        """Fit the model on the data.

        Parameters
        ----------
        X: np.ndarray of shape (n_timepoints, n_channels)
            The training time series, maybe with anomalies.
        y: np.ndarray of shape (n_timepoints,) or (n_timepoints, 1)
            Anomaly annotations for the training time series with values 0 or 1.
        """
        import tensorflow as tf

        self._check_params(X)

        # Create normal time series if not present
        if len(np.unique(y)) == 2:
            X_normal = X[y == 0]
            y_normal = y[y == 0]
            X_anomaly = X[y == 1]
        else:
            raise ValueError(
                "The training time series must have anomaly annotations with values"
                "0 for normal and 1 for anomaly."
            )

        # Divide the normal time series into train set and two validation sets for lstm
        X_train, X_val, y_train, y_val = train_test_split(
            X_normal, y_normal, test_size=0.2, shuffle=False
        )
        X_val1, X_val2, y_val1, y_val2 = train_test_split(
            X_val, y_val, test_size=0.5, shuffle=False
        )

        X_train_n, y_train_n = _create_sequences(
            X_train, self.window_size, self.prediction_horizon
        )
        y_train_n = y_train_n.reshape(-1, self.prediction_horizon * self.n_channels)
        X_val_1, y_val_1 = _create_sequences(
            X_val1, self.window_size, self.prediction_horizon
        )
        y_val_1 = y_val_1.reshape(-1, self.prediction_horizon * self.n_channels)
        X_val_2, y_val_2 = _create_sequences(
            X_val2, self.window_size, self.prediction_horizon
        )
        y_val_2 = y_val_2.reshape(-1, self.prediction_horizon * self.n_channels)

        X_anomalies, y_anomalies = _create_sequences(
            X_anomaly, self.window_size, self.prediction_horizon
        )
        y_anomalies = y_anomalies.reshape(-1, self.prediction_horizon * self.n_channels)

        # Fit LSTM model on the normal train set
        # input_shape = (self.window_size, self.n_channels)

        self.training_model_ = self.build_model()

        if self.save_init_model:
            self.training_model_.save(self.file_path + self.init_file_name + ".keras")

        if self.verbose:
            self.training_model_.summary()

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        self.callbacks_ = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            )
        ]

        self.history = self.training_model_.fit(
            X_train_n,
            y_train_n,
            validation_data=(X_val_1, y_val_1),
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        # Prediction errors on validation set 1 to calculate error vector
        predicted_vN1 = self.training_model_.predict(X_val_1)
        errors_vN1 = y_val_1 - predicted_vN1

        # Fit the error vectors to a Gaussian distribution
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(errors_vN1)

        # Mean and covariance matrix of the error distribution
        mu = cov_estimator.location_
        cov_matrix = cov_estimator.covariance_

        # Create a Gaussian Normal Distribution
        self.distribution = multivariate_normal(mean=mu, cov=cov_matrix)

        predicted_vN2 = self.training_model_.predict(X_val_2)
        predicted_vA = self.training_model_.predict(X_anomalies)

        errors_vN2 = y_val_2 - predicted_vN2
        errors_vA = y_anomalies - predicted_vA

        # Estimate the likelihood of the errors:
        p_vN2 = self.distribution.pdf(errors_vN2)
        p_vA = self.distribution.pdf(errors_vA)

        # Combine likelihoods and labels
        likelihoods = np.concatenate([p_vN2, p_vA])
        true_labels = np.concatenate(
            [np.zeros_like(p_vN2), np.ones_like(p_vA)]
        )  # 0 for normal, 1 for anomalous

        # Experiment with different thresholds and calculate Fβ-score
        self.best_tau = None
        self.best_fbeta = -1

        # Loop over different thresholds
        for tau in np.linspace(min(likelihoods), max(likelihoods), 100):
            # Classify as anomalous if likelihood < tau
            predictions = (likelihoods < tau).astype(int)

            # Calculate Fβ-score (arbitrarily use beta=1.0 for F1-score)
            fbeta = fbeta_score(true_labels, predictions, beta=1.0)

            # Track the best threshold and Fβ-score
            if fbeta > self.best_fbeta:
                self.best_tau = tau
                self.best_fbeta = fbeta

        try:
            if self.save_best_model:
                self.model_ = tf.keras.models.load_model(
                    self.file_path + self.file_name_ + ".keras", compile=False
                )
            else:
                os.remove(self.file_path + self.file_name_ + ".keras")
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        gc.collect()
        return self

    def _predict(self, X):
        X_, y_ = _create_sequences(X, self.window_size, self.prediction_horizon)
        y_ = y_.reshape(-1, self.prediction_horizon * self.n_channels)
        predict_test = self.training_model_.predict(X_)
        errors = y_ - predict_test
        likelihoods = self.distribution.pdf(errors)
        anomalies = (likelihoods < self.best_tau).astype(int)
        padding = np.zeros(X.shape[0] - len(anomalies))
        prediction = np.concatenate([padding, anomalies])
        return np.array(prediction, dtype=int)

    def _check_params(self, X: np.ndarray) -> None:
        if X.ndim == 1:
            self.n_channels = 1
        elif X.ndim == 2:
            self.n_channels = X.shape[1]
        else:
            raise ValueError(
                "The training time series must be of shape (n_timepoints,) or "
                "(n_timepoints, n_channels)."
            )

        if self.window_size < 1 or self.window_size > X.shape[0]:
            raise ValueError(
                "The window size must be at least 1 and at most the length of the "
                "time series."
            )
        if self.batch_size < 1 or self.batch_size > X.shape[0]:
            raise ValueError(
                "The batch size must be at least 1 and at most the length of the "
                "time series."
            )


# Create input and output sequences for lstm using sliding window
def _create_sequences(data, window_size, prediction_horizon):
    """Create input and output sequences using sliding window to train LSTM.

    Parameters
    ----------
    data: np.dnarray
        The time series of shape (n_timepoints, n_channels).
    window_size: int
        The length of the sliding window.
    prediction_horizon: int
        The number of time steps in future that would be predicted by the model.

    Returns
    -------
    X: np.ndarray
        The array of input sequences of shape
        (n_timepoints - window_size - 1, n_channels).
    y: np.ndarray
        The array of output sequences of shape
        (n_timepoints - window_size - 1, window_size).
    """
    X, y = [], []
    for i in range(len(data) - window_size - prediction_horizon + 1):
        X.append(data[i : (i + window_size)])
        y.append(data[(i + window_size) : (i + window_size + prediction_horizon)])
    return np.array(X), np.array(y)
