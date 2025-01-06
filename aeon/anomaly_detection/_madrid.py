"""MADRID anomaly detector. based on (https://sites.google.com/view/madrid-icdm-23/home)"""

__mentainer__ = ["acquayefrank"]
__all__ = ["MADRID"]

import numpy as np
import math
import time

from aeon.anomaly_detection.base import BaseAnomalyDetector


class MADRID(BaseAnomalyDetector):
    factor = 1
    _tags = {
        "fit_is_empty": False,
    }

    def __init__(
        self,
        min_length,
        max_length,
        step_size=1,
        split_psn=None,
        look_ahead=None,
        enable_output=False,
    ):
        self.min_length = min_length
        self.max_length = max_length
        if split_psn:
            self.split_psn = split_psn
        else:
            self.split_psn = (
                self.max_length + 1
            )  # should be greater than self.max_length
        self.step_size = step_size
        self.look_ahead = look_ahead
        self.enable_output = enable_output
        super().__init__(axis=1)

    def _fit(self, X: np.ndarray) -> "MADRID":
        self._check_params(X)
        self._inner_fit(X)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        X = X.squeeze()
        anomalies = self._inner_predict(X)
        return anomalies

    def _fit_predict(self, X: np.ndarray) -> np.ndarray:
        self._check_params(X)
        self._inner_fit(X)
        anomalies = self._inner_predict(X)
        return anomalies

    def _check_params(self, X: np.ndarray) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be greater than 0")

        if X.shape[0] < self.min_length:
            raise ValueError(
                f"Series length of X {X.shape[0]} is less than min_length "
                f"{self.min_length}"
            )

        if self._contains_constant_regions(X, self.min_length):
            error_message = (
                "BREAK: There is at least one region of length min_length that is constant, or near constant.\n\n"
                "To fix this issue:\n"
                "1) Choose a longer length for min_length.\n"
                "2) Add a small amount of noise to the entire time series (this will probably result in the current constant sections becoming top discords).\n"
                "3) Add a small linear trend to the entire time series (this will probably result in the current constant sections becoming motifs, and not discords).\n"
                "4) Carefully edit the data to remove the constant sections."
            )
            raise ValueError(error_message)

        if not isinstance(self.enable_output, bool):
            raise ValueError(f"{self.enable_output} should be a boolean")

    def _inner_fit(self, X: np.ndarray) -> None:
        len_x = len(X)
        factor = 1

        if (
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            )
            < 5000000
        ):
            # polynomial model (of order 6)
            p_1 = [
                -4.66922312132205e-45,
                1.54665628995475e-35,
                -1.29314859463985e-26,
                2.01592418847342e-18,
                -2.54253065977245e-11,
                9.77027495487874e-05,
                -1.27055582771851e-05,
            ]
            p_2 = [
                -3.79100071825804e-42,
                3.15547030055575e-33,
                -6.62877819318290e-25,
                2.59437174380763e-17,
                -8.10970871564789e-11,
                7.25344313152170e-05,
                4.68415490390476e-07,
            ]
        else:
            # linear model
            p_1 = [3.90752957831437e-05, 0]
            p_2 = [1.94005690535588e-05, 0]

        p_4 = [1.26834880558841e-05, 0]
        p_8 = [1.42210521045333e-05, 0]
        p_16 = [1.82290885539705e-05, 0]

        # Prediction
        factor = 16
        predicted_execution_time_16 = np.polyval(
            p_16,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 8
        predicted_execution_time_8 = np.polyval(
            p_8,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 4
        predicted_execution_time_4 = np.polyval(
            p_4,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 2
        predicted_execution_time_2 = np.polyval(
            p_2,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 1
        predicted_execution_time_1 = np.polyval(
            p_1,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        if predicted_execution_time_1 < 10:
            self.factor = 1
        else:
            test_data = self._test_data_madrid()
            factor = 16
            actual_measurement_16 = self._inner_predict(test_data)
            scaling_factor = actual_measurement_16 / 0.65461
            predicted_execution_time_16_scaled = (
                predicted_execution_time_16 * scaling_factor
            )
            predicted_execution_time_8_scaled = (
                predicted_execution_time_8 * scaling_factor
            )
            predicted_execution_time_4_scaled = (
                predicted_execution_time_4 * scaling_factor
            )
            predicted_execution_time_2_scaled = (
                predicted_execution_time_2 * scaling_factor
            )
            predicted_execution_time_1_scaled = (
                predicted_execution_time_1 * scaling_factor
            )

            execution_times = [
                (predicted_execution_time_16_scaled, 16),
                (predicted_execution_time_8_scaled, 8),
                (predicted_execution_time_4_scaled, 4),
                (predicted_execution_time_2_scaled, 2),
                (predicted_execution_time_1_scaled, 1),
            ]

            # Sort execution times based on predicted execution time
            execution_times.sort(key=lambda x: x[0])

            for _, factor in execution_times:
                if (self.min_length + factor - 1) // factor >= 2 and (
                    self.max_length + factor - 1
                ) // factor >= 2:
                    self.factor = factor
                    break
            else:
                raise ValueError(
                    f"No valid factor found that meets the criteria. Because:"
                    f"{self.min_length}/{self.factor} < 2 and "
                    f"{self.max_length}/{self.factor} < 2"
                )

    def _inner_predict(self, X):
        bfs_seed = float("-inf")  # used for first time run of dump_topk
        k = 1
        time_bf = 0

        # Initialize arrays
        num_rows = int(
            np.ceil((self.max_length + 1 - self.min_length) / self.step_size)
        )
        num_cols = len(X)
        multilength_discord_table = np.full((num_rows, num_cols), -np.inf)
        bsf = np.zeros((num_rows, 1))
        bsf_loc = np.full((num_rows, 1), np.nan)

        # Data for creating convergence plots
        time_sum_bsf = [0, 0]
        percent_sum_bsf = [0, 0]

        # Start timer
        start_time = time.time()

        # Set m values
        m_set = np.arange(self.min_length, self.max_length + 1, self.step_size)
        m_pointer = int(np.ceil(len(m_set) / 2))
        m = m_set[m_pointer]

        # Call DAMP_2_0 function (you need to define it in Python)
        discord_score, position, left_mp = self._dump_2_0(X, m)

        # anomalies = np.zeros(X.shape[0], dtype=bool)
        # return anomalies

    def _contains_constant_regions(self, X, sub_sequence_length):
        bool_vec = False  # in the origianl matlab code they use 0,1 but trus, false is a better representation
        X = np.asarray(X)

        constant_indices = np.where(np.diff(X) != 0)[0] + 1
        constant_indices = np.concatenate(([0], constant_indices, [len(X)]))
        constant_lengths = np.diff(constant_indices)

        constant_length = max(constant_lengths)

        if constant_length >= sub_sequence_length or np.var(X) < 0.2:
            bool_vec = True

        return bool_vec

    def _dump_2_0(self, X, subsequence_length, *args):
        if self.look_ahead is None:
            self.look_ahead = 2 ** np.ceil(np.log2(16 * subsequence_length))
        left_mp = -np.inf * np.ones_like(X)
        left_mp[: self.split_psn] = np.nan

        best_so_far = -np.inf
        bool_vec = np.ones(len(X), dtype=bool)

        for i in range(self.split_psn, self.split_psn + (16 * subsequence_length) + 1):
            if not bool_vec[i]:
                left_mp[i] = left_mp[i - 1] - 0.00001
                continue
            if i + subsequence_length - 1 > len(X):
                break
            query = X[i : i + subsequence_length]
            left_mp[i] = np.min(np.real(self._mass_v2(X[:i], query)))

            best_so_far = np.max(left_mp)

            # If lookahead is 0, then it is a pure online algorithm with no pruning
            if self.look_ahead != 0:
                # Perform forward MASS for pruning
                # The index at the beginning of the forward mass should be avoided in the exclusion zone
                start_of_mass = min(i + subsequence_length, len(X))
                end_of_mass = min(start_of_mass + self.look_ahead - 1, len(X))

                # The length of lookahead should be longer than that of the query
                if (end_of_mass - start_of_mass + 1) > subsequence_length:
                    distance_profile = np.real(
                        self._mass_v2(X[start_of_mass:end_of_mass], query)
                    )

                    # Find the subsequence indices less than the best so far discord score
                    dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]

                    # Converting indexes on distance profile to indexes on time series
                    ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1

                    # Update the Boolean vector
                    bool_vec[ts_index_less_than_BSF] = 0

        # Remaining test data except for the prefix
        for i in range(
            self.split_psn + (16 * subsequence_length) + 1,
            len(X) - subsequence_length + 1,
        ):
            if not bool_vec[i]:
                left_mp[i] = left_mp[i - 1] - 0.00001
                continue

            # Initialization for classic DAMP
            # Approximate leftMP value for the current subsequence
            approximate_distance = float("inf")

            # X indicates how long a time series to look backwards
            X = 2 ** (8 * subsequence_length).bit_length()

            # flag indicates if it is the first iteration of DAMP
            flag = 1

            # expansion_num indicates how many times the search has been expanded backward
            expansion_num = 0

            if i + subsequence_length - 1 > len(X):
                break

            # Extract the current subsequence from T
            query = X[i : i + subsequence_length]

            # Classic DAMP
            while approximate_distance >= best_so_far:
                # Case 1: Execute the algorithm on the time series segment farthest from the current subsequence
                # Arrived at the beginning of the time series
                if i - X + 1 + (expansion_num * subsequence_length) < 1:
                    approximate_distance = min(np.real(self._mass_v2(X[:i], query)))
                    left_mp[i] = approximate_distance
                    # Update the best discord so far
                    if approximate_distance > best_so_far:
                        # The current subsequence is the best discord so far
                        best_so_far = approximate_distance
                    break
                else:
                    if flag == 1:
                        # Case 2: Execute the algorithm on the time series segment closest to the current subsequence
                        flag = 0
                        approximate_distance = min(
                            np.real(self._mass_v2(X[i - X + 1 : i], query))
                        )
                    else:
                        # Case 3: All other cases
                        X_start = i - X + 1 + (expansion_num * subsequence_length)
                        X_end = i - (X // 2) + (expansion_num * subsequence_length)
                        approximate_distance = min(
                            np.real(self._mass_v2(X[X_start:X_end], query))
                        )

                    if approximate_distance < best_so_far:
                        # If a value less than the current best discord score exists on the distance profile, stop searching
                        left_mp[i] = approximate_distance
                        break
                    else:
                        # Otherwise expand the search
                        X *= 2
                        expansion_num += 1

            # If lookahead is 0, then it is a pure online algorithm with no pruning
            if self.look_ahead  != 0:
                # Perform forward MASS for pruning
                # The index at the beginning of the forward MASS should be avoided in the exclusion zone
                start_of_mass = min(i + subsequence_length, len(X))
                end_of_mass = min(start_of_mass + self.look_ahead - 1, len(X))
                
                # The length of lookahead should be longer than that of the query
                if (end_of_mass - start_of_mass + 1) > subsequence_length:
                    distance_profile = np.real(self._mass_v2(X[start_of_mass:end_of_mass], query))
                    
                    # Find the subsequence indices less than the best so far discord score
                    dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]
                    
                    # Converting indexes on distance profile to indexes on time series
                    ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1
                    
                    # Update the Boolean vector
                    bool_vec[ts_index_less_than_BSF] = 0# Get pruning rate

        PV = bool_vec[self.split_psn : (len(X) - subsequence_length + 1)]
        PR = (len(PV) - sum(PV)) / len(PV)

        # Get top discord
        discord_score = max(left_mp)
        position = left_mp.index(discord_score)

            

           

    def _mass_v2(self, x, y):
        # x is the data, y is the query
        m = len(y)
        n = len(x)

        # Compute y stats -- O(n)
        meany = np.mean(y)
        sigmay = np.std(y, ddof=1)  # Use ddof=1 for sample standard deviation

        # Compute x stats -- O(n)
        meanx = np.convolve(x, np.ones(m) / m, mode="valid")  # Moving average
        sigmax = np.sqrt(
            np.convolve((x - np.mean(x)) ** 2, np.ones(m) / m, mode="valid")
        )  # Moving std dev

        y_reversed = y[::-1]  # Reverse the query
        y_padded = np.pad(
            y_reversed, (0, n - m), mode="constant", constant_values=0
        )  # Append zeros

        # The main trick of getting dot products in O(n log n) time
        X = np.fft.fft(x)
        Y = np.fft.fft(y_padded)
        Z = X * Y
        z = np.fft.ifft(Z).real  # Take the real part of the inverse FFT

        dist = 2 * (
            m - (z[m - 1 : n] - m * meanx[m - 1 :] * meany) / (sigmax[m - 1 :] * sigmay)
        )
        dist = np.sqrt(dist)

        return dist

    def _test_data_madrid(self):
        np.random.seed(123456789)

        Fs = 10000
        t = np.arange(0, 10 + 1 / Fs, 1 / Fs)  # Time vector
        f_in_start = 50
        f_in_end = 60
        f_in = np.linspace(f_in_start, f_in_end, len(t))  # Linearly spaced frequency
        phase_in = np.cumsum(f_in / Fs)  # Cumulative phase

        y = np.sin(2 * np.pi * phase_in)  # Sinusoidal signal
        y += np.random.randn(len(y)) / 12  # Add noise

        EndOfTrain = len(y) // 2  # End of training segment

        # Add medium anomaly
        anomaly_start_1 = EndOfTrain + 1200
        y[anomaly_start_1 : anomaly_start_1 + 64] += np.random.randn(64) / 3

        # Add another anomaly
        anomaly_start_2 = EndOfTrain + 4180
        y[anomaly_start_2 : anomaly_start_2 + 160] += np.random.randn(160) / 4

        # Add long duration anomaly
        anomaly_start_3 = EndOfTrain + 8200
        anomaly_end_3 = EndOfTrain + 8390
        y[anomaly_start_3:anomaly_end_3] *= 0.5

        return y
