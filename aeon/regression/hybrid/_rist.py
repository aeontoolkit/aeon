from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import FunctionTransformer

from aeon.base.estimator.hybrid import BaseRIST
from aeon.regression import BaseRegressor
from aeon.utils.numba.general import first_order_differences_3d


class RISTRegressor(BaseRIST, BaseRegressor):
    """Randomised Interval-Shapelet Transformation (RIST) pipeline regressor.

    This regressor is a hybrid pipeline using the RandomIntervalTransformer using
    Catch22 features and summary stats, and the RandomDilatedShapeletTransformer.
    Both transforms extract features from different series transformations (1st Order
    Differences, PeriodogramTransformer, and ARCoefficientTransformer).
    An ExtraTreesRegressor with 200 trees is used as the estimator for the
    concatenated feature vector output.

    Parameters
    ----------
    n_intervals : int, callable or None, default=None,
        The number of intervals of random length, position and dimension to be
        extracted for the interval portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of intervals per `series_transformer` output.
        If None, extracts `int(np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5)`
        intervals where `Xt` is the series representation data.
    n_shapelets : int, callable or None, default=None,
        The number of shapelets of random dilation and position to be extracted for the
        shapelet portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of shapelets per `series_transformer` output.
        If None, extracts `int(np.sqrt(Xt.shape[2]) * 200 + 5)` shapelets where `Xt` is
        the series representation data.
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals and
        shapelets. If None, use the series as is. If "default", use [None, 1st Order
        Differences, PeriodogramTransformer, and ARCoefficientTransformer].

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    use_pycatch22 : bool, optional, default=True
        Wraps the C based pycatch22 implementation for aeon.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
    use_pyfftw : bool, default=True
        Whether to use the pyfftw library for FFT calculations. Requires the pyfftw
        package to be installed.
    estimator : sklearn regressor, default=None
        An sklearn estimator to be built using the transformed data. Defaults to an
        ExtraTreesRegressor with 200 trees.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.

    Examples
    --------
    >>> from aeon.regression.hybrid import RISTRegressor
    >>> from aeon.testing.utils.data_gen import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> reg = RISTRegressor(random_state=0)
    >>> reg.fit(X, y)
    RISTRegressor(...)
    >>> reg.predict(X)
    array([0.31798318, 1.41426301, 1.06414747, 0.6924721 , 0.56660146,
           1.26538944, 0.52324808, 1.0939405 ])
    """

    def __init__(
        self,
        n_intervals=None,
        n_shapelets=None,
        series_transformers="default",
        use_pycatch22=False,
        use_pyfftw=False,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        d = []
        self.use_pycatch22 = use_pycatch22
        if use_pycatch22:
            d.append("pycatch22")
        self.use_pyfftw = use_pyfftw
        if use_pyfftw:
            d.append("pyfftw")

        if d:
            self.set_tags(**{"python_dependencies": d})

        super().__init__(
            n_intervals=n_intervals,
            n_shapelets=n_shapelets,
            series_transformers=series_transformers,
            use_pycatch22=use_pycatch22,
            use_pyfftw=use_pyfftw,
            estimator=estimator,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    @classmethod
    def get_test_params(cls, parameter_set=None):
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return {
            "series_transformers": [
                None,
                FunctionTransformer(func=first_order_differences_3d, validate=False),
            ],
            "n_intervals": 1,
            "n_shapelets": 2,
            "estimator": ExtraTreesRegressor(n_estimators=2),
        }
