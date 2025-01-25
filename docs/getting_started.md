# Getting Started

The following information is designed to get users up and running with `aeon` quickly.
If installation is required, please see our [installation guide](installation) for
installing `aeon`.

We assume basic familiarity with the [`scikit-learn`](https://scikit-learn.org/stable/index.html)
package. If you want help with `scikit-learn` you may want to view
[their getting started guides](https://scikit-learn.org/stable/getting_started.html).

`aeon` is an open-source toolkit for learning from time series. It provides access to
the very latest algorithms for time series machine learning, in addition to a range of
classical techniques for the following learning tasks:

- [**Classification**](api_reference/classification), where a collection of time series
  labelled with  a discrete value is used to train a model to predict unseen cases
  ([more details](examples/classification/classification.ipynb)).
- [**Regression**](api_reference/regression), where a collection of time series
  labelled with a continuous value is used to train a model to predict unseen cases
  ([more details](examples/regression/regression.ipynb)).
- [**Clustering**](api_reference/clustering), where a collection of time series without
  any labels are used to train a model to label cases
  ([more details](examples/clustering/clustering.ipynb)).
- [**Similarity search**](api_reference/similarity_search), where the goal is to find
  time series motifs or nearest neighbors in an efficient way for either single series
  or collections.
  ([more details](examples/similarity_search/similarity_search.ipynb)).
- [**Anomaly detection**](api_reference/anomaly_detection), where the goal is to find
  values or areas of a single time series that are not representative of the whole series.
- [**Forecasting**](api_reference/forecasting), where the goal is to predict future values
  of a single time series
  ([more details](examples/forecasting/forecasting.ipynb)).
- [**Segmentation**](api_reference/segmentation), where the goal is to split a single time
  series into regions where the series are sofind areas of a time series that are not
  representative of the whole series
  ([more details](examples/segmentation/segmentation.ipynb)).

`aeon` also provides core modules that are used by the modules above:

- [**Transformations**](api_reference/transformations), where a either a single series or collection is
  transformed into a different representation or domain. ([more details](examples/transformations/transformations.ipynb)).
- [**Distances**](api_reference/distances), which measure the dissimilarity between two time series or
  collections of series and include functions to align series ([more details](examples/distances/distances.ipynb)).
- [**Networks**](api_reference/networks), provides core models for deep learning for all time series tasks
- ([more details](examples/networks/deep_learning.ipynb)).

There are dedicated notebooks going into more detail for each of these modules. This
guide is meant to give you the briefest of introductions to the main concepts and
code for each task to get started. For more information on the variety of
estimators available for each task, see the links above, the [API](api_reference) and
[examples](https://www.aeon-toolkit.org/en/latest/examples.html) pages.

## A Single Time Series

A time series is a series of real valued data assumed to be ordered. A univariate
time series has a single value at each time point. For example,
the heartbeat ECG reading from a single sensor or the number of passengers using an
airline per month would form a univariate series. Single time series are stored
by default in a `np.ndarray` (which we try to use internally whenever possible).
We can also handle `pd.Series` and `pd.DataFrame` objects as inputs, but these may be
converted to `np.ndarray` internally. The airline series is a classic example of a
univariate series from the forecasting domain. The series is the monthly totals of
international airline passengers, 1949 to 1960, in thousands.

```{code-block} python
>>> from aeon.datasets import load_airline
>>> y = load_airline()  # load an example univariate series as an array
>>> y[:5]  # first five time points
606.0
508.0
461.0
390.0
432.0
```

A multivariate time series is made up of multiple series or channels, where each
observation is a vector of related recordings in the same time index. An example
would be a motion trace from a smartwatch with at least three dimensions (X,Y,Z
co-ordinates), or multiple financial statistics recorded over time. Single
multivariate series input typically follows the shape `(n_channels, n_timepoints)` by
default. Algorithms may have an `axis` parameter to change this, where `axis=1` assumes
the default shape and is the default setting, and `axis=0` assumes the shape
`(n_timepoints, n_channels)`.

```{code-block} python
>>> from aeon.datasets import load_uschange
>>> data = load_uschange()  # load an example multivariate series
>>> data[:,:5]  # all channels, first five time points
[[ 0.61598622  0.46037569  0.87679142 -0.27424514  1.89737076]
 [ 0.97226104  1.16908472  1.55327055 -0.25527238  1.98715363]
 [-2.45270031 -0.55152509 -0.35870786 -2.18545486  1.90973412]
 [ 4.8103115   7.28799234  7.28901306  0.98522964  3.65777061]
 [ 0.9         0.5         0.5         0.7        -0.1       ]]
```

We commonly refer to the number of observations for a time series as `n_timepoints`.
If a series is multivariate, we refer to the dimensions as channels
(to avoid confusion with the dimensions of array) and in code use `n_channels`. So
the US Change data loaded above has five channels and 187 time points. For more
details on our provided datasets and on how to load data into aeon compatible data
structures, see our [datasets](examples/datasets/datasets.ipynb) notebooks.

## Single Series Modules

Different `aeon` modules work with individual series or collections of series.
Estimators in the `anomaly detection`, `forecasting` and `segmentation` modules use
single series input (they inherit from `BaseSeriesEstimator`). The functions in
`distances` take two series as arguments.

### Anomaly Detection

Anomaly detection (AD) is the process of identifying observations that are significantly
different from the rest of the data. More details to follow soon, once we have
written the notebook.

```{code-block} python
>>> from aeon.datasets import load_airline
>>> from aeon.anomaly_detection import STOMP
>>> stomp = STOMP(window_size=200)
>>> scores = est.fit_predict(X) # Get the anomaly scores
```

### Segmentation

Time series segmentation (TSS) is the process of dividing a time series into
segments or regions that are dissimilar to each other. This could, for
example, be the problem of splitting the motion trace from a smartwatch into
different activities such as walking, running, and sitting. It is closely related to
the field of change point detection, which is a term used more in the statistics
literature.

```{code-block} python
>>> from aeon.datasets import load_airline
>>> from aeon.segmentation import ClaSPSegmenter
>>> series = load_airline()
>>> clasp = ClaSPSegmenter()  # An example segmenter
>>> clasp.fit(data)  # fit the segmenter on the data
>>> clasp.fit_predict(ts)
[51]
```

### Distances

Distances between time series is a primitive operation in very many time series
tasks. We have an extensive set of distance functions in the `aeon.distances` module,
all optimised using numba. They all work with multivariate and unequal length series.

```{code-block} python
>>> from aeon.datasets import load_japanese_vowels
>>> from aeon.distances import dtw_distance
>>> data = load_japanese_vowels()  # load an example multivariate series collection
>>> dtw_distance(data[0], data[1])  # calculate the dtw distance
14.416269807978
```

## Collections of Time Series

The default storage for collections of time series is a 3D `np.ndarray`.
If `n_timepoints` varies between cases, we store a collection in a `list` of
`np.ndarray` arrays, each with the same number of channels. We do not have the
capability to use collections of time series with varying numbers of channels.
We also assume series length is always the same for all channels of a single series.

```{code-block} python
>>> from aeon.datasets import load_italy_power_demand
>>> X, y = load_italy_power_demand()  # load an example univariate collection
>>> X.shape
(1096, 1, 24)
>>> X[:5, :, :5]
[[[-0.71051757 -1.1833204  -1.3724416  -1.5930829  -1.4670021 ]]
 [[-0.99300935 -1.4267865  -1.5798843  -1.6054006  -1.6309169 ]]
 [[ 1.3190669   0.56977448  0.19512825 -0.08585642 -0.17951799]]
 [[-0.81244429 -1.1575534  -1.4163852  -1.5314215  -1.5026624 ]]
 [[-0.97284033 -1.3905178  -1.5367049  -1.6202404  -1.6202404 ]]]
>>> y[:5]
['1' '1' '2' '2' '1']
```

We use the terms case and instance interchangeably when referring to a single time
series contained in a collection. The size of a time series collection is referred to as
`n_cases` in code. Collections have the shape `(n_cases, n_channels, n_timepoints)`.

We recommend storing collections in a 3D `np.ndarray` even if each time series is
univariate (i.e. `n_channels == 1`). Collection estimators will work with 2D input of
shape `(n_cases, n_timepoints)` as you would expect from `scikit-learn`, but it is
possible to confuse a collection of univariate series of shape `(n_cases, n_timepoints)`
with a single multivariate series of shape `(n_channels, n_timepoints)`. This potential
confusion is one reason we make the distinction between series and collection
estimators.

```{code-block} python
>>>from aeon.datasets import load_basic_motions, load_plaid, load_japanese_vowels
>>> X2, y2 = load_basic_motions() # example equal length multivariate collection
>>> X2.shape
(80, 6, 100)
>>> X3, y3 = load_plaid()  # example unequal length univariate collection
>>> type(X3)
<class 'list'>
>>> len(X3)
1074
>>> X3[0].shape
(1, 500)
>>> X4, y4 = load_japanese_vowels()  # example unequal length mutlivariate collection
>>> len(X4)
640
>>> X4[0].shape
(12, 20)
```

## Collection based modules

The estimators in the `classification`, `regression` and `clustering` modules learn
from collections of time  series (they inherit from the class
`BaseCollectionEstimator`). Collections of time series will often be accompanied by an
array of target variables for supervised learning. The module `similarity_search` also
works with collections of time series.

Collection estimators closely follow the `scikit-learn` estimator interface, using
`fit`, `predict`, `transform`, `predict_proba`, `fit_predict` and `fit_transform`
where appropriate. They are also designed to work directly  with `scikit-learn`
functionality for e.g. model evaluation, parameter searching and pipelines where
appropriate.

### Classification

Time series classification (TSC) involves training a model on a labelled collection
of time series. The labels, referred to as `y` in code, should be a `numpy` array of
type `int` or `str`.

The classification estimator interface should be familiar if you have worked with
`scikit-learn`. In this example we fit a [KNeighborsTimeSeriesClassifier](classification.distance_based.KNeighborsTimeSeriesClassifier)
with dynamic time warping (DTW) on our example data.

```{code-block} python
>>> import numpy as np
>>> from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
>>> X = [[[1, 2, 3, 4, 5, 6, 7]],  # 3D array example (univariate)
...      [[4, 4, 4, 5, 6, 7, 3]]]  # Two samples, one channel, seven series length
>>> y = [0, 1]  # class labels for each sample
>>> X = np.array(X)
>>> y = np.array(y)
>>> clf = KNeighborsTimeSeriesClassifier(distance="dtw")
>>> clf.fit(X, y)  # fit the classifier on train data
KNeighborsTimeSeriesClassifier()
>>> X_test = np.array([[2, 2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4, 4]])
>>> clf.predict(X_test)  # make class predictions on new data
[0 1]
```

Once the classifier has been fit using the training data and class labels, we can
predict the labels for new cases. Like `scikit-learn`, `predict_proba` methods are
available to predict class probabilities and a `score` method is present to
calculate accuracy on new data.

### Regression

Time series regression assumes that the target variable is not a discrete label as
with classification, but is instead a continuous variable, or target variable. The
same input data considerations apply from the
classification section, and the modules function similarly. The target variable
should be a `numpy` array of type `float`.

Time series regression is a term commonly used in forecasting when used in
conjunction with a sliding
window. However, the term also includes "time series extrinsic regression" where the
target variable is not future values but some external variable.
In the following example we use a [KNeighborsTimeSeriesRegressor](regression.distance_based.KNeighborsTimeSeriesRegressor)
on an example time series regression problem called [Covid3Month](https://zenodo.org/record/3902690).

```{code-block} python
>>> from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
>>> from aeon.datasets import load_covid_3month
>>> from sklearn.metrics import mean_squared_error
>>> X_train, y_train = load_covid_3month(split="train")
>>> X_test, y_test = load_covid_3month(split="test")
>>> reg = KNeighborsTimeSeriesRegressor(distance="dtw")
>>> reg.fit(X_train, y_train)  # fit the regressor on train data
KNeighborsTimeSeriesRegressor()
>>> y_pred = reg.predict(X_test)  # make label predictions on new data
>>> y_pred[:6]
[0.04218472 0.01459854 0.         0.0164468  0.06254257 0.11111111]
>>> mean_squared_error(y_test, y_pred)
0.002921957478363366
```

### Clustering

Like classification and regression, time series clustering (TSCL) aims to follow the
`scikit-learn` interface where possible. The same input data format is used as in
the TSC and TSER modules. This example fits a [TimeSeriesKMeans](clustering._k_means.TimeSeriesKMeans)
clusterer on the [ArrowHead](http://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) dataset.

```{code-block} python
>>> from aeon.clustering import TimeSeriesKMeans
>>> from aeon.datasets import load_arrow_head
>>> from sklearn.metrics import rand_score
>>> X, y = load_arrow_head()
>>> kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw")
>>> kmeans.fit(X) # fit the clusterer
TimeSeriesKMeans(n_clusters=3)
>>> kmeans.labels_[0:10]  # cluster labels
[2 1 1 0 1 1 0 1 1 0]
>>> rand_score(y, kmeans.labels_)
0.6377792823290453
```

After calling `fit`, the `labels_` attribute contains the cluster labels for
each time series. The `predict` method can be used to predict the cluster labels for
new data.

### Similarity Search

The similarity search module in `aeon` offers a set of functions and estimators to solve
tasks related to time series similarity search. The estimators can be used standalone
or as parts of pipelines, while the functions give you the tools to build your own
estimators that would rely on similarity search at some point.

The estimators are inheriting from the [BaseSimiliaritySearch](similarity_search.base.BaseSimiliaritySearch)
class accepts as inputs 3D time series (n_cases, n_channels, n_timepoints) for the
fit method. Univariate and single series can still be used, but will need to be reshaped
to this format.

This collection, asked for the fit method, is stored as a database. It will be used in
the predict method, which expects a single 2D time series as input
(n_channels, query_length). This 2D time series will be used as a query to search for in
the 3D database.

The result of the predict method will then depends on wheter you use the [QuerySearch](similarity_search.query_search.QuerySearch)
and the [SeriesSearch](similarity_search.series_search.SeriesSearch) estimator. In [QuerySearch](similarity_search.query_search.QuerySearch), the 2D series is a subsequence
for which we want to indentify the best (or worst !) matches in the 3D database.
For [SeriesSearch](similarity_search.series_search.SeriesSearch), we require a `length` parmater, and we will search for the best
matches of all subsequences of size `length` in the 2D series inside the 3D database.
By default, these estimators will use the Euclidean (or squared Euclidean) distance,
but more distance will be added in the future.

```{code-block} python
>>> import numpy as np
>>> from aeon.similarity_search import QuerySearch
>>> X = [[[1, 2, 3, 4, 5, 6, 7]],  # 3D array example (univariate)
...      [[4, 4, 4, 5, 6, 7, 3]]]  # Two samples, one channel, seven series length
>>> X = np.array(X) # X is of shape (2, 1, 7) : (n_cases, n_channels, n_timepoints)
>>> top_k = QuerySearch(k=2)
>>> top_k.fit(X)  # fit the estimator on train data
...
>>> q = np.array([[4, 5, 6]]) # q is of shape (1,3) :
>>> top_k.predict(q)  # Identify the two (k=2) most similar subsequences of length 3 in X
[(0, 3), (1, 2)]
```
The output of predict gives a list of size `k`, where each element is a set indicating
the location of the best matches in X as `(id_sample, id_timestamp)`. This is equivalent
to the subsequence `X[id_sample, :, id_timestamps:id_timestamp + q.shape[0]]`.

## Transformers

We split transformers into two categories: those that transform single time series
and those that transform a collection.

### Transformers for Single Time Series

Transformers inheriting from the [BaseSeriesTransformer](transformations.base.BaseSeriesTransformer)
in the `aeon.transformations.series` package transform a single (possibly multivariate)
time series into a different time series or a feature vector. More info to follow.

The following example shows how to use the
[AutoCorrelationSeriesTransformer](transformations.series.AutoCorrelationSeriesTransformer)
class to extract the autocorrelation terms of a time series.

```{code-block} python
>>> from aeon.transformations.series import AutoCorrelationSeriesTransformer
>>> from aeon.datasets import load_airline
>>> acf = AutoCorrelationSeriesTransformer()
>>> y = load_airline()  # load single series airline dataset
>>> res = acf.fit_transform(y)
>>> res[0][:5]
[0.96019465 0.89567531 0.83739477 0.7977347  0.78594315]
```

### Transformers for Collections of Time Series

The `aeon.transformations.collections` module contains a range of transformers for
collections of time series. These do not allow for single series input,
treat 2D input types as a collection of univariate series, and have no restrictions on
the datatype of output.

Most time series classification and regression algorithms are based on some form of
transformation into an alternative feature space. For example, we might extract some
summary time series features from each series, and fit a traditional classifier or
regressor on these features. For example, we could use
[Catch22](transformations.collection.feauture_based), which calculates 22 summary
statistics for each series.

```{code-block} python
>>> from aeon.transformations.collection.feature_based import Catch22
>>> import numpy as np
>>> X = np.random.RandomState().random(size=(4, 1, 10))  # four cases of 10 timepoints
>>> c22 = Catch22(replace_nans=True)  # transform to four cases of 22 features
>>> c22.fit_transform(X)[0]
[ 4.99485761e-01  4.12452579e-01  3.00000000e+00  1.00000000e-01
  0.00000000e+00  1.00000000e+00  2.00000000e+00  3.08148791e-34
  1.96349541e+00  2.56152262e-01 -1.09028518e-02  9.08908735e-01
  2.00000000e+00  1.00000000e+00  4.00000000e+00  1.88915916e+00
  1.00000000e+00  5.95334611e-01  0.00000000e+00  0.00000000e+00
  8.23045267e-03  0.00000000e+00]
```

There are also series-to-series transformations, such as the
[Padder](transformations.collection) to lengthen
series and process unequal length collections.

```{code-block} python
>>> from aeon.transformations.collection import Padder
>>> from aeon.testing.data_generation import make_example_3d_numpy_list
>>> X, _ = make_example_3d_numpy_list(  # unequal length data with 8-12 timepoints
...     n_cases=2,
...     min_n_timepoints=8,
...     max_n_timepoints=12,
...     random_state=0,
... )
>>> print(X[0])
[[0.         1.6885315  1.71589124 1.69450348 1.24712739 0.76876341
  0.59506921 0.11342595 0.54531259 0.95533023 1.62433746 0.95995434]]
>>> print(X[1])
[[2.         0.28414423 0.3485172  0.08087359 3.33047938 3.112627
  3.48004859 3.91447337 3.19663426]]
>>> pad = Padder(pad_length=12, fill_value=0)  # pad to length 12
>>> pad.fit_transform(X)
[[[0.         1.6885315  1.71589124 1.69450348 1.24712739 0.76876341
   0.59506921 0.11342595 0.54531259 0.95533023 1.62433746 0.95995434]]
 [[2.         0.28414423 0.3485172  0.08087359 3.33047938 3.112627
   3.48004859 3.91447337 3.19663426 0.         0.         0.        ]]]
```

## Pipelines for aeon estimators

Like `scikit-learn`, `aeon` provides pipeline classes which can be used to chain
transformations and estimators together.

For machine learning tasks such as classification, regression and clustering, the
`scikit-learn` `make_pipeline` functionality can be used if the transformer outputs
a valid input type.

The following example uses the [Catch22](transformations.collection.catch22.Catch22)
feature extraction transformer and a random forest classifier to classify.

```{code-block} python
>>> from aeon.datasets import load_italy_power_demand
>>> from aeon.transformations.collection.feature_based import Catch22
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.metrics import accuracy_score
...
>>> # Load the italy power demand dataset
>>> X_train, y_train = load_italy_power_demand(split="train")
>>> X_test, y_test = load_italy_power_demand(split="test")
...
>>> # Create and fit the pipeline
>>> pipe = make_pipeline(
...     Catch22(replace_nans=True),
...     RandomForestClassifier(random_state=42),
... )
>>> pipe.fit(X_train, y_train)
Pipeline(steps=[('catch22', Catch22(replace_nans=True)),
                ('randomforestclassifier',
                 RandomForestClassifier(random_state=42))])
>>> # Make predictions like any other sklearn estimator
>>> accuracy_score(pipe.predict(X_test), y_test)
0.8989310009718173
```

Like with pipelines, tasks such as classification, regression and clustering can use
the available `scikit-learn` functionality.

```{code-block} python
>>> from sklearn.metrics import accuracy_score
>>> from sklearn.model_selection import GridSearchCV, KFold
>>> from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
>>> from aeon.datasets import load_italy_power_demand
...
>>> # Load the italy power demand dataset
>>> X_train, y_train = load_italy_power_demand(split="train")
>>> X_test, y_test = load_italy_power_demand(split="test")
...
>>> knn = KNeighborsTimeSeriesClassifier()
>>> param_grid = {"n_neighbors": [1, 5], "distance": ["euclidean", "dtw"]}
...
>>> gscv = GridSearchCV(knn, param_grid, cv=KFold(n_splits=4))
>>> gscv.fit(X_train, y_train)
...
>>> y_pred = gscv.predict(X_test)
>>> y_pred[:6]
['2' '2' '2' '2' '2' '1']
>>> accuracy_score(y_test, y_pred)
0.9523809523809523
>>> gscv.best_params_
{'distance': 'euclidean', 'n_neighbors': 5}
```
