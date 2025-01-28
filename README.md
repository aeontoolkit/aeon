<p align="center">
    <a href="https://aeon-toolkit.org"><img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/logo/aeon-logo-blue-compact.png" width="50%" alt="aeon logo" /></a>
</p>

# ⌛ Welcome to aeon

`aeon` is an open-source toolkit for learning from time series. It is compatible with
[scikit-learn](https://scikit-learn.org) and provides access to the very latest
algorithms for time series machine learning, in addition to a range of classical
techniques for learning tasks such as forecasting and classification.

We strive to provide a broad library of time series algorithms including the
latest advances, offer efficient implementations using numba, and interfaces with other
time series packages to provide a single framework for algorithm comparison.

The latest `aeon` release is `v1.0.0`. You can view the full changelog
[here](https://www.aeon-toolkit.org/en/stable/changelog.html).

Our webpage and documentation is available at https://aeon-toolkit.org.

The following modules are still considered experimental, and the [deprecation policy](https://www.aeon-toolkit.org/en/stable/developer_guide/deprecation.html)
does not apply:

- `anomaly_detection`
- `forecasting`
- `segmentation`
- `similarity_search`
- `visualisation`
- `transformations.collection.imbalance`

| Overview        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD**       | [![github-actions-release](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/release.yml?logo=github&label=build%20%28release%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/release.yml) [![github-actions-main](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/pr_pytest.yml?logo=github&branch=main&label=build%20%28main%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/pr_pytest.yml) [![github-actions-nightly](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/periodic_tests.yml?logo=github&label=build%20%28nightly%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/periodic_tests.yml) [![docs-main](https://img.shields.io/readthedocs/aeon-toolkit/stable?logo=readthedocs&label=docs%20%28stable%29)](https://www.aeon-toolkit.org/en/stable/) [![docs-main](https://img.shields.io/readthedocs/aeon-toolkit/latest?logo=readthedocs&label=docs%20%28latest%29)](https://www.aeon-toolkit.org/en/latest/) [![!codecov](https://img.shields.io/codecov/c/github/aeon-toolkit/aeon?label=codecov&logo=codecov)](https://codecov.io/gh/aeon-toolkit/aeon) [![openssf-scorecard](https://api.scorecard.dev/projects/github.com/aeon-toolkit/aeon/badge)](https://img.shields.io/ossf-scorecard/github.com/aeon-toolkit/aeon?label=openssf%20scorecard&style=flat) |
| **Code**        | [![!pypi](https://img.shields.io/pypi/v/aeon?logo=pypi&color=blue)](https://pypi.org/project/aeon/) [![!conda](https://img.shields.io/conda/vn/conda-forge/aeon?logo=anaconda&color=blue)](https://anaconda.org/conda-forge/aeon) [![!python-versions](https://img.shields.io/pypi/pyversions/aeon?logo=python)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![license](https://img.shields.io/badge/license-BSD%203--Clause-green?logo=style)](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE) [![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aeon-toolkit/aeon/main?filepath=examples)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Community**   | [![!slack](https://img.shields.io/static/v1?logo=slack&label=Slack&message=chat&color=lightgreen)](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA) [![!linkedin](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/aeon-toolkit/) [![!x-twitter](https://img.shields.io/static/v1?logo=x&label=X/Twitter&message=news&color=lightblue)](https://twitter.com/aeon_toolkit)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Affiliation** | [![numfocus](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

## ⚙️ Installation

`aeon` requires a Python version of 3.9 or greater. Our full installation guide is
available in our [documentation](https://www.aeon-toolkit.org/en/stable/installation.html).

The easiest way to install `aeon` is via pip:

```bash
pip install aeon
```

Some estimators require additional packages to be installed. If you want to install
the full package with all optional dependencies, you can use:

```bash
pip install aeon[all_extras]
```

Instructions for installation from the [GitHub source](https://github.com/aeon-toolkit/aeon)
can be found [here](https://www.aeon-toolkit.org/en/stable/developer_guide/dev_installation.html).

## ⏲️ Getting started

The best place to get started for all `aeon` packages is our [getting started guide](https://www.aeon-toolkit.org/en/stable/getting_started.html).

Below we provide a quick example of how to use `aeon` for classification and clustering.

### Classification/Regression

Time series classification looks to predict class labels fore unseen series using a
model fitted from a collection of time series. The framework for regression is similar,
replace the classifier with a regressor and the labels with continuous values.

```python
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

X = np.array([[[1, 2, 3, 4, 5, 5]],  # 3D array example (univariate)
             [[1, 2, 3, 4, 4, 2]],   # Three samples, one channel,
             [[8, 7, 6, 5, 4, 4]]])  # six series length
y = np.array(['low', 'low', 'high'])  # class labels for each sample

clf = KNeighborsTimeSeriesClassifier(distance="dtw")
clf.fit(X, y)  # fit the classifier on train data
>>> KNeighborsTimeSeriesClassifier()

X_test = np.array(
    [[[2, 2, 2, 2, 2, 2]], [[5, 5, 5, 5, 5, 5]], [[6, 6, 6, 6, 6, 6]]]
)
y_pred = clf.predict(X_test)  # make class predictions on new data
>>> ['low' 'high' 'high']
```

### Clustering

Time series clustering groups similar time series together from a collection of
time series.

```python
import numpy as np
from aeon.clustering import TimeSeriesKMeans

X = np.array([[[1, 2, 3, 4, 5, 5]],  # 3D array example (univariate)
             [[1, 2, 3, 4, 4, 2]],   # Three samples, one channel,
             [[8, 7, 6, 5, 4, 4]]])  # six series length

clu = TimeSeriesKMeans(distance="dtw", n_clusters=2)
clu.fit(X)  # fit the clusterer on train data
>>> TimeSeriesKMeans(distance='dtw', n_clusters=2)

clu.labels_ # get training cluster labels
>>> array([0, 0, 1])

X_test = np.array(
    [[[2, 2, 2, 2, 2, 2]], [[5, 5, 5, 5, 5, 5]], [[6, 6, 6, 6, 6, 6]]]
)
clu.predict(X_test)  # Assign clusters to new data
>>> array([1, 0, 0])
```

## 💬 Where to ask questions

| Type                               | Platforms                         |
|------------------------------------|-----------------------------------|
| 🐛 **Bug Reports**                 | [GitHub Issue Tracker]            |
| ✨ **Feature Requests & Ideas**     | [GitHub Issue Tracker] & [Slack]  |
| 💻 **Usage Questions**             | [GitHub Discussions] & [Slack]    |
| 💬 **General Discussion**          | [GitHub Discussions] & [Slack]    |
| 🏭 **Contribution & Development**  | [Slack]                           |

[GitHub Issue Tracker]: https://github.com/aeon-toolkit/aeon/issues
[GitHub Discussions]: https://github.com/aeon-toolkit/aeon/discussions
[Slack]: https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA

For enquiries about the project or collaboration, our email is
[contact@aeon-toolkit.org](mailto:contact@aeon-toolkit.org).

## 🔨 Contributing to aeon

If you are interested in contributing to `aeon`, please see our [contributing guide](https://www.aeon-toolkit.org/en/latest/contributing.html)
and have a read through before assigning an issue and creating a pull request. Be
aware that the `latest` version of the docs is the development version, and the `stable`
version is the latest release.

The `aeon` developers are volunteers so please be patient with responses to comments and
pull request reviews. If you have any questions, feel free to ask using the above
mediums.

## 📚 Citation

If you use `aeon` we would appreciate a citation of the following [paper](https://jmlr.org/papers/v25/23-1444.html):

```bibtex
@article{aeon24jmlr,
  author  = {Matthew Middlehurst and Ali Ismail-Fawaz and Antoine Guillaume and Christopher Holder and David Guijo-Rubio and Guzal Bulatova and Leonidas Tsaprounis and Lukasz Mentel and Martin Walter and Patrick Sch{{\"a}}fer and Anthony Bagnall},
  title   = {aeon: a Python Toolkit for Learning from Time Series},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {289},
  pages   = {1--10},
  url     = {http://jmlr.org/papers/v25/23-1444.html}
}
```

If you let us know about your paper using `aeon`, we will happily list it [here](https://www.aeon-toolkit.org/en/stable/papers_using_aeon.html).

## 💬 Further information

`aeon` was forked from `sktime` `v0.16.0` in 2022 by an initial group of eight core
developers.
