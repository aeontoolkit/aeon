---
hide-toc: true
---

# Welcome to __aeon__

`aeon` is a `scikit-learn` compatible toolkit for time series machine learning tasks
such as classification, regression, clustering, anomaly detection,
segmentation and similarity search.

- We provide a broad library of time series algorithms, including the latest
  advances and state-of-the-art for many tasks.
- Our algorithms are implemented as efficiently as possible by, for example,
  using `numba`.
- `aeon` is built on top of `scikit-learn`, allowing for easy integration with other
  machine learning libraries and other time series packages.
- We provide a range of tools for reproducing benchmarking results and evaluating time
  series algorithms implemented in `aeon` and other `scikit-learn` compatible packages.

## Community Channels

**GitHub**: [github.com/aeon-toolkit/aeon](https://github.com/aeon-toolkit/aeon)

**Slack**: [aeon slack](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA)

**Twitter**: [twitter/aeon-toolkit](https://twitter.com/aeon_toolkit)

**LinkedIn**: [linkedin/aeon-toolkit](https://www.linkedin.com/company/aeon-toolkit)

**Email**: [contact@aeon-toolkit.org](mailto:contact@aeon-toolkit.org)

## Modules

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card}
:img-top: examples/classification/img/tsc.png
:class-img-top: aeon-card-image
:text-align: center

Get started with time series classification.

+++

```{button-ref} /examples/classification/classification.ipynb
:color: primary
:click-parent:
:expand:

Classification
```

:::

:::{grid-item-card}
:img-top: examples/regression/img/tser.png
:class-img-top: aeon-card-image
:text-align: center

Get started with time series extrinsic regression.

+++

```{button-ref} /examples/regression/regression.ipynb
:color: primary
:click-parent:
:expand:

Regression
```

:::

:::{grid-item-card}
:img-top: examples/clustering/img/tscl.png
:class-img-top: aeon-card-image
:text-align: center

Get started with time series clustering.

+++

```{button-ref} /examples/clustering/clustering.ipynb
:color: primary
:click-parent:
:expand:

Clustering
```

:::

:::{grid-item-card}
:img-top: examples/anomaly_detection/img/anomaly_detection.png
:class-img-top: aeon-card-image
:text-align: center

Get started with anomaly detection.

+++

```{button-ref} /examples/anomaly_detection/anomaly_detection.ipynb
:color: primary
:click-parent:
:expand:

Anomaly Detection
```

:::

:::{grid-item-card}
:img-top: examples/forecasting/img/forecasting.png
:class-img-top: aeon-card-image
:text-align: center

Get started with forecasting

+++

```{button-ref} /examples/forecasting/forecasting.ipynb
:color: primary
:click-parent:
:expand:

Forecasting
```

:::

:::{grid-item-card}
:img-top: examples/segmentation/img/segmentation.png
:class-img-top: aeon-card-image
:text-align: center

Get started with segmentation

+++

```{button-ref} /examples/segmentation/segmentation.ipynb
:color: primary
:click-parent:
:expand:

Segmentation
```

:::

:::{grid-item-card}
:img-top: examples/transformations/img/transformations.png
:class-img-top: aeon-card-image
:text-align: center

Get started with time series transformations.

+++

```{button-ref} /examples/transformations/transformations.ipynb
:color: primary
:click-parent:
:expand:

Transformations
```

:::

:::{grid-item-card}
:img-top: examples/distances/img/distances.png
:class-img-top: aeon-card-image
:text-align: center

Get started with time series distances.

+++

```{button-ref} /examples/distances/distances.ipynb
:color: primary
:click-parent:
:expand:

Distances
```

:::

:::{grid-item-card}
:img-top: examples/similarity_search/img/sim_search.png
:class-img-top: aeon-card-image
:text-align: center

Get started with time series similarity search

+++

```{button-ref} /examples/similarity_search/similarity_search.ipynb
:color: primary
:click-parent:
:expand:

Similarity Search
```

:::

:::{grid-item-card}
:img-top: examples/datasets/img/data.png
:class-img-top: aeon-card-image
:text-align: center

Data structures and containers used in `aeon`.

+++

```{button-ref} /examples/datasets/datasets.ipynb
:color: primary
:click-parent:
:expand:

Data
```

:::

:::{grid-item-card}
:img-top: examples/benchmarking/img/benchmarking.png
:class-img-top: aeon-card-image
:text-align: center

How to benchmark algorithms with `aeon`.

+++

```{button-ref} /examples/benchmarking/benchmarking.ipynb
:color: primary
:click-parent:
:expand:

Benchmarking
```

:::

:::{grid-item-card}
:img-top: examples/networks/img/Inception.png
:class-img-top: aeon-card-image
:text-align: center

`aeon` deep learning networks for time series.

+++

```{button-ref} /examples/networks/deep_learning.ipynb
:color: primary
:click-parent:
:expand:

Networks
```

:::

::::

## Experimental Modules

Some modules of `aeon` are still experimental and may have changing interfaces.
To support development on these modules, the [deprecation policy](developer_guide/deprecation.md)
is relaxed, so it is suggested that you integrate these modules with care. The current
experimental modules are:

- `anomaly_detection`
- `forecasting`
- `segmentation`
- `similarity_search`
- `visualisation`
- `transformations.collection.imbalance`

```{toctree}
:caption: Using aeon
:hidden:

installation.md
getting_started.md
api_reference.md
examples.md
```

```{toctree}
:caption: Developing aeon
:hidden:

contributing.md
developer_guide.md
mentoring.md
```

```{toctree}
:caption: The aeon team
:hidden:

contributors.md
about.md
governance.md
code_of_conduct.md
```

```{toctree}
:caption: Other
:hidden:

estimator_overview.md
changelog.md
papers_using_aeon.md
```
