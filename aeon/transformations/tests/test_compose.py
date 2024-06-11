"""Unit tests for transformer composition functionality attached to the base class."""

__maintainer__ = []
__all__ = []

import pandas as pd
from sklearn.preprocessing import StandardScaler

from aeon.datasets import load_airline, load_basic_motions
from aeon.testing.data_generation import get_examples
from aeon.testing.mock_estimators import MockTransformer
from aeon.testing.utils.deep_equals import deep_equals
from aeon.testing.utils.estimator_checks import _assert_array_almost_equal
from aeon.transformations.boxcox import LogTransformer
from aeon.transformations.collection.pad import PaddingTransformer
from aeon.transformations.compose import (
    ColumnConcatenator,
    FeatureUnion,
    InvertTransform,
    OptionalPassthrough,
    TransformerPipeline,
)
from aeon.transformations.impute import Imputer
from aeon.transformations.subset import ColumnSelect
from aeon.transformations.summarize import SummaryTransformer
from aeon.transformations.theta import ThetaLinesTransformer


def test_dunder_mul():
    """Test the mul dunder method."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t1 = MockTransformer(power=2)
    t2 = MockTransformer(power=5)
    t3 = MockTransformer(power=0.1)
    t4 = MockTransformer(power=1)

    t12 = t1 * t2
    t123 = t12 * t3
    t312 = t3 * t12
    t1234 = t123 * t4
    t1234_2 = t12 * (t3 * t4)

    assert isinstance(t12, TransformerPipeline)
    assert isinstance(t123, TransformerPipeline)
    assert isinstance(t312, TransformerPipeline)
    assert isinstance(t1234, TransformerPipeline)
    assert isinstance(t1234_2, TransformerPipeline)

    assert [x.power for x in t12.steps] == [2, 5]
    assert [x.power for x in t123.steps] == [2, 5, 0.1]
    assert [x.power for x in t312.steps] == [0.1, 2, 5]
    assert [x.power for x in t1234.steps] == [2, 5, 0.1, 1]
    assert [x.power for x in t1234_2.steps] == [2, 5, 0.1, 1]

    _assert_array_almost_equal(X, t123.fit_transform(X))
    _assert_array_almost_equal(X, t312.fit_transform(X))
    _assert_array_almost_equal(X, t1234.fit_transform(X))
    _assert_array_almost_equal(X, t1234_2.fit_transform(X))
    _assert_array_almost_equal(t12.fit_transform(X), t3.fit(X).inverse_transform(X))


def test_dunder_add():
    """Test the add dunder method."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t1 = MockTransformer(power=2)
    t2 = MockTransformer(power=5)
    t3 = MockTransformer(power=3)

    t12 = t1 + t2
    t123 = t12 + t3
    t123r = t1 + (t2 + t3)

    assert isinstance(t12, FeatureUnion)
    assert isinstance(t123, FeatureUnion)
    assert isinstance(t123r, FeatureUnion)

    assert [x.power for x in t12.transformer_list] == [2, 5]
    assert [x.power for x in t123.transformer_list] == [2, 5, 3]
    assert [x.power for x in t123r.transformer_list] == [2, 5, 3]

    _assert_array_almost_equal(t123r.fit_transform(X), t123.fit_transform(X))


def test_mul_sklearn_autoadapt():
    """Test auto-adapter for sklearn in mul."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t1 = MockTransformer(power=2)
    t2 = StandardScaler()
    t3 = MockTransformer(power=0.5)

    t123 = t1 * t2 * t3
    t123r = t1 * (t2 * t3)
    t123l = (t1 * t2) * t3

    assert isinstance(t123, TransformerPipeline)
    assert isinstance(t123r, TransformerPipeline)
    assert isinstance(t123l, TransformerPipeline)

    _assert_array_almost_equal(t123.fit_transform(X), t123l.fit_transform(X))
    _assert_array_almost_equal(t123r.fit_transform(X), t123l.fit_transform(X))


def test_missing_unequal_tag_inference():
    """Test that TransformerPipeline infers missing/unequal tags correctly."""
    t1 = MockTransformer() * PaddingTransformer() * MockTransformer()
    t2 = MockTransformer() * MockTransformer()
    t3 = Imputer() * MockTransformer()
    t4 = MockTransformer() * Imputer()

    assert t1.get_tag("capability:unequal_length")
    assert t1.get_tag("capability:unequal_length:removes")
    assert not t2.get_tag("capability:unequal_length:removes")
    assert t3.get_tag("capability:missing_values")
    assert t3.get_tag("capability:missing_values:removes")
    assert not t4.get_tag("capability:missing_values")
    assert not t4.get_tag("capability:missing_values:removes")


def test_featureunion_transform_cols():
    """Test FeatureUnion name and number of columns."""
    X = pd.DataFrame({"test1": [1, 2], "test2": [3, 4]})

    t1 = MockTransformer(power=2)
    t2 = MockTransformer(power=5)
    t3 = MockTransformer(power=3)

    t123 = t1 + t2 + t3

    Xt = t123.fit_transform(X)

    expected_cols = pd.Index(
        [
            "MockTransformer_1__test1",
            "MockTransformer_1__test2",
            "MockTransformer_2__test1",
            "MockTransformer_2__test2",
            "MockTransformer_3__test1",
            "MockTransformer_3__test2",
        ]
    )

    msg = (
        f"FeatureUnion creates incorrect column names for DataFrame output. "
        f"Expected: {expected_cols}, found: {Xt.columns}"
    )

    assert deep_equals(Xt.columns, expected_cols), msg


def test_sklearn_after_primitives():
    """Test that sklearn transformer after primitives is correctly applied."""
    t = SummaryTransformer() * StandardScaler()
    assert t.get_tag("output_data_type") == "Primitives"

    X = get_examples("pd-multiindex")[0]
    X_out = t.fit_transform(X)
    X_summary = SummaryTransformer().fit_transform(X)

    assert (X_out.index == X_summary.index).all()
    assert deep_equals(X_out.columns, X_summary.columns)
    # var_0 is the same for all three instances
    # so summary statistics are all the same, thus StandardScaler transforms to 0
    assert X_out.iloc[0, 0] > -0.01
    assert X_out.iloc[0, 0] < 0.01
    # var_1 has some variation between three instances
    # fix this to one value to tie the output to current behaviour
    assert X_out.iloc[0, 10] > -1.37
    assert X_out.iloc[0, 10] < -1.36


def test_pipeline_column_vectorization():
    """Test that pipelines vectorize properly over columns."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t = ColumnSelect([0, 1]) * ThetaLinesTransformer()

    X_theta = t.fit_transform(X)

    assert set(X_theta.columns) == {"a__0", "a__2", "b__0", "b__2"}


def test_pipeline_inverse():
    """Tests that inverse composition works, with inverse skips. Also see #3084."""
    X = load_airline()
    t = LogTransformer() * Imputer()

    # LogTransformer has inverse_transform, and does not skip inverse transform
    # therefore, pipeline should also not skip inverse transform, and have capability
    assert t.get_tag("capability:inverse_transform")
    assert not t.get_tag("skip-inverse-transform")

    t.fit(X)
    Xt = t.transform(X)
    Xtt = t.inverse_transform(Xt)

    _assert_array_almost_equal(X, Xtt)


def test_subset_getitem():
    """Test subsetting using the [ ] dunder, __getitem__."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    t = ThetaLinesTransformer()

    t_before = t["a"]
    t_before_with_colon = t[["a", "b"], :]
    t_after_with_colon = t[:, ["a__0", "a__2"]]
    t_both = t[["a", "b"], ["b__0", "b__2", "c__0", "c__2"]]
    t_none = t[:, :]

    assert isinstance(t_before, TransformerPipeline)
    assert isinstance(t_after_with_colon, TransformerPipeline)
    assert isinstance(t_before_with_colon, TransformerPipeline)
    assert isinstance(t_both, TransformerPipeline)
    assert isinstance(t_none, ThetaLinesTransformer)

    X_theta = t.fit_transform(X)

    _assert_array_almost_equal(t_before.fit_transform(X), X_theta[["a__0", "a__2"]])
    _assert_array_almost_equal(
        t_after_with_colon.fit_transform(X), X_theta[["a__0", "a__2"]]
    )
    _assert_array_almost_equal(
        t_before_with_colon.fit_transform(X), X_theta[["a__0", "a__2", "b__0", "b__2"]]
    )
    _assert_array_almost_equal(t_both.fit_transform(X), X_theta[["b__0", "b__2"]])
    _assert_array_almost_equal(t_none.fit_transform(X), X_theta)


def test_dunder_invert():
    """Test the invert dunder method, for wrapping in OptionalPassthrough."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t = MockTransformer(power=3)

    t_inv = ~t

    assert isinstance(t_inv, InvertTransform)
    assert isinstance(t_inv.get_params()["transformer"], MockTransformer)

    _assert_array_almost_equal(
        t_inv.fit_transform(X), MockTransformer(1 / 3).fit_transform(X)
    )


def test_dunder_neg():
    """Test the neg dunder method, for wrapping in OptionalPassthrough."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    t = MockTransformer(power=2)

    tp = -t

    assert isinstance(tp, OptionalPassthrough)
    assert not tp.get_params()["passthrough"]
    assert isinstance(tp.get_params()["transformer"], MockTransformer)

    _assert_array_almost_equal(tp.fit_transform(X), X)


def test_column_concatenator():
    X, y = load_basic_motions(split="train")
    n_cases, n_channels, n_timepoints = X.shape
    trans = ColumnConcatenator()
    Xt = trans.fit_transform(X)

    # check if transformed dataframe is univariate
    assert Xt.shape[1] == 1

    # check if number of time series observations are correct
    assert Xt.shape[2] == X.shape[1] * X.shape[2]

    # check specific observations
    assert X[0][-1][-3] == Xt[0][0][-3]
    assert X[0][0][3] == Xt[0, 0][3]
