"""Unit tests for regressors deep learners save/load functionalities."""

import inspect
import os
import tempfile
import time

import pytest

from aeon.regression import deep_learning
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = ["hadifawaz1999"]


_deep_rgs_classes = [
    member[1] for member in inspect.getmembers(deep_learning, inspect.isclass)
]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("deep_rgs", _deep_rgs_classes)
def test_saving_loading_deep_learning_rgs(deep_rgs):
    """Test Deep Regressor saving."""
    with tempfile.TemporaryDirectory() as tmp:
        if not (
            deep_rgs.__name__
            in [
                "BaseDeepRegressor",
                "InceptionTimeRegressor",
                "LITETimeRegressor",
                "TapNetRegressor",
            ]
        ):
            if tmp[-1] != "/":
                tmp = tmp + "/"
            curr_time = str(time.time_ns())
            last_file_name = curr_time + "last"
            best_file_name = curr_time + "best"
            init_file_name = curr_time + "init"

            X, y = make_example_3d_numpy()

            test_params = deep_rgs.get_test_params()[0]
            test_params["save_best_model"] = True
            test_params["save_last_model"] = True
            test_params["save_init_model"] = True
            test_params["best_file_name"] = best_file_name
            test_params["last_file_name"] = last_file_name
            test_params["init_file_name"] = init_file_name
            test_params["file_path"] = tmp

            deep_rgs_train = deep_rgs(**test_params)
            deep_rgs_train.fit(X, y)

            deep_rgs_best = deep_rgs(**test_params)
            deep_rgs_best.load_model(
                model_path=os.path.join(tmp, best_file_name + ".keras"),
            )
            ypred_best = deep_rgs_best.predict(X)
            assert len(ypred_best) == len(y)

            deep_rgs_last = deep_rgs(**test_params)
            deep_rgs_last.load_model(
                model_path=os.path.join(tmp, last_file_name + ".keras"),
            )
            ypred_last = deep_rgs_last.predict(X)
            assert len(ypred_last) == len(y)

            deep_rgs_init = deep_rgs(**test_params)
            deep_rgs_init.load_model(
                model_path=os.path.join(tmp, init_file_name + ".keras"),
            )
            ypred_init = deep_rgs_init.predict(X)
            assert len(ypred_init) == len(y)
