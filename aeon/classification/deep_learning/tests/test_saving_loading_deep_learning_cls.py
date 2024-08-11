"""Unit tests for classifiers deep learners save/load functionalities."""

import inspect
import os
import tempfile
import time

import numpy as np
import pytest

from aeon.classification import deep_learning
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = ["hadifawaz1999"]


_deep_cls_classes = [
    member[1] for member in inspect.getmembers(deep_learning, inspect.isclass)
]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("deep_cls", _deep_cls_classes)
def test_saving_loading_deep_learning_cls(deep_cls):
    """Test Deep Classifier saving."""
    with tempfile.TemporaryDirectory() as tmp:
        if not (
            deep_cls.__name__
            in [
                "BaseDeepClassifier",
                "InceptionTimeClassifier",
                "LITETimeClassifier",
                "TapNetClassifier",
            ]
        ):
            if tmp[-1] != "/":
                tmp = tmp + "/"
            curr_time = str(time.time_ns())
            last_file_name = curr_time + "last"
            best_file_name = curr_time + "best"
            init_file_name = curr_time + "init"

            X, y = make_example_3d_numpy()

            test_params = deep_cls.get_test_params()[0]
            test_params["save_best_model"] = True
            test_params["save_last_model"] = True
            test_params["save_init_model"] = True
            test_params["best_file_name"] = best_file_name
            test_params["last_file_name"] = last_file_name
            test_params["init_file_name"] = init_file_name
            test_params["file_path"] = tmp

            deep_cls_train = deep_cls(**test_params)
            deep_cls_train.fit(X, y)

            deep_cls_best = deep_cls(**test_params)
            deep_cls_best.load_model(
                model_path=os.path.join(tmp, best_file_name + ".keras"),
                classes=np.unique(y),
            )
            ypred_best = deep_cls_best.predict(X)
            assert len(ypred_best) == len(y)

            deep_cls_last = deep_cls(**test_params)
            deep_cls_last.load_model(
                model_path=os.path.join(tmp, last_file_name + ".keras"),
                classes=np.unique(y),
            )
            ypred_last = deep_cls_last.predict(X)
            assert len(ypred_last) == len(y)

            deep_cls_init = deep_cls(**test_params)
            deep_cls_init.load_model(
                model_path=os.path.join(tmp, init_file_name + ".keras"),
                classes=np.unique(y),
            )
            ypred_init = deep_cls_init.predict(X)
            assert len(ypred_init) == len(y)
