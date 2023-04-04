import copy
import pandas as pd
import numpy as np
import pytest

from explainability.structured.core.structured_manipulator import \
    StructuredManipulator


class TestStructuredManipulator:
    def test_init(self):
        _ = StructuredManipulator(
            pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}),
            label_column="col2")

    def test_execute_default_methods(self):
        sm = StructuredManipulator(
            pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}),
            label_column="col2")

        _ = sm.replace_random_values() \
            .duplicate_features() \
            .categorize() \
            .split_category_value() \
            .sort_values()

        _ = sm.trace

    def test_replace_random_values(self):
        sm = StructuredManipulator(
            pd.DataFrame(data={"col1": [1, 2, 3, 4, 5, 6],
                               "col2": ['a', 'a', 'a', 'b', 'b', 'b'],
                               "col3": [1, 1, 1, 1, 1, 1]}, ),
            label_column="col3")

        sm1 = copy.deepcopy(sm).replace_random_values()
        sm2 = copy.deepcopy(sm).replace_random_values("col1", 0.5, 3.5)
        assert(sm1.trace == sm2.trace)
        assert((sm1.df["col1"] == 3.5).sum() == 3)
        assert((sm2.df["col1"] == 3.5).sum() == 3)

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).replace_random_values("col2")

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).replace_random_values(proportion=-1)

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).replace_random_values(proportion=2)

    def test_duplicate_features(self):
        sm = StructuredManipulator(
            pd.DataFrame(data={"col1": [1, 2, 3, 4, 5, 6],
                               "col2": ['a', 'a', 'a', 'b', 'b', 'b'],
                               "col3": [1, 1, 1, 1, 1, 1]}, ),
            label_column="col3")

        sm1 = copy.deepcopy(sm).duplicate_features("col1")
        sm2 = copy.deepcopy(sm).duplicate_features("col1", 1, ["col11"])
        assert(sm1.trace == sm2.trace)
        assert((sm1.df["col1"] == sm1.df["col11"]).sum() == 6)

        sm3 = copy.deepcopy(sm).duplicate_features("col2")
        sm4 = copy.deepcopy(sm).duplicate_features("col2", 1, ["col21"])
        assert(sm3.trace == sm4.trace)
        assert((sm3.df["col2"] == sm3.df["col21"]).sum() == 6)

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).duplicate_features("col3")

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).duplicate_features(num_dups=-1)

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).duplicate_features(dup_col_names=["1", "2"])

    def test_categorize(self):
        sm = StructuredManipulator(
            pd.DataFrame(data={"col1": [1, 2, 3, 4, 5, 6],
                               "col2": ['a', 'a', 'a', 'b', 'b', 'b'],
                               "col3": [1, 1, 1, 1, 1, 1]}, ),
            label_column="col3")

        sm1 = copy.deepcopy(sm).categorize("col1")
        sm2 = copy.deepcopy(sm).categorize("col1", 2, np.array([1. - 1e-12, 3.5, 6.]))
        assert(sm1.trace == sm2.trace)
        assert(len(sm1.df["col1"].unique()) == 2)

        sm3 = copy.deepcopy(sm).categorize("col1", bin_names=["x", "y"])
        assert(len(sm3.df["col1"].unique()) == 2)
        assert("x" in sm3.df["col1"].unique())
        assert("y" in sm3.df["col1"].unique())

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).categorize("col2")

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).categorize(num_bins=1)

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).categorize(bins=np.arange(2))

        with pytest.raises(ValueError):
            _ = copy.deepcopy(sm).categorize(bin_names=["1"])
