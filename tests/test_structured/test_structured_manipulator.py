import copy
import pandas as pd
import numpy as np
import pytest

from explainability.structured.structured_manipulator import \
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
            pd.DataFrame(data={"col1": [1, 2, 3, 4, 5, 6], "col2": ['a', 'a', 'a', 'b', 'b', 'b']}),
            label_column="col2")

        sm1 = copy.deepcopy(sm).replace_random_values()
        sm2 = copy.deepcopy(sm).replace_random_values("col1", 0.5, 3.5)
        assert(sm1.trace == sm2.trace)
        assert((sm1.df["col1"] == 3.5).sum() == 3)
        assert((sm2.df["col1"] == 3.5).sum() == 3)

        with pytest.raises(ValueError):
            sm3 = copy.deepcopy(sm).replace_random_values("col2")

        with pytest.raises(ValueError):
            sm4 = copy.deepcopy(sm).replace_random_values(proportion=-1)

        with pytest.raises(ValueError):
            sm5 = copy.deepcopy(sm).replace_random_values(proportion=2)
