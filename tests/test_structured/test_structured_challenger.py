import copy
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from explainability.structured.core.structured_challenger import \
    SKChallenger
from explainability.structured.core.structured_manipulator import \
    StructuredManipulator


class SimpleChallenger(SKChallenger):
    def generate_challenges(self) -> None:
        sm = StructuredManipulator(self.df, self.label_column, self.random_state)
        self.challenges["replace_values"] = copy.deepcopy(sm).replace_random_values()
        self.challenges["duplicate_features"] = copy.deepcopy(sm).duplicate_features()


class TestSKChallenger:
    def test_init(self):

        model = LinearRegression()
        _ = SimpleChallenger(model,
            pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}),
            label_column="col2")
