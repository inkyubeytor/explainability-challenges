import copy
import pandas as pd
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

    def test_generate_challenges(self):
        model = LinearRegression()
        sc = SimpleChallenger(model,
                              pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}),
                              label_column="col2")
        sc.generate_challenges()
        assert("base" in sc.challenges)
        assert("replace_values" in sc.challenges)
        assert("duplicate_features" in sc.challenges)

        sm_base = sc.challenges["base"]
        sm_rv = sc.challenges["replace_values"]
        sm_df = sc.challenges["duplicate_features"]
        assert((sm_base.df.values != sm_rv.df.values).any())
        assert(len(sm_base.df.columns) + 1 == len(sm_df.df.columns))
