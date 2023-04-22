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
        assert(len(sc.challenges) == 3)

        sm_base = sc.challenges["base"]
        sm_rv = sc.challenges["replace_values"]
        sm_df = sc.challenges["duplicate_features"]
        assert((sm_base.df.values != sm_rv.df.values).any())
        assert(len(sm_base.df.columns) + 1 == len(sm_df.df.columns))

    def test_train_models(self):
        model = LinearRegression()
        sc = SimpleChallenger(model,
                              pd.DataFrame(data={"col1": [1, 2, 3, 4, 5, 6],
                                                 "col2": ['a', 'a', 'a', 'b', 'b', 'b'],
                                                 "col3": [1, 1, 1, 1, 1, 1]}, ),
                              label_column="col3")
        sc.generate_challenges()
        sc.train_models()
        assert(len(sc.models) == 3)

        for challenge in sc.challenges:
            x_train, y_train, x_test, y_test = \
                sc.challenges[challenge].train_test_split(test_proportion=0.34)
            pipe = sc.models[challenge]
            enc = pipe.named_steps["encoder"]
            scal = pipe.named_steps["scaler"]
            model = pipe.named_steps["model"]
            x_train = scal.transform(enc.transform(x_train))
            x_test = scal.transform(enc.transform(x_test))
            assert(model.score(x_train, y_train) == 1)
            assert(model.score(x_test, y_test) == 1)
