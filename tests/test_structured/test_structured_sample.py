import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from explainability.structured.samples.challengers.categorize_challenger \
    import CategorizeChallenger
from explainability.structured.samples.explainers.interpretml.morris_explainer \
    import MorrisExplainer


class TestSample:

    @pytest.mark.slow
    def test_categorize_challenger(self):
        df = pd.read_csv(
            "https://archive.ics.uci.edu/"
            "ml/machine-learning-databases/adult/adult.data",
            header=None, nrows=100)
        df.columns = [
            "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
            "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
            "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry",
            "Income"
        ]
        label_column = "Income"
        cc = CategorizeChallenger(DecisionTreeClassifier(max_depth=3),
                                  df, label_column)
        cc.generate_challenges()
        cc.train_models()

        me = MorrisExplainer()
        me.explain_challenge(cc, "morris_explainer.png")
