import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from explainability.structured.samples.challengers.categorize_challenger \
    import CategorizeChallenger
from explainability.structured.samples.challengers.replace_challenger \
    import ReplaceChallenger
from explainability.structured.samples.explainers.interpretml.morris_explainer \
    import MorrisExplainer
from explainability.structured.samples.explainers.interpretml.pdp_explainer \
    import PDPExplainer


class TestSample:

    @pytest.mark.slow
    def test_categorize_challenger_morris_explainer(self):
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
        df.drop(["fnlwgt", "EducationNum"], axis=1, inplace=True)
        label_column = "Income"
        cc = CategorizeChallenger(DecisionTreeClassifier(max_depth=3),
                                  df, label_column)
        cc.generate_challenges()
        cc.train_models()

        me = MorrisExplainer()
        me.explain_challenge(cc, "morris_explainer_categorize.png")

    @pytest.mark.slow
    def test_replace_challenger_pdp_explainer(self):
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
        df.drop(["fnlwgt", "EducationNum"], axis=1, inplace=True)
        label_column = "Income"
        cc = ReplaceChallenger(DecisionTreeClassifier(max_depth=3),
                               df, label_column)
        cc.generate_challenges()
        cc.train_models()

        pe = PDPExplainer()
        pe.explain_challenge(cc, "Age", "pdp_explainer_replace.png")
