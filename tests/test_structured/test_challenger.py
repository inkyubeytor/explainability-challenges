import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from explainability.structured.samples.challengers.categorize_challenger \
    import CategorizeChallenger


class TestSampleChallengers:

    @pytest.mark.slow
    def test_categorize_challenger(self):
        df = pd.read_csv(
            "https://archive.ics.uci.edu/"
            "ml/machine-learning-databases/adult/adult.data",
            header=None)
        df.columns = [
            "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
            "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
            "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry",
            "Income"
        ]
        label_column = "Income"
        cc = CategorizeChallenger(DecisionTreeClassifier(max_depth=1),
                                  df, label_column)
        cc.generate_challenges()
        cc.train_models()
