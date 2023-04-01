import pandas as pd

from explainability.structured.structured_manipulator import \
    StructuredManipulator


class TestStructuredManipulator:
    def test_init(self):
        _ = StructuredManipulator(
            pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}),
            label_column="col2")
