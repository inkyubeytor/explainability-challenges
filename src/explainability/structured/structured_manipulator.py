import random
from typing import List, Optional

import numpy as np
import pandas as pd


class StructuredManipulator:
    """
    A class for applying transformations to DataFrames representing tabular
    datasets for single-target predictive tasks.

    While a manipulator operates on a copy of a provided dataset, each
    manipulator makes a sequence of modifications in-place. Reuse of
    manipulators should be done with caution, as intermediate results will not
    be maintained.
    """

    def __init__(self, df: pd.DataFrame, label_column: str):
        """
        Initializes a new StructuredManipulator. Note that this creates a deep
        copy of the entire passed dataset.

        :param df: The dataframe to manipulate.
        :param label_column: The column corresponding to the prediction target.
            All other columns are assumed to be feature columns.
        """
        self.df = df.copy(deep=True)

        assert label_column in df.columns
        self.label_column = label_column

        self.feature_columns = [c for c in df.columns if c != label_column]

    def _validate_or_select_feature_column(self, column: Optional[str], *,
                                           dtypes: Optional[
                                               List[str]] = None) -> str:
        """
        Validate or provide a feature column argument for a transformation.

        :param column: The argument to validate as being a feature column in
            the DataFrame. If `None`, a value is selected from the available
            feature columns.
        :param dtypes: An optional list of datatypes to ensure the type of the
            column is part of.
        :return: The name of the selected column.
        """
        if column is None:
            if dtypes is None:
                columns = self.feature_columns
            else:
                columns = [c for c in self.feature_columns if
                           self.df[c].dtype in dtypes]
            column = random.choice(columns)
        else:
            if column not in self.feature_columns:
                raise ValueError("Provided column not a feature column.")
            if dtypes is not None:
                if self.df[column].dtype not in dtypes:
                    raise ValueError(
                        f"Provided column has type not in {dtypes}")
        return column

    def apply_missing_values(self, column=None, amount=0.1, value='mean'):
        column = self._validate_or_select_feature_column(column)

        idx = self.df.index.values
        np.random.shuffle(idx)
        if type(amount) == int:
            idx = idx[:amount]
        elif type(amount) == float:
            idx = idx[:int(amount * len(self.df))]
        else:
            raise ValueError("invalid type for 'amount' (not int or float)")

        if value == 'mean':
            value = self.df[column].mean()
        if type(value) == int or type(value) == float or np.issubdtype(
                type(value),
                np.number):
            self.df.loc[idx, column] = value
        else:
            raise ValueError("invalid type for 'value' (not numeric)")

        return self

    def apply_redundant_features(self, column=None, num_dups=1,
                                 dup_col_names=None):
        column = self._validate_or_select_feature_column(column)

        for dup in range(num_dups):
            if dup_col_names is not None:
                dup_col_name = dup_col_names[dup]
            else:
                dup_col_name = column + str(dup + 1)
            self.df[dup_col_name] = self.df[column]

        return self

    def apply_adhoc_categorization(self, column, num_bins, bins=None,
                                   bin_names=None, new_col_name=None):
        column = self._validate_or_select_feature_column(column,
                                                         dtypes=["float"])
        if bins is None:
            bins = np.linspace(self.df[column].min(), self.df[column].max(),
                               num=num_bins + 1)
        if new_col_name is None:
            new_col_name = column + "_new"
        self.df[new_col_name] = ""
        for i in range(num_bins):
            if bin_names is None:
                bin_name = column + '_' + str(bins[i]) + '_' + str(bins[i + 1])
            else:
                bin_name = bin_names[i]
            print(bin_name)
            self.df.loc[(self.df[column] >= bins[i]) & (
                    self.df[column] <= bins[i + 1]), new_col_name] = bin_name
        return self

    def apply_duplicate_data(self, column=None, dup_value=None):
        column = self._validate_or_select_feature_column(column,
                                                         dtypes=["object",
                                                                 "category"])
        if dup_value is None:
            dup_value = np.random.choice(self.df[column].unique())
        old_val = dup_value
        new_val = f"{old_val}_{np.random.randint(0, 1e12)}"
        old_col = self.df[column].copy()
        new_col = self.df[column].copy()
        new_col[new_col == old_val] = new_val
        self.df[column] = np.where(np.random.random(len(old_col)) < 0.5,
                                   old_col,
                                   new_col)
        return self
