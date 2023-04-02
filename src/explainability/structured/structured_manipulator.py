import random
from typing import List, Literal, Optional, Self, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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

    @property
    def _feature_columns(self):
        """
        :return: Dynamically updating list of feature columns.
        """
        return [c for c in self.df.columns if c != self.label_column]

    def _validate_or_select_feature_column(self, column: Optional[str], *,
                                           dtypes: Optional[Union[
                                               List[str],
                                               Literal["numeric"]
                                           ]] = None) -> str:
        """
        Validate or provide a feature column argument for a transformation.

        :param column: The argument to validate as being a feature column in
            the DataFrame. If `None`, a value is selected from the available
            feature columns.
        :param dtypes: An optional specifier for what datatypes the selected
            column should be. If `"string"` or `"numeric"`, uses the associated
            pandas function for checking the dtype. If a list, explicitly
            checks that the dtype of the column is in the list.
        :return: The name of the selected column.
        """
        if column is None:
            if dtypes is None:
                columns = self._feature_columns
            elif dtypes == "numeric":
                columns = [c for c in self._feature_columns if
                           is_numeric_dtype(self.df[c])]
            else:
                columns = [c for c in self._feature_columns if
                           self.df[c].dtype in dtypes]
            column = random.choice(columns)
        else:
            if column not in self._feature_columns:
                raise ValueError("Provided column not a feature column.")

            if dtypes == "numeric" and not is_numeric_dtype(self.df[column]):
                raise ValueError("Provided column not a numeric column.")
            elif type(dtypes) == List and self.df[column].dtype not in dtypes:
                raise ValueError(f"Provided column has type not in {dtypes}")

        return column

    def replace_random_values(self, column: Optional[str] = None,
                              proportion: float = 1.0,
                              value: Optional[float] = None) -> Self:
        """
        Replace a random subset of values in a numeric feature column according
            to the provided scheme. If `value` is given, the replacement value
            is the provided value. If not, the replacement value is the mean of
            the values of that column.
        :param column: The name of the column to manipulate.
        :param proportion: The proportion of values in the column to replace.
        :param value: The replacement value to use.
        :return: self
        """
        column = self._validate_or_select_feature_column(column,
                                                         dtypes="numeric")

        idx = self.df.index.values
        np.random.shuffle(idx)
        idx = idx[:int(proportion * len(self.df))]

        if value is not None:
            value = self.df[column].mean()
        self.df.loc[idx, column] = value

        return self

    def duplicate_features(self, column: Optional[str] = None,
                           num_dups: int = 1,
                           dup_col_names: Optional[List[str]] = None) -> Self:
        """
        Creates duplicates of feature columns.

        :param column: The column to duplicate. If no column is provided, a
            random feature column will be duplicated.
        :param num_dups: The number of duplicate columns to make.
        :param dup_col_names: Optional list of names to use for the duplicated
            columns. If provided, must have length equal to `num_dups`.
        :return: self
        """
        column = self._validate_or_select_feature_column(column)

        if dup_col_names is not None and len(dup_col_names) != num_dups:
            raise ValueError("Invalid number of column names provided.")

        for dup in range(num_dups):
            if dup_col_names is not None:
                dup_col_name = dup_col_names[dup]
            else:
                dup_col_name = column + str(dup + 1)
            self.df[dup_col_name] = self.df[column]

        return self

    def categorize(self, column: Optional[str] = None,
                   num_bins: int = 2,
                   bins: Optional[np.array] = None,
                   bin_names: Optional[
                       List[str]] = None) -> Self:
        """
        Applies binning to a numeric column.
        :param column: The column to bin.
        :param num_bins: The number of bins to make.
        :param bins: The boundaries for the bins.
        :param bin_names: Category labels for each bin.
        :return: self
        """

        column = self._validate_or_select_feature_column(column,
                                                         dtypes="numeric")
        if bins is None:
            bins = np.linspace(self.df[column].min(), self.df[column].max(),
                               num=num_bins + 1)
        elif len(bins) - 1 != num_bins:
            raise ValueError(
                f"Number of bin boundaries ({len(bins)} provided) must be"
                f"one more than number of bins ({num_bins} provided).")

        if bin_names is not None and len(bin_names) != len(bins) - 1:
            raise ValueError(
                f"{len(bin_names)} labels provided for {len(bins) - 1} bins.")

        if bin_names is not None:
            self.df[column] = pd.cut(self.df[column], bins, labels=bin_names)
        else:
            self.df[column] = pd.cut(self.df[column], bins)

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
