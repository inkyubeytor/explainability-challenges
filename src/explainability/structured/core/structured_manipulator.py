import pprint
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, ParamSpec, \
    Self, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

P = ParamSpec("P")
R = TypeVar("R")
TraceData = Dict[str, Any]


def _trace(f: Callable[P, Tuple[R, TraceData]]) -> Callable[P, R]:
    """
    Wraps all transformations to store a record of the undertaken action.

    :param f: The method to trace.
    :return: A function that extracts the trace data and returns the
        expected output.
    """

    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        out, trace_data = f(self, *args, **kwargs)
        self._traces.append(Trace(operation=f.__name__, data=trace_data))
        return out

    return _wrapper


@dataclass
class Trace:
    operation: str
    data: TraceData

    def __str__(self):
        return "\n".join(["", self.operation, pprint.pformat(self.data), ""])


class StructuredManipulator:
    """
    A class for applying transformations to DataFrames representing tabular
    datasets for single-target predictive tasks.

    While a manipulator operates on a copy of a provided dataset, each
    manipulator makes a sequence of modifications in-place. Reuse of
    manipulators should be done with caution, as intermediate results will not
    be maintained.
    """

    def __init__(self, df: pd.DataFrame, label_column: str,
                 random_state: Optional[int] = None) -> None:
        """
        Initializes a new StructuredManipulator. Note that this creates a deep
        copy of the entire passed dataset.

        :param df: The dataframe to manipulate.
        :param label_column: The column corresponding to the prediction target.
            All other columns are assumed to be feature columns.
        :param random_state: A random state for reproducibility in stochastic
            operations.
        :return: None
        """
        self.random_state = random_state

        self.df = df.copy(deep=True) \
            .sample(frac=1, random_state=self.random_state) \
            .reset_index(drop=True)

        self._traces: List[Trace] = []

        assert label_column in df.columns
        self.label_column = label_column

    @property
    def trace(self):
        return "\n".join(str(t) for t in self._traces)

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

            rng = np.random.default_rng(seed=self.random_state)
            column = rng.choice(columns)
        else:
            if column not in self._feature_columns:
                raise ValueError("Provided column not a feature column.")

            if dtypes == "numeric" and not is_numeric_dtype(self.df[column]):
                raise ValueError("Provided column not a numeric column.")
            elif type(dtypes) == List and self.df[column].dtype not in dtypes:
                raise ValueError(f"Provided column has type not in {dtypes}")

        return column

    @_trace
    def replace_random_values(self, column: Optional[str] = None,
                              proportion: float = 0.5,
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

        if proportion < 0 or proportion > 1:
            raise ValueError("Provided proportion is not between 0 and 1.")

        idx = self.df.index.values
        rng = np.random.default_rng(seed=self.random_state)
        rng.shuffle(idx)
        idx = idx[:int(proportion * len(self.df))]

        if value is None:
            value = self.df[column].mean()
        self.df.loc[idx, column] = value

        return self, {"column": column,
                      "proportion": proportion,
                      "value": value}

    @_trace
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

        if num_dups < 1:
            raise ValueError("Number of duplicates must be at least 1.")

        if dup_col_names is not None and len(dup_col_names) != num_dups:
            raise ValueError("Invalid number of column names provided.")

        dup_col_names = dup_col_names if dup_col_names is not None else \
            [column + str(dup + 1) for dup in range(num_dups)]

        for dup_col_name in dup_col_names:
            self.df[dup_col_name] = self.df[column]

        return self, {"column": column,
                      "num_dups": num_dups,
                      "dup_col_names": dup_col_names}

    @_trace
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
        if num_bins < 2:
            raise ValueError("Num bins must be greater than 1.")
        if self.df[column].min() == self.df[column].max():
            raise ValueError("Column has only one value.")
        if bins is None:
            epsilon = 1e-12
            bins = np.linspace(self.df[column].min() - epsilon,
                               self.df[column].max() + epsilon,
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
            self.df[column] = pd.cut(self.df[column], bins, duplicates="drop")

        bin_names = self.df[column].unique()

        return self, {"column": column,
                      "num_bins": num_bins,
                      "bins": bins,
                      "bin_names": bin_names}

    @_trace
    def split_category_value(self, column: Optional[str] = None,
                             dup_value: Optional[str] = None,
                             proportion: float = 0.5) -> Self:
        """
        Splits a category for a categorical feature by creating a new category
        and reassigning the feature for approximately half of the rows with
        the original feature value to the new feature value.

        :param column: The categorical column to split a feature on.
        :param dup_value: Optional specification of the particular feature
            value to split into two feature values.
        :param proportion: The proportion of values to replace with new values.
        :return: self
        """

        column = self._validate_or_select_feature_column(column,
                                                         dtypes=["object",
                                                                 "category"])

        values = list(self.df[column].unique())
        rng = np.random.default_rng(seed=self.random_state)
        if dup_value is None:
            dup_value = rng.choice(values)
        elif dup_value not in values:
            raise ValueError("Chosen value not in column.")

        old_val = dup_value
        new_val = f"{old_val}_{rng.integers(0, 1e6)}"
        values.append(new_val)
        new_col = pd.Categorical(self.df[column], categories=values)
        flags = (new_col == old_val) & (rng.random(len(new_col)) < proportion)
        new_col[flags] = new_val
        self.df[column] = new_col

        return self, {"column": column,
                      "dup_value": dup_value,
                      "new_value": new_val}

    @_trace
    def sort_values(self, column: Optional[str] = None,
                    ascending: bool = True) -> Self:
        """
        Sort the internal DataFrame by a column. This sort order is maintained
        when creating train-test splits, so sorting attempts to introduce
        out of distribution values in the given feature at test time.

        :param column: The column to sort on.
        :param ascending: Whether the sort should be ascending or descending.
        :return: self
        """
        column = self._validate_or_select_feature_column(column)
        self.df.sort_values(by=column, ascending=ascending, inplace=True)

        return self, {"column": column, "ascending": ascending}

    def train_test_split(self, test_proportion: float = 0.2) -> \
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Partition the data into training and testing data according to the
        given proportion.

        :param test_proportion: The proportion of data to use for testing.
        :return: Tuple (X_train, y_train, X_test, y_test) partitioning the
            manipulated DataFrame.
        """
        num_test = int(test_proportion * len(self.df))
        num_train = len(self.df) - num_test

        train_df = self.df.iloc[:num_test, :]
        x_train = train_df.loc[:, train_df.columns != self.label_column]
        y_train = train_df[self.label_column]

        test_df = self.df.iloc[num_train:, :]
        x_test = test_df.loc[:, test_df.columns != self.label_column]
        y_test = test_df[self.label_column]

        return x_train, y_train, x_test, y_test
