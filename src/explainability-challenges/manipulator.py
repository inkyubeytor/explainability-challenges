import pandas as pd
import numpy as np


class Manipulator:

    def __init__(self, df: pd.Data, label_column: str):
        self.df = df
        self.label_column = label_column
        self.feature_columns = [c for c in df.columns if c != label_column]

    def apply_missing_values(self, column=None, amount=0.1, value='mean'):
        df = self.df.copy()
        if column is None:  # randomly select a column
            column = np.random.choice(self.feature_columns)

        idx = df.index.values
        np.random.shuffle(idx)
        if type(amount) == int:
            idx = idx[:amount]
        elif type(amount) == float:
            idx = idx[:int(amount * len(df))]
        else:
            raise ValueError("invalid type for 'amount' (not int or float)")

        if value == 'mean':
            value = df[column].mean()
        if type(value) == int or type(value) == float or np.issubdtype(type(value),
                                                                       np.number):
            df.loc[idx, column] = value
        else:
            raise ValueError("invalid type for 'value' (not numeric)")

        return Manipulator(df, self.label_column)

    def apply_redundant_features(self, column=None, num_dups=1, dup_col_names=None):
        df = self.df.copy()
        if column is None:  # randomly select a column
            column = np.random.choice(self.feature_columns)

        for dup in range(num_dups):
            if dup_col_names is not None:
                dup_col_name = dup_col_names[dup]
            else:
                dup_col_name = column + str(dup+1)
            df[dup_col_name] = df[column]

        return Manipulator(df, self.label_column)

    def apply_adhoc_categorization(self, column, num_bins, bins=None, bin_names=None, new_col_name=None):
        df = self.df.copy()
        if bins is None:
            bins = np.linspace(df[column].min(), df[column].max(), num=num_bins+1)
        if new_col_name is None:
            new_col_name = column + "_new"
        df[new_col_name] = ""
        for i in range(num_bins):
            if bin_names is None:
              bin_name = column + '_' + str(bins[i]) + '_' + str(bins[i+1])
            else:
              bin_name = bin_names[i]
            print(bin_name)
            df.loc[(df[column] >= bins[i]) & (df[column] <= bins[i+1]), new_col_name] = bin_name
        return Manipulator(df, self.label_column)

    def apply_duplicate_data(self, column=None, dup_value=None):
        df = self.df.copy()
        cols = [c for c in self.feature_columns if df[c].dtype in ["object", "category"]]
        if column is not None and column not in cols:
            raise ValueError("Provided column is not categorical")
        if column is None:  # randomly select a column
            column = np.random.choice(cols)
            print(column)
        if dup_value is None:
            dup_value = np.random.choice(df[column].unique())
        old_val = dup_value
        new_val = f"{old_val}_{np.random.randint(0, 1e12)}"
        old_col = df[column].copy()
        new_col = df[column].copy()
        new_col[new_col == old_val] = new_val
        df[column] = np.where(np.random.random(len(old_col)) < 0.5, old_col, new_col)
        return Manipulator(df, self.label_column)
