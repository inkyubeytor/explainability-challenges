from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator, clone

from .structured_manipulator import StructuredManipulator


class SKChallenger(ABC):
    """
    Base class for challenges to explainability methods for supervised learning
    tasks using structured tabular data and scikit-learn models.
    """

    def __init__(self, model: BaseEstimator,
                 df: pd.DataFrame, label_column: str,
                 random_state: Optional[int] = None) -> None:
        """
        :param model: The model to implement challenges on.
        :param df: The dataset to implement challenges on.
        :param label_column: The target prediction column for the dataset.
        :param random_state: A random state for reproducibility in stochastic
            operations.
        :return: None
        """
        self.df = df
        self.label_column = label_column
        self.random_state = random_state
        self._model = model
        self.challenges: Dict[str, StructuredManipulator] = {
            "base": StructuredManipulator(df, label_column, random_state)
        }
        self.models: Dict[str, BaseEstimator] = {}

    @abstractmethod
    def generate_challenges(self) -> None:
        """
        Abstract method for populating `self.challenges` with
        `StructuredManipulator` objects representing each modified dataset
        for the challenge.

        :return: None
        """
        raise NotImplementedError

    def train_models(self) -> None:
        """
        Train a model on each dataset for the challenge.

        :return: None
        """
        for challenge_name, sm in self.challenges.items():
            x, y, _, _ = sm.train_test_split()
            self.models[challenge_name] = clone(self._model).fit(x, y)
