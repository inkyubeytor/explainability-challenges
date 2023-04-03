from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, TypeVar

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from .structured_manipulator import StructuredManipulator

T = TypeVar("T")


class _StructuredChallenger(Generic[T], ABC):
    """
    Base class for challenges to explainability methods for supervised learning
    tasks using structured tabular data.
    """

    def __init__(self, model: T, df: pd.DataFrame, label_column: str,
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
        self.model = model
        self.challenges: Dict[str, StructuredManipulator] = {
            "base": StructuredManipulator(df, label_column, random_state)
        }
        self.models: Dict[str, T] = {}

    @abstractmethod
    def generate_challenges(self) -> None:
        """
        Abstract method for populating `self.challenges` with
        `StructuredManipulator` objects representing each modified dataset
        for the challenge.

        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def train_models(self) -> None:
        """
        Abstract method for training a model on each dataset for the challenge.

        :return: None
        """
        raise NotImplementedError


class SKClassifierChallenger(_StructuredChallenger[ClassifierMixin]):
    """
    `StructuredChallenger` specifically for scikit-learn classification models.
    """

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
        Trains a model on each dataset for the challenge.

        :return: None
        """
        # TODO: implement for sklearn classifiers
        raise NotImplementedError


class SKRegressorChallenger(_StructuredChallenger[RegressorMixin]):
    """
    `StructuredChallenger` specifically for scikit-learn regression models.
    """

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
        Trains a model on each dataset for the challenge.

        :return: None
        """
        # TODO: implement for sklearn regressors
        raise NotImplementedError
