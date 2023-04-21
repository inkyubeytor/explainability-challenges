import math
import os
from abc import ABC, abstractmethod
from os import PathLike
from tempfile import TemporaryDirectory
from typing import Union

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from .structured_challenger import SKChallenger
from .structured_manipulator import StructuredManipulator

Path = Union[PathLike, str]


class SKExplainer(ABC):
    """
    Base class for explainers that operate on scikit-learn challengers.
    """

    @abstractmethod
    def explain_global(self, trained_model: Pipeline,
                       manipulator: StructuredManipulator,
                       path: Path) -> None:
        """
        Generates a global explanation visualization of a model on a dataset.

        :param trained_model: A trained model from a challenger trained on the
            given manipulator.
        :param manipulator: The manipulator holding the data for the challenge.
        :param path: The path to save the global explanation image to.
        """
        raise NotImplementedError

    def explain_challenge(self, c: SKChallenger,
                          path: Path) -> None:
        """
        Generates a grid of global explanations corresponding to the given
        challenge.

        :param c: The scikit-learn challenge to explain.
        :param path: A path to save the image grid.
        """
        with TemporaryDirectory() as d:
            for challenge_name in c.challenges:
                # construct individual explanations
                p = os.path.join(d, challenge_name)
                self.explain_global(c.models[challenge_name],
                                    c.challenges[challenge_name],
                                    p)

            # construct image grid, based on
            # https://stackoverflow.com/questions/20038648/
            # writting-a-file-with-multiple-images-in-a-grid

            images_dir = d
            result_figsize_resolution = 40  # 1 = 100px

            images_list = list(c.challenges.keys())
            images_count = len(images_list)

            # Calculate the grid size:
            grid_size = math.ceil(math.sqrt(images_count))

            # Create plt plot:
            fig, axes = plt.subplots(grid_size, grid_size,
                                     figsize=(result_figsize_resolution,
                                              result_figsize_resolution))

            for i, image_filename in enumerate(images_list):
                x_position = i % grid_size
                y_position = i // grid_size

                plt_image = plt.imread(os.path.join(images_dir,
                                                    images_list[i]))
                axes[x_position, y_position].imshow(plt_image)

            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
            plt.savefig(path)
