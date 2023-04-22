from typing import Any, List, Optional

import torch
import torchvision
from PIL import Image
from torch import Tensor

from ...lib.trace import Trace, trace as _trace

# hack to support Python 3.10
# for Python 3.11 and higher, import Self from typing
Self = Any


class ImageManipulator:
    """
    A class for applying transformations to images.

    While a manipulator operates on a copy of the input image, each
    manipulator makes a sequence of modifications in-place. Reuse of
    manipulators should be done with caution, as intermediate results will not
    be maintained.
    """

    def __init__(self, image: Tensor,
                 random_state: Optional[int] = None) -> None:
        """
        Initializes a new StructuredManipulator. Note that this creates a deep
        copy of the entire passed dataset.

        :param image: The image to manipulate.
        :param random_state: A random state for reproducibility in stochastic
            operations.
        :return: None
        """
        self.random_state = random_state

        self.image = torch.clone(image)

        self._traces: List[Trace] = []

    @property
    def trace(self) -> str:
        """
        Returns a traceback of all operations performed on the manipulator.

        :return: Newline-separated traceback of all operations performed.
        """
        return "\n".join(str(t) for t in self._traces)

    @_trace
    def noise_attack(self, std: float = 250.0) -> Self:
        """
        Apply random noise to an image.

        :param std: The standard deviation of the Gaussian noise.
        :return: self
        """
        mean = torch.zeros(self.image.shape)
        std = torch.ones(self.image.shape) * std
        noise = torch.normal(mean, std)

        self.image = torch.clip(self.image + noise, min=0, max=255).int()

        return self, {"std": std}

    @_trace
    def blur_attack(self) -> Self:
        """
        Apply a Gaussian blur to an image.

        :return: self
        """
        transform = torchvision.transforms.GaussianBlur(kernel_size=15)
        self.image = transform(self.image).int()

        return self, {}

    @_trace
    def occlusion_attack(self) -> Self:
        """
        Occlude a region of the image.

        :return: self
        """
        transform = torchvision.transforms.RandomErasing(p=1)
        self.image = transform(self.image).int()

        return self, {}

    @_trace
    def dual_class_attack(self, image2_path: str) -> Self:
        """
        Create a dual-class image by injecting another image.

        :param image2_path: The path to the image to inject.
        :return: self
        """
        to_pil = torchvision.transforms.ToPILImage()
        to_tensor = torchvision.transforms.ToTensor()
        background = to_pil(self.image.squeeze() / 255).convert("RGBA")
        foreground = Image.open(image2_path)
        bg_size = background.size
        new_size = int(bg_size[0] / 3)
        foreground = foreground.resize((new_size, new_size))

        background.paste(foreground, (0, 0), foreground)

        background = background.convert("RGB")

        self.image = to_tensor(background).unsqueeze(dim=0)

        return self, {"image2_path": image2_path}
