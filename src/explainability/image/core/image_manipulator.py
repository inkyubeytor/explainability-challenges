from typing import Any, List, Optional, Tuple

import torch
import torchvision
from torchvision import models
from torch import Tensor
from torchattacks import PGD

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

        self.image = torch.clip(self.image + noise, min=0, max=1.0).float()

        return self, {"std": std}

    @_trace
    def blur_attack(self, kernel_size: int = 15) -> Self:
        """
        Apply a Gaussian blur to an image.

        :param kernel_size: The size of the blur kernel to be applied.
        :return: self
        """
        transform = torchvision.transforms.GaussianBlur(kernel_size=kernel_size)
        self.image = transform(self.image)

        return self, {}

    @_trace
    def occlusion_attack(self) -> Self:
        """
        Occlude a region of the image.

        :return: self
        """
        transform = torchvision.transforms.RandomErasing(p=1)
        self.image = transform(self.image)

        return self, {}

    @_trace
    def dual_class_attack(self, image: Tensor, dataset: Tensor,
                          loc: Tuple[int, int] = None) -> Self:
        """
        Create a dual-class image by injecting another image.

        :param image: The image to be modified
        :param dataset: A dataset from which to draw image to be injected
        :param loc: Location of image to be injected. Defaults to random
        :return: self
        """
        to_pil = torchvision.transforms.ToPILImage()
        to_tensor = torchvision.transforms.ToTensor()
        i = torch.randint(len(dataset), size=(1, 1))[0]
        foreground = to_pil(dataset[i].squeeze()).convert("RGBA")
        background = to_pil(image.squeeze()).convert("RGBA")
        bg_size = background.size
        new_size = int(bg_size[0] / 3)

        if loc is None:
            loc1 = torch.randint(int(2 * bg_size[0] / 3), size=(1, 1))[0]
            loc2 = torch.randint(int(2 * bg_size[0] / 3), size=(1, 1))[0]
            loc = (loc1, loc2)

        foreground = foreground.resize((new_size, new_size))
        background.paste(foreground, loc, foreground)
        background = background.convert("RGB")

        self.image = to_tensor(background)
        return self, {"dataset": dataset,
                      "loc": loc}

    @_trace
    def adversarial_attack(self, model=None) -> Self:
        """
        Apply torch adversarial attack to an image.

        :param model: The torch model to use for the attack.
        :return: self
        """
        image = self.image.unsqueeze(dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            model = (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                        .to(device).eval())
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
        ])
        input = transform(image).to(device)
        logits = model(input)
        label = torch.nn.functional.softmax(logits, dim=1).argmax()
        label = label.unsqueeze(dim=0)
        atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    
        output = atk(image, label).cpu()

        self.image = output
        return self, {}
