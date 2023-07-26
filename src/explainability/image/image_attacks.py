import sys
import torch
from torchvision import models
from .core.image_manipulator import ImageManipulator

sys.path.insert(0, '..')


# TODO Standardize image i/o data type (float, int, (0,...,255), (0, 1))

def adversarial_attack(image, model=None):
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(pretrained=True).to(device).eval()
    return ImageManipulator(image).adversarial_attack(model).image


def noise_attack(image, std=250):
    return ImageManipulator(image).noise_attack(std=std).image


def blur_attack(image):
    return ImageManipulator(image).blur_attack().image


def ood_attack(ood_dataset):
    i = torch.randint(len(ood_dataset), size=(1, 1))[0]
    return (ood_dataset[i])


def occlusion_attack(image):
    return ImageManipulator(image).occlusion_attack().image


def dual_class_attack(image, dataset, loc=None):
    return ImageManipulator(image).dual_class_attack(image, dataset, loc).image
