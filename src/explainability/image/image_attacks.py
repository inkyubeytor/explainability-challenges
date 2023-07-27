import sys
import torch
from torchvision import models
from .core.image_manipulator import ImageManipulator

sys.path.insert(0, '..')


def adversarial_attack(image, model=None):
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(pretrained=True).to(device).eval()
    return ImageManipulator(image).adversarial_attack(model).image


def noise_attack(image, std=250):
    return ImageManipulator(image).noise_attack(std=std).image


def blur_attack(image, kernel_size=15):
    return ImageManipulator(image).blur_attack(kernel_size=kernel_size).image


def ood_attack(ood_dataset):
    i = torch.randint(len(ood_dataset), size=(1, 1))[0]
    return (ood_dataset[i])


def occlusion_attack(image):
    return ImageManipulator(image).occlusion_attack().image


def dual_class_attack(image, dataset, loc=None):
    return ImageManipulator(image).dual_class_attack(image, dataset, loc).image
