import sys
import torch
from torchvision import models
from .core.image_manipulator import ImageManipulator

sys.path.insert(0, '..')


def adversarial_attack(image, model=None):
    return ImageManipulator(image).adversarial_attack(model).image


def add_noise(image, std=0.98):
    return ImageManipulator(image).noise_attack(std=std).image


def add_blur(image, kernel_size=15):
    return ImageManipulator(image).blur_attack(kernel_size=kernel_size).image


def get_ood(ood_dataset):
    i = torch.randint(len(ood_dataset), size=(1, 1))[0]
    return (ood_dataset[i])


def add_occlusion(image):
    return ImageManipulator(image).occlusion_attack().image


def add_dual_class(image, dataset, loc=None):
    return ImageManipulator(image).dual_class_attack(image, dataset, loc).image
