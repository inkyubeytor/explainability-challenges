"""
Original file is located at
    https://colab.research.google.com/drive/1eA8vxKCCuQPzhxNmSDAFJ2nQ48PDQ2pd
"""
import torch
import torchvision
from PIL import Image
from torchattacks import PGD


# TODO Standardize image i/o data type (float, int, (0,...,255), (0, 1))
def adversarial_attack(model, image):
    label = torch.nn.functional.softmax(model(image.to("cpu"))).argmax()
    label = label.unsqueeze(dim=0)
    atk = PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
    # atk.set_normalization_used(mean=[0.485, 0.456, 0.406],
    #                            std=[0.229, 0.224, 0.225])
    return atk(image / 256, label).cpu()


def noise_attack(image, std=250):
    mean = torch.zeros(image.shape)
    std = torch.ones(image.shape) * std
    noise = torch.normal(mean, std)

    return torch.clip(image + noise, min=0, max=255)


def blur_attack(image):
    transform = torchvision.transforms.GaussianBlur(kernel_size=15)
    return transform(image) / 255


def ood_attack(ood_dataset):
    i = torch.randint(len(ood_dataset), size=(1, 1))[0]
    return ood_dataset[i] / 255


def occlusion_attack(image):
    transform = torchvision.transforms.RandomErasing(p=1)
    return transform(image) / 255


# TODO: Automatically download cat image
def dual_class_attack(image, image2_path, loc=(0, 0)):
    to_pil = torchvision.transforms.ToPILImage()
    to_tensor = torchvision.transforms.ToTensor()
    background = to_pil(image.squeeze() / 255).convert("RGBA")
    foreground = Image.open(image2_path)
    bg_size = background.size
    new_size = int(bg_size[0] / 3)
    foreground = foreground.resize((new_size, new_size))

    background.paste(foreground, (0, 0), foreground)

    return to_tensor(background)

# image = torchvision.io.read_image(
# "/content/imagenette2/val/n02102040/n02102040_1082.JPEG")\
# .float().unsqueeze(dim=0)
# ood_dataset = torchvision.io.read_image(
# "/content/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG")\
# .float().unsqueeze(dim=0)
# model = models.resnet50(pretrained=True).to(device).eval()

# test1 = adversarial_attack(model, image)
# test2 = noise_attack(image)
# test3 = blur_attack(image)
# test4 = ood_attack(ood_dataset)
# test5 = occlusion_attack(image)
# test6 = dual_class_attack(image, "/content/cat.png")

# plt.imshow(torch.moveaxis(test1.squeeze(), 0, 2))

# plt.imshow(torch.moveaxis(test2.squeeze(), 0, 2))

# plt.imshow(torch.moveaxis(test3.squeeze(), 0, 2))

# plt.imshow(torch.moveaxis(test4.squeeze(), 0, 2))

# plt.imshow(torch.moveaxis(test5.squeeze(), 0, 2))

# plt.imshow(torch.moveaxis(test6.squeeze(), 0, 2))
