from datasets import load_dataset
from torchvision import transforms
from src.explainability.image.image_attacks \
    import add_noise, add_blur, \
    add_occlusion, add_dual_class, adversarial_attack

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = load_dataset("frgfm/imagenette",
                "full_size",
                split="validation",
                streaming=True)

img = transform(next(iter(dataset))['image'])

class TestImageManipulator:
    def test_add_noise(self):
        noisy_img = add_noise(img)

        assert noisy_img.type() == 'torch.FloatTensor'
        assert noisy_img.max().item() <= 1.0000
        assert noisy_img.min().item() >= 0.0
        assert (img == noisy_img).all().item() is False

    def test_add_blur(self):
        blurry_img = add_blur(img)

        assert blurry_img.type() == 'torch.FloatTensor'
        assert blurry_img.max().item() <= 1.0000
        assert blurry_img.min().item() >= 0.0
        assert (img == blurry_img).all().item() is False

    def test_add_occlusion(self):
        occlueded_img = add_occlusion(img)

        assert occlueded_img.type() == 'torch.FloatTensor'
        assert occlueded_img.max().item() <= 1.0000
        assert occlueded_img.min().item() >= 0.0
        assert (img == occlueded_img).all().item() is False

    def test_add_dual_class(self):
        dual_dataset = load_dataset("frgfm/imagewoof",
                                    "full_size",
                                    split="validation",
                                    streaming=True)

        dual_img = transform(next(iter(dual_dataset))['image'])
        dual_class_img = add_dual_class(img, dual_img)

        assert dual_class_img.type() == 'torch.FloatTensor'
        assert dual_class_img.max().item() <= 1.0000
        assert dual_class_img.min().item() >= 0.0
        assert (img == dual_class_img).all().item() is False

    def test_adversarial_attack(self):
        adversarial_img = adversarial_attack(img)

        assert adversarial_img.type() == 'torch.FloatTensor'
        assert adversarial_img.max().item() <= 1.0000
        assert adversarial_img.min().item() >= 0.0
        assert (img == adversarial_img).all().item() is False
