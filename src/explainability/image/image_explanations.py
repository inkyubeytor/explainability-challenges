"""
Original file is located at
    https://colab.research.google.com/drive/1hHwBaMwJG86Q3Uaqx1hJJcjmMXLx4umt
"""

# https://github.com/jacobgil/pytorch-grad-cam


import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import \
    fasterrcnn_reshape_transform
from torch.autograd import Variable
from torch.nn import ReLU


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class GuidedBackprop:
    """
   Produces gradients generated with guided back propagation from the given
   image.
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(_, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(_, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return modified_grad_out,

        def relu_forward_hook_function(_, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def __preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # Mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as _:
            print("could not transform PIL_img to a PIL Image object."
                  "Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def __get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


"""# Eigen CAM Helper Code"""

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle',
              'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
              'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
              'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
              'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
              'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam,
                                      labels, classes):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np,
                                                    renormalized_cam,
                                                    use_rgb=True)
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes,
                                           eigencam_image_renormalized)
    return image_with_bounding_boxes


"""# Explainations"""


# Question: Does this only work with resnet50?

def grad_cam(image, model):
    args = DotDict({"use_cuda": True})
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers,
                  use_cuda=args.use_cuda)
    targets = None
    grayscale_cam = cam(input_tensor=image, targets=targets)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        np.moveaxis(image.numpy()[0] / 255, 0, -1), grayscale_cam, use_rgb=True)

    return visualization


def guided_backprop(image, model):
    to_pil = torchvision.transforms.ToPILImage()
    to_tensor = torchvision.transforms.ToTensor()

    class_int = torch.nn.functional.softmax(model(image), dim=1).argmax()

    original_image = to_pil(image.squeeze() / 255).convert("RGB")
    prep_img = __preprocess_image(original_image)

    GBP = GuidedBackprop(model)

    guided_grads = GBP.generate_gradients(prep_img, class_int)

    return __get_positive_negative_saliency(guided_grads)


def eigen_cam(image, model):
    image = torch.moveaxis(image.squeeze().int(), 0, 2).numpy().astype('uint8')
    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    input_tensor = transform(image)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)

    # Run the model and display the detections
    boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
    image = draw_boxes(boxes, labels, classes, image)

    target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(model,
                   target_layers,
                   use_cuda=torch.cuda.is_available(),
                   reshape_transform=fasterrcnn_reshape_transform)

    grayscale_cam = cam(input_tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    # cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    # # And let's draw the boxes again:
    # image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    # Image.fromarray(image_with_bounding_boxes)

    return renormalize_cam_in_bounding_boxes(boxes, image_float_np,
                                             grayscale_cam, labels, classes)


"""# Testing"""

# image = read_image("/content/test_img.JPEG").float().unsqueeze(dim=0)
# model = resnet50(pretrained=True)

# visualization = grad_cam(image, model)

# plt.imshow(visualization)

# image = torchvision.io.read_image("/content/test_img.JPEG")\
# .float().unsqueeze(dim=0)
# model = models.alexnet(pretrained=True)
# pos, neg = guided_backprop(image, model)

# plt.imshow(np.moveaxis(pos.squeeze(), 0, 2))

# plt.imshow(np.moveaxis(neg.squeeze(), 0, 2))

# image = read_image("/content/test_img.JPEG")\
# .float().unsqueeze(dim=0).to("cpu")
# model = torchvision.models.detection\
# .fasterrcnn_resnet50_fpn(pretrained=True).to("cpu")
# model.eval().to("cpu")

# visualization = eigen_cam(image, model)

# plt.imshow(visualization)
