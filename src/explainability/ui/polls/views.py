import base64
from io import BytesIO
from random import randrange

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from torchvision.models import alexnet, resnet50

from explainability.image.image_attacks import blur_attack, dual_class_attack, \
    noise_attack, occlusion_attack
from explainability.image.image_explanations import eigen_cam, grad_cam, \
    guided_backprop

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# TODO: Modify to include function
attack_dict = {"noise": 1,
               "blur": 2,
               "occlusion": 3,
               "ood": 4,
               "dual": 5,
               "adversarial": 99,
               "none": 0}

testset = torchvision.datasets.ImageFolder(
    root='/Users/jrast/Downloads/imagenette2/val', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)


def start(request):
    request.session['truth'] = []
    request.session['response'] = []
    return redirect('graphic')


@csrf_exempt
def submit(request):
    if 'truth' not in request.session:
        request.session['truth'] = []

    if 'response' not in request.session:
        request.session['response'] = []

    try:
        data = request.POST['previous']
    except KeyError:  # is this sufficient?
        # Warning! Failing silently.
        # TODO: Proper error handling
        return redirect("graphic")

    response, truth = (data.split('-'))
    request.session['truth'].append(truth)
    request.session['response'].append(response)

    if len(request.session['truth']) >= 10:
        return redirect("result")

    # My code breaks if I do not include this and I have no idea why
    request.session['data'] = 8

    return redirect('graphic')


def result(request):
    correct = 0
    incorrect = 0

    response = request.session['response']
    truth = request.session['truth']

    for i in range(len(response)):
        if truth[i] == response[i]:
            correct += 1
        else:
            incorrect += 1

    return render(request, 'polls/result.html', {"correct": correct,
                                                 "incorrect": incorrect})


def graphic(request):
    # image = read_image("/Users/jrast/Downloads/test_img.JPEG")\
    # .float().unsqueeze(dim=0)
    # image = dls[0].one_batch()[0]
    # Handle state data

    image = next(iter(testloader))[0]
    image = (image * 255).int().float()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    attack = randrange(0, 8)
    print(attack)

    if attack == 0:
        test2 = noise_attack(image)
    elif attack == 1:
        test2 = blur_attack(image)
    elif attack == 2:
        test2 = occlusion_attack(image)
    elif attack == 3:
        # test2 = ood_attack(ood_dataset)
        test2 = image
    elif attack == 4:
        test2 = dual_class_attack(image, "/Users/jrast/Downloads/cat.png") * 255
    else:
        test2 = image

    # random = randrange(0, 2)
    random = 0

    # print(random)

    if random == 0:
        # visualization = eigen_cam(test2, test2, model)
        visualization = eigen_cam(test2, image, model)
    elif random == 1:
        model = resnet50(pretrained=True)
        visualization = grad_cam(test2, model)
    else:
        model = alexnet(pretrained=True)
        pos, neg = guided_backprop(image, model)
        # import pdb; pdb.set_trace()
        visualization = np.moveaxis(pos, 0, 2)

    buffer = BytesIO()
    image_out = visualization
    im = Image.fromarray(image_out)
    im.save(buffer, "PNG")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    return render(request, 'polls/graphic.html', {'graphic': graphic,
                                                  "mapping": attack_dict,
                                                  "data":
                                                      request.session['truth'],
                                                  "data2":
                                                      request.session[
                                                          'response'],
                                                  'truth': attack})
