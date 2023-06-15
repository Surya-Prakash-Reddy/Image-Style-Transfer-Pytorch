
#importing libraries
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import skimage

import torch
from torchvision import transforms, models

REPO_ROOT = Path(__file__).parent
DST_DIR = REPO_ROOT / "output"
DST_DIR.mkdir(exist_ok=True)

def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image

#function to convert image from it's normalised form to back to regular form
def imconvert(tensor):
    tensor = tensor.cpu().clone().detach()
    tensor = tensor.numpy().squeeze()
    tensor = tensor.transpose(1,2,0)
    tensor = tensor * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    tensor = tensor.clip(0,1)
    return tensor


# defining the function to get layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1', '21': 'conv4_2'}

    features = {}
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image

    return features


# calculating the gram matrix
def gram_matrix(tensor):
    batch_size, depth, height, width = tensor.shape

    tensor = tensor.view(depth, -1)
    tensor = torch.mm(tensor, tensor.t())
    return tensor


def style_convert(style_image: Path, content_image: Path):
    vgg = models.vgg19(pretrained=True).features

    for param in vgg.parameters():
        param.requires_grad_(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Cuda Available: ', torch.cuda.is_available())
    vgg.to(device)

    #load content image
    content = load_image(str(content_image)).to(device)
    [height, width] = cv2.imread(str(content_image)).shape[:2]

    #load style image
    style = load_image(str(style_image), shape=content.shape[-2:]).to(device)

    print("loaded images")

    style_features = get_features(style, vgg)
    content_features = get_features(content, vgg)

    print("done get_features()")

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    #we could start with random image, but it would be good to start with content image
    target = content.clone().requires_grad_(True).to(device)


    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.8,
                     'conv3_1': 0.5,
                     'conv4_1': 0.3,
                     'conv5_1': 0.1}


    content_weight = 1  # alpha
    style_weight = 5e6  # beta


    optimizer = torch.optim.Adam([target], lr=0.003)

    steps = 200
    print_every = 40

    print("going to convert images")

    for i in range(1,steps+1):

        target_features = get_features(target, vgg)
        content_loss = torch.mean((content_features['conv4_2']-target_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer]*torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss/ (d*h*w)

        total_loss = style_weight*style_loss + content_weight*content_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % print_every==0:
            print(f'{i}: Total Loss: ', total_loss.item())
            dst_name0 = DST_DIR / f"{content_image.stem}_{i:05d}.jpg"
            dst_name = DST_DIR / f"{content_image.stem}_{i:05d}_original_size.jpg"
            converted = imconvert(target)
            skimage.io.imsave(str(dst_name0), converted)
            resized = cv2.resize(converted, (width, height), interpolation=cv2.INTER_LINEAR)
            skimage.io.imsave(str(dst_name), resized)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("style transfer")
    parser.add_argument("style", help="style image")
    parser.add_argument("content", help="image to style transfer")
    args = parser.parse_args()

    style_image = Path(args.style)
    content_image = Path(args.content)

    style_convert(style_image, content_image)
