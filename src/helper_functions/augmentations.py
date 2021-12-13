from PIL import ImageFilter
from torchvision import transforms
import random
import numpy as np
imageSize = 224
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
from facenet_pytorch import fixed_image_standardization

class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_aug_trnsform(imageSize = 80):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(imageSize, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    
    return transform


def get_eval_trnsform(img_size=imageSize):

    transform = transforms.Compose(
        [
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization,
        ]
    )
    return transform
