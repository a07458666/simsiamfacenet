import os
import argparse
import torch
import numpy as np
import PIL.Image as Image
import torchvision.models as models

from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from src.txt_loading.txt_loader import (
    readClassIdx,
    readTestImagesPath,
    splitDataList,
)
from src.helper_functions.augmentations import get_eval_trnsform


def main(args):
    device = checkGPU()
    class_to_idx = readClassIdx(args)
    test_images = readTestImagesPath(args)
    model = loadModel(args, device)
    trans = get_eval_trnsform()

    submission = []
    for img in tqdm(test_images):
        predicted_class = predict(
            args.data_path, img, trans, device, model, class_to_idx
        )
        submission.append([img, predicted_class])
    np.savetxt("answer.txt", submission, fmt="%s")


def checkGPU():
    print("torch version:" + torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Available GPUs: ", end="")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i), end=" ")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")
    return device


def loadModel(args, device):
    with torch.no_grad():
        model = torch.load(args.model_path)
        # model.load_state_dict(torch.load(args.model_path_dict)["state_dict"])
        model.eval().to(device)
    return model


def predict(root, path, trans, device, model, class_to_idx):
    img = Image.open(os.path.join(root, path)).convert("RGB")
    m = nn.Softmax(dim=1)
    data = trans(img).unsqueeze(0).to(device)
    y = model(data)
    temp = torch.argmax(m(y)).item()
    label = class_to_idx[temp]
    output = str(temp + 1).zfill(3) + "." + label
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="310551010 inference bird")
    parser.add_argument(
        "--data_path", type=str, default="../../dataset/bird_datasets/test"
    )
    parser.add_argument(
        "--classes_path",
        type=str,
        default="../../dataset/bird_datasets/classes.txt",
    )
    parser.add_argument(
        "--test_filename_path",
        type=str,
        default="../../dataset/bird_datasets/testing_img_order.txt",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/model_bird_vit_AllData/checkpoint.pth.tar",
    )
    parser.add_argument("--output", type=str, default="answer.txt")
    args = parser.parse_args()

    main(args)
