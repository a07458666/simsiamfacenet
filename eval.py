import os
import argparse
import torch
import numpy as np
import math
import glob
from os import listdir
from os import walk
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

from src.data_loading.data_loader import BirdImageLoader
from src.txt_loading.txt_loader import (
    readClassIdx,
    readTrainImages,
    splitDataList,
)
from src.helper_functions.augmentations import get_eval_trnsform


def main(args):
    device = checkGPU()
    class_to_idx = readClassIdx(args)
    data_list = readTrainImages(args)
    _, val_data_list, _ = splitDataList(data_list)
    model = loadModel(args, device)
    trans = get_eval_trnsform()
    loader = create_dataloader(args, val_data_list, class_to_idx, trans)
    val_loss, val_acc_top1, val_acc_top5 = eval_model(
        args, model, loader, device
    )
    print(
        "val_loss ",
        val_loss,
        "val_acc_top1",
        val_acc_top1,
        "val_acc_top5 ",
        val_acc_top5,
    )


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
    for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pass_epoch(model, loader, device, mode="Train"):
    loss = 0
    loss_acc_top1 = 0
    loss_acc_top5 = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for i_batch, image_batch in tqdm(enumerate(loader)):
        x, y = image_batch[0].to(device), image_batch[1].to(device)
        if mode == "Train":
            model.train()
        elif mode == "Eval":
            model.eval()
        else:
            print("error model mode!")
        y_pred = model(x)

        loss_batch = loss_fn(y_pred, y)
        loss_batch_acc_top = accuracy(y_pred, y, topk=(1, 5))

        if mode == "Train":
            model_optimizer.zero_grad()
            loss_batch.backward()
            model_optimizer.step()

        loss += loss_batch.detach().cpu()
        loss_acc_top1 += loss_batch_acc_top[0]
        loss_acc_top5 += loss_batch_acc_top[1]

    loss /= i_batch + 1
    loss_acc_top1 /= i_batch + 1
    loss_acc_top5 /= i_batch + 1

    return loss, loss_acc_top1, loss_acc_top5


def eval_model(args, model, loader, device):
    with torch.no_grad():
        val_loss, val_acc_top1, val_acc_top5 = pass_epoch(
            model, loader, device, "Eval"
        )
        torch.cuda.empty_cache()
    return (
        val_loss.to("cpu").numpy(),
        val_acc_top1.to("cpu").numpy(),
        val_acc_top5.to("cpu").numpy(),
    )


def create_dataloader(args, data_list, class_to_idx, trans):
    dataset_test = BirdImageLoader(
        args.data_path, data_list, class_to_idx, transform=trans
    )

    loader = DataLoader(
        dataset_test,
        num_workers=args.batch_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    return loader


def loadModel(args, device):
    with torch.no_grad():
        model = torch.load(args.model_path)
        model.eval().to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="310551010 eval bird")
    parser.add_argument(
        "--data_path", type=str, default="../../dataset/bird_datasets/train"
    )
    parser.add_argument(
        "--classes_path",
        type=str,
        default="../../dataset/bird_datasets/classes.txt",
    )
    parser.add_argument(
        "--training_labels_path",
        type=str,
        default="../../dataset/bird_datasets/training_labels.txt",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/model_bird_vit_AllData/checkpoint.pth.tar",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    args = parser.parse_args()
    main(args)
