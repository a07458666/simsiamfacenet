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
from src.helper_functions.augmentations import get_eval_trnsform
from src.helper_functions.helper import checkGPU, update_loss_hist


def main(args):
    print("=====Inference=====")
    device = checkGPU()
    model = loadModel(args).to(device)
    trans = get_eval_trnsform()
    loader = create_dataloader(args, trans)
    result, target = inference_model(args, model, loader, device)

    if (args.output_foloder == ""):
        args.output_foloder = os.path.abspath(os.path.join(args.model_path, os.pardir))
        
    writerCSV(args, result, target)
    
    torch.cuda.empty_cache()

def pass_epoch(model, loader, device):
    y_pred_list = []
    y_list = []
    with torch.no_grad():
        for i_batch, image_batch in tqdm(enumerate(loader)):
            x = image_batch[0].to(device)
            y = image_batch[1]

            y_pred, _ = model(x) #model架構不同要修改
            # y_pred = model(x) #model架構不同要修改
            y_pred = y_pred.cpu().detach().numpy()
            for j, data in enumerate(y_pred):
                y_pred_list.append(data)
                y_list.append(int(y[j]))
    return torch.Tensor(y_pred_list).to(device), torch.Tensor(y_list).to(device)
    # return y_pred_list, y_list


def inference_model(args, model, loader, device):
    y_pre, y = pass_epoch(model, loader, device)
    return y_pre, y

def create_dataloader(args, trans):
    dataset_test = datasets.ImageFolder(args.data_path, transform=trans)

    loader = DataLoader(
        dataset_test,
        num_workers=args.batch_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    return loader

def loadModel(args):
    with torch.no_grad():
        model = torch.load(args.model_path).eval()
    return model

def pdist(v):
    dist = torch.norm(v[:, None] - v, dim=2, p=2)
    return dist

def writerCSV(args, results, targets):
    import csv
    filePath = os.path.join(args.output_foloder, 'result.csv')
    print("filePath", filePath)
    print("args.output_foloder", args.output_foloder)
    with open(filePath, 'w', newline='') as csvfile:

        # 以空白分隔欄位，建立 CSV 檔寫入器
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(['feature vector', 'target'])

        for result, target in zip(results, targets):
            writer.writerow([str(result.tolist()), str(target.tolist())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument(
        "--data_path", type=str, default="../../dataset/face_cleaned_data/test"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/model_baseline_new_data/checkpoint.pth.tar",
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
    parser.add_argument(
        "--output_foloder",
        type=str,
        default="",
    )
    args = parser.parse_args()
    main(args)
