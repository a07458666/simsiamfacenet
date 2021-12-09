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
    print("=====Eval=====")
    device = checkGPU()
    model = loadModel(args).to(device)
    trans = get_eval_trnsform()
    loader = create_dataloader(args, trans)
    hitRatioList, kList, sameDist, diffDist, valList, farList = eval_model(args, model, loader, device)
    print("===sameDist, diffDist, S/D===")
    print(sameDist, diffDist, sameDist / diffDist)
    print("===hitRatioList(k=1,k=5)===")
    print(hitRatioList[0], hitRatioList[4])

    if (args.output_foloder == ""):
        args.output_foloder = os.path.abspath(os.path.join(args.model_path, os.pardir))
        
    writerCSV(args, sameDist, diffDist, hitRatioList, valList, farList)

    update_loss_hist_lim(args, {'HitRatio' : [kList, hitRatioList]}, "HitRatio", "k", "hitRatio")
    update_loss_hist_lim(args, {'Dist' : [farList, valList]}, "VAL_FAR", "FAR", "VAL")
    
    torch.cuda.empty_cache()

def update_loss_hist_lim(args, data, name="result", xlabel = "Epoch", ylabel = "Loss"):
    from matplotlib import pyplot as plt
    legend_list = []
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0,1.1])
    for key in data.keys():
        legend_list.append(key)
        if (len(data[key]) == 2):
            plt.plot(data[key][0], data[key][1])
            plt.scatter(data[key][0], data[key][1])
            if (name == 'HitRatio'):
                for x,y in zip(data[key][0],data[key][1]):
                    label = "{:.2f}".format(y * 100)
                    plt.annotate(label, # this is the text
                                (x,y), # these are the coordinates to position the label
                                textcoords="offset points", # how to position the text
                                xytext=(0,-10), # distance from text to points (x,y)
                                ha='center') # horizontal alignment can be left, right or center
        else:
            plt.plot(data[key])
            plt.scatter(data[key])
    
    plt.savefig("{}/{}.png".format(args.output_foloder, name))
    plt.show()
    plt.clf()

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


def eval_model(args, model, loader, device):
    y_pre, y = pass_epoch(model, loader, device)

    # dist = pdist(y_pre)
    dist = pairwise_distance_torch(y_pre, device)
    dist[dist == 0] = float('nan')

    mask_pos, mask_neg = makemask(y)   

    pos = dist * mask_pos.float().to(device)
    pos[pos == 0] = float('nan')
    neg = dist * mask_neg.float().to(device)
    neg[neg == 0] = float('nan')
    psame = torch.sum(mask_pos == True)
    pdiff = torch.sum(mask_neg == True)

    hitRatioList, kList = calculateHitRatio(dist, mask_pos, mask_neg)
    sameDist, diffDist = calculateClusterDistClose(pos, neg, psame, pdiff)
    valList, farList = calculateClusterVAL_FAR(pos, neg, psame, pdiff)
    return hitRatioList, kList, sameDist, diffDist, valList, farList

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

def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def makemask(targets):
    n = targets.shape[0]

    # find the hardest positive and negative
    mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
    mask_neg = ~mask_pos
#     print("mask_pos", mask_pos.size())
#     print("mask_neg", mask_neg.size())
    mask_pos[torch.eye(n).byte()] = 0
    return mask_pos, mask_neg

def calculateClusterDistClose(pos, neg, psame, pdiff):
    sameD_val = torch.nansum(pos)
    diffD_val = torch.nansum(neg)
    
    sameD = sameD_val / psame
    diffD = diffD_val / pdiff
#     print("sameD_val ", sameD_val)
#     print("diffD_val", diffD_val)

    return sameD.cpu().detach().numpy(), diffD.cpu().detach().numpy()

def calculateClusterVAL_FAR(pos, neg, psame, pdiff):
    valList = []
    farList = []

    for i in tqdm(range(21)):
        d = i / 10
        ta = torch.sum(pos <= d)
        fa = torch.sum(neg <= d)

        val = ta / psame
        far = fa / pdiff
        valList.append(float(val.cpu().detach().numpy()))
        farList.append(float(far.cpu().detach().numpy()))
    return valList, farList

def calculateHitRatio(dist, mask_pos, mask_neg, kMax = (5 + 1)):
    hitRatioList = []
    kList = []
    for k in range(1, kMax):
        kList.append(k)
        values, indices = torch.topk(dist, k, largest = False)
        hitCount = torch.sum(torch.gather(mask_pos, 1, indices) == True, 1)
        hitRatio = torch.mean(hitCount / k)
        hitRatioList.append(float(hitRatio.cpu().detach().numpy()))

    return hitRatioList, kList

def writerCSV(args, sameDist, diffDist, hitRatioList, valList, farList):
    import csv
    filePath = os.path.join(args.output_foloder, 'output.csv')
    print("filePath", filePath)
    print("args.output_foloder", args.output_foloder)
    with open(filePath, 'w', newline='') as csvfile:

        # 以空白分隔欄位，建立 CSV 檔寫入器
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(['sameDist', 'diffDist', 'S/D', 'hitRatio', 'VAL', 'FAR'])
        writer.writerow([str(sameDist), str(diffDist), str(sameDist/diffDist), hitRatioList, valList, farList])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval")
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
