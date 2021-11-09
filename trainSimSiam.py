import os
import argparse
import torch
import numpy as np
import math

from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast

from src.data_loading.data_loader import FaceImages
from torch.utils.tensorboard import SummaryWriter
from src.helper_functions.helper import set_parameter_requires_grad, checkGPU
from src.helper_functions.helper import checkOutputDirectoryAndCreate,update_loss_hist
from src.helper_functions.tensorboardWriter import create_writer

def main(args):
    print("=====SimSiam=====")
    writer = create_writer(args)
    device = checkGPU()
    model = create_model(args).to(device)
    train_loader, val_loader = create_dataloader(args)
    checkOutputDirectoryAndCreate(args.output_foloder)
    train(args, model, train_loader, val_loader, writer, device)

def create_model(args):
    from facenet_pytorch import InceptionResnetV1
    from src.models.simsiam import SimSiam

    backbone = InceptionResnetV1()

    if args.pretrain_model_path != "":
        backbone = torch.load(args.pretrain_model_path).to(device)
        # checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")
        # msg = backbone.load_state_dict(checkpoint["model"], strict=False)
        # backbone.load_state_dict(torch.load(args.pretrain_model_path)['model']).to(device)

    set_parameter_requires_grad(backbone, args.fix_backbone)

    model = SimSiam(backbone, dim = args.dim, prev_dim = 512, pred_dim = 512)
    
    return model


def create_dataloader(args):
    from src.helper_functions.augmentations import (
        get_aug_trnsform,
        get_eval_trnsform,
    )

    trans_aug = get_aug_trnsform()
    trans_eval = get_eval_trnsform()
    dataset_train = FaceImages(args.data_path, transform=trans_aug)
    dataset_val = FaceImages(args.data_path, transform=trans_eval)

    train_loader = DataLoader(
        dataset_train,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset_val,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

    print("train len", dataset_train.__len__())
    print("val len", dataset_val.__len__())
    return train_loader, val_loader

def pass_epoch(model, loader, model_optimizer, loss_fn, scaler, device, mode="Train"):
    loss = 0

    for i_batch, image_batch in tqdm(enumerate(loader)):
        x1, x2 = image_batch[0].to(device), image_batch[1].to(device)
        if mode == "Train":
            model.train()
        elif mode == "Eval":
            model.eval()
        else:
            print("error model mode!")

        p1, p2, z1, z2 = model(x1, x2)
        loss_batch = -(loss_fn(p1, z2).mean() + loss_fn(p2, z1).mean()) * 0.5
        # loss_batch_acc_top = accuracy(y_pred, y, topk=(1, 5))

        if mode == "Train":
            model_optimizer.zero_grad()
            scaler.scale(loss_batch).backward()
            scaler.step(model_optimizer)
            scaler.update()
            model_optimizer.step()

        loss += loss_batch.item()
    loss /= i_batch + 1
    return loss


def train(args, model, train_loader, val_loader, writer, device):
    train_loss_history = []
    train_acc_top1_history = []
    train_acc_top5_history = []
    model_optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr * (args.batch_size / 256),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model_scheduler = CosineAnnealingLR(model_optimizer, T_max=20)
    torch.save(model, "{}/checkpoint.pth.tar".format(args.output_foloder))
    loss_fn = nn.CosineSimilarity(dim=1).to(device)
    scaler = GradScaler()
    stop = 0
    min_train_loss = math.inf

    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        print("-" * 10)
        train_loss = pass_epoch(
            model,
            train_loader,
            model_optimizer,
            loss_fn,
            scaler,
            device,
            "Train",
        )

        model_scheduler.step()

        writer.add_scalars(
            "loss", {"train": train_loss}, epoch
        )
        writer.flush()

        train_loss_history.append(train_loss)
        update_loss_hist(args, {"train": train_loss_history}, "Loss")

        if train_loss <= min_train_loss:
            min_train_loss = train_loss
            print("Best, save model, epoch = {}".format(epoch))
            torch.save(model.encoder,"{}/checkpoint.pth.tar".format(args.output_foloder))
            stop = 0
        else:
            stop += 1
            if stop > 100:
                print("early stopping")
                break
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimSiam")
    parser.add_argument(
        "--data_path", type=str, default="../../dataset/face_cleaned_data/train"
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
        "--lr",
        type=float,
        default=0.05,
        help="lr_new = lr * (batch_size / 256)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--pretrain_model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_foloder",
        type=str,
        default="model/model_define",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--fix_backbone",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
    )
    args = parser.parse_args()

    main(args)
