import os
import argparse
import torch
import numpy as np
import math

from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from src.loss_functions.triplet_loss import TripletLoss
from src.data_loading.data_loader import TripletImageLoader
from torch.utils.tensorboard import SummaryWriter
from src.helper_functions.helper import set_parameter_requires_grad, checkGPU
from src.helper_functions.helper import checkOutputDirectoryAndCreate,update_loss_hist, accuracy
from src.helper_functions.tensorboardWriter import create_writer

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


def main(args):
    print("=====Facenet=====")
    os.environ["WANDB_WATCH"] = "false"
    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if (wandb != None):
        wandb.init(project="FaceSSL", entity="andy-su", name=args.output_foloder)
        wandb.config.update(args)
        wandb.define_metric("loss", summary="min")
        wandb.define_metric("cross", summary="min")
        wandb.define_metric("triplet", summary="min")
        wandb.define_metric("acc", summary="max")

    writer = create_writer(args)
    device = checkGPU()
    model = create_model(args).to(device)
    train_loader, val_loader = create_dataloader(args)
    checkOutputDirectoryAndCreate(args.output_foloder)
    train(args, model, train_loader, val_loader, writer, device)

def create_model(args):
    from facenet_pytorch import InceptionResnetV1
    from src.models.facenet import Facenet

    if (args.pretrain == "casia-webface"):
        backbone = InceptionResnetV1(pretrained = 'casia-webface')
    elif (args.pretrain == "vggface2"):
        backbone = InceptionResnetV1(pretrained = 'vggface2')
    else:
        backbone = InceptionResnetV1()


    if args.pretrain_model_path != "":
        backbone = torch.load(args.pretrain_model_path).encoder
        # checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")
        # msg = backbone.load_state_dict(checkpoint["model"], strict=False)
        # backbone.load_state_dict(torch.load(args.pretrain_model_path)['model']).to(device)
    

    set_parameter_requires_grad(backbone, args.fix_backbone)

    model = Facenet(backbone, dim = args.dim, prev_dim = 512, pred_dim = 512)
    if args.resume_model_path != "":
        model = torch.load(args.resume_model_path)
    return model


def create_dataloader(args):
    from src.helper_functions.augmentations import (
        get_aug_trnsform,
        get_eval_trnsform,
    )

    trans_aug = get_eval_trnsform()
    trans_eval = get_eval_trnsform()
    dataset_train = TripletImageLoader(args.data_path, transform=trans_aug)
    img_inds = np.arange(len(dataset_train))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset_train,
        num_workers=args.workers,
        batch_size=args.batch_size,
        # shuffle=True,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset_train,
        num_workers=args.workers,
        batch_size=args.batch_size,
        # shuffle=True,
        sampler=SubsetRandomSampler(val_inds)
    )
    print("====")
    print("class count: ", len(np.unique(dataset_train.targets, return_counts=True)[0]))
    print("train len", train_inds.__len__())
    print("val len", val_inds.__len__())
    return train_loader, val_loader

def pass_epoch(args, model, loader, model_optimizer, tripletLoss_fn, crossEntropyLoss_fn, scaler, device, mode="Train"):
    loss = 0
    loss_triplet = 0
    loss_cross = 0
    acc_top1 = 0
    acc_top5 = 0

    for i_batch, image_batch in tqdm(enumerate(loader)):
        x, y = torch.cat(image_batch[0], 0).to(device), torch.cat(image_batch[1], 0).to(device)
        if mode == "Train":
            model.train()
        elif mode == "Eval":
            model.eval()
        else:
            print("error model mode!")
        
        if torch.isnan(x).sum() > 0:
            print("input data has nan")
            raise NameError('input data has nan')
        projector_out,  predictor_out = model(x)
        if torch.isnan(projector_out).sum() > 0:
            print("projector_out  has nan")
            raise NameError('projector_out  has nan')
        # compute loos
        loss_batch_triplet = tripletLoss_fn(projector_out, y)

        loss_batch_cross = crossEntropyLoss_fn(predictor_out, y)
        if torch.isnan(loss_batch_triplet):
            print("loss_batch_triplet is nan")
            print(predictor_out)
            print(y)
            raise NameError('loss_batch_triplet is nan')
        if torch.isinf(loss_batch_triplet):
            print("loss_batch_triplet is inf")
        if torch.isnan(loss_batch_cross):
            print("loss_batch_cross is nan")
        if torch.isinf(loss_batch_cross):
            print("loss_batch_cross is inf")
            
        loss_batch = loss_batch_cross * args.alpha + loss_batch_triplet * (1. - args.alpha)

        loss_batch_acc_top = accuracy(predictor_out, y, topk=(1, 5))
        
        if mode == "Train":
            model_optimizer.zero_grad()
            scaler.scale(loss_batch).backward()
            if (args.max_norm != -1):
                clip_grad_norm_(model.parameters(), max_norm=args.max_norm, error_if_nonfinite = False)
            scaler.step(model_optimizer)
            scaler.update()
            model_optimizer.step()

        loss += loss_batch.item()
        loss_triplet += loss_batch_triplet.item()
        loss_cross += loss_batch_cross.item()
        acc_top1 += loss_batch_acc_top[0].cpu()
        acc_top5 += loss_batch_acc_top[1].cpu()

    loss /= i_batch + 1
    loss_triplet /= i_batch + 1
    loss_cross /= i_batch + 1
    acc_top1 /= i_batch + 1
    acc_top5 /= i_batch + 1
    return loss, loss_triplet, loss_cross, acc_top1, acc_top5

def train(args, model, train_loader, val_loader, writer, device):
    train_loss_history = []
    train_acc_top1_history = []
    train_acc_top5_history = []
    val_loss_history = []
    val_acc_top1_history = []
    val_acc_top5_history = []

    model_optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr * (args.batch_size / 256),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model_scheduler = CosineAnnealingLR(model_optimizer, T_max=args.epochs)
    tripletLoss_fn = TripletLoss(device)
    crossEntropyLoss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()
    stop = 0
    min_val_loss = math.inf
    torch.save(model, "model/{}/checkpoint.pth.tar".format(args.output_foloder))
    
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        print("-" * 10)
        train_loss, train_loss_triplet, train_loss_cross, train_acc_top1, train_acc_top5 = pass_epoch(
            args,
            model,
            train_loader,
            model_optimizer,
            tripletLoss_fn,
            crossEntropyLoss_fn,
            scaler,
            device,
            "Train",
        )
        with torch.no_grad():
            val_loss, val_loss_triplet, val_loss_cross, val_acc_top1, val_acc_top5 = pass_epoch(
                args,
                model,
                val_loader,
                model_optimizer,
                tripletLoss_fn,
                crossEntropyLoss_fn,
                scaler,
                device,
                "Eval",
            )
        model_scheduler.step()

        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/train"] = train_loss
            logMsg["loss/val"] = val_loss
            logMsg["triplet/train"] = train_loss_triplet
            logMsg["triplet/val"] = val_loss_triplet
            logMsg["cross/train"] = train_loss_cross
            logMsg["cross/val"] = val_loss_cross
            logMsg["top1/train"] = train_acc_top1
            logMsg["top1/val"] = val_acc_top1
            logMsg["top5/train"] = train_acc_top5
            logMsg["top5/val"] = val_acc_top5
            wandb.log(logMsg)
            wandb.watch(model,log = "all", log_graph=True)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("triplet", {"train": train_loss_triplet, "val": val_loss_triplet}, epoch)
        writer.add_scalars("cross", {"train": train_loss_cross, "val": val_loss_cross}, epoch)
        writer.add_scalars("top1", {"train": train_acc_top1, "val": val_acc_top1}, epoch)
        writer.add_scalars("top5", {"train": train_acc_top5, "val": val_acc_top5}, epoch)
        writer.flush()

        train_loss_history.append(train_loss)
        train_acc_top1_history.append(train_acc_top1)
        train_acc_top5_history.append(train_acc_top5)

        val_loss_history.append(val_loss)
        val_acc_top1_history.append(val_acc_top1)
        val_acc_top5_history.append(val_acc_top5)

        update_loss_hist(args, {"train": train_loss_history, "val": val_loss_history}, "Loss")
        update_loss_hist(args, {"train": train_acc_top1_history, "val": val_acc_top1_history}, "Top1")
        update_loss_hist(args, {"train": train_acc_top5_history, "val": val_acc_top5_history}, "Top5")

#         torch.save(model, "model/{}/checkpoint.pth.tar".format(args.output_foloder))
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            print("Best, save model, epoch = {}".format(epoch))
            torch.save(model, "model/{}/checkpoint.pth.tar".format(args.output_foloder))
#             stop = 0
#         else:
#             stop += 1
#             if stop > 10:
#                 print("early stopping")
#                 break
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facenet")
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
        "--resume_model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_foloder",
        type=str,
        default="model_define",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--fix_backbone",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--sim_weight",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--var_weight",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--cov_weight",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=-1,
    )
    args = parser.parse_args()

    main(args)
