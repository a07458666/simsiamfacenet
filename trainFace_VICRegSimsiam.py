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

from src.data_loading.data_loader import TripletSSLImageLoader
from src.loss_functions.vicreg import simsiam_vicreg_loss_func
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
        wandb.define_metric("ssl", summary="min")
        wandb.define_metric("acc", summary="max")

    writer = create_writer(args)
    device = checkGPU()
    model = create_model(args).to(device)
    loaders = create_dataloader(args)
    checkOutputDirectoryAndCreate(args.output_foloder)
    train(args, model, loaders, writer, device)

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
        get_aug_trnsform_noCrop,
        get_aug_trnsform,
        get_eval_trnsform,
    )
    
    if args.noCrop:
        trans_aug = get_aug_trnsform_noCrop()
    else:
        trans_aug = get_aug_trnsform()
    trans_eval = get_eval_trnsform()
    dataset_train = TripletSSLImageLoader(args.data_path, transform=trans_aug)
    img_inds = np.arange(len(dataset_train))
    train_inds = img_inds[:int(0.9 * len(img_inds))]
    val_never_inds = img_inds[int(0.9 * len(img_inds)):]
    np.random.shuffle(train_inds)
    np.random.shuffle(val_never_inds)
    val_inds = train_inds[:len(val_never_inds)]
    train_inds = train_inds[len(val_never_inds):]
    np.random.shuffle(train_inds)
    np.random.shuffle(val_inds)
    
    dataLoaders = {}
    
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
    val_loader_never = DataLoader(
        dataset_train,
        num_workers=args.workers,
        batch_size=args.batch_size,
        # shuffle=True,
        sampler=SubsetRandomSampler(val_never_inds)
    )
    dataLoaders["train"] = train_loader
    dataLoaders["val"] = val_loader
    dataLoaders["val_never"] = val_loader_never
    print("====")
    print("class count: ", len(np.unique(dataset_train.targets, return_counts=True)[0]))
    print("data len", img_inds.__len__())
    print("train len", train_inds.__len__())
    print("val len", val_inds.__len__())
    print("val never len", val_never_inds.__len__())
    return dataLoaders

def pass_epoch(args, model, loader, model_optimizer, scaler, device, mode="Train"):
    loss = 0
    loss_ssl = 0
    loss_cross = 0
    acc_top1 = 0
    acc_top5 = 0
    
    crossEntropyLoss_fn = torch.nn.CrossEntropyLoss()

    for i_batch, image_batch in tqdm(enumerate(loader)):
        x1, x2, y = torch.cat(image_batch[0], 0).to(device), torch.cat(image_batch[1], 0).to(device), torch.cat(image_batch[2], 0).to(device)
        if mode == "Train":
            model.train()
        elif mode == "Eval":
            model.eval()
        else:
            print("error model mode!")
        
        if torch.isnan(x1).sum() > 0:
            print("input data has nan")
            raise NameError('input data has nan')
        if torch.isnan(x2).sum() > 0:
            print("input data has nan")
            raise NameError('input data has nan')
        z1, z2, p1, p2, ps1, ps2 = model.forwardSSL(x1, x2)

        # compute loos
        loss_batch_cross = crossEntropyLoss_fn(p1, y)
        loss_batch_ssl, _, _, _ = simsiam_vicreg_loss_func(z1, z2, ps1, ps2)
        
        loss_batch = loss_batch_cross * args.alpha + loss_batch_ssl * (1. - args.alpha)

        loss_batch_acc_top = accuracy(p1, y, topk=(1, 5))
        
        if mode == "Train":
            model_optimizer.zero_grad()
            scaler.scale(loss_batch).backward()
            if (args.max_norm != -1):
                clip_grad_norm_(model.parameters(), max_norm=args.max_norm, error_if_nonfinite = False)
            scaler.step(model_optimizer)
            scaler.update()
            model_optimizer.step()

        loss += loss_batch.item()
        loss_ssl += loss_batch_ssl.item()
        loss_cross += loss_batch_cross.item()
        acc_top1 += loss_batch_acc_top[0].cpu()
        acc_top5 += loss_batch_acc_top[1].cpu()

    loss /= i_batch + 1
    loss_ssl /= i_batch + 1
    loss_cross /= i_batch + 1
    acc_top1 /= i_batch + 1
    acc_top5 /= i_batch + 1
    return loss, loss_ssl, loss_cross, acc_top1, acc_top5

def train(args, model, loaders, writer, device):
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
    scaler = GradScaler()
    stop = 0
    min_val_loss = math.inf
    min_val_never_loss = math.inf
    torch.save(model, "model/{}/checkpoint.pth.tar".format(args.output_foloder))
    
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        print("-" * 10)
        train_loss, train_loss_ssl, train_loss_cross, train_acc_top1, train_acc_top5 = pass_epoch(
            args,
            model,
            loaders["train"],
            model_optimizer,
            scaler,
            device,
            "Train",
        )
        with torch.no_grad():
            val_loss, val_loss_ssl, val_loss_cross, val_acc_top1, val_acc_top5 = pass_epoch(
                args,
                model,
                loaders["val"],
                model_optimizer,
                scaler,
                device,
                "Eval",
            )
            _, val_never_loss_ssl, _, _, _ = pass_epoch(
                args,
                model,
                loaders["val_never"],
                model_optimizer,
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
            logMsg["ssl/train"] = train_loss_ssl
            logMsg["ssl/val"] = val_loss_ssl
            logMsg["cross/train"] = train_loss_cross
            logMsg["cross/val"] = val_loss_cross
            logMsg["top1/train"] = train_acc_top1
            logMsg["top1/val"] = val_acc_top1
            logMsg["top5/train"] = train_acc_top5
            logMsg["top5/val"] = val_acc_top5
            logMsg["ssl_never/val"] = val_never_loss_ssl
            wandb.log(logMsg)
            wandb.watch(model,log = "all", log_graph=True)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("ssl", {"train": train_loss_ssl, "val": val_loss_ssl}, epoch)
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

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            print("Best, save model, epoch = {}".format(epoch))
            torch.save(model, "model/{}/checkpoint.pth.tar".format(args.output_foloder))
        if val_never_loss_ssl <= min_val_never_loss:
            min_val_never_loss = val_never_loss_ssl
            print("Best, save never model, epoch = {}".format(epoch))
            torch.save(model, "model/{}/checkpoint_never.pth.tar".format(args.output_foloder))
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
    parser.add_argument('--noCrop', dest='noCrop', action='store_true')

    args = parser.parse_args()

    main(args)
