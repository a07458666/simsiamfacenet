import os
import argparse
import torch
import numpy as np
import math

from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    MultiStepLR,
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
)
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast

from src.data_loading.data_loader import BirdImageLoader
from src.txt_loading.txt_loader import (
    readClassIdx,
    readTrainImages,
    splitDataList,
)
from src.loss_functions.CrossEntropyLS import CrossEntropyLS
from torch.utils.tensorboard import SummaryWriter
from src.models.swin_transformer import SwinTransformer
from bottleneck_transformer_pytorch import BottleStack


def main(args):
    writer = create_writer(args)
    device = checkGPU()
    class_to_idx = readClassIdx(args)
    data_list = readTrainImages(args)
    train_data_list, val_data_list, _ = splitDataList(data_list, 0.9, 0.1)
    model = create_model(args).to(device)
    train_loader, val_loader = create_dataloader(
        args, train_data_list, val_data_list, class_to_idx
    )
    checkOutputDirectoryAndCreate(args)
    train(args, model, train_loader, val_loader, writer, device)


def checkOutputDirectoryAndCreate(args):
    if not os.path.exists(args.output_foloder):
        os.makedirs(args.output_foloder)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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


def update_loss_hist(args, train_list, val_list, name="result"):
    plt.plot(train_list)
    plt.plot(val_list)
    plt.title(name)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="center right")
    plt.savefig("{}/{}.png".format(args.output_foloder, name))
    plt.clf()


def create_model_BotNet(args):
    from torchvision.models import resnet50

    layer = BottleStack(
        dim=256,
        fmap_size=56,  # set specifically for imagenet's 224 x 224
        dim_out=2048,
        proj_factor=4,
        downsample=True,
        heads=4,
        dim_head=128,
        rel_pos_emb=True,
        activation=nn.ReLU(),
    )

    resnet = resnet50(pretrained=True)

    # model surgery

    backbone = list(resnet.children())

    model = nn.Sequential(
        *backbone[:5],
        layer,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(2048, 200),
    )
    # use the 'BotNet'

    # img = torch.randn(2, 3, 224, 224)
    # preds = model(img) # (2, 1000)
    return model


def create_model(args):
    import timm

    # backbone = timm.create_model(
    #     "vit_base_patch16_224_miil_in21k", pretrained=True
    # )
    backbone = SwinTransformer(
        img_size=224,
        window_size=7,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        num_classes=21841,
        drop_path_rate=0.2,
    )
    if args.pretrain_model_path != "":
        # backbone = torch.load(args.pretrain_model_path).to(device)
        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")
        msg = backbone.load_state_dict(checkpoint["model"], strict=False)
        # backbone.load_state_dict(torch.load(args.pretrain_model_path)['model']).to(device)
        # set_parameter_requires_grad(backbone, True)
    projector = nn.Sequential(
        nn.Linear(21841, 2048),
        nn.BatchNorm1d(2048),
        nn.LeakyReLU(),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(),
        nn.Linear(512, 200),
    )
    model = nn.Sequential(backbone, projector)
    return model


def create_dataloader(args, train_data_list, val_data_list, class_to_idx):
    from src.helper_functions.augmentations import (
        get_aug_trnsform,
        get_eval_trnsform,
        get_all_in_aug,
    )

    trans_aug = get_all_in_aug()
    trans_eval = get_eval_trnsform()
    dataset_train = BirdImageLoader(
        args.data_path, train_data_list, class_to_idx, transform=trans_aug
    )
    dataset_val = BirdImageLoader(
        args.data_path, val_data_list, class_to_idx, transform=trans_eval
    )

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

    print("class_to_idx ", len(class_to_idx))
    print("train len", dataset_train.__len__())
    print("val len", dataset_val.__len__())
    return train_loader, val_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top
    predictions for the specified values of k
    """
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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def pass_epoch(
    model, loader, model_optimizer, loss_fn, scaler, device, mode="Train"
):
    loss = 0
    acc_top1 = 0
    acc_top5 = 0

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
            scaler.scale(loss_batch).backward()
            scaler.step(model_optimizer)
            scaler.update()
            model_optimizer.step()

        loss += loss_batch.detach().cpu()
        acc_top1 += loss_batch_acc_top[0]
        acc_top5 += loss_batch_acc_top[1]

    loss /= i_batch + 1
    acc_top1 /= i_batch + 1
    acc_top5 /= i_batch + 1
    return loss, acc_top1, acc_top5


def create_writer(args):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter("runs/" + args.output_foloder)
    msg = ""
    for key in vars(args):
        msg += "{} = {}<br>".format(key, vars(args)[key])
    writer.add_text("Parameter", msg, 0)
    writer.flush()
    writer.close()
    return writer


def train(args, model, train_loader, val_loader, writer, device):
    train_loss_history = []
    train_acc_top1_history = []
    train_acc_top5_history = []
    val_loss_history = []
    val_acc_top1_history = []
    val_acc_top5_history = []
    model_optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model_scheduler = CosineAnnealingLR(model_optimizer, T_max=20)
    torch.save(model, "{}/checkpoint.pth.tar".format(args.output_foloder))
    loss_fn = CrossEntropyLS(args.label_smooth)
    scaler = GradScaler()
    stop = 0
    min_val_loss = math.inf
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        print("-" * 10)
        train_loss, train_acc_top1, train_acc_top5 = pass_epoch(
            model,
            train_loader,
            model_optimizer,
            loss_fn,
            scaler,
            device,
            "Train",
        )
        with torch.no_grad():
            val_loss, val_acc_top1, val_acc_top5 = pass_epoch(
                model,
                val_loader,
                model_optimizer,
                loss_fn,
                scaler,
                device,
                "Eval",
            )
        model_scheduler.step()

        writer.add_scalars(
            "loss", {"train": train_loss, "val": val_loss}, epoch
        )
        writer.add_scalars(
            "top1", {"train": train_acc_top1, "val": val_acc_top1}, epoch
        )
        writer.add_scalars(
            "top5", {"train": train_acc_top5, "val": val_acc_top5}, epoch
        )
        writer.flush()

        train_loss_history.append(train_loss)
        train_acc_top1_history.append(train_acc_top1)
        train_acc_top5_history.append(train_acc_top5)

        val_loss_history.append(val_loss)
        val_acc_top1_history.append(val_acc_top1)
        val_acc_top5_history.append(val_acc_top5)

        update_loss_hist(args, train_loss_history, val_loss_history, "Loss")
        update_loss_hist(
            args, train_acc_top5_history, val_acc_top5_history, "Top5"
        )
        update_loss_hist(
            args, train_acc_top1_history, val_acc_top1_history, "Top1"
        )
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            print("Best, save model, epoch = {}".format(epoch))
            torch.save(
                model,
                "{}/checkpoint.pth.tar".format(args.output_foloder),
            )
            stop = 0
        else:
            stop += 1
            if stop > 10:
                print("early stopping")
                break
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="310551010 train bird")
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
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.007,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--label_smooth",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--pretrain_model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_foloder",
        type=str,
        default="model/model_test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    main(args)
