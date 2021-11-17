import os
import torch
from matplotlib import pyplot as plt

def checkOutputDirectoryAndCreate(output_foloder):
    if not os.path.exists('checkpoints/' + output_foloder):
        os.makedirs('checkpoints/' + output_foloder)

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

def update_loss_hist(args, data, name="result", xlabel = "Epoch", ylabel = "Loss"):
    legend_list = []
    for key in data.keys():
        legend_list.append(key)
        if (len(data[key]) == 2):
            plt.plot(data[key][0], data[key][1])
        else:
            plt.plot(data[key])
    plt.title(name)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend_list, loc="center right")
    plt.savefig("{}/{}.png".format(args.output_foloder, name))
    plt.clf()

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