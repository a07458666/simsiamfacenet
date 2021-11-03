
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