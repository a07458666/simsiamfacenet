
class EarlyStopping(nn.Module):
    def __init__(self, limit = 10):
        self.limit = limit
        self.min_val_loss = 99999
        self.stopCount = 0
    def early_stopping_func(val_loss):
        isEarlyStoping = False
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.stopCount = 0
        else:
            self.stopCount += 1
            if self.stopCount > 10:
                print("early stopping")
                isEarlyStoping = True
        return isEarlyStoping
