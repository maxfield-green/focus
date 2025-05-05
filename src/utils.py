from torch.utils.tensorboard import SummaryWriter

class EarlyStopper:
    """
    Stop training when validation loss hasn't improved after 'patience' epochs.
    """
    def __init__(self, patience=10, delta=0.0):
        self.patience, self.delta = patience, delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def log_tensorboard(writer: SummaryWriter, epoch: int, train_loss: float, val_loss: float):
    writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
