import pytorch_lightning as pl
import torch as T
from torchvision.utils import make_grid


class DrawSTNTransform(pl.Callback):
    """Callback to draw STN transformations
    """
    def on_validation_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule) -> None:
        """Graph STN transformations per epoch

        Args:
            trainer (pl.Trainer): PL Trainer
            pl_module (pl.LightningModule): Model
        """
        # Getting images (x) and transforming it (x_t)
        with T.no_grad():
            x, _ = next(iter(trainer.datamodule.val_dataloader()))
            x_t = pl_module.stn(x)

        # De-Norm images: adding mean & std
        mu = T.tensor(0.1307)
        std = T.tensor(0.3081)
        x = std * x + mu
        x_t = std * x_t + mu

        # Clipping
        x = x.clamp(0, 1)
        x_t = x_t.clamp(0, 1)

        # Making a grids for TB drawing
        x_grid = make_grid(x)
        xt_grid = make_grid(x_t)

        # Logging images in TB
        epoch = trainer.current_epoch
        trainer.logger.experiment.add_image("STN/input", x_grid, epoch)
        trainer.logger.experiment.add_image("STN/transform", xt_grid, epoch)
