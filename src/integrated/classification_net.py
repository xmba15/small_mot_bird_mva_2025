import numpy as np
import pytorch_lightning as pl
import timm
import torch
from src.utils import get_object_from_dict
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from torchvision.utils import make_grid

__all__ = ("BaseClassificationModelPl",)


class BaseClassificationModelPl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self._setup_model()

    def setup(self, stage=None):
        self._setup_loss()
        self._setup_metric()

    def _setup_model(self):
        encoder_name = self.hparams["model"]["encoder_name"]
        pretrained = self.hparams["model"]["pretrained"]

        self.model = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=1,
        )

    def _setup_loss(self):
        self.cls_loss = nn.BCEWithLogitsLoss()

    def _setup_metric(self):
        self.metric = BinaryAccuracy()

    def common_step(self, batch, batch_idx):
        logits = self.model(batch["image"])
        labels = batch["label"].view(-1, 1)

        loss_cls = self.cls_loss(
            logits,
            labels.float(),
        )

        with torch.no_grad():
            preds = torch.sigmoid(logits)
            acc = self.metric(preds, labels)

        out = {
            "loss": loss_cls,
            "acc": acc,
        }

        return out

    def training_step(self, batch, batch_idx):
        out = self.common_step(batch, batch_idx)

        if batch_idx % 100 == 0:
            global_step = (
                self.current_epoch * self.trainer.num_training_batches + batch_idx
            )
            b = batch["image"].shape[0]

            images = batch["image"].detach().cpu()
            labels = batch["label"].detach().cpu().numpy()

            images_with_labels = annotate_images_with_labels(images, labels)

            self.logger.experiment.add_image(
                "train_image",
                make_grid(images_with_labels, nrow=b),
                global_step=global_step,
            )

        self.log(
            "train_loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train_acc",
            out["acc"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return out

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = self.common_step(batch, batch_idx)

        if batch_idx == 0:
            global_step = self.current_epoch
            b = batch["image"].shape[0]

            images = batch["image"].detach().cpu()
            labels = batch["label"].detach().cpu().numpy()

            images_with_labels = annotate_images_with_labels(images, labels)

            self.logger.experiment.add_image(
                "val_image",
                make_grid(images_with_labels, nrow=b),
                global_step=global_step,
            )

        self.log(
            "val_loss",
            out["loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val_acc",
            out["acc"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return out

    def configure_optimizers(self):
        optimizer = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.parameters() if x.requires_grad],
        )

        scheduler = {
            "scheduler": get_object_from_dict(
                self.hparams["scheduler"],
                optimizer=optimizer,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]


def annotate_images_with_labels(images, labels):
    import cv2

    annotated = []
    for img, label in zip(images, labels):
        np_img = img.permute(1, 2, 0).cpu().numpy()
        np_img = (np_img * 255).astype("uint8")
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        cv2.putText(
            np_img,
            str(label),
            (10, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 255, 0),
            2,  # Thickness
            cv2.LINE_AA,
        )

        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        annotated.append(torch.from_numpy(np_img).permute(2, 0, 1))

    return torch.stack(annotated)
