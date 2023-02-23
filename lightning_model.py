from typing import Tuple

import wandb

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import AdamW

from lightning import LightningModule

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_jaccard_index,
)

from model import FPNSwinTransformerV2
from scheduler import CosineLR
from helpers import train2colour, NUM_CLASSES


class LitSemanticSegmentationModel(LightningModule):
    """
    LightningModule which parameterize how the training is going to be conducted.
    """
    def __init__(self, hparams):
        """
        We setup the model, criterion and metrics to be monitored.

        :param hparams:
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.model.name == "FPNSwinTransformerV2":
            self.model = FPNSwinTransformerV2(
                backbone=self.hparams.model.backbone,
                pretrained=self.hparams.model.pretrained,
                n_classes=NUM_CLASSES,
            )
        # elif self.hparams.model.name == "xyz":
        #     model = ...
        else:
            raise ValueError(f"Unknwon model: {self.hparams.model.name}")

        self.criterion = CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=self.hparams.loss.label_smoothing,
        )

        self.training_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=NUM_CLASSES,
                    average="micro",
                    top_k=1,
                    multidim_average="global",
                    ignore_index=-1,
                ),
                "iou": MulticlassJaccardIndex(
                    num_classes=NUM_CLASSES,
                    average="micro",
                    ignore_index=-1,
                ),
            },
            prefix="train/",
        )

        self.validation_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=NUM_CLASSES,
                    average="micro",
                    top_k=1,
                    multidim_average="global",
                    ignore_index=-1,
                ),
                "iou": MulticlassJaccardIndex(
                    num_classes=NUM_CLASSES,
                    average="micro",
                    ignore_index=-1,
                ),
            },
            prefix="val/",
        )

        self.test_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=NUM_CLASSES,
                    average="micro",
                    top_k=1,
                    multidim_average="global",
                    ignore_index=-1,
                ),
                "iou": MulticlassJaccardIndex(
                    num_classes=NUM_CLASSES,
                    average="micro",
                    ignore_index=-1,
                ),
            },
            prefix="test/",
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Definition of the forward pass.
        :param x:
        :return:
        """
        n, _, h, w = x.shape
        ih, iw = self.model.in_size
        oh, ow = self.model.out_size
        if (h, w) == (ih, iw):
            # If the input matches the expected size, then we do as planned.
            x = self.model(x)
        else:
            # Otherwise, we chunk the input into tiles, and we reassemble it after inference.
            assert h % ih == 0 and w % iw == 0
            ph, pw = h // ih, w // iw
            # x.shape = (n, -1, h, w)
            x = x.reshape(n, -1, ph, ih, pw, iw)
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(n * ph * pw, -1, ih, iw)
            # x.shape = (n', -1, ih, iw)

            # Normally we would only do:
            # x = self.model(x)
            # But the GPU on my laptop is too small for large validation images, so we split the inference into sub-batches:
            if self.training:
                x = self.model(x)
            else:
                x = torch.cat(
                    [self.model(x[i : i + n]) for i in range(0, n * ph * pw, n)],
                    dim=0,
                )

            # x.shape = (n', -1, oh, ow)
            x = x.reshape(n, ph, pw, -1, oh, ow)
            x = x.permute(0, 3, 1, 4, 2, 5)
            x = x.reshape(n, -1, ph * oh, pw * ow)
            # x.shape = (n', -1, h', w')
        return x

    def training_step(self, input: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Defintion of a training step, where we perform the forward pass, compute and log the losses and the metrics,
        and if it is the first iteration of the epoch, we make a visualisation of the current data examples.

        :param input:
        :param batch_idx:
        :return:
        """
        image, target_mask = input
        predicted_map = self(image)
        predicted_map = F.interpolate(
            predicted_map,
            size=target_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        train_loss = self.criterion(input=predicted_map, target=target_mask)

        self.training_metrics.accuracy(preds=predicted_map, target=target_mask)
        self.training_metrics.iou(preds=predicted_map, target=target_mask)

        self.log("train_loss", train_loss)
        self.log_dict(self.training_metrics)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.make_visualisation(
                image=image,
                predicted_mask=predicted_map.argmax(dim=-3),
                target_mask=target_mask,
                mode="train",
            )

        return train_loss

    def validation_step(self, input: Tuple[Tensor, Tensor], batch_idx: int):
        """
        Defintion of a validation step, where we perform the forward pass, compute and log the metrics,
        and if it is the first iteration of the epoch, we make a visualisation of the current data examples.

        :param input:
        :param batch_idx:
        :return:
        """
        image, target_mask = input
        predicted_map = self(image)
        predicted_map = F.interpolate(
            predicted_map,
            size=target_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        self.validation_metrics.accuracy(preds=predicted_map, target=target_mask)
        self.validation_metrics.iou(preds=predicted_map, target=target_mask)

        self.log("val_metric", self.validation_metrics.iou)
        self.log_dict(self.validation_metrics)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.make_visualisation(
                image=image,
                predicted_mask=predicted_map.argmax(dim=-3),
                target_mask=target_mask,
                mode="val",
            )

    def test_step(self, input: Tuple[Tensor, Tensor], batch_idx: int):
        """
        Defintion of a test step, where we perform the forward pass, compute and log the metrics.

        :param input:
        :param batch_idx:
        :return:
        """
        image, target_mask = input
        predicted_map = self(image)
        predicted_map = F.interpolate(
            predicted_map,
            size=target_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        self.test_metrics.accuracy(preds=predicted_map, target=target_mask)
        self.test_metrics.iou(preds=predicted_map, target=target_mask)

        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        """
        Configures the optimizer and the learning rate scheduler according to the hyperparameters.

        :return:
        """
        opt = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.wd,
            amsgrad=True,
        )
        sch = CosineLR(
            optimizer=opt,
            warm_up_epochs=self.hparams.scheduler.warm_up_epochs,
            warm_up_factor=self.hparams.scheduler.warm_up_factor,
            cosine_epochs=self.hparams.scheduler.cosine_epochs,
            cosine_factor=self.hparams.scheduler.cosine_factor,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "name": "AdamW+CosineLR",
            },
        }

    @torch.no_grad()
    def make_visualisation(
        self,
        image: Tensor,
        predicted_mask: Tensor,
        target_mask: Tensor,
        mode: str,
    ):
        """
        Makes a visualisation for the logger in order to interpret the training dynamics and results.

        :param image:
        :param predicted_mask:
        :param target_mask:
        :param mode:
        :return:
        """
        device = torch.device("cpu")
        n = image.shape[0]

        examples = []
        for i in range(min(n, 4)):
            im = image[i].to(device=device)
            pm = predicted_mask[i].to(device=device)
            tm = target_mask[i].to(device=device)
            im = (im * 255.0).to(dtype=torch.uint8)
            pc = train2colour(pm).to(dtype=torch.uint8)
            tc = train2colour(tm).to(dtype=torch.uint8)
            ex = torch.cat([im, pc, tc], dim=-1)

            acc = multiclass_accuracy(
                preds=pm[None, ...],
                target=tm[None, ...],
                num_classes=NUM_CLASSES,
                average="micro",
                multidim_average="global",
                ignore_index=-1,
            )
            iou = multiclass_jaccard_index(
                preds=pm[None, ...],
                target=tm[None, ...],
                num_classes=NUM_CLASSES,
                average="micro",
                ignore_index=-1,
            )
            caption = f"acc: {round(acc.tolist(), 4)}, iou: {round(iou.tolist(), 4)}"

            example = wandb.Image(ex.permute([1, 2, 0]).numpy(), caption=caption)
            examples.append(example)

        self.logger.experiment.log({f"{mode}/examples": examples})
