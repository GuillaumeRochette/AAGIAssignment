from typing import Optional, Union

from pathlib import Path
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from torchvision.datasets import Cityscapes

from lightning import LightningDataModule

from transform import CityscapesTrainTransform, CityscapesEvalTransform


class CityscapesDataModule(LightningDataModule):
    """
    DataModule structure which promotes re-usability and portability by contains the dataset splits, the training and evaluation transforms, and the dataloaders methods.
    Moreover, this structure allows to seamlessly replicate over multiples nodes and machines, thanks to the setup() method.
    Regarding the dataset splits, it re-uses the built-in torchvision.datasets.Cityscapes to not re-invent the wheel.
    """
    def __init__(
        self,
        hparams: DictConfig,
        root: Union[str, Path],
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        if not isinstance(root, Path):
            root = Path(root)

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = CityscapesTrainTransform(
            crop_size=hparams.data.crop_size
        )
        self.eval_transform = CityscapesEvalTransform()

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = Cityscapes(
                root=str(self.root),
                split="train",
                mode="fine",
                target_type="semantic",
                transforms=self.train_transform,
            )
            self.val_dataset = Cityscapes(
                root=str(self.root),
                split="val",
                mode="fine",
                target_type="semantic",
                transforms=self.eval_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = Cityscapes(
                root=str(self.root),
                split="test",
                mode="fine",
                target_type="semantic",
                transforms=self.eval_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import os

    datamodule = CityscapesDataModule(
        hparams=OmegaConf.create({"data": {"crop_size": (256, 256)}}),
        root=os.getenv("CITYSCAPES_DATASET"),
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup()
    print(len(datamodule.train_dataloader()))
    print(len(datamodule.val_dataloader()))
    print(len(datamodule.test_dataloader()))

    image, mask = next(iter(datamodule.train_dataloader()))
    print(image.shape, image.dtype, image.min(), image.max())
    print(mask.shape, mask.dtype, mask.min(), mask.max())

    image, mask = next(iter(datamodule.val_dataloader()))
    print(image.shape, image.dtype, image.min(), image.max())
    print(mask.shape, mask.dtype, mask.min(), mask.max())

    # image, mask = next(iter(datamodule.test_dataloader()))
    # print(image.shape, image.dtype, image.min(), image.max())
    # print(mask.shape, mask.dtype, mask.min(), mask.max())
