from typing import Tuple

from PIL import Image

import numpy as np
import albumentations as A

import torch
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from helpers import original2train


class CityscapesTransform(object):
    """
    Data transform being applied to both the image and the semantic map.
    """
    def __init__(self):
        self.transform = A.Compose([])

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        image = np.array(image)
        mask = np.array(mask)

        data = self.transform(image=image, mask=mask)
        image = data["image"]
        mask = data["mask"]

        image = to_tensor(image)
        mask = torch.tensor(mask, dtype=torch.int64)
        mask = original2train(mask)

        return image, mask


class CityscapesTrainTransform(CityscapesTransform):
    """
    Training transform with additional data augmentation.
    """
    def __init__(self, crop_size: Tuple[int, int]):
        super().__init__()
        self.transform = A.Compose(
            [
                A.RandomCrop(
                    height=crop_size[0], width=crop_size[1], always_apply=True
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(),
            ]
        )


class CityscapesEvalTransform(CityscapesTransform):
    pass
