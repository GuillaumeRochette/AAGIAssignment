import torch
from torch import Tensor
from torchvision.datasets import Cityscapes

LABELS = Cityscapes.classes

ORIGINAL_IDS = torch.tensor([label.id for label in LABELS])
ORIGINAL_COLOURS = torch.tensor([label.color for label in LABELS])
TRAIN_IDS = torch.tensor(
    [label.train_id if label.train_id not in [-1, 255] else -1 for label in LABELS]
)
TRAIN_COLOURS = torch.tensor(
    [label.color if label.train_id not in [-1, 255] else (0, 0, 0) for label in LABELS]
)

NUM_CLASSES = sum([1 for label in LABELS if label.train_id not in [-1, 255]])


def original2train(original_mask: Tensor) -> Tensor:
    """
    Converts the mask of the original ids to the ids used in training because not all classes are evaluated.

    :param original_mask:
    :return:
    """
    assert original_mask.ndim >= 2
    shape = original_mask.shape
    dtype = original_mask.dtype
    device = original_mask.device
    train_mask = torch.zeros(shape, dtype=dtype, device=device)
    for orginal_id, train_id in zip(ORIGINAL_IDS, TRAIN_IDS):
        train_mask[original_mask == orginal_id] = train_id
    return train_mask


def ids2colours(mask, ids, colours):
    """
    Converts a mask of ids to a coloured mask for visualisation purposes.

    :param mask:
    :param ids:
    :param colours:
    :return:
    """
    assert mask.ndim >= 2
    shape = mask.shape
    dtype = mask.dtype
    device = mask.device
    image = torch.zeros(shape + (3,), dtype=dtype, device=device)
    for id, colour in zip(ids, colours):
        image[mask == id] = colour
    dims = list(range(image.ndim))
    dims = dims[:-3] + dims[-1:] + dims[-3:-1]
    image = image.permute(dims).contiguous()
    return image


def original2colour(mask: Tensor) -> Tensor:
    return ids2colours(mask, ORIGINAL_IDS, ORIGINAL_COLOURS)


def train2colour(mask: Tensor) -> Tensor:
    return ids2colours(mask, TRAIN_IDS, TRAIN_COLOURS)


def min_max(x, m=None, M=None):
    if m is not None and M is not None:
        assert m <= M
    if m is not None:
        x = max(x, m)
    if M is not None:
        x = min(x, M)
    return x
