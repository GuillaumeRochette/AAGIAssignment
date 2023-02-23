import argparse
from typing import Union, Sequence, Sized
import random
from pathlib import Path
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset, Sampler, DataLoader

from torchvision.transforms import ToTensor


class FiniteDataset(Dataset):
    """
    A simple finite dataset class inherited from torch.utils.data.Dataset.
    It recursively globs the images with the specified file_formats from the data_root.
    The user can also provide a composition of transforms to augment the images.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        file_formats: Sequence[str] = (".png", ".jpg"),
        transform=ToTensor(),
    ):
        if not isinstance(data_root, Path):
            data_root = Path(data_root)
        self.image_paths = sorted(
            p for p in data_root.rglob(f"*") if p.suffix in file_formats
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item: int) -> Tensor:
        """
        Opens an image using PIL, and applies it the transform if defined.

        :param item:
        :return: A PIL/Tensor object.
        """
        image = Image.open(self.image_paths[item])
        if self.transform is not None:
            image = self.transform(image)
        return image


class InfiniteSampler(Sampler):
    """
    An infinite sampler inherited from torch.utils.data.Sampler.
    Rather than defining an infinite generator on the data itself, we define an infinite generator, which will provide indexes to the torch.utils.data.DataLoader object, and therefore enabling the multi-processing and pinning to memory, etc...
    We enable the possibility to shuffle the list of indexes, it will happen once at the beginning and each time we have exhausted the finite dataset.
    """

    def __init__(self, dataset: Sized, shuffle: bool = False):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self) -> int:
        """
        Yields indefinitely an index from a (shuffled) list of indexes mapping to each image of the finite dataset.
        :return: An integer value.
        """
        if self.shuffle:
            indexes = random.sample(range(len(self.dataset)), len(self.dataset))
        else:
            indexes = list(range(len(self.dataset)))
        i = 0
        while True:
            yield indexes[i]
            i += 1
            if i >= len(indexes):
                i = 0
                if self.shuffle:
                    indexes = random.sample(range(len(self.dataset)), len(self.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Path where the data will be stored",
    )
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, required=True, help="Number of workers."
    )
    args = parser.parse_args()

    # We first create a finite dataset.
    dataset = FiniteDataset(data_root=args.data_root)
    print(len(dataset))

    # We then create the infinite sampler.
    sampler = InfiniteSampler(dataset, shuffle=True)

    # We pass the
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dataloader = iter(dataloader)

    i = 0
    while True:
        batch = next(dataloader)
        print(i, batch.shape)
        i += 1


if __name__ == "__main__":
    main()
