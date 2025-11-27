from typing import Optional

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    """LightningDataModule for MNIST.

    Reusable pattern: prepare_data, setup, train/val/test_dataloader.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        """Download MNIST if needed. Called only from one process."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Split train into train/val and set up test dataset."""
        if stage in (None, "fit"):
            full_train = datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            n_total = len(full_train)
            n_val = int(self.val_split * n_total)
            n_train = n_total - n_val
            self.mnist_train, self.mnist_val = random_split(
                full_train, [n_train, n_val]
            )

        if stage in (None, "test", "predict"):
            self.mnist_test = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
