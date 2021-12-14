import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    """MNIST Data module: train / val / test sets & data loaders
    """
    def __init__(self, data_dir: str=".", batch_size: int=64,
                 num_workers: int=4) -> None:
        """Loading MNIST dataset and data loaders

        MNIST data is coded as:

        * Sample: image of shape 28 x 28 x 1
        * Target: class id represented as integer

        Args:
            data_dir (str, optional): Path to store data. Defaults to ".".
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Loader processes. Defaults to 4.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Logging
        logging.info(f"data_dir: {data_dir}")
        logging.info(f"batch_size: {batch_size}")
        logging.info(f"num_workers: {num_workers}")

    def setup(self, stage: str=None) -> None:
        """Set up the datasets subsets: train / valid / test

        Args:
            stage (str, optional): Used internally by PL. Defaults to None.
        """
        # MNIST train set
        mnist = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        # Splitting train set into train/valid
        self.mnist_train, self.mnist_valid = random_split(mnist, [55000, 5000])

        # MNIST test set
        self.mnist_test = datasets.MNIST(
            root='.',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        # Logging
        logging.info("MNIST Data Module")
        logging.info(f"Train: {len(self.mnist_train)}")
        logging.info(f"Valid: {len(self.mnist_valid)}")
        logging.info(f"Test: {len(self.mnist_test)}")

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the train dataset"""
        return DataLoader(self.mnist_train, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation dataset"""
        return DataLoader(self.mnist_valid, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Returns the dataloader for the test dataset"""
        return DataLoader(self.mnist_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


def test_MNISTDataModule() -> None:
    """Simple code test for data module

    This test will be run automatically with pytest
    """
    # Testing intance and loading
    dm = MNISTDataModule(batch_size=8)
    dm.setup()

    # Testing dataset sizes
    assert len(dm.mnist_train) == 55000
    assert len(dm.mnist_valid) == 5000
    assert len(dm.mnist_test) == 10000

    # Testing data loading
    batch_sample_train, batch_target_train = next(iter(dm.train_dataloader()))
    batch_sample_valid, batch_target_valid = next(iter(dm.val_dataloader()))
    batch_sample_test, batch_target_test = next(iter(dm.test_dataloader()))

    # Testing batch shapes
    assert batch_sample_train.shape == (8, 1, 28, 28)
    assert batch_target_train.shape == (8,)
    assert batch_sample_valid.shape == (8, 1, 28, 28)
    assert batch_target_valid.shape == (8,)
    assert batch_sample_test.shape == (8, 1, 28, 28)
    assert batch_target_test.shape == (8,)


if __name__ == "__main__":
    # Quick testing
    logging.basicConfig(level=logging.INFO, format= '%(asctime)s [%(levelname)s] %(message)s')
    test_MNISTDataModule()
