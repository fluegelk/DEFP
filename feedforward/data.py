import pathlib

import torch
import torchvision


class SynthDataset(torch.utils.data.Dataset):
    """A map-style dataset based on a .pt file created by DRTPs synth_dataset_gen.py"""

    def __init__(self, name, path='./Dataset'):
        """
        Initialize and load the dataset from the given path and name (i.e. load the dataset from {path}/{name}.pt).

        Args:
            name: Name of the file to load.
            path: Path to load the dataset from.
        """
        file_path = pathlib.Path(path) / f'{name}.pt'
        self.dataset, self.input_size, self.input_channels, self.label_features = torch.load(file_path)
        self.classes = list(range(self.label_features))

    def __len__(self):
        """
        The length of the dataset (i.e. the number of samples).

        Returns:
            The length of the dataset.
        """
        return len(self.dataset[1])

    def __getitem__(self, index):
        """
        Get the data sample for the given index.

        Args:
            index: The index of the requested sample.

        Returns:
            The sample at the given index as tuple of the input data and the target.
        """
        return self.dataset[0][index], self.dataset[1][index]


class DatasetWithIndex(torch.utils.data.Dataset):
    """
    Wraps a given dataset to get item returns the triple (index, data, target) instead of the tuple (data, target).
    """

    def __init__(self, dataset):
        """
        Create the wrapping dataset.

        Args:
            dataset: The dataset to wrap.
        """
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Get the data for the given index as triple (index, data, target)
        Args:
            index: The index of the data to get.

        Returns:
            The data for the given index as triple (index, data, target)
        """
        data, target = self.dataset[index]
        return index, data, target

    def __len__(self):
        """
        The length of the dataset (i.e. the number of samples).

        Returns:
            The length of the dataset.
        """
        return len(self.dataset)


def create_data_loaders(train_set, train_test_set, test_set, batch_size, test_batch_size, shuffle_train_set=True,
                        **kwargs):
    """
    Create data loaders for the given train, train-test, and test set. The dataset are first wrapped in
    DatasetWithIndex, so the data loaders return triples (index, data, target) instead of the usual (data, target).

    Args:
        train_set: The train set.
        train_test_set: The train set but for validation (e.g. with other augmentations etc.). If None, the train set
        will be used.
        test_set: The test set.
        batch_size: The batch size for training.
        test_batch_size: The batch size for testing.
        shuffle_train_set: Whether to shuffle the train set, default True.
        **kwargs: Additional arguments to the DataLoader constructor.

    Returns:
        The three created data loaders for the train, train-test, and test set.
    """
    if train_test_set is None:
        train_test_set = train_set

    train_loader = torch.utils.data.DataLoader(
        DatasetWithIndex(train_set), batch_size=batch_size, shuffle=shuffle_train_set, **kwargs)
    train_test_loader = torch.utils.data.DataLoader(
        DatasetWithIndex(train_test_set), batch_size=test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        DatasetWithIndex(test_set), batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, train_test_loader, test_loader


def load_synthetic_classification_set(path='./DATASETS/classification'):
    """
    Load the synthetic classification dataset.

    Args:
        path: Path to the directory containing the train.pt and test.pt files.

    Returns:
        The train, train-test, and test set.
    """
    train_set = SynthDataset("train", path)
    test_set = SynthDataset("test", path)
    return train_set, train_set, test_set


def load_mnist(path='./DATASETS'):
    """
    Load the MNIST dataset. Will download the data from the web if it cannot be found in the given directory.

    Args:
        path: Path to the directory containing data/to which to download the data if none is found.

    Returns:
        The train, train-test, and test set.
    """
    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.0,), (1.0,))])
    train_set = torchvision.datasets.MNIST(path, train=True, download=True, transform=transform_mnist)
    test_set = torchvision.datasets.MNIST(path, train=False, download=True, transform=transform_mnist)
    return train_set, train_set, test_set


def load_cifar10(path='./DATASETS', augment=False):
    """
    Load the CIFAR10 dataset. Will download the data from the web if it cannot be found in the given directory.

    Args:
        path: Path to the directory containing data/to which to download the data if none is found.
        augment: Whether to augment the training data by performing random horizontal flips with a probability of 50%.
        Default is False.

    Returns:
        The train, train-test, and test set.
    """
    normalize = torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_cifar10 = transform_cifar10_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                 normalize, ])
    if augment:
        transform_cifar10 = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                            torchvision.transforms.ToTensor(), normalize, ])

    train_set = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=transform_cifar10)
    train_test_set = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=transform_cifar10_test)
    test_set = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=transform_cifar10_test)
    return train_set, train_test_set, test_set


def get_data_dimensions(dataset):
    """
    Get the input and target dimensions of the given classification dataset.

    Args:
        dataset: The dataset for which to get the dimensions.

    Returns:
        The input size (i.e. the shape of one input sample) and the number of classes.
    """
    data, _ = dataset[0]
    return data.shape, len(dataset.classes)


def prepare_data(dataset_name, batch_size, test_batch_size, shuffle_train_set=True, data_loader_kwargs=None):
    """
    Load the given dataset, create data loaders (for the train, train-test, and test set), and determine the model
    in- and output size corresponding to the selected dataset.

    Args:
        dataset_name: The dataset to prepare, one of: "classification_synth", "MNIST", "CIFAR10", "CIFAR10aug"
        batch_size: The batch size for training.
        test_batch_size: The batch size for testing.
        shuffle_train_set: Whether to shuffle the training set, default is True.
        data_loader_kwargs: Optional additional arguments to the data loader constructor.

    Returns:
        A 5-tuple of: train data loader, train-test data loader, test data loader, shape of one input sample, number of
        classes.
    """
    data_loader_kwargs = {} if data_loader_kwargs is None else data_loader_kwargs

    load_dataset = {
        "classification_synth": lambda: load_synthetic_classification_set(),
        "MNIST": lambda: load_mnist(),
        "CIFAR10": lambda: load_cifar10(),
        "CIFAR10aug": lambda: load_cifar10(augment=True),
    }

    if dataset_name not in load_dataset:
        raise ValueError(f'Invalid dataset {dataset_name}. Valid datasets are: {", ".join(load_dataset.keys())}.')
    train_set, train_test_set, test_set = load_dataset[dataset_name]()

    train_loader, train_test_loader, test_loader = create_data_loaders(
        train_set, train_test_set, test_set, batch_size, test_batch_size, shuffle_train_set, **data_loader_kwargs)

    input_size, number_of_classes = get_data_dimensions(train_set)

    return train_loader, train_test_loader, test_loader, input_size, number_of_classes
