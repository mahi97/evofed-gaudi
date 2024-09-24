import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

# Dataset Preprocessing
def preprocess_dataset(dataset):
    """Normalize and preprocess the dataset."""
    images = torch.tensor(dataset.data).float() / 255.0  # Normalize images and convert to torch tensors
    if len(images.shape) == 3:  # If grayscale images (e.g., MNIST), add channel dimension
        images = images.unsqueeze(1)
    labels = torch.tensor(dataset.targets)
    return {'image': images, 'label': labels}

# IID data distribution (for simplicity, we can treat as regular batching here)
def distribute_data_iid(dataset, n_clients):
    """Distribute data IID among clients."""
    client_data = []
    num_samples_per_client = len(dataset['image']) // n_clients

    for _ in range(n_clients):
        indices = np.random.choice(len(dataset['image']), num_samples_per_client, replace=False)
        shard = {'image': dataset['image'][indices], 'label': dataset['label'][indices]}
        client_data.append(shard)
    
    return client_data

# Function to dynamically load datasets based on the dataset name
def load_dataset(dataset_name: str, train=True):
    """
    Load datasets based on dataset name dynamically.
    :param dataset_name: Name of the dataset ('mnist', 'fmnist', 'cifar10').
    :param train: Load training set if True, otherwise load test set.
    :return: The loaded dataset.
    """
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset_name.lower() == "fmnist" or dataset_name.lower() == "fashionmnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return dataset

# Main function to load dataset and distribute among clients
def get_datasets(dataset_name: str, n_clients: int = 1):
    """
    Load and preprocess datasets with options for federated learning settings or regular batching.
    :param dataset_name: Name of the dataset (e.g., "MNIST", "FMNIST", "CIFAR10").
    :param n_clients: Number of clients for federated learning or regular use.
    :return: train_clients (train dataset), test_clients (test dataset)
    """
    # Load the train and test datasets
    train_ds = load_dataset(dataset_name, train=True)
    test_ds = load_dataset(dataset_name, train=False)

    # Preprocess datasets
    train_ds = preprocess_dataset(train_ds)
    test_ds = preprocess_dataset(test_ds)

    # If more than one client is specified, distribute the dataset across clients
    if n_clients > 1:
        train_clients = distribute_data_iid(train_ds, n_clients)
        test_clients = distribute_data_iid(test_ds, n_clients)
    else:
        train_clients = [train_ds]  # No distribution, treat as a single client
        test_clients = [test_ds]

    return train_clients[0], test_clients[0]  # Return the dataset for a single client or regular use

