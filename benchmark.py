import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import numpy as np

# Set device function
def set_device(device_type='cuda'):
    """Set device to CUDA or Habana."""
    if device_type == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_type == 'habana':
        try:
            import habana_frameworks.torch.core as htcore
            device = torch.device('hpu')
        except ImportError:
            raise ImportError("Habana device not found. Please install necessary packages.")
    else:
        device = torch.device('cpu')
    return device

# Preprocess the dataset
def preprocess_dataset(dataset):
    """Normalize and preprocess the dataset."""
    dataset.data = dataset.data.float() / 255.0  # Normalize the images
    return dataset

# Pad or trim client data
def pad_or_trim_data(client_data, target_size):
    """Pad or trim client data to have the same number of samples."""
    for client in client_data:
        num_samples = len(client['image'])
        if num_samples < target_size:
            # Padding
            padding_images = torch.zeros((target_size - num_samples,) + client['image'].shape[1:])
            padding_labels = torch.zeros((target_size - num_samples,) + client['label'].shape[1:])
            client['image'] = torch.cat([client['image'], padding_images])
            client['label'] = torch.cat([client['label'], padding_labels])
        elif num_samples > target_size:
            # Trimming
            client['image'] = client['image'][:target_size]
            client['label'] = client['label'][:target_size]
    return client_data

# IID Data distribution
def distribute_data_iid(dataset, n_clients, n_shards_per_client):
    """Distribute data IID among clients."""
    client_data = []
    num_samples_per_client = len(dataset.data) // n_clients

    for _ in range(n_clients):
        client_shards = []
        for _ in range(n_shards_per_client):
            indices = np.random.choice(len(dataset.data), num_samples_per_client, replace=False)
            shard = {'image': dataset.data[indices], 'label': dataset.targets[indices]}
            client_shards.append(shard)

        client_data.append({'image': torch.cat([shard['image'] for shard in client_shards], dim=0),
                            'label': torch.cat([shard['label'] for shard in client_shards], dim=0)})
    return client_data

# Non-IID Data distribution
def distribute_data_non_iid(dataset, n_clients, n_shards_per_client):
    """Distribute data non-IID among clients."""
    client_data = []
    num_samples_per_shard = len(dataset.data) // (n_clients * n_shards_per_client)
    num_classes = len(torch.unique(dataset.targets))
    class_indices = [torch.where(dataset.targets == i)[0] for i in range(num_classes)]

    for _ in range(n_clients):
        client_shards = []
        for _ in range(n_shards_per_client):
            selected_classes = np.random.choice(num_classes, 2, replace=False)
            indices = torch.cat([torch.tensor(np.random.choice(class_indices[cls], num_samples_per_shard // 2, replace=False)) for cls in selected_classes])
            shard = {'image': dataset.data[indices], 'label': dataset.targets[indices]}
            client_shards.append(shard)

        client_data.append({'image': torch.cat([shard['image'] for shard in client_shards], dim=0),
                            'label': torch.cat([shard['label'] for shard in client_shards], dim=0)})
    return client_data

# Load datasets and distribute them based on IID or non-IID
def get_datasets(dataset_name: str,
                 n_clients: int = 1,
                 n_shards_per_client: int = 2,
                 iid: bool = True,
                 use_max_padding: bool = False,
                 device_type: str = 'cuda'):
    """
    Load and preprocess datasets with options for federated learning settings.
    """
    device = set_device(device_type)

    # Load dataset (using MNIST as example)
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported!")

    # Preprocess the datasets
    train_dataset = preprocess_dataset(train_dataset)

    if n_clients > 1:
        if iid:
            train_clients = distribute_data_iid(train_dataset, n_clients, n_shards_per_client)
        else:
            train_clients = distribute_data_non_iid(train_dataset, n_clients, n_shards_per_client)

        # Determine the size to pad or trim
        sizes = [len(client['image']) for client in train_clients]
        target_size = max(sizes) if use_max_padding else min(sizes)
        train_clients = pad_or_trim_data(train_clients, target_size)
        reshaped_clients = {'image': [], 'label': []}
        for client in train_clients:
            reshaped_clients['image'].append(client['image'].to(device))
            reshaped_clients['label'].append(client['label'].to(device))

        reshaped_clients['image'] = torch.stack(reshaped_clients['image'])
        reshaped_clients['label'] = torch.stack(reshaped_clients['label'])
        return reshaped_clients
    else:
        train_dataset.data = train_dataset.data.unsqueeze(0).to(device)
        return train_dataset

# Create batches of data
def create_batches(x, y, batch_size, device_type='cpu'):
    """
    Create batches of data with the given batch size using PyTorch.
    """
    device = set_device(device_type)
    
    # Ensure data is moved to the correct device
    x = x.to(device)
    y = y.to(device)
    
    train_ds_size = x.shape[0]
    
    # Adjust batch size to ensure it doesn't exceed the dataset size
    if batch_size > train_ds_size:
        batch_size = train_ds_size
        print(f"Batch size adjusted to {batch_size} because it exceeds the dataset size")

    steps_per_epoch = train_ds_size // batch_size
    
    # Create a permutation of the indices
    perms = torch.randperm(train_ds_size, device=device)[:steps_per_epoch * batch_size]
    perms = perms.view(steps_per_epoch, batch_size)
    
    # Rearrange the data according to the permuted indices
    xb = x[perms]
    yb = y[perms]
    
    return xb, yb

# Time expensive matrix operation
def expensive_matrix_operation(x):
    """Example of an expensive matrix operation."""
    return torch.matmul(x, x.transpose(-2, -1))

# Benchmarking expensive matrix operation
def benchmark_operation(xb, yb, device_type='cuda'):
    device = set_device(device_type)
    xb = xb.to(device)
    yb = yb.to(device)
    
    start_time = time.time()
    
    for i in range(xb.shape[0]):  # Loop over batches
        batch_x = xb[i]
        result = expensive_matrix_operation(batch_x)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for matrix operation on all batches: {elapsed_time:.4f} seconds")

# Benchmark loading datasets in IID and non-IID modes
def benchmark_data_loading_iid(dataset_name, n_clients, n_shards_per_client, use_max_padding, batch_size, device_type):
    start_time = time.time()
    
    # Load IID datasets
    reshaped_clients = get_datasets_with_batches(dataset_name, 
                                                 n_clients=n_clients, 
                                                 n_shards_per_client=n_shards_per_client, 
                                                 iid=True, 
                                                 use_max_padding=use_max_padding, 
                                                 batch_size=batch_size, 
                                                 device_type=device_type)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to load IID dataset with {n_clients} clients and {n_shards_per_client} shards: {elapsed_time:.4f} seconds")
    return reshaped_clients

def benchmark_data_loading_non_iid(dataset_name, n_clients, n_shards_per_client, use_max_padding, batch_size, device_type):
    start_time = time.time()
    
    # Load Non-IID datasets
    reshaped_clients = get_datasets_with_batches(dataset_name, 
                                                 n_clients=n_clients, 
                                                 n_shards_per_client=n_shards_per_client, 
                                                 iid=False, 
                                                 use_max_padding=use_max_padding, 
                                                 batch_size=batch_size, 
                                                 device_type=device_type)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to load Non-IID dataset with {n_clients} clients and {n_shards_per_client} shards: {elapsed_time:.4f} seconds")
    return reshaped_clients

def get_datasets_with_batches(dataset_name: str,
                              n_clients: int = 1,
                              n_shards_per_client: int = 2,
                              iid: bool = True,
                              use_max_padding: bool = False,
                              batch_size: int = 32,
                              device_type: str = 'cuda'):
    """
    Load and preprocess datasets with options for federated learning settings and batching.
    """
    reshaped_clients = get_datasets(dataset_name, n_clients, n_shards_per_client, iid, use_max_padding, device_type)

    if n_clients == 1:
        xb, yb = create_batches(reshaped_clients.data, reshaped_clients.targets, batch_size, device_type)
        return xb, yb
    else:
        batched_clients = []
        for client in reshaped_clients['image']:
            xb, yb = create_batches(client, reshaped_clients['label'], batch_size, device_type)
            batched_clients.append({'image': xb, 'label': yb})
        return batched_clients

# Main benchmarking loop
if __name__ == '__main__':
    dataset_name = "mnist"
    n_clients_list = [1, 5, 10]
    n_shards_per_client_list = [2, 5]
    batch_size = 64
    device_type = 'cuda'
    use_max_padding = False
    
    for n_clients in n_clients_list:
        for n_shards_per_client in n_shards_per_client_list:
            # Benchmark IID data loading
            reshaped_clients_iid = benchmark_data_loading_iid(dataset_name, n_clients, n_shards_per_client, use_max_padding, batch_size, device_type)
            
            # Benchmark Non-IID data loading
            reshaped_clients_non_iid = benchmark_data_loading_non_iid(dataset_name, n_clients, n_shards_per_client, use_max_padding, batch_size, device_type)
            
            # Benchmark batch creation for IID data
            print(f"Benchmarking batch creation for IID data with {n_clients} clients...")
            if n_clients == 1:
                batch_x_iid, batch_y_iid = reshaped_clients_iid  # Now handle as batch tuple (xb, yb)
            else:
                for client in reshaped_clients_iid['image']:
                    batch_x_iid, batch_y_iid = create_batches(client, reshaped_clients_iid['label'], batch_size, device_type)
            
            # Benchmark batch creation for Non-IID data
            print(f"Benchmarking batch creation for Non-IID data with {n_clients} clients...")
            if n_clients == 1:
                batch_x_non_iid, batch_y_non_iid = reshaped_clients_non_iid  # Handle as batch tuple (xb, yb)
            else:
                for client in reshaped_clients_non_iid['image']:
                    batch_x_non_iid, batch_y_non_iid = create_batches(client, reshaped_clients_non_iid['label'], batch_size, device_type)
