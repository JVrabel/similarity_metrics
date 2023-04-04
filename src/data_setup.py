"""
Contains functionality for creating PyTorch DataLoaders for 
LIBS benchmark classification dataset.
"""

import os
import torch
from torch.utils.data import DataLoader
from src.load_libs_data import load_contest_train_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
from src.siamese_net import prepare_triplets


NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int=128, 
    num_workers: int=NUM_WORKERS, 
    split_rate: float=0.3,
    random_st: int=42,
    spectra_count: int=50
    ):
    """Creates training and validation DataLoaders.

    Takes in a training directory directory path and split the data
    to train/validation. After, it turns them into PyTorch Datasets and 
    then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader, class_labels).
    Example usage:
        train_dataloader, test_dataloader, class_labels, wavelengths = \
        = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """



X, y, samples = load_contest_train_dataset('datasets/train.h5', spectra_count)
wavelengths = X.columns

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_rate, random_state=random_st, stratify=y, shuffle = True)
del X, y, samples

if True:
  scaler =  Normalizer(norm = 'max')
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.fit_transform(X_val)

# Convert data to torch tensors
X_train = torch.from_numpy(X_train).unsqueeze(1).float() # Add extra dimension for channels
X_val = torch.from_numpy(X_val).unsqueeze(1).float() # Add extra dimension for channels
y_train = torch.from_numpy(np.array(y_train)).long()
y_val = torch.from_numpy(np.array(y_val)).long()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If available, move data to the GPU
X_train.to(device)
X_val.to(device) 
y_train.to(device)
y_val.to(device)

# Prepare triplets for the training, validation
train_triplets = prepare_triplets(X_train, y_train)
val_triplets = prepare_triplets(X_val, y_val)


# Create PyTorch DataLoader objects for the training and validation sets
train_loader = DataLoader(train_triplets, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_triplets, batch_size=batch_size, shuffle=True)


return train_loader, val_loader, y_train
