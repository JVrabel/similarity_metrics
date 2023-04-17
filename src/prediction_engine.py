"""
Contains functionality for creating PyTorch DataLoaders for 
LIBS benchmark classification dataset.
"""

import os
import torch
from torch.utils.data import DataLoader
from load_libs_data import load_contest_test_dataset, load_contest_train_dataset
from sklearn.preprocessing import Normalizer, MinMaxScaler
import numpy as np


def create_dataloaders(
    test_dir: str, 
    test_labels_dir: str, 
    batch_size: int, 
    device: torch.device,
    pred_test: bool,
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

    if pred_test:
        X_test = load_contest_test_dataset(test_dir)
        y_test = np.loadtxt(test_labels_dir, delimiter = ',')
    else: # use with caution, only for predicting training embeddings
        X_test, y_test, _ = load_contest_train_dataset(test_dir)

    if True:
      scaler =  Normalizer(norm = 'max')
      X_test = scaler.fit_transform(X_test)

    # Convert data to torch tensors
    X_test = torch.from_numpy(X_test).unsqueeze(1).float() # Add extra dimension for channels
    y_test = torch.from_numpy(np.array(y_test)).long()


    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If available, move data to the GPU
    X_test.to(device)
    y_test.to(device)



    # Create PyTorch DataLoader objects for the training and validation sets
    pred_test_loader = DataLoader(X_test, batch_size=batch_size)


    return pred_test_loader, y_test

def predict_test(
                model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader,
                device: torch.device,
                test_dir: str, 
                test_labels_dir: str,
                batch_size: int,
                y_test
                ):
    X_test_pred=[]
    with torch.no_grad():
        for data in dataloader:
            input = data.to(device)
            output = (model.forward_once(input)).cpu()
            output = np.array(output)
            X_test_pred.append(output)
    X_test_pred = np.concatenate(X_test_pred, axis = 0)
    return X_test_pred




