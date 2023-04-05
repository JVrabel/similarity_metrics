"""
Contains functions for training and validating a PyTorch model.
"""
import torch
import wandb
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        training loss
    """
    model.train()
    # Initialize running loss and number of batches for the training dataset
    running_loss_train = 0.0
    num_batches_train = 0

    # Iterate over the training data loader
    for i, data in enumerate(dataloader, 0):
        # Get the inputs and labels for the current batch
        inputs = [input.to(device) for input in data]
        
        input_a = inputs[0]
        input_p = inputs[1]
        input_n = inputs[2]
        
        # Compute the embeddings for the current batch
        outputs = model(input_a, input_p, input_n)


        # Split the embeddings into anchor, positive, and negative examples
        anchor = outputs[0]
        positive = outputs[1]
        negative = outputs[2]
        
        # Compute the triplet loss for the current batch
        loss_train = loss_fn(anchor, positive, negative)

        # Backpropagate the loss and update the network parameters
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update the running loss and number of batches for the training dataset
        running_loss_train += loss_train.item()
        num_batches_train += 1

    # Compute the average loss for the training dataset
    epoch_loss_train = running_loss_train / num_batches_train
    return epoch_loss_train

def val_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               device: torch.device):
    """validates a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    test loss

    """
    model.eval() 


    # Initialize running loss and number of batches for the validation dataset
    running_loss_val = 0.0
    num_batches_val = 0

    # Turn on inference context manager
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # Get the inputs and labels for the current batch

            inputs_val = [input.to(device) for input in data]

            input_a = inputs_val[0]
            input_p = inputs_val[1]
            input_n = inputs_val[2]
            # Compute the embeddings for the current batch
            outputs = model(input_a, input_p, input_n)

            # Split the embeddings into anchor, positive, and negative examples
            anchor = outputs[0]
            positive = outputs[1]
            negative = outputs[2]

            # Compute the triplet loss for the current batch
            loss_val = loss_fn(anchor, positive, negative)

            # Update the running loss and number of batches for the validation dataset
            running_loss_val += loss_val.item()
            num_batches_val += 1

    # Compute the average loss for the validation dataset
    epoch_loss_val = running_loss_val / num_batches_val
    return epoch_loss_val
        
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and validates a PyTorch model.

  Passes a target PyTorch models through train_step() and val_step()
  functions for a number of epochs, training and validating the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    val_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and validating loss as well as training and
    validating accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  val_loss: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  val_loss: [1.2641, 1.5706]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [], "val_loss": []}

  wandb.init(project='siamese_net_contest', entity='jakubv')
  wandb.watch(model)  
  # Loop through training and validating steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss = train_step(model=model,
                            dataloader=train_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            device=device)
      val_loss = val_step(model=model,
                            dataloader=val_dataloader,
                            loss_fn=loss_fn,
                            device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"val_loss: {val_loss:.4f} "
      )
      wandb.log({
        "epoch": epoch+1,
        "training_loss": train_loss, 
        "validation_loss": val_loss
        })  
      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["val_loss"].append(val_loss)
  wandb.finish()  
  # Return the filled results at the end of the epochs
  return results
