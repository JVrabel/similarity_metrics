"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, siamese_net, utils
import numpy as np
import torch.nn as nn


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 128
HIDDEN_UNITS = 10
LEARNING_RATE = 0.0001
INPUT_SIZE = 40000
CHANNELS=50
KERNEL_SIZES=[50, 10]
STRIDES=[2, 2]
PADDINGS=[1, 1]
HIDDEN_SIZES=[256,128,64]

# Setup directories
train_dir = "datasets/train.h5"
#test_dir = "datasets/test.h5"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader, train_labels = data_setup.create_dataloaders(
    train_dir=train_dir,
    #test_dir=test_dir,
    batch_size=BATCH_SIZE
)

# Create model with help from siamese_net.py
# margin = 1.0

model = siamese_net.SiameseNetwork(
    input_size=INPUT_SIZE, 
    output_size=len(np.unique(train_labels)), 
    channels=CHANNELS, 
    kernel_sizes=KERNEL_SIZES, 
    strides=STRIDES, 
    paddings=PADDINGS, 
    hidden_sizes=HIDDEN_SIZES
).to(device)

# Set loss and optimizer
loss_fn = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             val_dataloader=val_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="modularity_test1.pth")
