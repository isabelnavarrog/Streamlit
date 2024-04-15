from cnn import CNN
import torchvision
from cnn import load_data
from cnn import load_model_weights
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Deep Learning",
    name="resnet50-30epoch-20unfreeze",

    # track hyperparameters and run metadata
    config={
    "dataset": "dataset",
    "epochs": 30,
    "configuration": "custom",
    "batch_size": 32,
    "img_size": 224,
    "unfreezed_layers" :20}
)

# Load data and model 
train_dir = './dataset/training'
valid_dir = './dataset/validation'

train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=32, 
                                                    img_size=224) # ResNet50 requires 224x224 images
model = CNN(torchvision.models.resnet152(weights='DEFAULT'), num_classes, unfreezed_layers=20)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
history = model.train_model(train_loader, valid_loader, optimizer, criterion, epochs=30)
model.save('resnet50-30epoch-20unfreeze')

# simulate training
for epoch in range(len(history['train_loss'])):
    wandb.log({"train_loss": history['train_loss'][epoch],
               "train_accuracy": history['train_accuracy'][epoch],
               "valid_loss": history['valid_loss'][epoch],
               "valid_accuracy": history['valid_accuracy'][epoch],
               "epoch":epoch+1})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()