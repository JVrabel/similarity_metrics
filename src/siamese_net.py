import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SiameseNetwork(nn.Module):
    def __init__(self, output_size=10, channels=50):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 50, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, 10, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(1*25*50*400-750, 256) # flattened channels -> 10 (assumes input has dim 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        
    def forward_once(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu2(x)
        x = self.fc1(x.flatten(start_dim=1)) # flatten here

        x = self.relu3(x)
        x = self.fc2(x)
        
        return x
        
    def forward(self, anchor, positive, negative):
        anchor_embedding = self.forward_once(anchor)
        positive_embedding = self.forward_once(positive)
        negative_embedding = self.forward_once(negative)
        return anchor_embedding, positive_embedding, negative_embedding

    def pairwise_similarity(self, x1, x2):
        # Get embeddings for the two inputs
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        sim = torch.cdist(emb1, emb2, p=2)
        return sim

def triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin): #not used in the end, there is an pytorch function for this
    distance_positive = torch.norm(anchor_embedding - positive_embedding, dim=1)
    distance_negative = torch.norm(anchor_embedding - negative_embedding, dim=1)
    loss = torch.relu(distance_positive - distance_negative + margin)
    return loss.mean()


# Prepare triplet data
def prepare_triplets(data, labels):
    labels = np.array(labels)
    data = np.array(data)
    triplets = []
    label_to_indices = {}
    for i, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
    
    for i in range(len(data)):
        anchor = data[i]
        anchor_label = labels[i]
        positive_index = i
        while positive_index == i:
            positive_index = label_to_indices[anchor_label][torch.randint(len(label_to_indices[anchor_label]), size=(1,)).item()]
        positive = data[positive_index]
        
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = labels[torch.randint(len(labels), size=(1,)).item()]
        negative_index = label_to_indices[negative_label][torch.randint(len(label_to_indices[negative_label]), size=(1,)).item()]
        negative = data[negative_index]
        
        triplets.append((anchor, positive, negative))
    
    return triplets

# Train the network
def train(net, optimizer, criterion, train_loader, margin, epochs, batch_size):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            anchor_batch, positive_batch, negative_batch = batch
            # optimizer.zero_grad()
            # anchor_embedding, positive_embedding, negative_embedding = net(anchor_batch, positive_batch, negative_batch)
            # loss = criterion(anchor_embedding, positive_embedding, negative_embedding, margin)
            # loss.backward()
            # optimizer.step()
            # running_loss += loss.item()
        
        print('Epoch %d, loss: %.3f' % (epoch+1, running_loss / len(train_loader)))

