import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# class SiameseNetwork(nn.Module):
#     def __init__(self, output_size=10, channels=50):
#         super(SiameseNetwork, self).__init__()
#         self.conv1 = nn.Conv1d(1, channels, 50, stride=2, padding=1)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv1d(channels, channels, 10, stride=2, padding=1)
#         self.relu2 = nn.ReLU()
#         self.fc1 = nn.Linear(383450, 256) # flattened channels -> 10 (assumes input has dim 50)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 10)
        
#     def forward_once(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         #print(x.shape)
#         x = self.relu2(x)
#         x = self.fc1(x.flatten(start_dim=1)) # flatten here

#         x = self.relu3(x)
#         x = self.fc2(x)
        
#         return x

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, output_size=10, channels=50, kernel_sizes=[50, 10], strides=[2, 2], paddings=[1, 1], hidden_sizes=[256,128]):
        super(SiameseNetwork, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList() 
        self.fc_layers = nn.ModuleList() 
        self.relu_fc_layers = nn.ModuleList() 
        in_channels = 1
        for i in range(len(kernel_sizes)):
            self.conv_layers.append(nn.Conv1d(in_channels, channels, kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            self.relu_layers.append(nn.ReLU())
            in_channels = channels
        # define the input size for the Linear layer by calculating the flattened output of convolution layers 
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.fc_layers.append(nn.Linear(self.calculate_flattened_size(input_size), hidden_sizes[i]))
                self.relu_fc_layers.append(nn.ReLU())
            else:
                self.fc_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                self.relu_fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], output_size))
            

    def forward_once(self, x):
        for conv_layer, relu_layer in zip(self.conv_layers, self.relu_layers):
            x = conv_layer(x)
            x = relu_layer(x)

        x = x.flatten(start_dim=1)
        for fc_layer, relu_fc_layer in zip(self.fc_layers[:-1], self.relu_fc_layers):
            x = fc_layer(x)
            x = relu_fc_layer(x)
        x = self.fc_layers[-1](x)
        return x
        

    def calculate_flattened_size(self, input_size):
          # use a random tensor to calculate the flattened size at runtime
          with torch.no_grad():
              x = torch.zeros(1, 1, input_size)
              for conv_layer, relu_layer in zip(self.conv_layers, self.relu_layers):
                  x = conv_layer(x)
                  x = relu_layer(x)
              return int(torch.prod(torch.tensor(x.size())))

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



# def triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin): #not used in the end, there is an pytorch function for this
#     distance_positive = torch.norm(anchor_embedding - positive_embedding, dim=1)
#     distance_negative = torch.norm(anchor_embedding - negative_embedding, dim=1)
#     loss = torch.relu(distance_positive - distance_negative + margin)
#     return loss.mean()



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


# Prepare triplet data
def prepare_triplets(data, labels):
    labels = np.array(labels)
    data = np.array(data)
    triplets = []
    label_to_indices = {} # a list of indices is provided for each label, ordering is from the most populated label to the smallest. Note that label_to_indices[i] gives all indices for the label 'i', NOT the i-th element!
    for i, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
    
    for i in range(len(data)):
        anchor = data[i]
        anchor_label = labels[i]

        # choose positive sample from same class as anchor
        positive_index = i
        while positive_index == i:
            positive_index = label_to_indices[anchor_label][torch.randint(len(label_to_indices[anchor_label]), size=(1,)).item()]
        positive = data[positive_index]
        
        # choose negative sample from a different class than anchor
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = labels[torch.randint(len(labels), size=(1,)).item()]
        negative_index = label_to_indices[negative_label][torch.randint(len(label_to_indices[negative_label]), size=(1,)).item()]
        negative = data[negative_index]
        
        triplets.append((anchor, positive, negative))
    
    return triplets


from collections import Counter
import random

def prepare_balanced_triplets(data, labels):
    """
    data: numpy array of data samples
    labels: numpy array of corresponding sample labels
    """

    # Count number of samples for each class
    label_count = dict(Counter(labels))

    # Find label indices for each element in data
    label_to_indices = np.empty((len(np.unique(labels)), max(label_count.values())), dtype=int)
    label_to_indices.fill(-1)

    row_idx_map = {}
    for i, label in enumerate(labels):
        if label not in row_idx_map:
            row_idx_map[label] = 0
        idx = row_idx_map[label]
        label_to_indices[label][idx] = i
        row_idx_map[label] += 1
    
    # Set threshold using maximum number of samples among all classes
    max_sample_count = max(label_count.values())
    threshold = int(max_sample_count * 0.3) # Consider samples up to 80% of max
    
    balanced_triplets = []
    for i in range(len(data)):
        anchor = data[i]
        anchor_label = labels[i]

        # Sample positive example from the same class
        anchor_positive_indices = label_to_indices[anchor_label][:label_count[anchor_label]]
        positive_index = np.random.choice(anchor_positive_indices)
        while positive_index == i:
            positive_index = np.random.choice(anchor_positive_indices)
        positive = data[positive_index]
        
        # Sample negative example
        for _ in range(10): # attempt a few times to find a suitable negative example
            random_label = np.random.choice(labels)
            if label_count[random_label] < threshold: # undersampled class
                random_index = label_to_indices[random_label][np.random.choice(label_count[random_label])] 
            else: # oversampled class
                negative_indices = label_to_indices[random_label][label_to_indices[random_label] != positive_index]
                random_index = np.random.choice(negative_indices)                                      
            negative = data[random_index]                               
            if np.linalg.norm(anchor - negative) > np.linalg.norm(anchor - positive):
                break

        # Add triplet to list
        balanced_triplets.append((anchor, positive, negative))

    return balanced_triplets
