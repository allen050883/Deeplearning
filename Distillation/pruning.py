import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from time import time
import copy
import pandas as pd
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_MNIST():
    """Function to load and normalize MNIST data""" 
    train = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ]))
    test = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ]))
    print("MNIST datset loaded and normalized.")
    train_loader = torch.utils.data.DataLoader(dataset=train, shuffle=True, batch_size=100)
    test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=False, batch_size=100)
    print("PyTorch DataLoaders loaded.")
    return train, test, train_loader, test_loader


def visualize_MNIST(train_loader):
    """Function to visualize data given a DataLoader object"""
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print("image shape:", images.shape, "\n label shape:", labels.shape)
    # visualize data
    fig, ax = plt.subplots(2,5)
    for i, ax in enumerate(ax.flatten()):
        im_idx = np.argwhere(labels == i)[0][0]
        plottable_image = images[im_idx].squeeze()
        ax.imshow(plottable_image)


# load and visualize MNISt
train, test, train_loader, test_loader = load_MNIST()
visualize_MNIST(train_loader)



class Net(nn.Module):
    """A non-sparse neural network with four hidden fully-connected layers"""

    def __init__(self):
        super(Net,self).__init__()
        self.input_layer = nn.Linear(784, 1000, bias=False)
        self.hidden1_layer = nn.Linear(1000, 1000, bias=False)
        self.hidden2_layer = nn.Linear(1000, 500, bias=False)
        self.hidden3_layer = nn.Linear(500, 200, bias=False)
        self.hidden4_layer = nn.Linear(200, 10, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden1_layer(x)
        x = F.relu(x)
        x = self.hidden2_layer(x)
        x = F.relu(x)
        x = self.hidden3_layer(x)
        x = F.relu(x)
        x = self.hidden4_layer(x)
        output = F.log_softmax(x, dim=1)

        return output


def train(model, train_loader, epochs=3, learning_rate=0.001):
    """Function to train a neural net"""

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    time0 = time()
    total_samples = 0 

    for e in range(epochs):
        print("Starting epoch", e)
        total_loss = 0

        for idx, (images,labels) in enumerate(train_loader):
            # images = images.view(images.shape[0],-1) # flatten
            optimizer.zero_grad() # forward pass
            output = model(images)
            loss = lossFunction(output,labels) # calculate loss
            loss.backward() # backpropagate
            optimizer.step() # update weights

            total_samples += labels.size(0)
            total_loss += loss.item()

            if idx % 100 == 0:
                print("Running loss:", total_loss)

    final_time = (time()-time0)/60 
    print("Model trained in ", final_time, "minutes on ", total_samples, "samples")


model = Net()
print(summary(Net().cuda(), input_size=(1, 28 ,28), batch_size=-1))
train(model, train_loader)


def test(model, test_loader):
    """Test neural net"""

    correct = 0
    total = 0 

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.view(images.shape[0],-1) # flatten
            output = model(images)
            values, indices = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (labels == indices).sum().item()

        acc = correct / total * 100
        # print("Accuracy: ", acc, "% for ", total, "training samples")

    return acc

acc = test(model, test_loader)
print("The accuracy of our vanilla NN is", acc, "%")



def sparsify_by_weights(model, k):
    """Function that takes un-sparsified neural net and does weight-pruning
    by k sparsity"""

    # make copy of original neural net
    sparse_m = copy.deepcopy(model)

    with torch.no_grad():
        for idx, i in enumerate(sparse_m.parameters()): 
            if idx == 4: # skip last layer of 5-layer neural net
              break 
            # change tensor to numpy format, then set appropriate number of smallest weights to zero
            layer_copy = torch.flatten(i)
            layer_copy = layer_copy.detach().numpy()
            indices = abs(layer_copy).argsort() # get indices of smallest weights by absolute value
            indices = indices[:int(len(indices)*k)] # get k fraction of smallest indices 
            layer_copy[indices] = 0 

            # change weights of model
            i = torch.from_numpy(layer_copy)
    
    return sparse_m  


def l2(array):
  return np.sqrt(np.sum([i**2 for i in array]))

def sparsify_by_unit(model, k):
    """Creates a k-sparsity model with unit-level pruning that sets columns with smallest L2 to zero."""
    
    # make copy of original neural net
    sparse_m = copy.deepcopy(model)

    for idx, i in enumerate(sparse_m.parameters()):
        if idx == 4: # skip last layer of 5-layer neural net
            break
        layer_copy = i.detach().numpy()
        indices = np.argsort([l2(i) for i in layer_copy])
        indices = indices[:int(len(indices)*k)]
        layer_copy[indices,:] = 0
        i = torch.from_numpy(layer_copy)
    
    return sparse_m 

def get_pruning_accuracies(model, prune_type):
    """ Takes a model and prune type ("weight" or "unit") and returns a DataFrame of pruning accuracies for given sparsities."""

    df = pd.DataFrame({"sparsity": [], "accuracy": []})
    sparsities = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

    for s in sparsities:
        if prune_type == "weight":
            new_model = sparsify_by_weights(model, s)
        elif prune_type == "unit":
            new_model = sparsify_by_unit(model, s)
        else:
            print("Must specify prune-type.")
            return 
        acc = test(new_model, test_loader)
        df = df.append({"sparsity": s, "accuracy": acc}, ignore_index=True)
    # print(summary(new_model.cuda(), input_size=(1, 28 ,28), batch_size=-1))

    return df 


df_weight = get_pruning_accuracies(model, "weight")
df_unit = get_pruning_accuracies(model, "unit")

print("Accuracies for Weight Pruning")
print(df_weight)

print()

print("Accuracies for Unit Pruning")
print(df_unit)


plt.figure()
plt.title("Accuracy vs Sparsity")
plt.plot(df_unit["sparsity"], df_unit["accuracy"], label="Unit-pruning")
plt.plot(df_weight["sparsity"], df_weight["accuracy"], label="Weight-pruning")
plt.xlabel("Sparsity (as fraction)")
plt.ylabel("% Accuracy")
plt.legend()
