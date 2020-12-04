#!/usr/bin/env python
# coding: utf-8

###########################################################################
# Jupyter notebook in python form to run on a cluster :)
###########################################################################

import sys
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import scipy
import scipy.sparse

NUM_INGREDIENTS = 3500

ARGS = dict()
ARGS["seed"] = 1
ARGS["no_cuda"] = False
ARGS["log_interval"] = 50
ARGS["batch_size"] = 256
ARGS["test-batch-size"] = 1000

PARAMS = dict()
PARAMS["epochs"] = 10


###########################################################################
# Fxns for getting data/formatting
###########################################################################

def split_train_val_test(recipes, train=0.8, val=0.1, dataset_size=200000):
    shuffled = np.random.RandomState(0).permutation(recipes)
    n_train = int(len(shuffled) * train * dataset_size / len(recipes)) # * dataset_size / len(recipes) scale down
    n_val = int(len(shuffled) * val * dataset_size / len(recipes))
    return shuffled[:n_train], shuffled[n_train: n_train + n_val], shuffled[-n_val:]

def get_torch_tensor(data):
    x, y = convert_one_hot(data)
    x = x.tocoo()
    print(data.shape)
    x_tensor = torch.sparse_coo_tensor(torch.tensor([x.row.tolist(), x.col.tolist()]),
                                  torch.tensor(x.data.astype(np.float32)), 
                                   torch.Size([data.shape[0], NUM_INGREDIENTS]))
    y_tensor = torch.tensor(y.astype(np.int_))
    tensor = data_utils.TensorDataset(x_tensor, y_tensor)
    return tensor

# we need to convert our data into one-hot encoding
# we'll also return the x vector (all - 1 ingredient) and y (missing ingredient)
def convert_one_hot(array):
    # here i'm getting an array of zeros
    # num rows is the size of the input array (ie how many recipes)
    # num cols is num of ingredients total (so we can 1-hot them)
    inputs = scipy.sparse.dok_matrix((len(array), NUM_INGREDIENTS), dtype=np.int)
    targets = np.empty(len(array))
    
    for i in range(len(array)):
        if len(array[i]) > 0:
            # this is just indexing into the ith row of the array (ith recipe)
            # and saying all the values in the recipe we're gonna set to 1
#             one_hot[i][array[i]] = 1
            
            # randomly choose one of the ingredients
            leave_out_idx = np.random.randint(len(array[i]))
            leave_out = array[i][leave_out_idx]
            leave_out_array = np.delete(array[i], leave_out_idx)
            for ingredient_index in leave_out_array:
                inputs[i, ingredient_index] = 1
            targets[i] = leave_out
            
        else:
            print("shouldn't get here ever")
        
#     return one_hot, inputs, targets

    inputs = scipy.sparse.csr_matrix(inputs)
    return inputs, targets

def get_data(dataset_size=100000):
    data = np.load('dataset.npz', allow_pickle=True)
    ingredients = data['ingredients']
    recipes = data['recipes']
    vectorized_len = np.vectorize(len)
    recipes = recipes[vectorized_len(recipes) > 0]

    train_recipes, val_recipes, test_recipes = split_train_val_test(recipes, dataset_size=dataset_size)

    # mini_size = 1000
    # mini_train_tensor = get_torch_tensor(train_recipes[:mini_size])

    train_tensor = get_torch_tensor(train_recipes)
    valid_tensor = get_torch_tensor(val_recipes)
    test_tensor = get_torch_tensor(test_recipes)

    mini_size = 10
    mini_tensor = get_torch_tensor(train_recipes[:mini_size])

    return train_tensor, valid_tensor, test_tensor, mini_tensor

###########################################################################
# Neural net architecture/training
###########################################################################

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params

    def forward(self, x):
        pass

class TwoHiddenNN(Net):
    def __init__(self, params):
        super(TwoHiddenNN, self).__init__(params)
        self.fc0 = nn.Linear(NUM_INGREDIENTS, self.params["hidden_1"])
        self.fc1 = nn.Linear(self.params["hidden_1"], self.params["hidden_2"])
        self.fc2 = nn.Linear(self.params["hidden_2"], NUM_INGREDIENTS)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ThreeHiddenNN(Net):
    def __init__(self, params):
        super(ThreeHiddenNN, self).__init__(params)
        self.fc0 = nn.Linear(NUM_INGREDIENTS, self.params["hidden_1"])
        self.fc1 = nn.Linear(self.params["hidden_1"], self.params["hidden_2"])
        self.fc2 = nn.Linear(self.params["hidden_2"], self.params["hidden_3"])
        self.fc3 = nn.Linear(self.params["hidden_3"], NUM_INGREDIENTS)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    
    losses = []

    batches = tqdm(enumerate(train_loader), total=len(train_loader))
    batches.set_description("Epoch NA: Loss (NA) Accuracy (NA %)")
    for batch_idx, (data, target) in batches:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
#         print(data.shape)
        output = model(data)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
#         loss = F.nll_loss(output, target)
        pred = output.max(1, keepdim=True)[1]
#         print(f"Prediction: {pred}, Actual: {target}, Loss: {loss}")
        correct = pred.eq(target.float().view_as(pred)).sum().item()
        sum_num_correct += correct
        sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        batches.set_description(
          "Epoch {:d}: Loss ({:.2e},  Accuracy ({:02.0f}%)".format(
            epoch, loss.item(), 100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size))
        )
        
    # return avg loss, accuracy of epoch
    return sum(losses)/len(losses), 100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # We use reduction = 'sum' here to ignore the impact of batch size and make 
            # this value comparable with the loss reported in the train loop above. Note,
            # though, that we divide by the len of the dataset below (so this is truly a per-element loss value)
#             test_loss += F.mse_loss(torch.clamp(output.view(target.shape), 0., 5.), target.float(), reduction='sum') # sum up the mean square loss
#             pred = torch.clamp(output, 0., 5.)
#             correct += pred.eq(target.float().view_as(pred)).sum().item()
            loss_fn = nn.CrossEntropyLoss()
            loss = torch.sum(loss_fn(output, target)).item() # sum up batch loss
            test_loss += loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.2e}\n'.format(test_loss))
    print('\nTest set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

    # loss, accuracy
    return test_loss, test_accuracy

def run_nn(params, args, tensors, nn_model_type=TwoHiddenNN):
    train_tensor, valid_tensor, test_tensor, mini_tensor = tensors
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        train_tensor,
        batch_size=args["batch_size"], shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        valid_tensor,
        batch_size=args["batch_size"], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_tensor,
        batch_size=args["batch_size"], shuffle=True, **kwargs)
    mini_loader = torch.utils.data.DataLoader(
        mini_tensor,
        batch_size=args["batch_size"], shuffle=True, **kwargs)

    model = nn_model_type(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    train_losses = []
    train_accuracies = []
    torch.manual_seed(args["seed"])
    for epoch in range(1, params["epochs"] + 1):
        # testing that this script works
        # train_loss, train_accuracy = train(model, device, mini_loader, optimizer, epoch)

        # use this for training
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

    # testing that this script works
    # valid_loss, valid_accuracy = test(model, device, mini_loader)

    # use this for actual validation
    valid_loss, valid_accuracy = test(model, device, valid_loader)

    return model, train_losses, train_accuracies, valid_loss, valid_accuracy

def train_and_validate(hidden_layer_count, params, args, tensors):
    # we will vary # of hidden layers, # of nodes in hidden layers, learning rate
    lr = params["lr"]
    # num_hidden_layers = [2, 3]
    # lrs = [0.01, 0.001, 0.0001]
    if hidden_layer_count == 2:
        num_hidden_nodes = [100, 500, 1000, NUM_INGREDIENTS]
    elif hidden_layer_count == 3:
        num_hidden_nodes = [100, 500, 1000]

    results = {}
    
    results[hidden_layer_count] = {lr: {}}
    # for hidden_layer_count in num_hidden_layers:
    #     results[hidden_layer_count] = results.get(hidden_layer_count, {})
    #     for lr in lrs:
    #         results[hidden_layer_count][lr] = results[hidden_layer_count].get(lr, {})
    #             params["lr"] = lr
    for hidden_1 in num_hidden_nodes:
        params["hidden_1"] = hidden_1
        for hidden_2 in num_hidden_nodes:
            params["hidden_2"] = hidden_2
            if hidden_layer_count == 2:
                results[hidden_layer_count][lr][(hidden_1, hidden_2)] = run_nn(params, args, tensors, TwoHiddenNN)
            elif hidden_layer_count == 3:
                for hidden_3 in num_hidden_nodes:
                    params["hidden_3"] = hidden_3
                    results[hidden_layer_count][lr][(hidden_1, hidden_2, hidden_3)] = run_nn(params, args, tensors, ThreeHiddenNN)

    return results

if __name__ == "__main__":
    hidden_layer_count = int(sys.argv[1])
    lr = float(sys.argv[2])
    print(hidden_layer_count, lr)

    tensors = get_data(dataset_size=100000)
    # tensors = get_data(dataset_size=100)
    
    PARAMS["lr"] = lr
    results = train_and_validate(hidden_layer_count, PARAMS, ARGS, tensors)

    # write this data to pickle so we don't have to go through this brutal process again
    with open(f"nn_train_valid_results_{hidden_layer_count}_{lr}.pkl", "wb+") as f:
        pkl.dump(results, f)