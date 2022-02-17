import numpy as np
import model_architectures
import torch.optim as optim
import torch
import torch.nn.functional as F
import os
import importlib
import sys


def check_file(file):
    if os.path.exists(file):
        print("read successful! starting training")
        return True
    else:
        print("cannot read! \n make sure you dont write .npy at the end")
        return False

def training(neural_net_name, data_folder_name, epochs, batches, optimizer, loss_function, learning_rate = 0.001):
    # get the specified training data
    file = os.getcwd() + "\\training_data\\{}".format(data_folder_name)
    # check if the file exists,
    # ----->if it does load it
    # ----->else end the function
    if check_file(file):
        training_data_X = np.load(file + f"\\{data_folder_name}X.npy", allow_pickle=True)
        training_data_Y = list(np.load(file + f"\\{data_folder_name}Y.npy", allow_pickle=True))
        print("loaded features and labels")
    else:
        return
    # get the specified model from model_architectures
    class_neuralnet = getattr(model_architectures, neural_net_name)
    neural_net = class_neuralnet()
    print("loaded neural network: ", neural_net)
    # get the specified optimizer from torch.optim class
    optimizer = getattr(optim, optimizer)(neural_net.parameters(), lr = learning_rate)

    # get the specified loss function from torch.nn.functional class
    loss_func = getattr(F, loss_function)
    print("loaded necessary attributes for neural network")
    road = []
    minimap = []
    speed = []

    # seperation of images in training_data_X
    print("separating data...")
    for data in training_data_X:
        road.append(data[0])
        minimap.append(data[1])
        speed.append(data[2])
    print("separation of data successful!")
    road, minimap, speed, training_data_Y = np.array(road), np.array(minimap), np.array(speed), np.array(training_data_Y)
    print("transformed list to arrays")

    # training starts here
    for epoch in range(epochs):
        print("epoch: ", epoch+1)
        permutation = torch.randperm(len(road))
        for i in range(0, len(road), batches):
            neural_net.zero_grad()
            indices = permutation[i:i+batches]
            road_batch, minimap_batch, speed_batch = \
                torch.tensor(road[indices]), torch.tensor(minimap[indices]), torch.tensor(speed[indices])
            y_batch = torch.argmax(torch.tensor(training_data_Y[indices]), dim=1)
            output = neural_net.forward(road_batch/255, minimap_batch/255, speed_batch/255)  # (, minimap_batch, speed_batch)
            loss = loss_func(output, y_batch.view(1000,))
            loss.backward()
            optimizer.step()


training("nvidia_arch", "keke", 10, 1000, "Adam", "nll_loss")
