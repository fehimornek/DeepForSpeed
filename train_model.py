import numpy as np
import model_architectures
import torch.optim as optim
import torch
import torch.nn.functional as F
import os

# training the neural network
def training(data_folder_name, neural_net_folder="default", neural_net_name="nvidia_arch",  epochs = 10, batches= 1000, optimizer="Adam", loss_function="nll_loss", learning_rate = 0.001):
    """
    - data_folder_name = is a string, used for choosing training folder. If the folders name is test write test do not write its
    content like testX, testY
    - neural_net_folder = is a string, which is the saved models name. If you write a new name instead of default it will start
    training a new neural net.
    - neural_net_name = is a string, used for choosing model from model_architectures.py
    - epochs = is an int, how many times neural network will see the whole data
    - batches = is an int, how many images will be used for training at every iteration
    - optimizer = is an int, is a string, used for picking an optimizer from pytorch.nn.optim
    - loss_function = is a string, used for picking a loss function from pytorch.nn.F
    - learning_rate = is a float, determines how much of an effect each gradient update has on the network
    """

    # get the training data
    file = os.getcwd() + "\\training_data\\{}".format(data_folder_name)
    # check if the file exists,
    # ----->if it does load it
    # ----->else end the function
    if os.path.exists(file):
        training_data_X = np.load(file + f"\\{data_folder_name}X.npy", allow_pickle=True)
        training_data_Y = list(np.load(file + f"\\{data_folder_name}Y.npy", allow_pickle=True))
    else:
        return

    # get the model from model_architectures
    class_neuralnet = getattr(model_architectures, neural_net_name)
    neural_net = class_neuralnet()
    print("loaded neural network: ", neural_net)

    nn_file = os.getcwd() + f"\\trained_models\\{neural_net_folder}.pth"
    if os.path.exists(nn_file):
        neural_net.load_state_dict(torch.load(os.getcwd() + f"\\trained_models\\{neural_net_folder}.pth"))
        print("loaded model weights of {}.pth".format(neural_net_folder))
    else:
        print("model weights initialized!")

    # get the specified optimizer from torch.optim class
    optimizer = getattr(optim, optimizer)(neural_net.parameters(), lr = learning_rate)
    # get the specified loss function from torch.nn.functional class
    loss_func = getattr(F, loss_function)

    road = []
    minimap = []
    speed = []

    # seperation of images in training_data_X
    print("separating data...")
    for data in training_data_X:
        road.append(data[0])
        minimap.append(data[1])
        speed.append(data[2])

    road, minimap, speed, training_data_Y = np.array(road), np.array(minimap), np.array(speed), np.array(training_data_Y)

    print("starting training!")
    # training starts here
    for epoch in range(epochs):
        print("epoch: ", epoch+1)
        # take random permutation of the data this will be used for mini batch training
        permutation = torch.randperm(len(road))
        for i in range(0, len(road), batches):
            neural_net.zero_grad()
            # take indices using the random permutation
            indices = permutation[i:i+batches]
            # take batches of data with the indices and then turn them into tensors because numpy arrays can't be fed to the neural net
            road_batch, minimap_batch, speed_batch = \
                torch.tensor(road[indices]), torch.tensor(minimap[indices]), torch.tensor(speed[indices])
            # y_batch = [0,0,0,0,0,0] sth like this and argmax turns it into y_batch = 6 this is need because the loss function expects target as an int
            y_batch = torch.argmax(torch.tensor(training_data_Y[indices]), dim=1)
            # we have to turn data into float because pytorch expects data to be float type. This can be achieved by dividing by 255 which also
            # regularizes the data
            output = neural_net.forward(road_batch/255, minimap_batch/255, speed_batch/255)
            # calculate the error
            loss = loss_func(output, y_batch)
            # backprop thank god pytorch for that am i right? lol
            loss.backward()
            optimizer.step()

    torch.save(neural_net.state_dict(), os.getcwd() + f"\\trained_models\\{neural_net_folder}.pth")
    print("saved neural network weights!")

training("keke", "keke")
