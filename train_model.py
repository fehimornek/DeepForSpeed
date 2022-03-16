import time
import model_architectures
import torch.optim as optim
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# training the neural network
def training(neural_net_name,  epochs, batches, learning_rate,neural_net_folder_name = "default"):
    """
    - neural_net_name = is a string, used for choosing model from model_architectures.py
    - epochs = is an int, how many times neural network will see the whole data
    - batches = is an int, how many images will be used for training at every iteration
    - learning_rate = is a float, determines how much of an effect each gradient update has on the network
    - neural_net_folder_name = is a string, which is the saved models name. If you write a new name instead of default it will start
    training a new neural net.
    """

    ## get the training data
    data_name = input("enter the name of the data: ")
    data_location = input("which folder do you want to get the data from? (options: raw - balanced - augmented): ")
    file = os.getcwd() + f"\\training_data\\{data_location}\\{data_name}"

    ## check if the training file exists,
    if os.path.exists(file):
        print("loading training data...")
        training_data_X = np.load(file + f"\\{data_name}X.npy", allow_pickle=True)
        training_data_Y = np.load(file + f"\\{data_name}Y.npy", allow_pickle=True)
    else:
        print("data doesnt exist at this path!")

    # get the model from model_architectures
    class_neuralnet = getattr(model_architectures, neural_net_name)
    neural_net = class_neuralnet()
    neural_net.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neural_net = neural_net.to(device)

    print("loaded neural network: ", neural_net)
    print("using: ", device)

    # get or set a new model weights
    nn_file = os.getcwd() + f"\\trained_models\\{neural_net_folder_name}.pth"
    if os.path.exists(nn_file):
        neural_net.load_state_dict(torch.load(os.getcwd() + f"\\trained_models\\{neural_net_folder_name}.pth"))
        print("loaded model weights of {}.pth".format(neural_net_folder_name))
    else:
        print("model weights initialized!")

    optimizer = optim.Adam(neural_net.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    # separation of images in training_data_X
    road = []
    minimap = []
    speed = []

    print("separating data...")
    for data in training_data_X:
        road.append(data[0])
        minimap.append(data[1])
        speed.append(data[2])

    # turn the training data to tensor and load them to gpu with .cuda()
    road, minimap, speed, training_data_Y = torch.tensor(road).cuda(), torch.tensor(minimap).cuda(), torch.tensor(speed).cuda(), torch.tensor(training_data_Y).cuda()
    loss_arr = []
    start_time = time.time()
    print("starting training!")
    # training starts here
    for epoch in range(epochs):
        # take random permutation of the data this will be used for mini batch training
        permutation = torch.randperm(len(road))
        print("epoch: ", epoch + 1)
        for i in range(0, len(road), batches):
            # take indices using the random permutation
            indices = permutation[i:i + batches]
            # take batches of data with the indices and then turn them into tensors because numpy arrays can't be fed to the neural net
            road_batch, minimap_batch, speed_batch, y_batch = road[indices], minimap[indices], speed[indices], \
                                                              training_data_Y[indices].float()

            # add a dummy dimension because we dont have a channel value
            road_batch = road_batch[None, :]
            minimap_batch = minimap_batch[None, :]
            speed_batch = speed_batch[None, :]

            # turn the data from (channel, batches, width, height ) to (batches, channel, width, height )
            road_batch = torch.permute(road_batch, (1, 0, 2, 3))
            minimap_batch = torch.permute(minimap_batch, (1, 0, 2, 3))
            speed_batch = torch.permute(speed_batch, (1, 0, 2, 3))

            """
            y_batch = [0,0,0,0,0,1] is sth like this and argmax turns it into y_batch = 6 this
            is needed because the loss function expects target as an int
            """
            y_batch = torch.argmax(y_batch, dim=1)

            """
            we have to turn data into float because pytorch expects data to be float type. This can be achieved by dividing by 255 which also
            regularizes the data.
            """
            output = neural_net(road_batch / 255, minimap_batch / 255, speed_batch / 255)
            # calculate the error
            loss = loss_func(output, y_batch)
            loss_arr.append(loss.detach().cpu().numpy())
            # backprop, thank god pytorch for that am i right? lol
            neural_net.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(neural_net.state_dict(), os.getcwd() + f"\\trained_models\\{neural_net_folder_name}.pth")
    print("saved neural network weights!")
    print("training took: ", (time.time() - start_time) / 60, "minutes")
    # show how loss changed
    plt.plot(loss_arr)
    plt.show()


if __name__ == "__main__":
    training(learning_rate=0.001, epochs=200, neural_net_name="nvidia_arch", batches=32, neural_net_folder_name="defaultBetter")