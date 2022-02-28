import os
import numpy as np
import random

"""
The training data is quite imbalanced because most of the time we are just
pressing the forward key and maybe only half of the time we are steering. If our convnet
sees this data it will be super biased to just going forward all the time so we need to change that.
Amount of do nothings is also quite big too this also needs to be taken care of.
"""

def preprocess():
    data_name = input("which data do you want to preprocess: ")
    file = os.getcwd() + f"\\training_data\\{data_name}"

    if os.path.exists(file):
        print("loading data!")
        training_dataX = list(np.load(file + f"\\{data_name}X.npy", allow_pickle=True))
        training_dataY = list(np.load(file + f"\\{data_name}Y.npy", allow_pickle=True))
    else:
        print("data doesnt exist!")
        return
    print(len(training_dataY))
    os.mkdir(os.getcwd() + f"\\training_data\\processed\\{data_name}")

    forward, right, left, forward_right, forward_left, do_nothing = [], [], [], [], [], []

    idx = 0

    print("separating data!")
    for data in training_dataX:
        # if data is a forward
        if training_dataY[idx][0] == 1:
            """
            append the data and then remove it from training_dataX this is done because
            we will remove most of the forwards and then add them back to training_dataX 
            if we dont remove them we will just have even more forwards which would be super bad.
            """
            forward.append(data)

        elif training_dataY[idx][1] == 1:
            left.append(data)

        elif training_dataY[idx][2] == 1:
            right.append(data)

        elif training_dataY[idx][3] == 1:       # forward lefts
            forward_left.append(data)

        elif training_dataY[idx][4] == 1:       # forward rights
            forward_right.append(data)

        elif training_dataY[idx][5] == 1:       # do nothing
            do_nothing.append(data)

        # used to move on to the next data
        idx += 1

    random.shuffle(forward),random.shuffle(forward_left),random.shuffle(forward_right)
    random.shuffle(do_nothing), random.shuffle(left),random.shuffle(right)

    lengths = [len(forward),len(forward_left),len(forward_right),len(do_nothing), len(left), len(right)]

    minimum_length = min(lengths)

    balanced_forward = [forward[i] for i in range(round(minimum_length))]
    balanced_forward_left = [forward_left[i] for i in range(round(minimum_length))]
    balanced_forward_right = [forward_right[i] for i in range(round(minimum_length))]
    balanced_do_nothing = [do_nothing[i] for i in range(round(minimum_length))]
    balanced_left = [left[i] for i in range(minimum_length)]
    balanced_right = [right[i] for i in range(minimum_length)]

    training_dataX = balanced_forward + balanced_left + balanced_right + \
                    balanced_forward_left + balanced_forward_right + balanced_do_nothing

    for i in range(minimum_length):
       training_dataY.append([1,0,0,0,0,0])
    for i in range(minimum_length):
       training_dataY.append([0,1,0,0,0,0])
    for i in range(minimum_length):
       training_dataY.append([0,0,1,0,0,0])
    for i in range(minimum_length):
       training_dataY.append([0,0,0,1,0,0])
    for i in range(minimum_length):
       training_dataY.append([0,0,0,0,1,0])
    for i in range(minimum_length):
       training_dataY.append([0,0,0,0,0,1])

    permutation = np.arange(len(training_dataX))
    np.random.shuffle(permutation)
    training_dataY = np.array(training_dataY)
    training_dataX = np.array(training_dataX)
    dataX_shuffled = training_dataX[permutation]
    dataY_shuffled = training_dataY[permutation]

    np.save(os.getcwd() + f"\\training_data\\processed\\{data_name}\\{data_name}X.npy", dataX_shuffled)
    np.save(os.getcwd() + f"\\training_data\\processed\\{data_name}\\{data_name}Y.npy", dataY_shuffled)
    print("saved the data!")

if __name__ == "__main__":
    preprocess()