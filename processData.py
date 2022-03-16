import os
import numpy as np
import random

from PIL import Image, ImageEnhance

"""
The training data is quite imbalanced because most of the time we are just
pressing the forward key and maybe only half of the time we are steering. If our convnet
sees this data it will be super biased to just going forward all the time so we need to change that.
Amount of do nothings is also quite big too this also needs to be taken care of.
"""


def preprocess():
    # this code piece is added for safety and ensures that there always will be the folder
    check_processed_folder = os.getcwd() + "\\training_data\\balanced"
    if not os.path.exists(check_processed_folder):
        os.mkdir(check_processed_folder)

    data_name = input("which data do you want to preprocess: ")
    file = os.getcwd() + f"\\training_data\\raw\\{data_name}"

    if os.path.exists(file):
        print("loading data!")
        training_dataX = list(np.load(file + f"\\{data_name}X.npy", allow_pickle=True))
        training_dataY = list(np.load(file + f"\\{data_name}Y.npy", allow_pickle=True))
    else:
        print("data doesnt exist!")
        return

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

    np.save(os.getcwd() + f"\\training_data\\balanced\\{data_name}\\{data_name}X.npy", dataX_shuffled)
    np.save(os.getcwd() + f"\\training_data\\balanced\\{data_name}\\{data_name}Y.npy", dataY_shuffled)
    print("balanced data saved!")

"""
Augmentations
1-change brightness values
2-flip the images when going either to right or left to create more turning data
might add rotation by a small degree
"""

def augmentData():
    # this code piece is added for safety and ensures that there always will be the folder
    check_augment_folder = os.getcwd() + "\\training_data\\augmented"
    if not os.path.exists(check_augment_folder):
        os.mkdir(check_augment_folder)

    data_name = input("what data do you want to augment: ")
    data_location = input("will you augment raw data or processed data? (options: raw - augmented): ")
    img_path = os.getcwd() + f"\\training_data\\{data_location}\\{data_name}"

    # load data in a safe manner
    if os.path.exists(img_path):
        print("loading data!")
        imageData = np.load(img_path + f"\\{data_name}X.npy", allow_pickle=True)
        labels = np.load(img_path + f"\\{data_name}Y.npy", allow_pickle=True)
    else:
        print("data doesnt exist!")
        return

    new_images = []
    new_labels = []

    print("processing darker versions of images...")
    # for every image add that image that's %40 darker
    for i in range(len(imageData)):
        # get every individual image
        image = np.array(imageData[i][0])
        image_normal = Image.fromarray(image, mode="L")
        # get a new image that has %60 of the originals brightness
        img = ImageEnhance.Brightness(image_normal)
        image_dark = img.enhance(0.6)
        # add the new image to the data list
        image_arr = np.array(image_dark)
        new_images.append([image_arr, imageData[i][1], imageData[i][2]])
        new_labels.append(labels[i])

    """
    for images where label is either right, left, forward right or forward left we will flip the images.
    example: if the images label is left then when we flip it, the flipped image will have label right.
    """
    print("processing flipped images...")
    for i in range(len(imageData)):
        label = labels[i]
        # if label is not forward and do nothing
        if label[0] == 0 and label[-1] == 0:
            # take road and minimap images and flip them
            road_image = np.array(imageData[i][0])
            minimap_image = np.array(imageData[i][1])
            road_pil = Image.fromarray(road_image, mode="L")
            minimap_pil = Image.fromarray(minimap_image, mode="L")
            road_flipped = road_pil.transpose(Image.FLIP_LEFT_RIGHT)
            minimap_flipped = minimap_pil.transpose(Image.FLIP_LEFT_RIGHT)
            # add the flipped images to the list
            new_images.append([np.array(road_flipped), np.array(minimap_flipped), imageData[i][2]])
            # get the correct label for the flipped image (logic is explained above)
            if label[1] == 1:
                new_labels.append([0,0,1,0,0,0])
            elif label[2] == 1:
                new_labels.append([0,1,0,0,0,0])
            elif label[3] == 1:
                new_labels.append([0,0,0,0,1,0])
            elif label[4] == 1:
                new_labels.append([0,0,0,1,0,0])

    new_image_data = np.concatenate((imageData, new_images))
    new_label_data = np.concatenate((labels, new_labels))
    perm = np.arange((len(new_image_data)))
    np.random.shuffle(perm)
    image_shuffled = new_image_data[perm]
    label_shuffled = new_label_data[perm]
    print("operations complete! previous data amount was", len(imageData), "new data amount is", len(image_shuffled))
    folder = input("what your data should be saved as: ")
    print("saving data!")
    os.mkdir(os.getcwd() + f"\\training_data\\augmented\\{folder}")
    np.save(os.getcwd() + f"\\training_data\\augmented\\{folder}\\{folder}X.npy", image_shuffled)
    np.save(os.getcwd() + f"\\training_data\\augmented\\{folder}\\{folder}Y.npy", label_shuffled)
    print("augmented data saved!")

augmentData()