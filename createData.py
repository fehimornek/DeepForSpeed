import cv2
import time
import mss
import numpy as np
import keyboard
import os


"""
this script records road, minimap and speed-o-meter with the correct key presses associated with them. 
I can get around 20fps with this script with i5 9th generation processor. If your fps is less than this
it may not create enough data or play the game right. To check your fps you can uncomment the lines that show fps.
"""


class CreateData:
    def __init__(self):
        # places of the screen that are relevant to us
        self.road = {'left': 150, 'top': 310, 'width': 700, 'height': 200}
        self.minimap = {'left': 50, 'top': 525, 'width': 210, 'height': 215}
        self.speed = {'left': 827, 'top': 676, 'width': 73, 'height': 37}
        # variable for seeing fps
        self.last_time = time.time()

    def key_press(self):
        onehotencoding = np.array([0,0,0,0,0,0])  # elements of the list on the left corresponds to [w, a, d, wa, wd, nothing]
        if keyboard.is_pressed("w"):
            if keyboard.is_pressed("a"):
                onehotencoding[3] = 1
            elif keyboard.is_pressed("d"):
                onehotencoding[4] = 1
            else:
                onehotencoding[0] = 1
        elif keyboard.is_pressed("a"):
            onehotencoding[1] = 1
        elif keyboard.is_pressed("d"):
            onehotencoding[2] = 1
        else:
            onehotencoding[5] = 1
        return onehotencoding

    def get_screen(self):
        with mss.mss() as sct:
                # take the screenshot of the relevant places on the screen
                road_sct = sct.grab(self.road)
                minimap_sct = sct.grab(self.minimap)
                speed_sct = sct.grab(self.speed)
                # turn into array for resizing
                road_arr = np.array(road_sct)
                minimap_arr = np.array(minimap_sct)
                speed_arr = np.array(speed_sct)
                # resize
                road_arr = cv2.resize(road_arr,(120,60))
                minimap_arr = cv2.resize(minimap_arr, (50, 50))
                speed_arr = cv2.resize(speed_arr, (15, 15))

                # uncomment the next lines to test if road is visible if it is not make the needed adjustments
                #cv2.imshow("window", road_arr)
                #if cv2.waitKey(25) & 0xFF == ord("q"):
                #   cv2.destroyAllWindows()


        # return list of images as an numpy array
        return np.array([road_arr, minimap_arr, speed_arr], dtype=object)

    def main(self):
        name = input("enter the name for the training file: ")

        # get current directory and add the data to the training_data folder
        file = os.getcwd() + "\\DeepForSpeed\\training_data\\{}".format(name)

        if not os.path.exists(file):
            print("new data created!")
            os.mkdir(file)
            training_data_X = np.array([])
            training_data_Y = np.array([])

        else:
            print("data already exists! appending to the data.")
            training_data_X = np.load(file + f"\\{name}X.npy", allow_pickle=True)
            training_data_Y = np.load(file + f"\\{name}y.npy", allow_pickle=True)

        for i in range(3, 0, -1):
            print("data collection starts in {}".format(i))
            time.sleep(1)

        print("started collecting data!")
        while True:
            key = self.key_press()          # key is an one hot encoded array
            image = self.get_screen()       # image is an array of three elements
            training_data_X = np.append(training_data_X, image)
            training_data_Y = np.append(training_data_Y, key)

            # code snippet by sentdex <3
            if len(training_data_X) % 500 == 0:
                print("training files saved! frame count: {}".format(len(training_data_X)))
                np.save(file + f"\\{name}X.npy", training_data_X)
                np.save(file + f"\\{name}Y.npy", training_data_Y)

            if keyboard.is_pressed("q"):            # if q is pressed data collection ends
                break

            # you can test your fps through uncommenting these lines
            # print("fps:", round(1/(time.time() - self.last_time)))
            # self.last_time = time.time()

a = CreateData()
a.main()