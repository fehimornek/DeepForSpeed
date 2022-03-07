from play_util import *
from createData import CreateData
import model_architectures
import torch
import os
import keyboard
import time
import numpy as np

def calcProb(tensor):
    return (np.array(tensor)*100) / sum(sum(np.array(tensor)))


# function that acts as an api between game and the neural network
def playGame(modelName, trainedModelName):
    # get the neuralnet from model architectures
    neural_net = getattr(model_architectures, modelName)
    neuralnet = neural_net()
    # check if trained model exists if it does load them else return from the function
    nn_location = os.getcwd() + f"\\trained_models\\{trainedModelName}.pth"
    if os.path.exists(nn_location):
        neuralnet.load_state_dict(torch.load(nn_location))
        neuralnet.eval()
    else:
        print("that trained model does not exist")
        return

    # read the screen
    dataloader = CreateData()

    # wait a little before running the script to give time to open the game
    for i in range(5,0,-1):
        print(i)
        time.sleep(1)

    pause = False
    # main loop
    with torch.no_grad():
        while True:
            while not pause:
                screen = dataloader.get_screen()
                # turn the screen to tensors
                road, minimap, speed = torch.tensor(screen[0]), torch.tensor(screen[1]), torch.tensor(screen[2])
                # some dummy dimension for pytorch
                road = road[None, None]
                minimap = minimap[None, None]
                speed = speed[None, None]
                output = neuralnet.forward(road/255, minimap/255, speed/255)
                print(calcProb(output))
                index = torch.argmax(output)
                """
                directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
                0x11 = w
                0x1E = A
                0x20 = D
                """
                if index == 0:          # press w
                    PressKey(0x11)
                    time.sleep(0.3)
                    ReleaseKey(0x11)
                    print("forward")
                elif index == 1:        # press a
                    PressKey(0x1E)
                    time.sleep(0.05)
                    ReleaseKey(0x1E)
                    print("left")
                elif index == 2:        # press d
                    PressKey(0x20)
                    time.sleep(0.05)
                    ReleaseKey(0x20)
                    print("right")
                elif index == 3:        # press wa
                    PressKey(0x11)
                    PressKey(0x1E)
                    time.sleep(0.05)
                    ReleaseKey(0x11)
                    ReleaseKey(0x1E)
                    print("forward left")
                elif index == 4:        # press wd
                    PressKey(0x11)
                    PressKey(0x20)
                    time.sleep(0.05)
                    ReleaseKey(0x11)
                    ReleaseKey(0x20)
                    print("forward right")

                elif index == 5:        # nothing
                    time.sleep(0.05)
                    print("do nothing")

                # if q is pressed quit
                if keyboard.is_pressed("q"):
                    return

               # pause the game
                if keyboard.is_pressed("z"):
                    pause = True

            if keyboard.is_pressed("z"):
                pause = False

playGame("nvidia_arch", "defaultRaw" )