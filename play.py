from play_util import *
from createData import CreateData
from model_architectures import *
import torch
import os
import keyboard

model = nvidia_arch()
model.state_dict(torch.load(os.getcwd() + f"\\trained_models\\default.pth"))
dataloader = CreateData()
for i in range(3,0,-1):
    print(i)
    time.sleep(1)
while True:
    screen = dataloader.get_screen()
    road, minimap, speed = torch.tensor(screen[0]), torch.tensor(screen[1]), torch.tensor(screen[2])
    road= torch.permute(road, (2, 1, 0))
    minimap = torch.permute(minimap, (2, 1, 0))
    speed = torch.permute(speed, (2, 1, 0))
    road = road[None, :]
    minimap = minimap[None, :]
    speed = speed[None, :]
    output = model.forward(road/255, minimap/255, speed/255)

    # directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
    # 0x11 = w
    # 0x1E = A
    # 0x20 = D
    index = torch.argmax(output)
    if index == 0:          # press w
        PressKey(0x11)
        time.sleep(1)
        ReleaseKey(0x11)
        print("forward")
    elif index == 1:        # press a
        PressKey(0x1E)
        time.sleep(1)
        ReleaseKey(0x1E)
        print("left")
    elif index == 2:        # press d
        PressKey(0x20)
        time.sleep(1)
        ReleaseKey(0x20)
        print("right")
    elif index == 3:        # press wa
        PressKey(0x11)
        PressKey(0x1E)
        time.sleep(1)
        ReleaseKey(0x11)
        ReleaseKey(0x1E)
        print("forward left")
    elif index == 4:        # press wd
        PressKey(0x11)
        PressKey(0x20)
        time.sleep(1)
        ReleaseKey(0x11)
        ReleaseKey(0x20)
        print("forward right")

    elif index == 5:        # nothing
        time.sleep(1)
        print("do nothing")

    # if q is pressed quit
    if keyboard.is_pressed("q"):
        break