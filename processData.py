import numpy as np
import cv2
import random

training_dataX = list(np.load("training_data/train1/train1X.npy", allow_pickle=True))
training_dataY = list(np.load("training_data/train1/train1Y.npy", allow_pickle=True))

idx =0
forward, left, right, forright, forleft, nthg = [], [], [], [], [], []
for data in training_dataX:
    if training_dataY[idx][0] == 1:
        forward.append(data)
        training_dataX.pop(idx)
        training_dataY.pop(idx)
    elif training_dataY[idx][4] == 1:
        forright.append(data)
    elif training_dataY[idx][5] == 1:
        nthg.append(data)
        training_dataX.pop(idx)
        training_dataY.pop(idx)
    idx += 1

random.shuffle(forward)
random.shuffle(nthg)
forward = [forward[i] for i in range(round(len(forright)*1.5))]
nthg = [nthg[i] for i in range(len(forright))]

training_dataX = training_dataX + forward + nthg

for i in range(len(forward)):
    training_dataY.append([1,0,0,0,0,0])
for i in range(len(nthg)):
    training_dataY.append([0,0,0,0,0,1])

np.save("\\training_data\\processed\\tX.npy", training_dataX)
np.save("\\training_data\\processed\\tY.np", training_dataY)