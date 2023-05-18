import numpy as np
import torch

direction = np.load('directions/smile.npy')
# print(direction[0])
direction = np.repeat(direction, 18, axis=0)
# print(direction[4])
direction = np.expand_dims(direction, axis=0)
# print(direction[0,0])
direction = torch.from_numpy(direction).to(torch.float32)
# print(direction.shape)
# print(direction[0,0])