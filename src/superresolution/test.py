import numpy as np

direction = np.load('directions/smile.npy')
direction = np.tile(direction, (18, 1))
print(direction.shape)