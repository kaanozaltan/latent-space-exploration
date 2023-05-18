import numpy as np

direction = np.load('directions/smile.npy')
direction = np.repeat(direction, 18, axis=0)
direction = np.expand_dims(direction, axis=0)
print(direction.shape)