import numpy as np


class AnyNumpyArrayOfShape(object):
    def __init__(self, shape):
        self.shape = shape

    def __eq__(self, other):
        return self.shape == other.shape


class EqualNumpyArray(object):

    def __init__(self, arr):
        self.arr = arr

    def __eq__(self, other):
        return np.all(self.arr == other)
