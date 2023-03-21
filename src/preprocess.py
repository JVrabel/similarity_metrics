import numpy as np

def efficiency_fcn(x, x0, A, k):
    """
    Calculate the (1-exponential decay of x).
    :param x: numpy array of the independent variable
    :param A: amplitude or initial value of the dependent variable
    :param k: decay constant
    :return: y, numpy array of the dependent variable
    """
    y =1 - A * np.exp(-k * (x-x0))
    return y

def relu(efficiency_fcn):
    """Applies the rectified linear unit (ReLU) function to the given efficiency function. t"""
    return np.maximum(0, efficiency_fcn)

