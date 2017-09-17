import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

scores = [3.0, 1.0, 0.2]

"""Softmax."""
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_yi = np.exp(x)
    suma_e_yj = sum(e_yi)
    soft = e_yi/suma_e_yj
    return soft

print(softmax(scores))
