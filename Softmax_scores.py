"""Softmax."""

import numpy as np
scores = [1.0, 2.0, 3.0]


"""
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
"""
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_yi = np.exp(x)
    suma_e_yj = sum(e_yi)
    soft = e_yi/suma_e_yj
    return soft

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()


