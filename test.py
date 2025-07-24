import matplotlib.pyplot as plt
import numpy as np
import noise


def fast_noise(x, y):
    return noise.pnoise2(x * 0.04, y)



for i in range(1000):
    x = i*0.5
    plt.scatter(x, fast_noise(x, 0))


plt.show()