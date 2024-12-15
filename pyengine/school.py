import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.log2(x)


N = 500
x_axis = np.linspace(0.1, N, N)
y_axis = f(x_axis)
values = []
for n in range(N):
    max_list = []
    for i in range(40):
        arr = np.random.choice([0, 1], size=n)
        max_ones = max(map(len, ''.join(map(str, arr)).split("0")))
        max_total = max(len(s) for s in ''.join(map(str, arr)).split('1') + ''.join(map(str, arr)).split('0'))
        max_list.append(max_total)
    values.append(sum(max_list) / len(max_list))

plt.plot(x_axis, y_axis, color="orange")
plt.scatter(x_axis, values, s=3)
plt.title("log2(x) en langste reeks kop / munt")
plt.show()
