import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])
T = np.linspace(0.01, 5, num=100, endpoint=True)
min_X = min(X)

P = np.zeros((100, 5))


def calculate_prob(i, j):
    tmp_array = [(x / min_X) for x in X]
    scale = (np.power(tmp_array, (-1 / T[i])))
    return scale[j] / sum(scale)


for i in range(len(T)):
    for j in range(len(X)):
        P[i][j] = calculate_prob(i, j)

#print(P)

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
