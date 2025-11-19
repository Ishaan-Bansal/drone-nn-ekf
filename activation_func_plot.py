import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 400)
relu = np.maximum(0, x)
tanh = np.tanh(x)

plt.plot(x, relu, label='ReLU', linewidth=2)
plt.plot(x, tanh, label='tanh', linewidth=2)
plt.title('ReLU vs tanh Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()