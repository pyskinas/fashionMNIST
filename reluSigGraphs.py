import matplotlib.pyplot as plt
import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# So that it applies the function to each element individually
relu = np.vectorize(relu)
sigmoid = np.vectorize(sigmoid)

# Our range of x values
x = np.linspace(-5, 5, 100)

# Change the function according to which one you want to see
y = sigmoid(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.title("Sigmoid(x)")
plt.grid()
plt.show()
