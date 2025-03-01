import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

X = [1, 2, 3, 4]


Wx = np.array([[0.5], [0.3], [0.2], [0.1]])  
Wh = np.array([[0.4], [0.2], [0.3], [0.1]])
b = np.array([0.1, 0.2, 0.1, 0.3])

h_t = 0
C_t = 0


for t, x_t in enumerate(X):
    f_t = sigmoid(Wx[0] * x_t + Wh[0] * h_t + b[0])
    i_t = sigmoid(Wx[1] * x_t + Wh[1] * h_t + b[1])
    C_tilde_t = tanh(Wx[2] * x_t + Wh[2] * h_t + b[2])
    C_t = f_t * C_t + i_t * C_tilde_t
    o_t = sigmoid(Wx[3] * x_t + Wh[3] * h_t + b[3])
    h_t = o_t * tanh(C_t)
    print(f"Time Step {t+1}: h_t = {h_t}, C_t = {C_t}")


W_y = 0.9
b_y = 0.5
y_hat = W_y * h_t + b_y
print(f"Predicted Next Value: {y_hat}")
