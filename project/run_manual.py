"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import random

import minitorch
import minitorch.operators as mop


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(2, 1)

    def forward(self, x):
        y = self.linear(x)
        return minitorch.operators.sigmoid(y[0])


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        random.seed(100)
        self.weights = []
        self.bias = []
        for i in range(in_size):
            weights = []
            for j in range(out_size):
                w = self.add_parameter(f"weight_{i}_{j}", 2 * (random.random() - 0.5))
                weights.append(w)
            self.weights.append(weights)
        for j in range(out_size):
            b = self.add_parameter(f"bias_{j}", 2 * (random.random() - 0.5))
            self.bias.append(b)

    def forward(self, inputs):
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y


class ManualTrain:
    def __init__(self, hidden_layers):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward((x[0], x[1]))


# Based: https://github.com/kyegomez/xLSTM/blob/main/xlstm_torch/main.py
class sLSTMCell(minitorch.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        self.weights = []
        self.bias = []

        for i in range(input_size):
            weights = []
            for j in range(hidden_size):
                w_i = self.add_parameter(f"w_i_{i}_{j}", 2 * (random.random()) - 0.5)
                w_f = self.add_parameter(f"w_f_{i}_{j}", 2 * (random.random()) - 0.5)
                w_o = self.add_parameter(f"w_o_{i}_{j}", 2 * (random.random()) - 0.5)
                w_z = self.add_parameter(f"w_z_{i}_{j}", 2 * (random.random()) - 0.5)
                weights.append([w_i, w_f, w_o, w_z])
            self.weights.append(weights)
        for j in range(hidden_size):
            b_i = self.add_parameter(f"b_i_{j}", 2 * (random.random() - 0.5))
            b_f = self.add_parameter(f"b_f_{j}", 2 * (random.random() - 0.5))
            b_o = self.add_parameter(f"b_o_{j}", 2 * (random.random() - 0.5))
            b_z = self.add_parameter(f"b_z_{j}", 2 * (random.random() - 0.5))
            self.bias.append([b_i, b_f, b_o, b_z])

    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states

        i_tilda = ...
        f_tilda = ...
        o_tilda = ...
        z_tilda = ...

        i_t = mop.exp(i_tilda)
        f_t = mop.sigmoid(f_tilda)  # choose either sigmoid or exp based on context

        # stabilizer state update
        m_t = mop.max(mop.log(f_t) + m_prev, mop.log(i_t))

        # stabilized gates
        i_prime = mop.exp(mop.log(i_t) - m_t)
        f_prime = mop.exp(mop.log(f_t) + m_prev - m_t)

        c_t = f_prime * c_prev + i_prime * mop.tanh(z_tilda)
        n_t = f_prime * n_prev + i_prime

        c_hat = c_t / n_t
        h_t = mop.sigmoid(o_tilda) * mop.tanh(c_hat)

        return h_t, (h_t, c_t, n_t, m_t)


class sLSTM(minitorch.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super().__init__()
        self.layers = [
            sLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def forward(self, x, initial_states=None):
        pass
