"""NumPy双层LSTM，显式BPTT与Adam；门顺序i/f/g/o，末层Dropout。"""

import numpy as np
from scipy.special import expit


class NumpyLSTM:
    def __init__(self, inputs=2, hidden=32, seed=42):
        self.hidden = hidden
        self.rng = np.random.default_rng(seed)
        self.params = {}
        for layer, dim in enumerate([inputs, hidden]):
            bound = np.sqrt(6 / (dim + hidden + 4 * hidden))
            self.params[f"W{layer}"] = self.rng.uniform(
                -bound, bound, (dim + hidden, 4 * hidden)
            )
            self.params[f"b{layer}"] = np.zeros(4 * hidden)
            self.params[f"b{layer}"][hidden : 2 * hidden] = 1
        self.params["Wo"] = self.rng.uniform(
            -np.sqrt(6 / (hidden + inputs)),
            np.sqrt(6 / (hidden + inputs)),
            (hidden, inputs),
        )
        self.params["bo"] = np.zeros(inputs)
        self.first = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.second = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def layer(self, x, level, cache=True):
        B, T, D = x.shape
        H = self.hidden
        h = np.zeros((B, H))
        c = h.copy()
        history = []
        outputs = []
        W, b = self.params[f"W{level}"], self.params[f"b{level}"]
        for t in range(T):
            combined = np.concatenate([x[:, t], h], axis=1)
            gates = combined @ W + b
            i = expit(gates[:, :H])
            f = expit(gates[:, H : 2 * H])
            g = np.tanh(gates[:, 2 * H : 3 * H])
            o = expit(gates[:, 3 * H :])
            prev = c
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            if cache:
                history.append((combined, prev, i, f, g, o, tc))
            outputs.append(h)
        return np.stack(outputs, axis=1), history

    def forward(self, x, training=False, cache=True):
        h0, c0 = self.layer(x, 0, cache)
        h1, c1 = self.layer(h0, 1, cache)
        mask = (
            self.rng.binomial(1, 0.8, h1[:, -1].shape) / 0.8
            if training
            else np.ones_like(h1[:, -1])
        )
        last = h1[:, -1] * mask
        return last @ self.params["Wo"] + self.params["bo"], (c0, c1, last, mask)

    def backward_layer(self, dout, history, level):
        B, T, H = dout.shape
        W = self.params[f"W{level}"]
        D = W.shape[0] - H
        dx = np.zeros((B, T, D))
        dW = np.zeros_like(W)
        db = np.zeros(4 * H)
        dh = np.zeros((B, H))
        dc = dh.copy()
        for t in range(T - 1, -1, -1):
            combined, prev, i, f, g, o, tc = history[t]
            dh = dh + dout[:, t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc * tc)
            di = dc * g
            df = dc * prev
            dg = dc * i
            dc = dc * f
            gates = np.concatenate(
                [
                    di * i * (1 - i),
                    df * f * (1 - f),
                    dg * (1 - g * g),
                    do * o * (1 - o),
                ],
                axis=1,
            )
            dW += combined.T @ gates
            db += gates.sum(axis=0)
            back = gates @ W.T
            dx[:, t] = back[:, :D]
            dh = back[:, D:]
        return dx, dW, db

    def loss_grad(self, x, y, training=False):
        pred, (c0, c1, last, mask) = self.forward(x, training=training)
        loss = np.mean((pred - y) ** 2)
        dp = 2 * (pred - y) / y.size
        grads = {"Wo": last.T @ dp, "bo": dp.sum(axis=0)}
        dh = np.zeros((len(x), x.shape[1], self.hidden))
        dh[:, -1] = (dp @ self.params["Wo"].T) * mask
        dx, grads["W1"], grads["b1"] = self.backward_layer(dh, c1, 1)
        _, grads["W0"], grads["b0"] = self.backward_layer(dx, c0, 0)
        return loss, grads

    def update(self, grads, lr=0.001):
        self.step += 1
        norm = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
        factor = min(1.0, 5 / (norm + 1e-12))
        for key, g in grads.items():
            g = g * factor
            self.first[key] = 0.9 * self.first[key] + 0.1 * g
            self.second[key] = 0.999 * self.second[key] + 0.001 * g * g
            self.params[key] -= (
                lr
                * (self.first[key] / (1 - 0.9**self.step))
                / (np.sqrt(self.second[key] / (1 - 0.999**self.step)) + 1e-8)
            )

    def predict(self, x):
        return self.forward(x, cache=False)[0]

    def check_gradient(self, x, y):
        """用真实训练窗口核对每组参数中心差分，不合成输入。"""
        _, grad = self.loss_grad(x, y)
        errors = []
        for key, p in self.params.items():
            for flat in set([0, int(np.argmax(np.abs(grad[key])))]):
                idx = np.unravel_index(flat, p.shape)
                old = p[idx]
                eps = 1e-5
                p[idx] = old + eps
                plus = np.mean((self.predict(x) - y) ** 2)
                p[idx] = old - eps
                minus = np.mean((self.predict(x) - y) ** 2)
                p[idx] = old
                numerical = (plus - minus) / (2 * eps)
                errors.append(abs(numerical - grad[key][idx]))
                assert np.isclose(numerical, grad[key][idx], rtol=1e-4, atol=1e-7), (
                    key,
                    idx,
                )
        return max(errors)
