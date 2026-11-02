"""Differentiable Logic Gate Network (LGN) in JAX — eigene, minimale Implementierung.

Jedes Neuron hat zwei feste Eingänge (a, b) und lernt, welches der 16 möglichen
2-Input-Logikgatter es sein soll. Im Training ist das Gatter eine Softmax-Mischung
aller 16 Gatter (differenzierbar), danach wird pro Neuron das wahrscheinlichste
Gatter fest gewählt -> reine Logikschaltung.
"""
import jax
import jax.numpy as jnp
import numpy as np

# Gatter g als Polynom in a, b:  g = c0 + c1*a + c2*b + c3*a*b
# Die Gatter-ID ist zugleich das Bitmuster (f(0,0), f(0,1), f(1,0), f(1,1)).
GATE_NAMES = [
    "ZERO", "AND", "A_AND_NOT_B", "A", "NOT_A_AND_B", "B", "XOR", "OR",
    "NOR", "XNOR", "NOT_B", "A_OR_NOT_B", "NOT_A", "NOT_A_OR_B", "NAND", "ONE",
]
GATE_COEFS = np.array([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0],
    [0, 0, 1, -1], [0, 0, 1, 0], [0, 1, 1, -2], [0, 1, 1, -1],
    [1, -1, -1, 1], [1, -1, -1, 2], [1, 0, -1, 0], [1, 0, -1, 1],
    [1, -1, 0, 0], [1, -1, 0, 1], [1, 0, 0, -1], [1, 0, 0, 0],
], dtype=np.float32)


class LGN:
    def __init__(self, in_dim, width, depth, n_classes, tau, seed=0):
        assert width % n_classes == 0
        rng = np.random.default_rng(seed)
        dims = [in_dim] + [width] * depth
        self.wiring = []
        for i in range(depth):
            # Zufällige, feste Verdrahtung; jeder Eingang wird möglichst gleich oft genutzt
            idx = rng.permutation(2 * dims[i + 1]) % dims[i]
            self.wiring.append((idx[: dims[i + 1]].astype(np.int32), idx[dims[i + 1]:].astype(np.int32)))
        self.n_classes, self.tau = n_classes, tau
        key = jax.random.PRNGKey(seed)
        self.params = [jax.random.normal(k, (width, 16)) for k in jax.random.split(key, depth)]

    def apply(self, params, x, hard=False):
        coefs = jnp.asarray(GATE_COEFS)
        for logits, (ia, ib) in zip(params, self.wiring):
            p = jax.nn.one_hot(jnp.argmax(logits, -1), 16) if hard else jax.nn.softmax(logits, -1)
            k = p @ coefs
            a, b = x[:, ia], x[:, ib]
            x = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * a * b
        # GroupSum: Ausgänge in n_classes Gruppen aufteilen und zählen
        return x.reshape(x.shape[0], self.n_classes, -1).sum(-1) / self.tau

    @staticmethod
    def gates(params):
        return [np.asarray(jnp.argmax(l, -1)) for l in params]
