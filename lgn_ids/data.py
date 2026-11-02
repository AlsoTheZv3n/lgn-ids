"""UNSW-NB15 laden und Flow-Features in Bits kodieren.

Numerische Features -> Thermometer-Kodierung (x > Schwelle, Schwellen = Quantile
der Trainingsdaten). Kategorische Features -> One-Hot (x == Wert).
Dieselbe Kodierung wird 1:1 in den C-Code exportiert.
"""
import csv
import json
import numpy as np

CATEGORICAL = ["xProt", "xServ", "xState"]
DROP = ["id", "label"]


def load_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = np.array([[float(v) for v in row] for row in r], dtype=np.float64)
    y = rows[:, header.index("label")].astype(np.int64)
    feats = [h for h in header if h not in DROP]
    X = rows[:, [header.index(h) for h in feats]]
    return X, y, feats


class BitEncoder:
    """Liste von Bits, jedes Bit = (feature, op, wert) mit op in {'gt', 'eq'}."""

    def __init__(self, bits=None):
        self.bits = bits or []

    @classmethod
    def fit(cls, X, feats, n_thresholds=15, min_cat_count=50):
        bits = []
        q = np.arange(1, n_thresholds + 1) / (n_thresholds + 1)
        for j, name in enumerate(feats):
            col = X[:, j]
            if name in CATEGORICAL:
                vals, counts = np.unique(col, return_counts=True)
                for v in vals[counts >= min_cat_count]:
                    bits.append((name, "eq", float(v)))
            else:
                for t in np.unique(np.quantile(col, q, method="lower")):
                    frac = (col > t).mean()
                    if 0.001 < frac < 0.999:  # nutzlose (konstante) Bits weglassen
                        bits.append((name, "gt", float(t)))
        return cls(bits)

    def transform(self, X, feats):
        idx = {f: i for i, f in enumerate(feats)}
        out = np.empty((X.shape[0], len(self.bits)), dtype=np.uint8)
        for k, (name, op, v) in enumerate(self.bits):
            col = X[:, idx[name]]
            out[:, k] = (col > v) if op == "gt" else (col == v)
        return out

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.bits, f, indent=0)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls([tuple(b) for b in json.load(f)])
