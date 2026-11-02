"""Vergleichsmodelle auf denselben Daten: FT-Transformer (JAX) und Gradient Boosting.

python -m lgn_ids.baselines train --epochs 6     # trainieren + Genauigkeit
python -m lgn_ids.baselines bench                 # CPU-Durchsatz messen (Threads über XLA_FLAGS)
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from .data import CATEGORICAL, load_csv
from .train import metrics

D, LAYERS, HEADS = 64, 3, 8


class Prep:
    """Numerisch: sign*log1p, standardisiert. Kategorisch: Index (seltene Werte -> 0)."""

    def __init__(self, X, feats, min_count=50):
        self.num = [i for i, f in enumerate(feats) if f not in CATEGORICAL]
        self.cat = [i for i, f in enumerate(feats) if f in CATEGORICAL]
        z = self._log(X[:, self.num])
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-6
        self.vocab = []
        for i in self.cat:
            v, c = np.unique(X[:, i], return_counts=True)
            self.vocab.append({float(x): k + 1 for k, x in enumerate(v[c >= min_count])})

    @staticmethod
    def _log(z):
        return np.sign(z) * np.log1p(np.abs(z))

    def __call__(self, X):
        xn = ((self._log(X[:, self.num]) - self.mu) / self.sd).astype(np.float32)
        xc = np.stack([[voc.get(float(v), 0) for v in X[:, i]] for i, voc in zip(self.cat, self.vocab)], 1)
        return xn, xc.astype(np.int32)


def init_ft(key, n_num, cat_sizes):
    import jax
    import jax.numpy as jnp
    ks = iter(jax.random.split(key, 64))
    nrm = lambda shape, s: jax.random.normal(next(ks), shape) * s
    p = {"w": nrm((n_num, D), 0.1), "b": jnp.zeros((n_num, D)), "cls": nrm((D,), 0.1),
         "emb": [nrm((s, D), 0.1) for s in cat_sizes], "layers": [],
         "head": {"g": jnp.ones(D), "bn": jnp.zeros(D), "W": nrm((D, 2), D ** -0.5), "bh": jnp.zeros(2)}}
    for _ in range(LAYERS):
        p["layers"].append({
            "g1": jnp.ones(D), "n1": jnp.zeros(D), "Wqkv": nrm((D, 3 * D), D ** -0.5), "bqkv": jnp.zeros(3 * D),
            "Wo": nrm((D, D), D ** -0.5), "bo": jnp.zeros(D), "g2": jnp.ones(D), "n2": jnp.zeros(D),
            "W1": nrm((D, 2 * D), D ** -0.5), "b1": jnp.zeros(2 * D), "W2": nrm((2 * D, D), (2 * D) ** -0.5), "b2": jnp.zeros(D)})
    return p


def ft_apply(p, xn, xc):
    """Feature Tokenizer + Pre-Norm-Transformer (Gorishniy et al., 2021)."""
    import jax
    import jax.numpy as jnp
    ln = lambda h, g, b: (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5) * g + b
    tok = xn[..., None] * p["w"] + p["b"]
    cat = jnp.stack([e[xc[:, i]] for i, e in enumerate(p["emb"])], 1)
    h = jnp.concatenate([jnp.broadcast_to(p["cls"], (xn.shape[0], 1, D)), tok, cat], 1)
    B, T, _ = h.shape
    for L in p["layers"]:
        q, k, v = jnp.split(ln(h, L["g1"], L["n1"]) @ L["Wqkv"] + L["bqkv"], 3, -1)
        sh = lambda z: z.reshape(B, T, HEADS, D // HEADS).transpose(0, 2, 1, 3)
        att = jax.nn.softmax(sh(q) @ sh(k).transpose(0, 1, 3, 2) / np.sqrt(D // HEADS), -1) @ sh(v)
        h = h + att.transpose(0, 2, 1, 3).reshape(B, T, D) @ L["Wo"] + L["bo"]
        h = h + jax.nn.gelu(ln(h, L["g2"], L["n2"]) @ L["W1"] + L["b1"]) @ L["W2"] + L["b2"]
    H = p["head"]
    return jax.nn.relu(ln(h[:, 0], H["g"], H["bn"])) @ H["W"] + H["bh"]


def split(a):
    X, y, feats = load_csv(f"{a.data}/train.csv")
    Xte, yte, _ = load_csv(f"{a.data}/test.csv")
    perm = np.random.default_rng(a.seed).permutation(len(X)); n_val = len(X) // 10
    return X, y, feats, Xte, yte, perm[n_val:], perm[:n_val]


def gb_matrix(X, Xte, feats):
    """Kategorien als Index < 255 (Vorgabe von HistGradientBoosting)."""
    Xc, Xtec = X.copy(), Xte.copy()
    for j, f in enumerate(feats):
        if f in CATEGORICAL:
            _, inv = np.unique(np.concatenate([X[:, j], Xte[:, j]]), return_inverse=True)
            inv = np.minimum(inv, 250); Xc[:, j] = inv[:len(X)]; Xtec[:, j] = inv[len(X):]
    return Xc, Xtec


def gb_speed(gb, Xtec):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        gb.predict(Xtec[:1000]); t0 = time.perf_counter(); gb.predict(Xtec); dt = time.perf_counter() - t0
        one = []
        for i in range(300):
            t1 = time.perf_counter(); gb.predict(Xtec[i:i + 1]); one.append(time.perf_counter() - t1)
    return {"ns_per_flow_batch": dt / len(Xtec) * 1e9, "flows_per_s_batch": len(Xtec) / dt,
            "latency_single_flow_us": float(np.median(one) * 1e6)}


def train(a):
    import jax
    import jax.numpy as jnp
    import optax
    os.makedirs("results", exist_ok=True)
    X, y, feats, Xte, yte, tr, va = split(a)
    out = {}

    # --- Gradient Boosting: die stärkste klassische Baseline für Tabellendaten ---
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xc, Xtec = gb_matrix(X, Xte, feats)
    gb = HistGradientBoostingClassifier(max_iter=300, categorical_features=[f in CATEGORICAL for f in feats],
                                        class_weight="balanced", random_state=a.seed)
    gb.fit(Xc[tr], y[tr])
    with open("results/gradient_boosting.pkl", "wb") as f:
        pickle.dump(gb, f)
    out["gradient_boosting"] = {"test": metrics(gb.predict(Xtec), yte), "speed_1thread": gb_speed(gb, Xtec)}
    print("Gradient Boosting:", json.dumps(out["gradient_boosting"]), flush=True)

    # --- FT-Transformer ---
    prep = Prep(X[tr], feats)
    xn, xc = prep(X[tr]); vxn, vxc = prep(X[va]); txn, txc = prep(Xte)
    ytr = y[tr].astype(np.int32)
    cw = jnp.asarray(len(tr) / (2 * np.bincount(y[tr], minlength=2)), jnp.float32)   # wie beim LGN
    params = init_ft(jax.random.PRNGKey(a.seed), xn.shape[1], [len(v) + 1 for v in prep.vocab])
    opt = optax.adamw(1e-3, weight_decay=1e-5)
    state = opt.init(params)

    @jax.jit
    def step(p, s, a_, b_, y_):
        loss, g = jax.value_and_grad(
            lambda q: (cw[y_] * optax.softmax_cross_entropy_with_integer_labels(ft_apply(q, a_, b_), y_)).mean())(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, loss

    fwd = jax.jit(lambda p, a_, b_: ft_apply(p, a_, b_).argmax(-1))
    pred = lambda p, a_, b_: np.concatenate([np.asarray(fwd(p, a_[i:i + 4096], b_[i:i + 4096])) for i in range(0, len(a_), 4096)])
    rng = np.random.default_rng(a.seed); bs = 512; n_full = len(xn) // bs * bs
    best, best_p = -1, None
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); order = rng.permutation(len(xn))[:n_full]
        for i in range(0, n_full, bs):
            b = order[i:i + bs]
            params, state, _ = step(params, state, xn[b], xc[b], ytr[b])
        acc = metrics(pred(params, vxn, vxc), y[va])["balanced_accuracy"]
        print(f"FT-Transformer Epoche {ep:2d} | Val bal. {acc:.4f} | {time.time() - t0:.0f}s", flush=True)
        if acc > best: best, best_p = acc, jax.tree_util.tree_map(np.asarray, params)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(best_p))
    out["ft_transformer"] = {"params": int(n_params), "test": metrics(pred(best_p, txn, txc), yte)}
    print("FT-Transformer:", json.dumps(out["ft_transformer"]), flush=True)
    with open("results/ft_transformer.pkl", "wb") as f:
        pickle.dump({"params": best_p, "prep": prep}, f)
    with open("results/baselines.json", "w") as f:
        json.dump(out, f, indent=2)


def bench(a):
    """Inferenz-Durchsatz des FT-Transformers (XLA, jit-kompiliert). Vorverarbeitung NICHT mitgezählt."""
    import jax
    with open("results/ft_transformer.pkl", "rb") as f:
        d = pickle.load(f)
    _, _, feats, Xte, _, _, _ = split(a)
    txn, txc = d["prep"](Xte)
    fwd = jax.jit(lambda p, a_, b_: ft_apply(p, a_, b_).argmax(-1))
    res = []
    for batch, total in ((1, 2000), (64, 20000), (1024, 40000)):
        a_, b_ = txn[:batch], txc[:batch]
        for _ in range(5): fwd(d["params"], a_, b_).block_until_ready()
        reps = total // batch; t0 = time.perf_counter()
        for _ in range(reps): fwd(d["params"], a_, b_).block_until_ready()
        dt = time.perf_counter() - t0
        res.append({"batch": batch, "threads": a.threads_label, "ns_per_flow": dt / (reps * batch) * 1e9,
                    "flows_per_s": reps * batch / dt})
        print(json.dumps(res[-1]), flush=True)
    if a.threads_label == "1" and os.path.exists("results/gradient_boosting.pkl"):
        X, _, _, _, _, _, _ = split(a)
        gb = pickle.load(open("results/gradient_boosting.pkl", "rb"))
        bl = json.load(open("results/baselines.json"))
        bl["gradient_boosting"]["speed_1thread"] = gb_speed(gb, gb_matrix(X, Xte, feats)[1])
        json.dump(bl, open("results/baselines.json", "w"), indent=2)
        print("Gradient Boosting:", json.dumps(bl["gradient_boosting"]["speed_1thread"]), flush=True)
    path = "results/transformer_speed.json"
    allr = json.load(open(path)) if os.path.exists(path) else []
    allr = [r for r in allr if r["threads"] != a.threads_label] + res
    json.dump(allr, open(path, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "bench"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads-label", default="alle")
    a = ap.parse_args()
    train(a) if a.mode == "train" else bench(a)
