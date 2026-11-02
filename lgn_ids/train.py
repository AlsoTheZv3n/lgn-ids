"""LGN auf UNSW-NB15 trainieren (Angriff vs. normal) und als Logikschaltung speichern.

python -m lgn_ids.train --width 2000 --depth 4 --tau 10 --epochs 20
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import BitEncoder, load_csv
from .lgn import LGN


def metrics(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12), "false_positive_rate": fp / max(fp + tn, 1),
        "balanced_accuracy": (rec + tn / max(tn + fp, 1)) / 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model")
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--thresholds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="Klassen nicht gewichten (Standard: beide Klassen gleich gewichtet)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, y, feats = load_csv(f"{a.data}/train.csv")
    Xte, yte, _ = load_csv(f"{a.data}/test.csv")
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(X)); n_val = len(X) // 10
    va, tr = perm[:n_val], perm[n_val:]

    enc = BitEncoder.fit(X[tr], feats, n_thresholds=a.thresholds)
    Btr, Bva, Bte = enc.transform(X[tr], feats), enc.transform(X[va], feats), enc.transform(Xte, feats)
    print(f"{len(feats)} Features -> {len(enc.bits)} Eingangsbits | train {len(tr)} val {n_val} test {len(Xte)}", flush=True)

    model = LGN(len(enc.bits), a.width, a.depth, n_classes=2, tau=a.tau, seed=a.seed)
    opt = optax.adam(a.lr)
    params, opt_state = model.params, opt.init(model.params)

    # Klassengewichte: Im Training sind 68 % Angriffe, im echten Netz ist der Anteil viel kleiner.
    # Gleiche Gewichtung verhindert, dass das Modell diesen Anteil als Vorwissen lernt.
    cw = jnp.asarray(len(tr) / (2 * np.bincount(y[tr], minlength=2)) if a.balance else np.ones(2), jnp.float32)

    def loss_fn(p, xb, yb):
        return (cw[yb] * optax.softmax_cross_entropy_with_integer_labels(model.apply(p, xb), yb)).mean()

    @jax.jit
    def step(p, s, xb, yb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb.astype(jnp.float32), yb)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    pred_fn = jax.jit(lambda p, xb, hard: model.apply(p, xb.astype(jnp.float32), hard).argmax(-1), static_argnums=2)

    def predict(p, B, hard=True, bs=8192):
        return np.concatenate([np.asarray(pred_fn(p, B[i:i + bs], hard)) for i in range(0, len(B), bs)])

    ytr = y[tr].astype(np.int32)
    n_full = (len(Btr) // a.batch) * a.batch            # feste Batchgrösse -> nur 1x kompilieren
    best, best_params = -1.0, None
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); order = rng.permutation(len(Btr))[:n_full]; losses = []
        for i in range(0, n_full, a.batch):
            b = order[i:i + a.batch]
            params, opt_state, loss = step(params, opt_state, Btr[b], ytr[b])
            losses.append(loss)
        bal_soft = metrics(predict(params, Bva, hard=False), y[va])["balanced_accuracy"]
        bal_hard = metrics(predict(params, Bva, hard=True), y[va])["balanced_accuracy"]
        print(f"Epoche {ep:3d} | Loss {float(np.mean(losses)):.4f} | Val bal. soft {bal_soft:.4f} | "
              f"Val bal. hart {bal_hard:.4f} | {time.time() - t0:.0f}s", flush=True)
        if bal_hard > best:   # Auswahl nach balancierter Genauigkeit der gehärteten Schaltung
            best, best_params = bal_hard, [np.asarray(l) for l in params]

    pred = predict(best_params, Bte, hard=True)
    m = metrics(pred, yte)
    print("Test (gehärtete Logikschaltung):", json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)

    enc.save(f"{a.out}/encoder.json")
    gates = LGN.gates(best_params)
    np.savez(
        f"{a.out}/circuit.npz",
        **{f"ia{i}": w[0] for i, w in enumerate(model.wiring)},
        **{f"ib{i}": w[1] for i, w in enumerate(model.wiring)},
        **{f"gate{i}": g for i, g in enumerate(gates)},
    )
    np.save(f"{a.out}/test_pred_python.npy", pred.astype(np.uint8))
    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump({"config": vars(a), "val_balanced_acc_hard": float(best), "test": m,
                   "n_input_bits": len(enc.bits), "features": feats}, f, indent=2)


if __name__ == "__main__":
    main()
