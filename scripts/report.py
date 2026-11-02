"""Fasst alle Messungen in results/RESULTS.md zusammen."""
import json
import re

m = json.load(open("model/metrics.json"))
ex = json.load(open("model/export_stats.json"))
bl = json.load(open("results/baselines.json"))
tf = json.load(open("results/transformer_speed.json"))
c = open("results/c_bench.txt").read()
c_ns = float(re.search(r"([\d.]+) ns/Flow", c).group(1))
c_lat = open("results/c_latency.txt").read()
c_one = float(re.search(r"([\d.]+) ns", c_lat).group(1))

tf1 = {r["batch"]: r for r in tf if r["threads"] == "1"}
tf_best = min(r["ns_per_flow"] for r in tf1.values())   # beste Batchgrösse, zugunsten des Transformers
gb = bl["gradient_boosting"]


def de(x, nd):  # Schweizer Format: Dezimalkomma, Apostroph als Tausendertrenner
    return f"{x:,.{nd}f}".replace(",", "'").replace(".", ",")


def t(ns):  # Zeit mit passender Einheit
    if ns < 1e3: return f"{de(ns, 0)} ns"
    if ns < 1e5: return f"{de(ns / 1e3, 1)} µs"
    if ns < 1e6: return f"{de(ns / 1e3, 0)} µs"
    return f"{de(ns / 1e6, 1)} ms"


def rate(ns):
    r = 1e9 / ns
    return f"{de(r / 1e6, 2)} Mio." if r >= 1e6 else de(r, 0)


def row(name, m_, speed_ns, lat_ns, size):
    return (f"| {name} | {de(m_['accuracy'] * 100, 1)} % | {de(m_['f1'], 3)} | {de(m_['false_positive_rate'] * 100, 1)} % | "
            f"{rate(speed_ns)} | {t(speed_ns)} | {t(lat_ns)} | {size} |")


lines = [
    "| Modell | Genauigkeit | F1 | Fehlalarme | Flows/s (1 Kern) | Zeit/Flow | Latenz 1 Flow | Grösse |",
    "|---|---|---|---|---|---|---|---|",
    row("**LGN (dieses Projekt, C)**", m["test"], c_ns, c_one, f"{de(ex['gates_after_optimization'], 0)} Gatter"),
    row("FT-Transformer (JAX/XLA)", bl["ft_transformer"]["test"], tf_best, tf1[1]["ns_per_flow"],
        f"{de(bl['ft_transformer']['params'], 0)} Parameter"),
    row("Gradient Boosting (sklearn)", gb["test"], gb["speed_1thread"]["ns_per_flow_batch"],
        gb["speed_1thread"]["latency_single_flow_us"] * 1e3, "300 Bäume"),
]
sp_tf = tf_best / c_ns
sp_gb = gb["speed_1thread"]["ns_per_flow_batch"] / c_ns
txt = "\n".join(lines) + (f"\n\nDurchsatz pro Kern: LGN ist **{de(sp_tf, 0)}× schneller als der Transformer** "
                           f"und **{de(sp_gb, 0)}× schneller als Gradient Boosting**.\n")
open("results/RESULTS.md", "w").write(txt)
print(txt)
