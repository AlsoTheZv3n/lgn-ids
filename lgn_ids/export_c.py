"""Gehärtetes LGN -> reiner C-Code aus Bit-Operationen.

- Bit-Slicing: ein uint64_t trägt dasselbe Bit von 64 Flows -> 64 Flows pro Instruktion
- Konstanten-Faltung, Alias-Auflösung (Gatter A/B), Common-Subexpression-Elimination
- Dead-Code-Elimination: nur Gatter, die den Ausgang beeinflussen, werden erzeugt
- GroupSum als bit-geslicter Zähler; Neuronen, die sich zwischen den Klassen aufheben, entfallen

python -m lgn_ids.export_c --model model --out c
"""
import argparse
import json
from collections import Counter

import numpy as np

from .data import BitEncoder

# Gatter-ID g ist zugleich das Bitmuster (f(0,0), f(0,1), f(1,0), f(1,1)) -> TT[g][a][b]
TT = [[[(g >> (3 - (2 * a + b))) & 1 for b in (0, 1)] for a in (0, 1)] for g in range(16)]
EXPR = {  # Bitmuster -> C-Ausdruck
    0b0001: "{a} & {b}", 0b0010: "{a} & ~{b}", 0b0100: "~{a} & {b}", 0b0110: "{a} ^ {b}",
    0b0111: "{a} | {b}", 0b1000: "~({a} | {b})", 0b1001: "~({a} ^ {b})", 0b1011: "{a} | ~{b}",
    0b1101: "~{a} | {b}", 0b1110: "~({a} & {b})",
}
SYMMETRIC = {0b0001, 0b0110, 0b0111, 0b1000, 0b1001, 0b1110}


def check_truth_tables():
    """Stellt sicher, dass TT exakt dem Polynom-Training (GATE_COEFS) entspricht."""
    from .lgn import GATE_COEFS
    for g in range(16):
        c = GATE_COEFS[g].tolist()
        for a in (0, 1):
            for b in (0, 1):
                assert int(c[0] + c[1] * a + c[2] * b + c[3] * a * b) == TT[g][a][b], g


class Circuit:
    def __init__(self, n_inputs):
        self.exprs, self.deps = [], []      # erzeugte Knoten
        self.cse = {}
        self.inputs = [("in", k) for k in range(n_inputs)]

    def node(self, pattern, a, b):
        if pattern in SYMMETRIC and b < a:
            a, b = b, a
        key = (pattern, a, b)
        if key not in self.cse:
            self.cse[key] = ("t", len(self.exprs))
            self.exprs.append((pattern, a, b)); self.deps.append([a, b])
        return self.cse[key]

    def gate(self, g, A, B):
        """Gatter g auf Referenzen A, B anwenden (mit Faltung). Referenz = ('c',0|1) | ('in',k) | ('t',k)."""
        f = lambda a, b: TT[g][a][b]
        def unary(ref, f0, f1):            # Funktion einer Variablen: f(0), f(1)
            if f0 == f1: return ("c", f0)
            if (f0, f1) == (0, 1): return ref
            return self.node(0b1100, ref, ref)  # ~ref (Muster 1100 = NOT a)
        if A[0] == "c" and B[0] == "c": return ("c", f(A[1], B[1]))
        if A[0] == "c": return unary(B, f(A[1], 0), f(A[1], 1))
        if B[0] == "c": return unary(A, f(0, B[1]), f(1, B[1]))
        if A == B: return unary(A, f(0, 0), f(1, 1))
        pat = sum(f(a, b) << (3 - (2 * a + b)) for a in (0, 1) for b in (0, 1))
        if pat in (0b0000, 0b1111): return ("c", pat & 1)
        if pat == 0b0011: return A
        if pat == 0b0101: return B
        if pat == 0b1100: return self.node(0b1100, A, A)
        if pat == 0b1010: return self.node(0b1100, B, B)
        return self.node(pat, A, B)


def build(circ_npz, n_inputs):
    z = np.load(circ_npz)
    depth = sum(1 for k in z.files if k.startswith("gate"))
    c = Circuit(n_inputs)
    refs = c.inputs
    for l in range(depth):
        ia, ib, gates = z[f"ia{l}"], z[f"ib{l}"], z[f"gate{l}"]
        refs = [c.gate(int(g), refs[int(a)], refs[int(b)]) for a, b, g in zip(ia, ib, gates)]
    return c, refs


def ref_c(r):
    return {"c": lambda k: "0ULL" if k == 0 else "~0ULL", "in": lambda k: f"x[{k}]", "t": lambda k: f"t[{k}]"}[r[0]](r[1])


def emit(model_dir, out_dir, chunk=3000):
    check_truth_tables()
    enc = BitEncoder.load(f"{model_dir}/encoder.json")
    meta = json.load(open(f"{model_dir}/metrics.json"))
    feats = meta["features"]
    c, out_refs = build(f"{model_dir}/circuit.npz", len(enc.bits))

    # GroupSum: erste Hälfte = Klasse 0 (normal), zweite = Klasse 1 (Angriff)
    half = len(out_refs) // 2
    weight = Counter()
    const_votes = [0, 0]
    for cls, refs in ((0, out_refs[:half]), (1, out_refs[half:])):
        for r in refs:
            if r[0] == "c": const_votes[cls] += r[1]
            else: weight[r] += 1 if cls == 1 else -1
    weight = {r: w for r, w in weight.items() if w != 0}   # heben sich auf -> weg

    # Dead-Code-Elimination
    live = [False] * len(c.exprs)
    stack = [r[1] for r in weight if r[0] == "t"]
    while stack:
        k = stack.pop()
        if live[k]: continue
        live[k] = True
        stack += [d[1] for d in c.deps[k] if d[0] == "t" and not live[d[1]]]
    remap, order = {}, []
    for k in range(len(c.exprs)):
        if live[k]: remap[k] = len(order); order.append(k)
    rr = lambda r: ("t", remap[r[1]]) if r[0] == "t" else r

    n_nodes = len(order)
    n_feat_bits = len(enc.bits)
    lines = []
    w = lines.append
    w("/* AUTOMATISCH ERZEUGT von lgn_ids/export_c.py — nicht von Hand bearbeiten. */")
    w('#include "lgn_model.h"\n')
    w(f"const char *const LGN_FEATURE_NAMES[LGN_N_FEATURES] = {{{', '.join(json.dumps(f) for f in feats)}}};\n")
    fidx = {f: i for i, f in enumerate(feats)}
    w(f"static const unsigned short BIT_FEAT[{n_feat_bits}] = {{{', '.join(str(fidx[b[0]]) for b in enc.bits)}}};")
    w(f"static const unsigned char BIT_EQ[{n_feat_bits}] = {{{', '.join('1' if b[1] == 'eq' else '0' for b in enc.bits)}}};")
    w(f"static const double BIT_VAL[{n_feat_bits}] = {{{', '.join(repr(float(b[2])) for b in enc.bits)}}};\n")
    w("#if defined(__AVX512F__) || defined(__AVX2__)")
    w("#include <immintrin.h>")
    w("#endif")
    w("/* Rohe Features (64 Flows) -> bit-geslicte Eingänge. Spaltenweise + SIMD: 8 (AVX-512) bzw. 4 (AVX2) Vergleiche pro Instruktion. */")
    w("void lgn_encode64(const double *rows, int n, uint64_t *x) {")
    w("    static _Thread_local double col[LGN_N_FEATURES][64] __attribute__((aligned(64)));")
    w("    for (int j = 0; j < 64; j++)")
    w("        for (int f = 0; f < LGN_N_FEATURES; f++) col[f][j] = j < n ? rows[j * LGN_N_FEATURES + f] : 0.0;")
    w("    for (int k = 0; k < LGN_N_INPUT_BITS; k++) {")
    w("        const double *c = col[BIT_FEAT[k]]; const double v = BIT_VAL[k]; uint64_t m = 0;")
    w("#if defined(__AVX512F__) && defined(__AVX512BW__)")
    w("        const __m512d vv = _mm512_set1_pd(v);  /* 8 Masken à 8 Bit direkt in Maskenregistern zu 64 Bit zusammensetzen */")
    w("        __mmask8 q[8];")
    w("        if (BIT_EQ[k]) for (int j = 0; j < 8; j++) q[j] = _mm512_cmp_pd_mask(_mm512_load_pd(c + 8 * j), vv, _CMP_EQ_OQ);")
    w("        else           for (int j = 0; j < 8; j++) q[j] = _mm512_cmp_pd_mask(_mm512_load_pd(c + 8 * j), vv, _CMP_GT_OQ);")
    w("        __mmask32 lo = _mm512_kunpackw(_mm512_kunpackb(q[3], q[2]), _mm512_kunpackb(q[1], q[0]));")
    w("        __mmask32 hi = _mm512_kunpackw(_mm512_kunpackb(q[7], q[6]), _mm512_kunpackb(q[5], q[4]));")
    w("        m = (uint64_t)_mm512_kunpackd(hi, lo);")
    w("#elif defined(__AVX512F__)")
    w("        const __m512d vv = _mm512_set1_pd(v);")
    w("        if (BIT_EQ[k]) for (int j = 0; j < 64; j += 8) m |= (uint64_t)_mm512_cmp_pd_mask(_mm512_load_pd(c + j), vv, _CMP_EQ_OQ) << j;")
    w("        else           for (int j = 0; j < 64; j += 8) m |= (uint64_t)_mm512_cmp_pd_mask(_mm512_load_pd(c + j), vv, _CMP_GT_OQ) << j;")
    w("#elif defined(__AVX2__)")
    w("        const __m256d vv = _mm256_set1_pd(v);")
    w("        if (BIT_EQ[k]) for (int j = 0; j < 64; j += 4) m |= (uint64_t)_mm256_movemask_pd(_mm256_cmp_pd(_mm256_load_pd(c + j), vv, _CMP_EQ_OQ)) << j;")
    w("        else           for (int j = 0; j < 64; j += 4) m |= (uint64_t)_mm256_movemask_pd(_mm256_cmp_pd(_mm256_load_pd(c + j), vv, _CMP_GT_OQ)) << j;")
    w("#else")
    w("        if (BIT_EQ[k]) { for (int j = 0; j < 64; j++) m |= (uint64_t)(c[j] == v) << j; }")
    w("        else           { for (int j = 0; j < 64; j++) m |= (uint64_t)(c[j] >  v) << j; }")
    w("#endif")
    w("        x[k] = m;")
    w("    }")
    w("}\n")

    n_chunks = (n_nodes + chunk - 1) // chunk
    for ci in range(n_chunks):
        w(f"static void part{ci}(const uint64_t *restrict x, uint64_t *restrict t) {{")
        for idx in range(ci * chunk, min((ci + 1) * chunk, n_nodes)):
            pat, a, b = c.exprs[order[idx]]
            if pat == 0b1100:
                w(f"    t[{idx}] = ~{ref_c(rr(a))};")
            else:
                w(f"    t[{idx}] = {EXPR[pat].format(a=ref_c(rr(a)), b=ref_c(rr(b)))};")
        w("}\n")

    # GroupSum als Addierer-Baum (Carry-Save / Wallace), komplett verzweigungsfrei.
    # score = sum_pos w*b + sum_neg |w|*(1-b) - neg_total + c1 - c0   mit (1-b) = ~b
    from collections import defaultdict, deque
    cols, neg_total = defaultdict(deque), 0
    for r, wt in sorted(weight.items()):
        term = ref_c(rr(r)) if wt > 0 else "~" + ref_c(rr(r))
        if wt < 0: neg_total += -wt
        for sh in range(abs(wt).bit_length()):
            if (abs(wt) >> sh) & 1: cols[sh].append(term)
    adder, out_bits, nv, sh = [], [], 0, 0
    while sh <= max(cols, default=-1):
        col = cols[sh]
        while len(col) >= 2:
            if len(col) >= 3:
                a1, b1, c1 = col.popleft(), col.popleft(), col.popleft()
                adder.append(f"    const uint64_t u{nv} = {a1} ^ {b1};")
                adder.append(f"    const uint64_t s{nv} = u{nv} ^ {c1}, k{nv} = ({a1} & {b1}) | (u{nv} & {c1});")
            else:
                a1, b1 = col.popleft(), col.popleft()
                adder.append(f"    const uint64_t s{nv} = {a1} ^ {b1}, k{nv} = {a1} & {b1};")
            col.append(f"s{nv}"); cols[sh + 1].append(f"k{nv}"); nv += 1
        out_bits.append(col[0] if col else "0ULL")
        sh += 1
    out_bits = out_bits or ["0ULL"]
    offset = const_votes[1] - const_votes[0] - neg_total

    w("void lgn_predict64(const double *rows, int n, uint8_t *pred, int *score) {")
    w("    uint64_t x[LGN_N_INPUT_BITS];")
    w("    static _Thread_local uint64_t t[LGN_N_NODES > 0 ? LGN_N_NODES : 1];")
    w("    lgn_encode64(rows, n, x);")
    for ci in range(n_chunks):
        w(f"    part{ci}(x, t);")
    lines.extend(adder)
    w(f"    const uint64_t tot[{len(out_bits)}] = {{{', '.join(out_bits)}}};")
    w("    for (int j = 0; j < n; j++) {")
    w("        int v = 0;")
    w(f"        for (int k = 0; k < {len(out_bits)}; k++) v |= (int)((tot[k] >> j) & 1) << k;")
    w(f"        score[j] = v + ({offset});   /* Stimmen Angriff minus Stimmen normal */")
    w("        pred[j] = score[j] > 0;        /* 1 = Angriff (bei Gleichstand: normal) */")
    w("    }")
    w("}")

    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/lgn_model.c", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"{out_dir}/lgn_model.h", "w") as f:
        f.write(f"""/* AUTOMATISCH ERZEUGT von lgn_ids/export_c.py */
#ifndef LGN_MODEL_H
#define LGN_MODEL_H
#include <stdint.h>

#define LGN_N_FEATURES   {len(feats)}
#define LGN_N_INPUT_BITS {n_feat_bits}
#define LGN_N_NODES      {n_nodes}   /* Logikgatter nach Optimierung */

extern const char *const LGN_FEATURE_NAMES[LGN_N_FEATURES];

/* Bis zu 64 Flows (rows: n x LGN_N_FEATURES, Reihenfolge wie LGN_FEATURE_NAMES)
   -> pred[j] (1 = Angriff), score[j] (Stimmen Angriff - Stimmen normal). */
void lgn_predict64(const double *rows, int n, uint8_t *pred, int *score);
void lgn_encode64(const double *rows, int n, uint64_t *x);

#endif
""")
    z = np.load(f"{model_dir}/circuit.npz")
    total = sum(len(z[k]) for k in z.files if k.startswith("gate"))
    stats = {"gates_trained": int(total), "gates_after_optimization": n_nodes,
             "output_terms": len(weight), "adder_cells": nv, "input_bits": n_feat_bits}
    print(json.dumps(stats))
    with open(f"{model_dir}/export_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--out", default="c")
    a = ap.parse_args()
    emit(a.model, a.out)
