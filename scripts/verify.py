"""Vergleicht die Vorhersagen des C-Programms mit denen des Python-Modells (müssen identisch sein)."""
import sys

import numpy as np

c = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1, dtype=np.int64)[:, 0]
p = np.load(sys.argv[2]).astype(np.int64)
diff = int((c != p).sum())
print(f"{len(p)} Flows verglichen, {diff} Abweichungen -> {'IDENTISCH' if diff == 0 else 'FEHLER'}")
sys.exit(1 if diff else 0)
