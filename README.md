# LGN-IDS: Netzwerkangriffe in Echtzeit erkennen, ohne GPU

LGN-IDS klassifiziert Netzwerk-Flows als **Angriff** oder **normal**. Das Modell ist ein *Logic Gate Network* (LGN): Nach dem Training besteht es nur noch aus Logikgattern (AND, XOR, NAND …) und wird als C-Code aus reinen Bit-Operationen exportiert. Keine Gleitkomma-Mathematik im Netz, keine Matrizen, keine Runtime, keine Abhängigkeiten.

**Das Problem:** Ein Intrusion-Detection-System muss jeden Flow prüfen, bei 10–100 Gbit/s können das Hunderttausende bis Millionen pro Sekunde sein. Transformer sind für diese Aufgabe genau, aber auf CPUs um Grössenordnungen zu langsam und brauchen für solchen Durchsatz GPUs.

## Ergebnisse

UNSW-NB15, offizieller Test-Split (82'332 Flows), Angriff vs. normal. Alle Zeiten auf **einem** CPU-Kern (Intel Xeon, 2,1 GHz).

| Modell | Genauigkeit | F1 | Fehlalarme | Flows/s (1 Kern) | Zeit/Flow | Latenz 1 Flow | Grösse |
|---|---|---|---|---|---|---|---|
| **LGN (dieses Projekt, C)** | 90,0 % | 0,912 | 15,1 % | 3,91 Mio. | 256 ns | 11,4 µs | 5'337 Gatter |
| FT-Transformer (JAX/XLA) | 91,4 % | 0,924 | 12,8 % | 2'799 | 357 µs | 839 µs | 114'242 Parameter |
| Gradient Boosting (sklearn) | 90,8 % | 0,921 | 16,9 % | 65'515 | 15,3 µs | 3,7 ms | 300 Bäume |

Durchsatz pro Kern: LGN ist **1'396× schneller als der Transformer** und **60× schneller als Gradient Boosting**.

Der Transformer ist 1,4 Prozentpunkte genauer und hat weniger Fehlalarme. Um denselben Durchsatz wie **ein** CPU-Kern mit LGN zu erreichen, bräuchte er aber rund 1'400 Kerne oder eine GPU. Die Transformer-Zeit ist mit der günstigsten Batchgrösse und ohne Vorverarbeitung gemessen, die LGN-Zeit enthält die Feature-Kodierung. „Fehlalarme“ ist der Anteil normaler Flows, die als Angriff markiert werden; mit `-t` lässt sich dieser Wert zulasten der Erkennungsrate senken.

### Fehlalarme vs. Erkennungsrate

Der `score` jedes Flows ist die Differenz der Gatter-Stimmen (Angriff minus normal). Mit `-t` verschiebt man die Alarmschwelle:

| Schwelle `-t` | Erkennungsrate | Fehlalarme | Precision |
|---|---|---|---|
| 0 (Standard) | 94,1 % | 15,1 % | 88,4 % |
| 20 | 88,0 % | 5,0 % | 95,5 % |
| 40 | 80,8 % | 0,9 % | 99,1 % |
| 60 | 70,6 % | 0,1 % | 99,8 % |

Die Schwellen sind Beispiele auf dem Test-Split. Für den Betrieb wählt man die Schwelle auf eigenen Daten.

## So funktioniert es

1. **Kodierung:** Jedes der 42 Flow-Features wird in Bits übersetzt (Schwellenwerte aus den Trainingsdaten, z. B. `sbytes > 1006`; Kategorien als `xProt == 4`, also TCP).
2. **Training (JAX):** Jedes Neuron hat zwei feste Eingänge und lernt per Softmax, welches der 16 möglichen Logikgatter es sein soll.
3. **Härten:** Pro Neuron wird das wahrscheinlichste Gatter fest gewählt. Ergebnis: eine reine Logikschaltung.
4. **Export nach C:** Konstanten-Faltung, Alias-Auflösung und Dead-Code-Elimination entfernen überflüssige Gatter. Jedes verbleibende Gatter ist **eine** Bit-Operation auf `uint64_t`, damit laufen 64 Flows gleichzeitig (Bit-Slicing). Die Abstimmung am Ausgang ist ein verzweigungsfreier Addierer-Baum.
5. **Verifikation:** `make verify` prüft, dass das kompilierte C-Programm auf allen Test-Flows **exakt** dieselben Entscheidungen trifft wie das trainierte Modell.

## Schnellstart

```bash
pip install -r requirements.txt
make data        # UNSW-NB15 laden
make train       # LGN trainieren (CPU, ca. 10 Min.)
make export      # Logikschaltung -> c/lgn_model.c
make verify      # C == Python, Flow für Flow
make bench       # Durchsatz messen
make baselines   # Transformer + Gradient Boosting zum Vergleich
```

Der exportierte C-Code liegt bereits in `c/`. Ohne Training reicht `make build`.

```bash
./c/lgn_ids flows.csv              # pred,score pro Flow
./c/lgn_ids flows.csv -t 50        # Alarm erst ab score > 50 (weniger Fehlalarme)
./c/lgn_ids flows.csv --bench 50   # Durchsatz
```

## Einbinden

```c
#include "lgn_model.h"

double rows[64][LGN_N_FEATURES];   /* Features in der Reihenfolge von LGN_FEATURE_NAMES */
uint8_t pred[64]; int score[64];
lgn_predict64(&rows[0][0], n, pred, score);   /* n <= 64; pred: 1 = Angriff */
```

`c/lgn_model.c` und `c/lgn_model.h` sind die einzigen Dateien, die man braucht. Mit `-march=native` nutzt der Code AVX-512 bzw. AVX2. Ohne SIMD läuft er portabel als reines C11 (auch auf ARM), dann mit rund 1,4 Mio. Flows/s pro Kern.

## Grenzen

- **Feature-Extraktion:** Das Modell erwartet die 42 UNSW-NB15-Flow-Features (erzeugt mit Argus und Zeek). In einem echten Netz ist die Feature-Extraktion voraussichtlich teurer als die Klassifikation. Für ein eigenes Netz neu trainieren, z. B. auf Zeek-`conn.log` oder NetFlow/IPFIX.
- **Datensatz:** UNSW-NB15 ist synthetisch erzeugter Verkehr (2015). Trainings- und Test-Split unterscheiden sich in der Verteilung, deshalb liegt die Test-Genauigkeit aller Modelle deutlich unter der Validierungs-Genauigkeit (ca. 94 %). Ein grösseres Netz (4000 × 5, ohne Klassengewichtung) passte sich dem Trainings-Split besser an, erreichte auf dem Test-Split aber nur 84,7 % bei 31,5 % Fehlalarmen. Deshalb ist das kompakte Netz der Standard, und beide Klassen werden im Training gleich gewichtet (gilt auch für die Vergleichsmodelle).
- **Latenz vs. Durchsatz:** Bit-Slicing rechnet immer 64 Flows gleichzeitig. Ein einzelner Flow kostet deshalb etwa gleich viel wie 64. Bei hohem Verkehrsaufkommen spielt das keine Rolle, weil 64 Flows in Mikrosekunden zusammenkommen.

## Nächste Schritte

- **FPGA:** Dieselbe Logikschaltung als Verilog exportieren. Dann läuft die Klassifikation ohne Instruktionen direkt in Hardware, z. B. auf einer SmartNIC.
- **Eigene Daten:** Training auf Zeek- oder NetFlow-Features aus einem realen Netz.
- **Angriffstyp:** Mehrklassen-Variante (DoS, Exploits, Reconnaissance …).

## Struktur

```
lgn_ids/data.py       Laden + Bit-Kodierung der Features
lgn_ids/lgn.py        Logic Gate Network (JAX)
lgn_ids/train.py      Training, Härten, Speichern
lgn_ids/export_c.py   Logikschaltung -> optimierter C-Code
lgn_ids/baselines.py  FT-Transformer und Gradient Boosting zum Vergleich
c/lgn_ids.c           CLI (Vorhersage, Benchmark, Latenz)
c/lgn_model.{c,h}     generiertes Modell
```

Methode: Petersen et al., *Deep Differentiable Logic Gate Networks*, NeurIPS 2022. Datensatz: Moustafa & Slay, *UNSW-NB15*, 2015.
