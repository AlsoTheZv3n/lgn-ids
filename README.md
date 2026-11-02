# LGN-IDS

**Netzwerkangriffe in Echtzeit erkennen, ohne GPU.**

[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-blue)](LICENSE)
![C11](https://img.shields.io/badge/C-C11-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)

LGN-IDS klassifiziert Netzwerk-Flows als **Angriff** oder **normal**. Das Modell ist ein *Logic Gate Network* (LGN): Nach dem Training besteht es nur noch aus Logikgattern (AND, XOR, NAND …) und wird als C-Code aus reinen Bit-Operationen exportiert. Keine Gleitkomma-Mathematik im Netz, keine Matrizen, keine Runtime, keine Abhängigkeiten.

Auf **einem** CPU-Kern schafft es 3,9 Mio. Flows pro Sekunde, rund 1'400× mehr als ein Transformer auf derselben Hardware, bei 1,4 Prozentpunkten weniger Genauigkeit.

**Das Problem:** Ein Intrusion-Detection-System muss jeden Flow prüfen, bei 10–100 Gbit/s können das Hunderttausende bis Millionen pro Sekunde sein. Transformer sind für diese Aufgabe genau, aber auf CPUs um Grössenordnungen zu langsam und brauchen für solchen Durchsatz GPUs.

> Forschungsprototyp, trainiert und getestet auf dem öffentlichen Datensatz UNSW-NB15. Was für den Einsatz in einem echten Netz fehlt, steht unter [Grenzen](#grenzen).

## Ergebnisse

UNSW-NB15, offizieller Test-Split (82'332 Flows), Angriff vs. normal. Alle Zeiten auf **einem** CPU-Kern (Intel Xeon, 2,1 GHz).

| Modell | Genauigkeit | F1 | Fehlalarme | Flows/s (1 Kern) | Zeit/Flow | Latenz 1 Flow | Grösse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LGN (dieses Projekt, C)** | 90,0 % | 0,912 | 15,1 % | 3,91 Mio. | 256 ns | 11,4 µs | 5'337 Gatter |
| FT-Transformer (JAX/XLA) | 91,4 % | 0,924 | 12,8 % | 2'799 | 357 µs | 839 µs | 114'242 Parameter |
| Gradient Boosting (sklearn) | 90,8 % | 0,921 | 16,9 % | 65'515 | 15,3 µs | 3,7 ms | 300 Bäume |

Durchsatz pro Kern: LGN ist **1'396× schneller als der Transformer** und **60× schneller als Gradient Boosting**.

Der Transformer ist 1,4 Prozentpunkte genauer und hat weniger Fehlalarme. Um denselben Durchsatz wie **ein** CPU-Kern mit LGN zu erreichen, bräuchte er aber rund 1'400 Kerne oder eine GPU. Die Transformer-Zeit ist mit der günstigsten Batchgrösse und ohne Vorverarbeitung gemessen, die LGN-Zeit enthält die Feature-Kodierung. „Fehlalarme“ ist der Anteil normaler Flows, die als Angriff markiert werden; mit `-t` lässt sich dieser Wert zulasten der Erkennungsrate senken.

Auf anderer Hardware ändern sich nur die Zeiten. Die Vorhersagen des C-Programms und damit alle Genauigkeitswerte bleiben gleich.

### Fehlalarme vs. Erkennungsrate

Der `score` jedes Flows ist die Differenz der Gatter-Stimmen (Angriff minus normal). Mit `-t` verschiebt man die Alarmschwelle:

| Schwelle `-t` | Erkennungsrate | Fehlalarme | Precision |
| --- | --- | --- | --- |
| 0 (Standard) | 94,1 % | 15,1 % | 88,4 % |
| 20 | 88,0 % | 5,0 % | 95,5 % |
| 40 | 80,8 % | 0,9 % | 99,1 % |
| 60 | 70,6 % | 0,1 % | 99,8 % |

Die Schwellen sind Beispiele auf dem Test-Split. Für den Betrieb wählt man die Schwelle auf eigenen Daten.

## So funktioniert es

1. **Kodierung:** Jedes der 42 Flow-Features wird in Bits übersetzt (Schwellenwerte aus den Trainingsdaten, z. B. `sbytes > 1006`; Kategorien als `xProt == 4`, also TCP). Insgesamt sind es 409 Eingangsbits.
2. **Training (JAX):** Jedes Neuron hat zwei feste Eingänge und lernt per Softmax, welches der 16 möglichen Logikgatter es sein soll.
3. **Härten:** Pro Neuron wird das wahrscheinlichste Gatter fest gewählt. Ergebnis: eine reine Logikschaltung.
4. **Export nach C:** Konstanten-Faltung, Alias-Auflösung und Dead-Code-Elimination entfernen überflüssige Gatter (8'000 trainiert, 5'337 übrig). Jedes verbleibende Gatter ist **eine** Bit-Operation auf `uint64_t`, damit laufen 64 Flows gleichzeitig (Bit-Slicing). Die Abstimmung am Ausgang ist ein verzweigungsfreier Addierer-Baum.
5. **Verifikation:** `make verify` prüft, dass das kompilierte C-Programm auf allen Test-Flows **exakt** dieselben Entscheidungen trifft wie das trainierte Modell.

## Voraussetzungen

| Wofür | Was |
| --- | --- |
| C-Programm bauen und nutzen | GCC oder Clang (C11), `make` |
| Testdaten laden (`make data`) | `bash`, `curl`, `unzip` |
| Training, Export, Verifikation | Python 3.10+ mit den Paketen aus `requirements.txt` (NumPy, JAX, Optax, scikit-learn) |
| Vergleichsmodelle (`make baselines`) | zusätzlich `taskset` (Linux, Paket util-linux) |

Getestet unter Ubuntu 24.04 (WSL) mit GCC 13. Das Kommandozeilenprogramm nutzt POSIX-Funktionen (`getline`, `clock_gettime`), unter Windows deshalb WSL verwenden. Der Modellcode selbst (`c/lgn_model.c`) braucht kein POSIX, aber GCC oder Clang; MSVC wird nicht unterstützt.

## Schnellstart

Das trainierte Modell liegt bereits im Repo, als Schaltung in `model/` und als fertiger C-Code in `c/`. Zum Ausprobieren reicht ein C-Compiler:

```bash
git clone https://github.com/AlsoTheZv3n/lgn-ids.git
cd lgn-ids
make build                     # baut c/lgn_ids
make data                      # lädt UNSW-NB15 nach data/
./c/lgn_ids data/test.csv -q   # Zusammenfassung mit Metriken
```

Ausgabe (die Zeiten hängen von der CPU ab):

```text
82332 Flows, 48240 als Angriff erkannt | 255.9 ns/Flow | 3.91 Mio. Flows/s (1 Thread, inkl. Kodierung)
Accuracy 0.8997 | Precision 0.8843 | Recall 0.9410 | F1 0.9117 | False-Positive-Rate 0.1509
```

## Kommandozeile

```text
./c/lgn_ids <flows.csv> [Optionen]
```

| Option | Wirkung |
| --- | --- |
| *(keine)* | Pro Flow eine Zeile `pred,score` auf stdout, Zusammenfassung auf stderr |
| `-` statt Dateiname | Flows von stdin lesen |
| `-q` | Nur die Zusammenfassung |
| `-t T` | Alarm erst ab `score > T` (Standard 0). Höher = weniger Fehlalarme, siehe [Tabelle oben](#fehlalarme-vs-erkennungsrate) |
| `--bench N` | Durchsatz messen, gemittelt über N Durchläufe, 1 Thread |
| `--latency` | Median-Latenz für einen einzelnen Flow messen |

```bash
./c/lgn_ids flows.csv > alarme.csv     # pred,score pro Flow
./c/lgn_ids flows.csv -t 40 -q         # strengere Schwelle, nur Zusammenfassung
cat flows.csv | ./c/lgn_ids -          # von stdin
```

**Eingabe:** CSV mit Kopfzeile, Komma als Trenner, ohne Anführungszeichen. Nötig sind die 42 Feature-Spalten des Modells, erkannt am Namen. Reihenfolge und zusätzliche Spalten (z. B. `id`) spielen keine Rolle. Die Namen stehen in `model/metrics.json` unter `features`. Fehlt eine Spalte, bricht das Programm mit `Spalte fehlt: <name>` ab. Zeilen mit weniger Spalten als die Kopfzeile werden übersprungen.

Gibt es zusätzlich eine Spalte `label` (0 = normal, 1 = Angriff), gibt das Programm auch Accuracy, Precision, Recall, F1 und die Fehlalarmrate aus.

**Ausgabe:** `pred` ist 1 für Angriff und 0 für normal, in derselben Reihenfolge wie die Eingabe. `score` ist die Differenz der Gatter-Stimmen (Angriff minus normal): Je höher, desto deutlicher stimmt die Schaltung für Angriff.

## Als C-Bibliothek einbinden

Für die eigene Anwendung braucht man nur `c/lgn_model.c` und `c/lgn_model.h`:

```c
#include "lgn_model.h"

double  rows[64 * LGN_N_FEATURES];   /* n Flows, Features in der Reihenfolge von LGN_FEATURE_NAMES */
uint8_t pred[64];                    /* 1 = Angriff */
int     score[64];                   /* Stimmen Angriff minus Stimmen normal */

lgn_predict64(rows, n, pred, score);   /* n <= 64 */
```

```bash
gcc -O2 -march=native -std=c11 -c c/lgn_model.c
```

- Mit `-march=native` nutzt der Code AVX-512 bzw. AVX2. Ohne SIMD läuft er portabel (auch auf ARM), auf der Referenz-CPU dann mit rund 1,4 Mio. Flows/s pro Kern.
- `-march=native` optimiert für die CPU des Build-Rechners. Für Binaries, die auf anderen Rechnern laufen sollen, gezielt z. B. mit `-mavx2` oder ganz ohne SIMD-Flags bauen.
- `lgn_predict64` ist thread-sicher, die Zwischenpuffer sind `_Thread_local`. Pro Thread lässt sich also ein eigener Strom von Flows klassifizieren.
- `lgn_encode64` liefert nur die Bit-Kodierung (409 Eingangsbits für bis zu 64 Flows), falls man sie separat braucht.

## Alles selbst reproduzieren

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make data        # UNSW-NB15 laden -> data/train.csv, data/test.csv
make train       # LGN trainieren (CPU, ca. 10 Min.) -> model/
make export      # Logikschaltung -> c/lgn_model.c, c/lgn_model.h
make verify      # C == Python, Flow für Flow
make bench       # Durchsatz messen
make baselines   # FT-Transformer + Gradient Boosting zum Vergleich
```

`make verify` muss immer melden:

```text
82332 Flows verglichen, 0 Abweichungen -> IDENTISCH
```

Die Ergebnistabelle in `results/RESULTS.md` entsteht nach `make baselines` so:

```bash
./c/lgn_ids data/test.csv --bench 50 2> results/c_bench.txt
./c/lgn_ids data/test.csv --latency  2> results/c_latency.txt
python3 scripts/report.py
```

Hinweise:

- `make train` überschreibt `model/`, `make export` überschreibt `c/lgn_model.c` und `c/lgn_model.h`. Mit anderen Bibliotheksversionen kann ein neues Training leicht andere Zahlen liefern; `make verify` muss trotzdem 0 Abweichungen melden.
- Zeiten sind nur auf einem sonst ruhigen System vergleichbar. `make baselines` bindet die Transformer-Messung per `taskset` an einen Kern.

## Datensatz

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (UNSW Canberra, 2015): 42 Flow-Features pro Verbindung, erzeugt mit Argus und Bro (heute Zeek). Der offizielle Split hat 175'341 Trainings- und 82'332 Test-Flows. Für die Modellauswahl werden 10 % des Trainings-Splits als Validierung abgetrennt.

Die Daten sind nicht im Repo. `make data` lädt eine aufbereitete Kopie von [InitRoot/UNSW_NB15](https://github.com/InitRoot/UNSW_NB15), in der die kategorischen Spalten (Protokoll, Dienst, Verbindungszustand) als Zahlen kodiert sind: `xProt`, `xServ`, `xState`. In dieser Kopie sind die Dateinamen von Training und Test vertauscht, das Skript korrigiert das.

Laut den Autoren ist der Datensatz für akademische Forschung frei nutzbar. Kommerzielle Nutzung muss mit ihnen abgesprochen werden. Wer Ergebnisse veröffentlicht, soll ihre Arbeiten zitieren (Liste auf der [offiziellen Seite](https://research.unsw.edu.au/projects/unsw-nb15-dataset)).

## Grenzen

- **Forschungsprototyp:** Das Modell ist auf einem öffentlichen Benchmark trainiert und ersetzt kein produktives IDS.
- **Feature-Extraktion:** Das Modell erwartet die 42 UNSW-NB15-Flow-Features. In einem echten Netz ist die Feature-Extraktion voraussichtlich teurer als die Klassifikation. Für ein eigenes Netz neu trainieren, z. B. auf Zeek-`conn.log` oder NetFlow/IPFIX.
- **Datensatz:** UNSW-NB15 ist synthetisch erzeugter Verkehr (2015). Trainings- und Test-Split unterscheiden sich in der Verteilung, deshalb liegt die Test-Genauigkeit aller Modelle deutlich unter der Validierungs-Genauigkeit (ca. 94 %). Ein grösseres Netz (4000 × 5, ohne Klassengewichtung) passte sich dem Trainings-Split besser an, erreichte auf dem Test-Split aber nur 84,7 % bei 31,5 % Fehlalarmen. Deshalb ist das kompakte Netz der Standard, und beide Klassen werden im Training gleich gewichtet (gilt auch für die Vergleichsmodelle).
- **Latenz vs. Durchsatz:** Bit-Slicing rechnet immer 64 Flows gleichzeitig. Ein einzelner Flow kostet deshalb etwa gleich viel wie 64. Bei hohem Verkehrsaufkommen spielt das keine Rolle, weil 64 Flows in Mikrosekunden zusammenkommen.

## Nächste Schritte

- **FPGA:** Dieselbe Logikschaltung als Verilog exportieren. Dann läuft die Klassifikation ohne Instruktionen direkt in Hardware, z. B. auf einer SmartNIC.
- **Eigene Daten:** Training auf Zeek- oder NetFlow-Features aus einem realen Netz.
- **Angriffstyp:** Mehrklassen-Variante (DoS, Exploits, Reconnaissance …).

## Projektstruktur

```text
lgn_ids/              Python-Paket
  data.py             Laden + Bit-Kodierung der Features
  lgn.py              Logic Gate Network (JAX)
  train.py            Training, Härten, Speichern
  export_c.py         Logikschaltung -> optimierter C-Code
  baselines.py        FT-Transformer und Gradient Boosting zum Vergleich
c/
  lgn_ids.c           Kommandozeilenprogramm (Vorhersage, Benchmark, Latenz)
  lgn_model.c/.h      generiertes Modell, nicht von Hand bearbeiten
model/                trainierte Schaltung, Bit-Kodierung, Metriken, Python-Vorhersagen für verify
results/              Messwerte und RESULTS.md
scripts/
  download_data.sh    Datensatz laden
  verify.py           Vorhersagen von C und Python vergleichen
  report.py           Ergebnistabelle erzeugen
```

## Quellen

- **Methode:** Petersen et al., *Deep Differentiable Logic Gate Networks*, NeurIPS 2022, [arXiv:2210.08277](https://arxiv.org/abs/2210.08277). Die Implementierung hier ist eine eigene, minimale JAX-Version.
- **Datensatz:** Moustafa & Slay, *UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS 2015.
- **Vergleichsmodell:** Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021, [arXiv:2106.11959](https://arxiv.org/abs/2106.11959) (FT-Transformer).

## Lizenz

Code und Modell stehen unter der [MIT-Lizenz](LICENSE). Für den Datensatz UNSW-NB15 gelten die Bedingungen seiner Autoren (siehe [Datensatz](#datensatz)).
