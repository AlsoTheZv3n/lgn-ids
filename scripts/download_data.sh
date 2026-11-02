#!/usr/bin/env bash
# Lädt den UNSW-NB15-Datensatz (offizieller Train/Test-Split, Kategorien als Zahlen kodiert).
# Hinweis: In dieser Kopie sind die Dateinamen vertauscht. Die Datei mit 175'341 Zeilen
# ist der offizielle Trainings-Split, die mit 82'332 Zeilen der Test-Split.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data raw
curl -sSfL -o raw/UNSW_NB15.zip "https://raw.githubusercontent.com/InitRoot/UNSW_NB15/master/UNSW_NB15.zip"
unzip -o -q raw/UNSW_NB15.zip -d raw
mv raw/UNSW_NB15_testing-set.csv data/train.csv
mv raw/UNSW_NB15_training-set.csv data/test.csv
wc -l data/train.csv data/test.csv
