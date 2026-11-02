PY      ?= python3
CC      ?= gcc
CFLAGS  ?= -O2 -march=native -std=c11 -Wall

.PHONY: all data train export build verify bench baselines clean

all: build

data:
	bash scripts/download_data.sh

train:
	$(PY) -m lgn_ids.train --width 2000 --depth 4 --tau 10 --epochs 20 --out model

export:
	$(PY) -m lgn_ids.export_c --model model --out c

build: c/lgn_ids

c/lgn_ids: c/lgn_ids.c c/lgn_model.c c/lgn_model.h
	$(CC) $(CFLAGS) -o $@ c/lgn_ids.c c/lgn_model.c

# Prüft, dass das kompilierte C-Programm exakt dieselben Vorhersagen liefert wie Python
verify: build
	./c/lgn_ids data/test.csv > results/c_predictions.csv
	$(PY) scripts/verify.py results/c_predictions.csv model/test_pred_python.npy

bench: build
	./c/lgn_ids data/test.csv --bench 50

baselines:
	$(PY) -m lgn_ids.baselines train --epochs 6
	XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" taskset -c 0 \
		$(PY) -m lgn_ids.baselines bench --threads-label 1
	$(PY) -m lgn_ids.baselines bench --threads-label alle

clean:
	rm -f c/lgn_ids results/c_predictions.csv
