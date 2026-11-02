/* lgn_ids — Netzwerk-Flows mit einem Logic Gate Network klassifizieren.
 *
 *   lgn_ids flows.csv                 Vorhersage pro Flow nach stdout (pred,score)
 *   lgn_ids flows.csv -q              nur Zusammenfassung (mit Metriken, falls Spalte "label" existiert)
 *   lgn_ids flows.csv --bench [N]     Durchsatz messen (N Durchläufe, 1 Thread)
 *   lgn_ids flows.csv --latency       Latenz für einzelne Flows messen
 *   lgn_ids flows.csv -t T            Alarm erst ab score > T (weniger Fehlalarme, Standard 0)
 *
 * Die CSV braucht eine Kopfzeile mit den Feature-Namen (Reihenfolge egal).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lgn_model.h"

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int split(char *line, char **cols, int max) {
    int n = 0; char *p = line;
    while (n < max) {
        cols[n++] = p;
        char *c = strchr(p, ',');
        if (!c) break;
        *c = 0; p = c + 1;
    }
    char *nl = strpbrk(cols[n - 1], "\r\n"); if (nl) *nl = 0;
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Aufruf: %s flows.csv [-q] [-t SCHWELLE] [--bench N] [--latency]\n", argv[0]); return 1; }
    int quiet = 0, bench = 0, latency = 0, thr = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-q")) quiet = 1;
        else if (!strcmp(argv[i], "--bench")) { bench = (i + 1 < argc) ? atoi(argv[++i]) : 20; quiet = 1; }
        else if (!strcmp(argv[i], "--latency")) { latency = 1; quiet = 1; }
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) thr = atoi(argv[++i]);
    }
    FILE *f = strcmp(argv[1], "-") ? fopen(argv[1], "r") : stdin;
    if (!f) { perror(argv[1]); return 1; }

    char *line = NULL; size_t cap = 0; char *cols[512];
    if (getline(&line, &cap, f) < 0) { fprintf(stderr, "leere Datei\n"); return 1; }
    int nh = split(line, cols, 512), map[LGN_N_FEATURES], label_col = -1;
    for (int k = 0; k < LGN_N_FEATURES; k++) {
        map[k] = -1;
        for (int i = 0; i < nh; i++) if (!strcmp(cols[i], LGN_FEATURE_NAMES[k])) map[k] = i;
        if (map[k] < 0) { fprintf(stderr, "Spalte fehlt: %s\n", LGN_FEATURE_NAMES[k]); return 1; }
    }
    for (int i = 0; i < nh; i++) if (!strcmp(cols[i], "label")) label_col = i;

    size_t n = 0, ncap = 1 << 16;
    double *rows = malloc(ncap * LGN_N_FEATURES * sizeof(double));
    int *labels = malloc(ncap * sizeof(int));
    while (getline(&line, &cap, f) > 0) {
        if (n == ncap) { ncap *= 2; rows = realloc(rows, ncap * LGN_N_FEATURES * sizeof(double)); labels = realloc(labels, ncap * sizeof(int)); }
        int nc = split(line, cols, 512);
        if (nc < nh) continue;
        for (int k = 0; k < LGN_N_FEATURES; k++) rows[n * LGN_N_FEATURES + k] = strtod(cols[map[k]], NULL);
        labels[n] = label_col >= 0 ? atoi(cols[label_col]) : -1;
        n++;
    }
    if (f != stdin) fclose(f);

    uint8_t *pred = malloc(n + 64); int *score = malloc((n + 64) * sizeof(int));
    if (latency) {   /* Latenz für einen einzelnen Flow (Median) */
        int m = n < 20000 ? (int)n : 20000; double *lat = malloc(m * sizeof(double));
        for (int i = 0; i < m; i++) {
            double a = now(); lgn_predict64(rows + (size_t)i * LGN_N_FEATURES, 1, pred + i, score + i); lat[i] = now() - a;
        }
        for (int i = 1; i < m; i++) { double v = lat[i]; int j = i - 1; while (j >= 0 && lat[j] > v) { lat[j + 1] = lat[j]; j--; } lat[j + 1] = v; }
        fprintf(stderr, "Latenz 1 Flow (Median): %.0f ns\n", lat[m / 2] * 1e9);
        free(lat); return 0;
    }
    double t0 = now();
    int runs = bench > 0 ? bench : 1;
    for (int r = 0; r < runs; r++)
        for (size_t i = 0; i < n; i += 64) {
            int m = (n - i < 64) ? (int)(n - i) : 64;
            lgn_predict64(rows + i * LGN_N_FEATURES, m, pred + i, score + i);
        }
    double dt = (now() - t0) / runs;
    if (thr) for (size_t i = 0; i < n; i++) pred[i] = score[i] > thr;

    if (!quiet) {
        printf("pred,score\n");
        for (size_t i = 0; i < n; i++) printf("%d,%d\n", pred[i], score[i]);
    }
    size_t tp = 0, tn = 0, fp = 0, fn = 0, attacks = 0;
    for (size_t i = 0; i < n; i++) {
        attacks += pred[i];
        if (labels[i] < 0) continue;
        tp += pred[i] && labels[i]; tn += !pred[i] && !labels[i];
        fp += pred[i] && !labels[i]; fn += !pred[i] && labels[i];
    }
    fprintf(stderr, "%zu Flows, %zu als Angriff erkannt | %.1f ns/Flow | %.2f Mio. Flows/s (1 Thread, inkl. Kodierung)\n",
            n, attacks, dt / n * 1e9, n / dt / 1e6);
    if (label_col >= 0) {
        double prec = tp / (double)(tp + fp ? tp + fp : 1), rec = tp / (double)(tp + fn ? tp + fn : 1);
        fprintf(stderr, "Accuracy %.4f | Precision %.4f | Recall %.4f | F1 %.4f | False-Positive-Rate %.4f\n",
                (tp + tn) / (double)n, prec, rec, 2 * prec * rec / (prec + rec + 1e-12), fp / (double)(fp + tn ? fp + tn : 1));
    }
    free(rows); free(labels); free(pred); free(score); free(line);
    return 0;
}
