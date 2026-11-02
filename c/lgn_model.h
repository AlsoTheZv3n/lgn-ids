/* AUTOMATISCH ERZEUGT von lgn_ids/export_c.py */
#ifndef LGN_MODEL_H
#define LGN_MODEL_H
#include <stdint.h>

#define LGN_N_FEATURES   42
#define LGN_N_INPUT_BITS 409
#define LGN_N_NODES      5337   /* Logikgatter nach Optimierung */

extern const char *const LGN_FEATURE_NAMES[LGN_N_FEATURES];

/* Bis zu 64 Flows (rows: n x LGN_N_FEATURES, Reihenfolge wie LGN_FEATURE_NAMES)
   -> pred[j] (1 = Angriff), score[j] (Stimmen Angriff - Stimmen normal). */
void lgn_predict64(const double *rows, int n, uint8_t *pred, int *score);
void lgn_encode64(const double *rows, int n, uint64_t *x);

#endif
