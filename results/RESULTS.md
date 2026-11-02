| Modell | Genauigkeit | F1 | Fehlalarme | Flows/s (1 Kern) | Zeit/Flow | Latenz 1 Flow | Grösse |
|---|---|---|---|---|---|---|---|
| **LGN (dieses Projekt, C)** | 90,0 % | 0,912 | 15,1 % | 3,91 Mio. | 256 ns | 11,4 µs | 5'337 Gatter |
| FT-Transformer (JAX/XLA) | 91,4 % | 0,924 | 12,8 % | 2'799 | 357 µs | 839 µs | 114'242 Parameter |
| Gradient Boosting (sklearn) | 90,8 % | 0,921 | 16,9 % | 65'515 | 15,3 µs | 3,7 ms | 300 Bäume |

Durchsatz pro Kern: LGN ist **1'396× schneller als der Transformer** und **60× schneller als Gradient Boosting**.
