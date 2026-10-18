# RSFC — dendritic vs point-neuron

| model | overall | post-switch | steady |
|---|---|---|---|
| dendritic | 79.8% | 50.3% | 85.8% |
| dendritic_soma_only | 51.3% | 48.9% | 52.2% |
| dendritic_apical_only | 49.9% | 50.2% | 49.9% |
| dendritic_sst | 51.3% | 48.9% | 52.2% |
| single_ltc | 76.4% | 53.0% | 79.9% |
| single_ltc_48 | 82.4% | 59.8% | 85.8% |
| gru | 65.7% | 52.0% | 68.4% |
| mlp | 53.4% | 45.9% | 55.8% |

Verdict: FAIL (dendritic vs best baseline -2.7 pts; SST drop 1.4 pts)
