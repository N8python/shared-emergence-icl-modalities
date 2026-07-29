# 128-trial task-correlation analysis

- Data root: `results_128`
- Posterior draws: 10,000
- Paired task-bootstrap draws: 10,000
- Posterior-propagation intervals: Jeffreys uncertainty for each task's Bernoulli success probability; the 100-task suite is held fixed.
- Task-bootstrap intervals: uncertainty from resampling the aligned 100 tasks; every modality and condition uses the same sampled indices.

## Overall exact-match accuracy

| Modality | Clean | Deranged | Gap | Trials/task/condition |
|---|---:|---:|---:|---:|
| Language | 0.2996 | 0.1608 | 0.1388 | 128 |
| Genome | 0.3323 | 0.1523 | 0.1801 | 128 |
| Integer | 0.4813 | 0.1672 | 0.3141 | 128 |
| Image | 0.4350 | 0.1826 | 0.2524 | 128 |
| Time | 0.3257 | 0.1402 | 0.1855 | 128 |
| Protein | 0.2642 | 0.1651 | 0.0991 | 128 |

## Discrete-modality observed Spearman ranges

- clean: 0.694 to 0.969
- deranged: 0.852 to 0.898
- gap: 0.431 to 0.886
- residual: 0.391 to 0.894

## Discrete-modality clean-minus-deranged pairs

| Pair | Observed rho | Posterior 95% interval | Task-bootstrap 95% interval | Reference rho |
|---|---:|---:|---:|---:|
| Language–Genome | 0.672 | [0.552, 0.696] | [0.524, 0.787] | 0.436 |
| Language–Integer | 0.431 | [0.352, 0.486] | [0.255, 0.579] | 0.340 |
| Language–Protein | 0.568 | [0.382, 0.585] | [0.406, 0.703] | 0.265 |
| Genome–Integer | 0.679 | [0.587, 0.687] | [0.546, 0.778] | 0.459 |
| Genome–Protein | 0.886 | [0.697, 0.843] | [0.823, 0.925] | 0.317 |
| Integer–Protein | 0.630 | [0.466, 0.624] | [0.481, 0.743] | 0.321 |

## Discrete-modality residualized-clean pairs

| Pair | Observed rho | Posterior 95% interval | Task-bootstrap 95% interval | Reference rho |
|---|---:|---:|---:|---:|
| Language–Genome | 0.675 | [0.559, 0.698] | [0.523, 0.784] | 0.457 |
| Language–Integer | 0.391 | [0.332, 0.450] | [0.216, 0.556] | 0.352 |
| Language–Protein | 0.562 | [0.391, 0.591] | [0.405, 0.700] | 0.302 |
| Genome–Integer | 0.634 | [0.568, 0.658] | [0.498, 0.753] | 0.496 |
| Genome–Protein | 0.894 | [0.715, 0.850] | [0.826, 0.930] | 0.394 |
| Integer–Protein | 0.589 | [0.457, 0.601] | [0.436, 0.715] | 0.383 |
