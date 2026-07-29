# Bernoulli sampling-noise correlation simulation

- Calibration data: `results_128`
- Tasks per simulated experiment: 100
- Monte Carlo draws: 20,000
- True correlation 1 uses the same latent task accuracy in both modalities but independent Bernoulli trials.
- True correlation 0 uses independent latent task accuracies with the same fitted marginal distribution.

## Pooled canonical calibration

| True latent corr. | Trials/task | Metric | Mean observed | 95% simulation interval |
|---:|---:|---|---:|---:|
| 0 | 8 | pearson | -0.001 | [-0.196, 0.199] |
| 0 | 8 | spearman | -0.001 | [-0.197, 0.199] |
| 0 | 128 | pearson | -0.000 | [-0.199, 0.200] |
| 0 | 128 | spearman | -0.000 | [-0.199, 0.200] |
| 1 | 8 | pearson | 0.848 | [0.780, 0.902] |
| 1 | 8 | spearman | 0.833 | [0.759, 0.891] |
| 1 | 128 | pearson | 0.989 | [0.983, 0.993] |
| 1 | 128 | spearman | 0.984 | [0.977, 0.990] |
