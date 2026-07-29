# One-sided permutation tests for positive gap correlation

- Null hypothesis: task labels are exchangeable between modalities (`H0: rho <= 0`).
- Aligned tasks: 100
- Monte Carlo permutations: 1,000,000
- Seed: 20260728
- Holm and Benjamini-Hochberg corrections cover all 15 modality pairs.
- Minimum reportable add-one permutation p-value: 1.00e-06.

| Pair | Spearman rho | One-sided permutation p | Holm p | BH-FDR p | Asymptotic p |
|---|---:|---:|---:|---:|---:|
| Language–Genome | 0.672 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 9.64e-15 |
| Language–Protein | 0.568 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 3.51e-10 |
| Genome–Integer | 0.679 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 4.01e-15 |
| Genome–Time | 0.664 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 2.44e-14 |
| Genome–Protein | 0.886 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 7.62e-35 |
| Integer–Time | 0.549 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 1.67e-09 |
| Integer–Protein | 0.630 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 1.11e-12 |
| Time–Protein | 0.689 | ≤1.0e-06 | 1.50e-05 | 1.87e-06 | 1.19e-15 |
| Language–Image | 0.441 | 2.00e-06 | 1.50e-05 | 3.33e-06 | 2.24e-06 |
| Language–Integer | 0.431 | 3.00e-06 | 1.80e-05 | 4.50e-06 | 3.89e-06 |
| Language–Time | 0.354 | 1.63e-04 | 8.15e-04 | 2.22e-04 | 1.53e-04 |
| Genome–Image | 0.149 | 0.0686 | 0.2744 | 0.0858 | 6.88e-02 |
| Image–Time | 0.113 | 0.1312 | 0.3937 | 0.1514 | 1.31e-01 |
| Image–Protein | 0.084 | 0.2023 | 0.4045 | 0.2167 | 2.03e-01 |
| Integer–Image | 0.055 | 0.2915 | 0.4045 | 0.2915 | 2.93e-01 |
