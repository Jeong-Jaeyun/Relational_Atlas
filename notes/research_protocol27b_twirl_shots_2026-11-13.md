# I01 stage 27b: finite-shot twirl on/off (prediction frozen before acquisition)

Date: 2026-11-13. Builds on [stage 27a](research_result27a_exact_design_2026-11-13.md) and the [revised claim](topic_decision_revised_2026-11-13.tex). This protocol is committed to git before any 27b code exists. A failed prediction is reported as a failure; settings, thresholds, seeds and the analysis are not changed after acquisition. No dimension-selection claim is made.

## Question

Stage 27a showed exactly that the observed block contrast is an interference term. It reverses sign when the return pulse sign is flipped (σ = −1), and it is removed by a Pauli twirl of the pulses (random σ per shot), which leaves only $\sin^4(\varepsilon/2)D \approx 10^{-8}$–$10^{-9}$. Stage 27b asks whether finite-shot Aer data discriminate these three conditions as predicted.

## Frozen circuit and budget

- **Circuit.** Cutoff-one circuit of stages 17–19 and 27a: n = 3, β = 0.45, β_r = 0.15, nonlinear spectrum, τ = 0, shifts (0, 0.60), bias 1.25, L = 7. ε = π/2. Transpiled u3/cx template; source $R_x(\varepsilon)$ after barrier 1, return $R_x(-\sigma\varepsilon)$ after barrier 2; `measure_all`. The joint event is the all-zero 8-bit string, and clock acceptance is the clock bits both zero. The same counting is used as in stage 17.
- **Classes.** same_support (0,0), conjugate_partner (0,1), different_pair_same_sign (0,2), different_pair_opposite_sign (0,3).
- **Conditions.**
  - `plus` (σ = +1 every shot).
  - `minus` (σ = −1 every shot).
  - `twirl`: each shot independently σ = ±1 with probability 1/2. This is implemented by drawing $n_+\sim{\rm Bin}(n_l,1/2)$ per (class, block, axis, label) cell with seed 20261128, then running the σ = +1 circuit with $n_+$ shots and the σ = −1 circuit with $n_l-n_+$. The per-sign record is stored, but the confirmatory twirl analysis pools the two signs, i.e. discards the record, as standard randomized compiling does.
- **Budget.** 1,048,576 shots per label, skewed largest-remainder allocation. The total is 4 classes × 3 conditions × 2 blocks × 3 axes × 7,340,032 = **528,482,304 preparations**.
- **Seeds.** Every circuit is run individually with its own seed `20261127 + 1_000_000*class + 100_000*condition + 1_000*label + 10*(3*block + axis) + sign_index`. Bootstrap seed 20261129; calibration seed 20261130.

## Statistics

- **Block contrast.** For each axis, $\hat\Delta=\sum_l w_l(\hat r_{1l}-\hat r_{0l})$, with frozen design weights $w_l=n_l/\sum n_l$ and rates $\hat r$ = joint counts / shots of the cell. Its variance is $\hat V=\sum_l w_l^2[\hat r_{1l}(1-\hat r_{1l})/n_{1l}+\hat r_{0l}(1-\hat r_{0l})/n_{0l}]$.
- **Comparison test (new).** For conditions A and B of one class, the Wald statistic is $\sum_{\rm axes}(\hat\Delta_A-\hat\Delta_B)^2/(\hat V_A+\hat V_B)$, with p-value from the χ² distribution with 3 degrees of freedom.
- **Single-condition null test (as stages 17/18).** Bounded weak-null bootstrap of stage 4b (`weak_null_test`) with 10,000 bootstraps.

## Predictions

Λ values are exact noncentralities at this budget, computed from exact probabilities before acquisition.

**Positive predictions.** Family of 6, Bonferroni threshold 0.05/6 = 0.008333. A test passes if p ≤ 0.008333.

| ID | Class | Test | Exact Λ |
|---|---|---|---:|
| S1 | same_support | plus vs minus | 668.7 |
| S2 | different_pair_same_sign | plus vs minus | 173.0 |
| S3 | different_pair_opposite_sign | plus vs minus | 502.4 |
| W1 | same_support | plus vs twirl | 149.9 |
| W2 | different_pair_same_sign | plus vs twirl | 41.3 |
| W3 | different_pair_opposite_sign | plus vs twirl | 117.5 |

**Null predictions.** Each is judged at the nominal 0.05, which is stricter for a prediction of no detection. A test passes if p > 0.05.

| ID | Class | Condition | Exact contrast |
|---|---|---|---|
| Z1–Z3 | three allowed classes | twirl (weak-null bootstrap) | ≤ 2.4×10⁻⁸ |
| N1–N3 | conjugate_partner | plus, minus, twirl (weak-null bootstrap) | ≤ 10⁻¹⁶ |

**Secondary prediction (D1–D3).** For each allowed class, the estimated difference vector $\hat\Delta_{\rm plus}-\hat\Delta_{\rm minus}$ has a positive inner product with the exact interference vector $C$.

**Pass rule.** The interference account passes 27b if all 6 positive predictions and all 6 null predictions pass. Under the exact model, the chance that at least one of the six true-null tests fails by chance is about 1 − 0.95⁶ ≈ 0.26. Such a failure is reported and discussed; it is not reinterpreted.

## Calibration audit (run before acquisition, inside the same pipeline)

1. **Comparison test.** For each allowed class, 2,000 panels of two independent binomial samples from the same exact `plus` probabilities at the design budget. The false-positive rate at 0.05 must lie in [0.035, 0.065]; the rate at 0.008333 is reported.
2. **Weak-null test.** 200 panels each from the exact conjugate `plus` probabilities and from the exact same_support `twirl` probabilities (null to 2.4×10⁻⁸), with 999 bootstraps. The false-positive rate at 0.05 must lie in [0.02, 0.09].

If a calibration check fails, acquisition still runs, but the corresponding confirmatory conclusions are marked withheld.

## Exploratory (no criteria)

- **Twirl with the record kept.** Using the stored per-sign counts inside the twirl condition, compare σ = +1 vs σ = −1 shots (Wald, as S). This tests the 27a/[Vy26] expectation that the discarded twirl record still carries the interference.
- Standardized residuals of all $\hat\Delta$ against exact values; the projection ratio of $\hat\Delta_{\rm twirl}$ onto $\hat\Delta_{\rm plus}$; and the clock-acceptance check (exact 1/4).

## Outputs

Code: `src/relational_atlas_i01/holdout/qiskit_rank_holdout27b_twirl_shots.py`, resumable with a checkpoint. Results: `results/qiskit_rank_holdout27b_twirl_shots.json`, holding raw counts, splits, calibration, every test and pass/fail. A result note and regression tests follow.
