# I01 stage 27a: exact design checks (prediction frozen before computation)

Date: 2026-11-13. Motivation: [literature review](literature_review_2026-11-13.md). Stage 27a is shot-free. Its job is to test four exact claims that decide the scope and budget of a later finite-shot stage 27b. No dimension-selection claim is made.

The protocol is committed to git before any 27a code exists. A failed criterion is reported as a failure; the model, settings and thresholds will not be changed after seeing results.

## Common frozen settings

Unless stated otherwise, these are the stage-15/19/20 cutoff-one circuit settings:
- n = 3; two-bit (plus, minus) encoding per axis; prepared and return states strictly on P = 0.
- β = 0.45, β_r = 0.15, nonlinear spectrum, τ = 0.
- Common-phase blocks SHIFTS = (0, 0.60); bias 1.25 on one axis at a time.
- L = 7 labels with the frozen skewed weights.
- Four support classes (source, return) = (0,0) same, (0,1) conjugate, (0,2) other-axis same sign, (0,3) other-axis opposite sign.
- The contrast Δ is (block 1 − block 0) of the joint clock-zero/return-zero probability, label-pooled with the frozen weights, for each bias axis. "max|Δ|" means the maximum over the three bias axes.
- Clock acceptance is 1/4.

Aer runs use 6 workers. "Zero" means a numerical zero below the stated tolerance.

## A. Label aliasing (hypothesis H-L)

Data: the stored prevalidation-1 exact grid (`results/qiskit_rank_holdout_grid.json`, 128 rows: n = 3–6, two β, two biases, τ ∈ {0, π/2}, L ∈ {2n−1, 2n+1}, unrestricted or P = 0), recomputed from `qiskit_rank_holdout_grid.model`.

For one bias axis, write $a_x = w_x e^{i(\tau E_x + k_x\cdot\vartheta_0)}$, where $\vartheta_0$ carries the bias only. The label phase is $\theta_l = 2\pi l/L$ on every axis, so

$$p_\alpha(l)=\Big|\sum_x a_x e^{i(\alpha+\theta_l)P(x)}\Big|^2 .$$

Uniform label averaging keeps only charge differences $q=P(x)-P(y)\equiv0 \pmod L$. The label-erased common-shift change for α = 0.23 is therefore

$$\delta_{\rm alias}=\sum_{q\equiv0\,(L),\,q\neq0}(e^{i\alpha q}-1)S_q,\qquad S_q=\sum_{P(x)-P(y)=q}a_x\bar a_y .$$

Predictions:
- **A1.** For every one of the 128 rows, the maximum over axes of $|\delta_{\rm alias}|$ equals the stored `erased_common_shift_max_probability_change` within 1e-12.
- **A2.** $\delta_{\rm alias}$ is identically zero (no admissible q) for every L = 2n+1 row and every P = 0 row.
- **A3.** Among the 32 short-label unrestricted rows, the set with $|\delta_{\rm alias}|>10^{-10}$ equals the stored set with change > 1e-10.

The τ-dependence of that set is not predicted. Any explanation of it will be labelled post hoc.

## B. Pulse-sign twirl

The source pulse is $R_x(\varepsilon)$ on qubit a after the preparation barrier. The return pulse is $R_x(-\sigma\varepsilon)$ on qubit b after the science layer. σ = +1 is the standard configuration. Write $c=\cos\frac\varepsilon2$ and $s=\sin\frac\varepsilon2$. A Pauli twirl of each pulse is equivalent to averaging over σ = ±1. The derived predictions are

$$p_\sigma = 2^{-c_n}\,|c^2A_0+\sigma s^2A_{ab}|^2,\qquad
\Delta_\sigma=\sigma\,\tfrac12\sin^2\!\varepsilon\,C+\sin^4\!\tfrac\varepsilon2\,D,\qquad
\Delta_{\rm tw}=\tfrac12(\Delta_++\Delta_-)=\sin^4\!\tfrac\varepsilon2\,D,$$

with the stage-20 coefficients C and D.

The grid is ε ∈ {1.0, π/2} × σ ∈ {±1} × 4 classes × 42 settings = 672 independent Aer statevector circuits (transpiled u3/cx template).

- **B1.** Sparse $p_\sigma$ vs Aer: max absolute difference ≤ 1e-11.
- **B2.** The Aer $\Delta_\sigma$ obeys the two-term law with sign σ within 1e-12.
- **B3.** The Aer $\Delta_{\rm tw}$ equals $\sin^4(\varepsilon/2)D$ within 1e-12. The interference coefficient C does not enter.
- **B4.** Conjugate class: max|Δ_σ| ≤ 1e-14 for both signs and both angles.
- **B5.** Stage-26 code-preserving circuit (cutoff 2; frozen stage-24/26 settings ε = 0.83, τ = π/2, additive extension). The return rotation is replaced by the code-preserving rotation with angle −σε. The prediction is $p_\sigma=2^{-c}|B_0+\sigma s^2B_{ab}|^2$, where $B_0=\langle r|D_bUD_a|w\rangle$ is common-phase invariant, so $\Delta_{\rm tw}=s^4D'$. Both signs are run over the full grid in Aer (336 circuits). Criteria: sparse vs Aer ≤ 1e-11; $\Delta_{\rm tw}$ vs $s^4D'$ ≤ 1e-12; conjugate max|Δ| ≤ 1e-14.
- **B6 (design information, no criterion).** Approximate noncentrality at the stage-17 budget (131072 shots/label skewed allocation; Λ = Σ_axes δ² N / (2 p̄(1−p̄))) for $\Delta_+$, $\Delta_-$ and $\Delta_{\rm tw}$, at ε = π/2 for the cutoff-one classes.

## C. Naive repetition does not amplify against a null background

The k-repeated circuit is: preparation, then k copies of [source $R_x(\varepsilon)$, science layer $U(\theta)$, return $R_x(-\varepsilon)$], then clock decoding and the return projection. The same θ is used in every copy.

For k ≥ 2, a source flip in one copy and another source flip in a later copy return x to itself through an out-of-sector intermediate that experiences the common phase; the same holds for two return flips. These self-return paths exist for every class, including the conjugate class.

- **C1.** k = 1, ε = 1.0: the sparse contrasts reproduce the stored stage-15 Aer n = 3 contrasts within 1e-13.
- **C2.** For k ≥ 2 the conjugate class has 0 direct transfer paths and $2T_3(0)=14$ self-return paths (by enumeration). Its sparse max|Δ| at ε = π/2 exceeds 1e-10 for k = 2, 3 and 4. The structural null is lost, so naive repetition cannot amplify noise-induced conjugate leakage against a zero ideal background.
- **C3.** Independent Aer statevector (state set directly, no multiplexer) vs sparse for k ∈ {2, 3}, ε = π/2, 4 classes × 42 settings: ≤ 1e-11.
- **C4 (exploratory, no criterion).** max|Δ| for k = 1–6 at ε ∈ {0.1, π/2} for every class, with skewed and with uniform labels.

## D. Diagonal-noise null-preservation lemma

**Lemma.** Insert channels whose Kraus operators are diagonal in the computational basis (system and clock) anywhere between preparation and the return pulse. Each Kraus branch then keeps P = 0 input support, so odd-flip terms still vanish and the transfer path set is unchanged. Its direct term is common-phase independent. The conjugate-class contrast therefore stays zero.

A bit-changing error is not covered by the lemma. A coherent $R_x$ on a remote qubit before the source pulse followed by the return flip gives an even path through an out-of-sector intermediate during U. This path exists at n = 3, so the null should break.

The circuit is the stage-19 circuit at ε = π/2. Slot 1 is after preparation (before the source pulse); slot 2 is after the science layer (before the return pulse).
- **D1.** Coherent Z-type error: $R_z(\phi_q)$ on all 8 qubits at slot 1 and independent $R_z(\phi'_q)$ at slot 2, with φ, φ' ~ N(0, 0.3²) drawn once with seed 20261113 and fixed across settings. Aer statevector: conjugate max|Δ| ≤ 1e-14.
- **D2.** Phase damping p = 0.03 on all 8 qubits at slot 1 (stage 19 already covered slot 2). Aer density matrix: conjugate max|Δ| ≤ 1e-14.
- **D3 (positive control).** Coherent $R_x(0.1)$ on qubit 3 at slot 1. Aer statevector: conjugate max|Δ| > 1e-10.

## Outputs

Code: `src/relational_atlas_i01/holdout/qiskit_rank_holdout27a_exact_design.py`. Results: `results/qiskit_rank_holdout27a_exact_design.json`. The JSON records every criterion, the observed value and pass/fail, plus the exploratory tables. Regression tests go in `tests/`.
