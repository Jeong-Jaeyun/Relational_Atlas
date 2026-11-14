# I01 stage 26: code-preserving access (prediction frozen before acquisition)

Date: 2026-10-15. This is a focused follow-up to stages 24--25, not a new dimension-selection claim.

Implementation note before acquisition: Aer 0.17.2 did not assemble the `c3rx_o0` instruction emitted by the controlled-RX constructor. The implementation now inserts the exactly equivalent 16-by-16 local unitary (an RX block on 0 and the target one-hot word, identity on every other word). No prediction, model parameter, or pass/fail threshold changed.

Each axis uses the previous one-hot code for $k\in\{-2,-1,0,1,2\}$. Preparation, return reference, energy clock, $P=0$ support, thermal weights, phase grid, label allocation and angle $\epsilon=0.83$ are unchanged. Replace an unconditional bit-$X_j$ pulse by a code-preserving rotation $R_j(\epsilon)=\exp(-i\epsilon H_j/2)$, where $H_j=|0\rangle\langle j|+|j\rangle\langle0|$ on one local axis, with $j=+1$ or $-1$, and $H_j=0$ on the other three physical levels. On the four-qubit one-hot register this is an RX on the chosen bit controlled on the other three bits being zero. It is the identity on all other bit words, so no amplitude starting in a physical codeword can visit an invalid multi-hot word.

Freeze source $+1$ on axis 0 and four returns: the same access, $-1$ on axis 0, $+1$ on axis 1, $-1$ on axis 1. Use both earlier diagonal phase extensions, additive and neutral-invalid, without changing their physical-word action. The $n=3,K=2$ grid is $4\times2\times3\times7\times2=336$ joint return probabilities. An independent sparse amplitude calculation and Qiskit Aer gate evolution will both evaluate the full grid; workers 6, batch size 6.

The earlier involutive-bit identity must **not** be applied unchanged: $H_j^2=Q_j$, the projector onto $\{0,j\}$. Writing $c=\cos(\epsilon/2)$, $s=\sin(\epsilon/2)$ and $D_j=I+(c-1)Q_j$, odd-$H$ return terms vanish on strict $P=0$ support, and the new exact amplitude prediction is

\[
\langle r|R_b(-\epsilon)UR_a(\epsilon)|w\rangle
=\langle r|D_bUD_a|w\rangle+s^2\langle r|H_bUH_a|w\rangle.
\]

The transfer-path counts, by class (same / conjugate / other-same-sign / other-opposite-sign), are predicted as $T_{n-1}^{(K)}(0)+T_{n-1}^{(K)}(-1),0,2T_{n-2}^{(K)}(-1),2T_{n-2}^{(K)}(0)$. The frozen $K=2$ rows are $n=2:(2,0,0,2)$, $n=3:(9,0,2,2)$, $n=4:(37,0,8,10)$. Thus the cross-axis same-sign path first opens at $n=3$, while the $n=3$ same-access path count drops from stage 24's 19 to 9. Counts are necessary support conditions, not promises of positive pooled contrast.

Pass/fail criteria: enumerated graph counts equal the frozen rows; no one-pulse $H$ path connects $P=0$ input to return; every full-grid additive/neutral-invalid pair agrees pointwise within $10^{-12}$; every sparse/Aer joint probability agrees within $10^{-11}$; and the Aer state after the source pulse has invalid-code probability below $10^{-12}$. Any failure is reported without altering the model or thresholds. A null conjugate *transfer* term is predicted, but the direct return probability need not vanish. No shot-noise inference or error-location claim is part of this experiment.
