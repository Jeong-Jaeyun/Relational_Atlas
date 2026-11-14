# I01 문헌 검토: 27단계 설계 전 (2026-11-13)

목적은 두 가지다. (1) [주제 결정 원고](topic_decision_2026-10-06.tex)의 기여 주장을 선행연구에 비추어 다시 좁힌다. (2) 27단계(유한 샷·잡음) 사전등록에 필요한 설계 결정을 문헌 근거로 고정한다. 이 문서의 새 계산은 경로 수 무차별 열거 1건뿐이고, 나머지는 문헌 판독과 이미 저장된 결과의 재해석이다. 재해석 중 검증되지 않은 것은 **가설**로 표시했다.

## 0. 요약 판정

| # | 이 프로젝트의 결과 | 문헌상 위치 | 판정 |
|---|---|---|---|
| 1 | 짝 접근이 섹터 밖을 거쳐 P=0으로 돌아오는 전이 항 $\sin^2(\epsilon/2)A_{ab}$ | 게이지 위반 섭동이 섹터 내부에 2차 유효 과정을 만든다는 결과 [HH20], 코히런트 게이지 드리프트 [LLY20] | **메커니즘은 이미 알려짐.** 이 회로에서의 정확(비섭동) 두 항 공식과 관측량 정의만 고유 |
| 2 | 균일 위상 표지가 대비를 지움 | 무작위 게이지 변환 [LLY20], 대칭 변환 평균에 의한 대칭 보호 [TSCT21] | **같은 수학**(대칭군 평균 = twirl). "비균일 표지 자원"은 불완전한 twirl |
| 3 | 표지 유지 시 rank 4, 삭제 시 rank 3 | 외부 위상 기준 유무에 따른 QFI [JDD12], 주변화 Fisher 정보의 chain rule, RC 레이블 회수 [Vy26] | 일반 원리의 사례. 새 발견으로 주장하지 않는다 |
| 4 | 경로 수 표와 "n=3 최초 개방" | 삼항계수 OEIS A002426, A005717 | 계수 자체는 표준 조합론. n=3 개방은 **전하 −1을 보상할 관전 축이 하나 필요**하다는 전하 균형으로 설명됨(§4) |
| 5 | phase damping은 켤레 영 보존, amplitude damping·읽기 오류는 깸 | 경로 공식에서 바로 유도 가능 | 사후 발견이 아니라 **사전 예측 가능한 명제**로 다시 써야 함(§3) |
| 6 | 코드 밖 위상 의존(25단계), 코드 보존 접근(26단계) | one-hot/unary 인코딩의 무효 상태 문제 [Sa20; LY25; LLY20] | 문제 설정은 표준. 경로 클래스별 보호 조건 $\Gamma^{\rm inv}_{ab}=\varnothing$은 이 회로의 구체 결과 |
| 7 | 켤레 누출의 근사 비중심도 Λ≈0.002–0.007 | 코히런트 오류 증폭 [KLY15], RC [WE16; Ha21] | 샷 수만 늘리는 방식으로는 비현실적(§2). 증폭 설계나 다른 관측량이 필요 |

## 1. 위치 설정: 게이지 위반과 대칭 보호 문헌 (Q1)

**Halimeh–Hauke [HH20].** 게이지 위반 섭동 $\lambda H_1$이 있을 때, Gauss 법칙 위반은 1차 항이 소거되어 짧은 시간에 $(\lambda t)^2$로 자란다. 에너지 벌점 $V H_G$를 추가하면 섹터 내부 유효 해밀토니안이 $H_{\rm eff}=P_0H_0P_0-\lambda^2P_0H_1(1-P_0)(H_0+VH_G)^{-1}H_1P_0$가 된다. 즉 섹터 밖으로 나갔다 돌아오는 2차 과정이 게이지 불변이지만 원래와 다른 동역학을 만든다. 이 프로젝트의 $A_{ab}=\langle r|X_bUX_a|w\rangle$ 항은 이 2차 과정의 디지털·두 펄스 판이다. 차이는 다음과 같다. 이 회로에서는 두 펄스 전개가 유한하므로 섭동 근사 없이 **모든 ε에서 정확한 두 항 공식**이 성립하고, 그 결과 $\epsilon=\pi$의 잔차 항 $\sin^4(\epsilon/2)D_i$까지 예측된다.

**Lamm–Lawrence–Yamauchi [LLY20].** 코히런트 오류가 물리 부분공간 밖으로 진폭을 보내는 "게이지 드리프트"를 다룬다. Trotter 단계 사이에 무작위 게이지 변환을 넣으면, 섹터 밖 진폭이 변환을 겪는 동안 무작위 위상을 받아 선형 누적이 $\sqrt{t}$ 랜덤워크로 바뀐다. Z₂ 예에서는 $U_E\,\phi(g)\,U_E|0^+\rangle=|0^+\rangle$로 드리프트가 정확히 상쇄된다. 저자들이 밝힌 적용 범위는 "섹터 밖으로 나간 뒤 게이지 변환이 적용될 만큼 머문 오류"다. 이 프로젝트의 회로가 바로 그 구조다. 과학 위상층 $e^{i\theta P}$는 진폭이 섹터 밖 중간 상태(전하 ±1)에 있을 때 작용하는 대칭 변환이다. 같은 논문은 "spin-1을 큐비트 둘에 넣으면 $4^V-3^V$개의 비물리 상태가 생긴다"고 지적하는데, 이것이 이 프로젝트의 cutoff-1 인코딩과 같다.

**Tran–Su–Carney–Taylor [TSCT21].** 대칭 변환 $C_k$를 단계 사이에 끼우면 1차 오류가 $\bar V=\frac1r\sum_kC_k^\dagger VC_k$로 평균된다. 무작위 선택은 $O(1/\sqrt r)$, 최적 선택은 $O(1/r)$로 줄어든다. Schwinger 모형 수치에서는 **균일 Z₈ 변환이 균일 U(1) 변환보다 훨씬 덜 효과적**이었다. 유한 군은 평균으로 지울 수 있는 성분이 적기 때문이다. 또 이 보호는 시간 상관이 있는 코히런트 오류에만 효과가 있다. 짝 펄스 $R_x(\epsilon)$, $R_x(-\epsilon)$는 완전히 상관된 오류의 극단적인 경우다.

**함의 1: "균일 표지가 대비를 지운다"는 Z_L twirl이다.** 모든 저장 코드에서 표지는 공통 위상 $\theta_l=2\pi l/L$을 모든 축에 곱한다(`np.full(n, shift + 2*pi*label/L)`). 따라서 표지 $l$은 전하 $P$가 생성하는 U(1)의 Z_L 부분군 원소이고, 균일 평균은 전하 차이 $q\not\equiv0\pmod L$인 푸리에 성분을 모두 지운다.

**가설 H-L (검증 필요).** 비제약 상태에서 간섭항의 전하 차이는 $[-2n,2n]$ 범위에 있다. $L=2n+1$이면 0이 아닌 모든 $q$가 지워지지만, $L=2n-1$이면 $q=\pm(2n-1)$ 성분이 에일리어싱으로 살아남는다. 선검증 1에서 짧은 표지 $L=2n-1$의 비제약 32행 중 16행이 표지를 지워도 공통 위상에 반응했던 결과가 이것으로 설명될 수 있다. 저장된 정확 격자로 바로 확인할 수 있으므로, 27a에서 저비용 정확 계산으로 시험한다.

**대칭 검증 문헌.** 대칭 검사 사후선택에 의한 오류 완화 [BM18; Sa19], Gauss 법칙 오라클 [St19], 앤실라 제어 게이지 변환을 이용한 Zeno 사영 [Ba24]은 "대칭 섹터 밖으로 샌 진폭을 검출하거나 버리는" 방향의 연구다. 이 프로젝트의 복귀 확률 측정은 그 사영의 한 형태다. 다만 측정 대상이 사영을 통과한 진폭의 **간섭 대비**라는 점에서 목적이 다르다. 코히런트 오류를 신드롬이 없는 공간으로 사영하면 코히런트 논리 회전이 남는다는 QEC 결과 [Br18; Be18]도 같은 구조다.

## 2. 검출 가능성과 증폭 (Q2)

**무차별 샷 증가의 비용.** 비중심도는 표본 크기에 선형이고(Λ ∝ N), 17·18단계에서는 Λ=9.90일 때 검출에 실패했고 Λ=19.80일 때 검출했다. 주제 결정 원고의 Λ(17단계 예산 22,020,096 준비 기준)를 Λ≈20까지 선형 외삽하면 다음과 같다.

| 오류 채널 (p=0.03) | 현재 Λ | Λ≈20에 필요한 준비 수 |
|---|---:|---:|
| 대칭 읽기 오류 | 0.00677 | 약 6.5×10¹⁰ |
| 준비 amplitude damping | 0.00208 | 약 2.1×10¹¹ |

누출은 오류율 p에 1차로 비례하므로 Λ ∝ p²이다. 현실적인 p=0.001에서는 같은 Λ에 약 900배 더 많은 준비가 필요하다. 따라서 **27단계에서 켤레 누출의 위치 판정을 샷 수만으로 노리는 것은 범위에서 제외**한다.

**코히런트 증폭 [KLY15; Ni21].** Robust phase estimation은 게이트를 k번 반복해 회전각을 k배로 키우고, 총 게이트 수 N에 대해 $O(1/N)$(Heisenberg 스케일링)의 추정 오차를 얻는다. SPAM 같은 가법 오차가 $1/\sqrt8$보다 작으면 강건하다. 코히런트 오류는 깊이에 따라 최대 제곱으로 누적되고, 확률적 Pauli 오류는 선형으로 누적된다 [Ha21]. 따라서 **반복 깊이 k에 대한 스케일링으로 코히런트 성분과 비코히런트 누출을 분리**할 수 있다. 다만 이 회로에서 짝 접근 블록을 반복했을 때 전이 진폭이 실제로 누적되는지는 아직 계산하지 않았다. 27a에서 정확 계산으로 먼저 확인한다.

## 3. 잡음 모형의 현실성 (Q3)

**Twirling 예측 (해석적, 미검증).** 각 접근 펄스를 Pauli twirl하면 $R_x(\epsilon)$은 채널 $\rho\mapsto c^2\rho+s^2X\rho X$($c=\cos\frac\epsilon2$, $s=\sin\frac\epsilon2$)가 되고, 경로 사이의 간섭이 사라진다. 이때 복귀 확률은 $c^4|A_0|^2+s^4|A_{ab}|^2$이며, 공통 위상 블록 대비는

$$\Delta_i^{\rm tw}(\epsilon)=\sin^4\!\tfrac{\epsilon}{2}\,D_i \qquad\text{(코히런트: } \tfrac12\sin^2\!\epsilon\,C_i+\sin^4\!\tfrac{\epsilon}{2}\,D_i\text{)}$$

가 된다. 작은 ε에서 신호가 $O(\epsilon^2)$에서 $O(\epsilon^4)$로 떨어진다. 즉 하드웨어에서 randomized compiling [WE16; Ha21]을 쓰면 이 프로젝트의 주 신호인 간섭항 $C_i$가 사라진다. 반대로 twirl 기록(레이블)을 보존하면 코히런트 정보를 회수할 수 있다는 결과 [Vy26]는 이 프로젝트의 "위상 표지 보존" 결과와 같은 Fisher 정보 분해 $F_{\rm joint}=F_{\rm marg}+\mathbb E_b[F_{s|b}]$를 쓴다. [Vy26]은 2026년 단독 저자 프리프린트이고 동료심사를 거치지 않았다. 분해 자체는 표준 chain rule이다.

**대칭 채널 명제 (사전 예측으로 재서술).** 전이 진폭은 경로 합 $A_{ab}=\sum_{x\in\Gamma_{ab}}r^*_{X_bX_ax}w_xU_{X_ax}$이다. 계산 기저에서 대각인 Kraus 연산자를 가진 채널(phase damping 등)은 경로 집합을 바꾸지 않으므로, 경로가 0개인 켤레 클래스는 영으로 유지되어야 한다. 비트를 바꾸는 Kraus 연산자(amplitude damping)나 측정 사건을 바꾸는 읽기 오류는 새 경로를 열 수 있다. 19단계의 결과(phase damping 2.56×10⁻¹⁷)는 이 명제의 확인 사례다. 원고에서는 이를 "발견"이 아니라 "예측의 확인"으로 서술한다.

**읽기 오류 완화 [Bv21; Na21].** 22단계에서 읽기 누출의 1차 항은 거의 q1 단일 비트 결과에서 나왔다. 텐서곱형 읽기 완화(보정 행렬 역적용)는 이 1차 항을 제거할 것으로 예상된다. 따라서 "채널별 지문" 주장은 완화 후에도 남는지 명시해야 한다.

## 4. 인코딩과 조합론 (Q4)

**one-hot 인코딩의 무효 코드.** unary(one-hot)·Gray·이진 인코딩의 자원 비교 [Sa20], spin-S 사슬에서 direct(one-hot)·compact·Dicke 매핑의 하드웨어 비교 [LY25]가 있다. one-hot 매핑은 큐비트와 Hamiltonian 항이 많고, 잡음이 있을 때 무효 상태 누출을 사후선택으로 걸러야 한다. 25단계의 "코드 밖 위상 연장"과 26단계의 "코드 보존 접근"은 이 표준 문제의 한 사례다. 26단계 접근은 해밍 가중치 0↔1 사이만 회전하는 Givens형 연산에 해당한다. 새로움으로 주장할 수 있는 것은 경로 클래스별 보호 조건($\Gamma^{\rm inv}_{ab}=\varnothing$이면 코드 밖 위상에 완전히 불변)과, 같은 비트 클래스만 민감하다는 구체 결과뿐이다.

**경로 수는 삼항계수이고, n=3 개방은 전하 균형이다.** $T_m(q)=[z^q](z^{-1}+1+z)^m$의 열은 OEIS A002426(중심 삼항계수 1, 1, 3, 7, 19, 51, 141)과 A005717(1, 2, 6, 16, 45)이다. 이번에 cutoff-1 인코딩을 무차별 열거해 n=2–6의 네 클래스 표 전체가 공식과 일치함을 다시 확인했다. 열거 과정에서 다음 해석이 분명해졌다.

- **같은 큐비트 클래스** $T_n(0)$는 $X_aX_a=I$이므로 P=0 기저 상태의 총 개수일 뿐이다(자명).
- **다른 축·같은 부호 클래스.** 축 0의 +1 비트를 켜고(전하 +1) 축 1의 +1 비트를 끄면(전하 −1) 순 전하 변화는 0이다. 그런데 출발 상태가 P=0이려면 나머지 $n-2$개 축의 전하 합이 −1이어야 한다. $n=2$에서는 나머지 축이 없으니 불가능하고, $n\ge3$에서 처음 가능해진다.

따라서 "3은 이 경로의 최초 개방 차원"이라는 문장은 **"보상 전하를 담을 관전 축이 최소 하나 필요하다"**로 바꿔 쓰는 것이 정확하다. 이는 공간 차원 선택과의 연결을 더 약하게 만든다.

## 5. 위상 기준과 식별성 (Q5)

외부 위상 기준이 없을 때는 위상 평균 상태로 QFI를 계산해야 하며, 이를 무시하면 과대평가된다 [JDD12; BRS07]. 장애 모수(nuisance)가 있는 양자 추정의 일반론 [SYH20]과, 모수의 일부 함수에만 의존하는 "sloppy" 모형의 특이 Fisher 행렬 논의 [Fr24]가 Qiskit 3–7단계의 "미지 영점·드리프트로 rank 3" 결과의 일반적인 틀이다. 이 프로젝트의 기여는 원리가 아니라, 시계 조건화·드리프트·에일리어싱·확률적 브리지 각각에서 어떤 기준 자원이 rank를 회복하는지를 구체 회로로 보인 것이다.

## 6. 27단계 설계에 주는 함의

권장 순서는 **샷 없는 정확 계산(27a)으로 예측을 고정한 뒤에 유한 샷(27b)**이다.

**27a: 정확 계산, 저비용**
1. 가설 H-L 검증: 선검증 1의 $L=2n-1$ 실패 16행이 전하 차이 $q=\pm(2n-1)$ 푸리에 성분으로 정확히 재현되는가.
2. Twirl 예측 검증: 26단계 회로에서 코히런트 대 Pauli-twirled 접근의 대비가 $\Delta^{\rm tw}=\sin^4(\epsilon/2)D$를 따르는가.
3. 증폭 가능성: 짝 접근 블록을 k번 반복할 때 전이 진폭이 k에 대해 누적되는가. 누적된다면 코히런트(k²)와 비코히런트(k) 성분이 분리되는 깊이와 예산은 얼마인가.
4. 대칭 채널 명제를 정리로 쓰고 기존 19단계 결과로 대조한다.

**27b: 유한 샷 (27a 결과로 예산을 정한 뒤 사전등록)**
- 주 검정 후보: 26단계 회로에서 twirl 유무에 따른 간섭항 $C$의 검출. 신호 크기가 17단계와 같은 $10^{-4}$대일 것으로 예상되지만, 사전등록 전에 27a에서 정확한 Λ를 계산해 확정한다.
- 켤레 누출 위치 판정은 27a-3에서 증폭이 확인될 때만 다시 고려한다.

**주제 결정 원고 수정 권고**
- 선행연구 절에 [HH20; LLY20; TSCT21; St19; BM18; Br18]을 추가하고, 메커니즘의 일반형이 이미 알려져 있음을 명시한다.
- "n=3 최초 개방"을 §4의 전하 균형 서술로 바꾼다.
- 표지 결과는 Z_L twirl과 Fisher chain rule의 사례로 위치시킨다.
- phase damping 결과는 대칭 채널 명제의 확인으로 서술한다.

## 참고문헌

검증 표시: **[읽음]** 이번 검토에서 본문을 읽음 · **[서지 확인]** 저널·arXiv 정보를 웹에서 확인 · **[인용 목록]** 읽은 논문의 참고문헌 목록에서 서지 확인 · **[미검증]** 기억에 의존.

- [HH20] J. C. Halimeh, P. Hauke, "Reliability of lattice gauge theories," PRL 125, 030503 (2020). arXiv:2001.00024. **[읽음][서지 확인]**
- [LLY20] H. Lamm, S. Lawrence, Y. Yamauchi, "Suppressing coherent gauge drift in quantum simulations," arXiv:2005.12688 (2020). **[읽음]** 저널 게재 여부는 확인하지 못함.
- [TSCT21] M. C. Tran, Y. Su, D. Carney, J. M. Taylor, "Faster digital quantum simulation by symmetry protection," arXiv:2006.16248; PRX Quantum 2, 010323 (2021). **[읽음]** 저널 정보는 [미검증].
- [Ba24] C. Ball, "Suppressing gauge drift in quantum simulations with gauge transformations," arXiv:2409.04395; 관련: C. Ball, T. D. Cohen, PRA 110, 022417 (2024). **[읽음][인용 목록]**
- [St19] J. R. Stryker, "Oracles for Gauss's law on digital quantum computers," PRA 99, 042301 (2019). arXiv:1812.01617. **[인용 목록]**
- [BM18] X. Bonet-Monroig, R. Sagastizabal, M. Singh, T. E. O'Brien, "Low-cost error mitigation by symmetry verification," PRA 98, 062339 (2018). arXiv:1807.10050. **[서지 확인]**
- [Sa19] R. Sagastizabal et al., "Experimental error mitigation via symmetry verification in a variational quantum eigensolver," PRA 100, 010302 (2019). **[서지 확인]**
- [Br18] S. Bravyi, M. Englbrecht, R. König, N. Peard, "Correcting coherent errors with surface codes," arXiv:1710.02270; npj Quantum Inf. 4, 55 (2018). arXiv 존재만 확인, 저널 정보는 [미검증].
- [Be18] S. J. Beale, J. J. Wallman, M. Gutiérrez, K. R. Brown, R. Laflamme, arXiv:1805.08802; PRL 121, 190501 (2018), "Quantum error correction decoheres noise." arXiv 존재만 확인, 저널 정보는 [미검증].
- [KLY15] S. Kimmel, G. H. Low, T. J. Yoder, "Robust calibration of a universal single-qubit gate set via robust phase estimation," arXiv:1502.02677; PRA 92, 062315 (2015). **[읽음]** 저널 정보는 [미검증].
- [Ni21] E. Nielsen et al., "Gate set tomography," Quantum 5, 557 (2021). **[인용 목록]**
- [WE16] J. J. Wallman, J. Emerson, "Noise tailoring for scalable quantum computation via randomized compiling," PRA 94, 052325 (2016). **[인용 목록]**
- [Ha21] A. Hashim et al., "Randomized compiling for scalable quantum computing on a noisy superconducting quantum processor," PRX 11, 041039 (2021). arXiv:2010.00215. **[읽음][인용 목록]**
- [Vy26] K. Vyas, "Label and recover coherent errors: randomized compiling does not destroy coherent-error information," arXiv:2607.26756 (2026). **[읽음]** 단독 저자 프리프린트, 동료심사 전.
- [Bv21] S. Bravyi, S. Sheldon, A. Kandala, D. C. Mckay, J. M. Gambetta, "Mitigating measurement errors in multiqubit experiments," PRA 103, 042605 (2021). arXiv:2006.14044. **[서지 확인]**
- [Na21] P. D. Nation, H. Kang, N. Sundaresan, J. M. Gambetta, "Scalable mitigation of measurement errors on quantum computers," PRX Quantum 2, 040326 (2021). arXiv:2108.12518. **[미검증]**
- [Sa20] N. P. D. Sawaya et al., "Resource-efficient digital quantum simulation of d-level systems for photonic, vibrational, and spin-s Hamiltonians," npj Quantum Inf. 6, 49 (2020). arXiv:1909.12847. **[서지 확인]**
- [LY25] E. Lötstedt, K. Yamanouchi, "Comparison of encoding schemes for quantum computing of S > 1/2 spin chains," arXiv:2502.18838 (2025). **[읽음]**
- [JDD12] M. Jarzyna, R. Demkowicz-Dobrzański, "Quantum interferometry with and without an external phase reference," PRA 85, 011801(R) (2012). arXiv:1108.3844. **[서지 확인]**
- [BRS07] S. D. Bartlett, T. Rudolph, R. W. Spekkens, "Reference frames, superselection rules, and quantum information," Rev. Mod. Phys. 79, 555 (2007). 주제 결정 원고에 이미 인용됨.
- [SYH20] J. Suzuki, Y. Yang, M. Hayashi, "Quantum state estimation with nuisance parameters," J. Phys. A (2020). arXiv:1911.02790. **[서지 확인]** 권·쪽은 미확인.
- [Fr24] "Overcoming sloppiness for enhanced metrology in continuous-variable quantum statistical models," arXiv:2410.02989. 제목과 ID만 확인했고 본문은 읽지 않음. 저자 표기는 생략.
- OEIS A002426 (central trinomial coefficients), A005717 (trinomial triangle의 다음 열), A027907 (trinomial triangle). **[서지 확인]**
