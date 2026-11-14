# RELATIONAL ATLAS — I01 조건 지도

원본 아이디어: `D:\research_idea_notes_individual_2026-10-02\research_idea_notes\I01.md`.
이 폴더는 I01을 “가능/불가능”이 아니라 **어떤 조건에서 어떤 종류의 3차원성이 나타나는가**로 재검토한 독립 실험이다. 원본은 변경하지 않았다.

최신 판정과 다음 게이트는 [RESEARCH_STATUS.md](RESEARCH_STATUS.md)에 정리했다.

핵심 판정은 다음과 같다.

| 조건 | 관찰 현상 | 해석 |
|---|---|---|
| 정확한 대각 $U(1)$ 제약 $\sum_i p_i=0$ | $T^4/U(1)\cong T^3$, QFI rank 3 | 정확한 몫공간이지만 제약 자체는 가정 |
| $H_\lambda=\sum_i p_i^2+\lambda(\sum_i p_i)^2$, $\lambda>1$ 및 $2\le E_{\rm cut}<1+\lambda$ | 저에너지 QFI rank 3; 하전 gap 위에서 rank 4 복귀 | 에너지 의존 유효 차원 |
| 상대 위상만 측정 | 제약이 없는 상태도 Fisher rank 3 | 검출기 인공물; 절대 방향 측정 추가 시 rank 4 |
| 관계적 시계 조건화 | $\pi$ 재귀, 위상별 1차 조화 관측 암점 | 측정 contrast 변화; 위상 공간의 소멸 아님 |
| 순열대칭 활성화 퍼텐셜 $V(m)=am+bm(m-1)/2$, $-3b<a<-2b$ | 한 고정된 분기에서 $n=4,5,6$ 모두 활성 축 3개 | 축을 미리 고르지 않지만 3을 모수로 선택; 분기 고정 메커니즘 필요 |

가장 큰 열린 문제는 **왜 정확히 3인가**이다. 같은 단일 제약을 $n=5,6$에 적용하면 접근 차원은 4,5다. 별도의 대칭적 활성화 모형에서는 개수 3을 선택할 수 있으나, 모수 범위를 그에 맞춰 정해야 하고 한 방향 분기를 오래 고정할 장벽도 아직 없다. 따라서 이 아이디어는 “조건부 수학/토이 연구 프로그램”으로 유지하되, 우주의 실제 공간 차원이나 동역학적 발생을 입증한 것으로 보지 않는다.

상세한 수식, 수치 표, 한계, 후속 판별 실험은 [조건 지도 LaTeX 노트](notes/main.tex)에 있다. 2026-10-02 후속 연구는 [국소 조우와 접선 대조군](notes/why_three_stage1.tex), [세 활성 축 분기의 안정성](notes/why_three_stage2.tex), [주기적 토러스의 반대 감김 끈](notes/why_three_stage3.tex)에 있다. 마지막 실험은 실제 주기영상을 포함해 반대 감김쌍의 접촉률을 계산하고, 접촉 후 소멸 및 팽창을 별도 가정으로 분리했다. 각 계산의 그림과 JSON 원자료는 `results/`에 있다. 아직 반지름의 운동방정식이나 끈의 미시적 상호작용은 구현하지 않았다.

새로운 [Qiskit 회로 실험 노트](notes/why_three_qiskit1.tex)는 제약된 상태의 rank 감소와 검출기의 위상 기준 손실로 인한 겉보기 rank 감소를 분리한다. `n=3,4,5,6`의 제약 QFI rank는 `2,3,4,5`이고, `n=4` Aer 유한-shots 실험에서는 비제약/제약/전하-평균 검출기의 운영적 rank가 각각 `4/3/3`으로 나왔다. 이 결과는 3의 보편적 선택을 보여주지 않는다. 부분적인 위상 기준 손실에서는 정확 rank 4가 유지되지만 정해 둔 감도 임계 아래로 네 번째 방향이 숨을 수 있다. [유한 시계 회로](src/relational_atlas_i01/phase_reference/qiskit_clock_probe.py)는 에너지 매칭 제약을 명시적으로 준비해 조건부 변화와 $\pi$ 재귀를 확인했다. 재귀는 $P=0$ 섹터의 짝수 에너지 구조에서 온다. 원자료는 [조건 검출기 JSON](results/qiskit_condition_probe.json)과 [시계 JSON](results/qiskit_clock_probe.json)에 있다.

[위상 표지 개입 실험](notes/why_three_qiskit2.tex)은 같은 144개 Aer 회로의 raw counts를 표지별로 사용하거나 합쳐서 분석했다. 비제약 상태의 정확 CFI rank는 표지를 쓰면 4, 버리면 3이고, $P=0$ 대조군은 두 분석 모두 3이다. 공통 위상 $0\to0.3$ rad 변화의 순열 검정에서 비제약/표지 사용만 차이를 검출했다(2000회 재배분에서 $p\le0.00050$). 표지 사용은 보정된 외부 위상 기록을 자원으로 가정한다. 이 실험은 자발적 공간 차원 선택의 증거가 아니라 관측 rank의 출처를 구분한 결과다. [코드](src/relational_atlas_i01/phase_reference/qiskit_phase_label_intervention.py)와 [원자료](results/qiskit_phase_label_intervention.json)를 함께 저장했다.

[영점 보정·드리프트 실험](notes/why_three_qiskit3.tex)은 그 외부 자원 가정을 검증한다. 표지는 기록했지만 블록별 위상 영점이 미지이면 과학 데이터의 공통 방향은 영점과 정확히 혼동되어 유효 rank 3이다. 알려진 위상의 기준 측정을 같은 블록에 추가하면 rank 4가 회복된다. 두 블록의 실제 공통 위상 0.300 rad를 5회 Aer 반복에서 0.2980–0.3022 rad로 추정했다. 블록 내부의 기록되지 않은 위상 흔들림은 방향 정보의 세기를 줄인다. [코드](src/relational_atlas_i01/phase_reference/qiskit_phase_anchor_audit.py)와 [원자료](results/qiskit_phase_anchor_audit.json)에 설정·raw counts·반복 결과를 저장했다.

[유한 시계·표지·기준 위상 결합 실험](notes/why_three_qiskit4.tex)은 5준위 시계에 조건화한 상태에서도 rank 3의 세 출처를 분리한다. 비제약 상태는 표지 보존/삭제/미지 영점/기준 보정에 따라 정확 Fisher rank `4/3/3/4`, 미리 준비한 $P=0$ 대조군은 모두 3이다. $0\le\tau\le\pi$의 201점 스캔에서는 이 rank가 유지됐지만 $\tau\simeq\pi/2$에서 정보량이 크게 약해졌다. 432개 Aer 회로(각 4096 shots, CPU workers 6)의 기준 보정 추정은 주입값 0.300 rad에 대해 $\tau=0$에서 0.2956, $\tau=\pi/2$에서 0.3123 rad였다. 이는 시계 조건화 뒤에도 개입 판별이 가능하다는 토이 결과이며 공간 차원의 자발적 선택은 아니다. [코드](src/relational_atlas_i01/phase_reference/qiskit_clock_anchor_intervention.py)와 [raw counts·201점 스캔 JSON](results/qiskit_clock_anchor_intervention.json)을 저장했다.

[시계 조건부 드리프트 개입 실험](notes/why_three_qiskit5.tex)은 양 끝 기준 측정 사이에서 위상 영점이 휘는 경우를 검사한다. 실제 곡률을 nuisance로 허용하면 양 끝 기준점만으로 비제약 과학 위상의 공통 방향이 정확히 혼동되어 rank 3이고, 중간 기준점을 더하면 rank 4다. 곡률이 0이라고 잘못 고정한 분석은 rank 4를 주면서도 주입값 0.300 rad를 약 0.518 rad로 편향시켰다. 중간 기준점 보정 후에는 0.2975 rad가 나왔다($\tau=0$). $\tau=\pi/2$에서도 같은 방향이지만 불확실도가 더 크다. [코드](src/relational_atlas_i01/phase_reference/qiskit_clock_drift_intervention.py), [raw counts와 Fisher 판정](results/qiskit_clock_drift_intervention.json)을 저장했다. 이것은 ``항상 rank 3''이 하나의 원인이라는 뜻이 아니라, 기준 위상 자원의 시간 해상도가 추가 식별 조건임을 보여준다.

[균일 기준점 에일리어싱 실험](notes/why_three_qiskit6.tex)은 기준점을 늘려도 배치가 숨은 진동의 영점과 맞물리면 실패함을 보인다. $u=0,1/4,1/2,3/4,1$의 기준 5점에서는 주입한 사인 드리프트가 모두 0이라 비제약 정확 Fisher rank가 3이고, $u=1/8$ 검증점 또는 과학 시점 $u=3/8$을 포함한 9점에서는 4다. Aer 864회로(각 4096 shots, workers 6)에서 같은 데이터의 5점 보간은 참 위상 0.300 rad를 0.115 rad로 오독했지만, 과학 시점 기준 측정은 0.294 rad로 복원했다($\tau=0$). 이는 사인 모드를 알아서 적합한 값이 아니라 기준점 raw counts의 직접 대조다. [코드](src/relational_atlas_i01/phase_reference/qiskit_clock_alias_audit.py)와 [원자료](results/qiskit_clock_alias_audit.json)에 홀드아웃 기준점과 $P=0$ 대조군 반복을 기록했다.

[확률적 위상 브리지 실험](notes/why_three_qiskit7.tex)은 과학/기준 측정 시점이 떨어지면 완벽한 양쪽 기준점에도 중간 위상의 조건부 분산이 남음을 보였다. 선언된 영평균 지수상관 가우스 토이 모형에서 간격/상관시간 $h/\ell=0.05$와 1.0의 블록당 잔차 표준편차는 각각 0.0268, 0.1047 rad였다. 개별 블록의 미지 잔차를 자유 nuisance로 두면 비제약 정확 Fisher rank는 여전히 3이다. 알려진 확률 법칙과 독립 블록을 가정하면 여러 블록 평균으로 과학 위상을 추정할 수 있지만, 샷 오차만으로 만든 구간은 큰 간격에서 심하게 과소포괄된다. Aer 1728회로(각 4096 shots, workers 6)와 별도의 20,000-ensemble 근사 지도를 [코드](src/relational_atlas_i01/phase_reference/qiskit_clock_stochastic_bridge.py)·[원자료](results/qiskit_clock_stochastic_bridge.json)에 저장했다. [기존 $P=0$ Aer 대조군의 raw-count 점검](results/qiskit_pzero_sampling_audit.json)은 초기 32검정의 명목 양성 5개와 달리 추가 40검정에서는 1개만 발견했고, 전체 72검정의 표준화 분산은 정확 이항 기준 범위에 들었다. 초기 집중을 물리 신호로 해석하지 않는다.

[I01 선검증 1 결과 노트](notes/why_three_prevalidation1.tex)는 [사전 고정 프로토콜](notes/prevalidation_protocol_2026-10-04.tex)에 따라 원래의 $n=4$ 조건을 벗어난 128행을 시험한다. 충분한 위상 표지 $L=2n+1$의 64행에서는 예상 개입 rank 패턴 위반이 없었다. 짧은 표지 $L=2n-1$에서는 비제약 32행 중 16행이 표지를 지워도 공통 위상에 반응했다. 즉 표지 삭제 효과에는 해상도·시계의 조건이 붙는다. [정확 격자](results/qiskit_rank_holdout_grid.json), [독립 회로 대조](results/qiskit_rank_holdout_circuit.json), [Aer 유한 샷 원자료](results/qiskit_rank_holdout_shots.json)를 함께 저장했다. **현 판정은 토이 계열의 부분 검증이며, 관측 차원의 일반 이론이나 공간 차원 선택의 확증은 보류**다.

[I01 선검증 2 결과 노트](notes/why_three_prevalidation2.tex)는 [두 번째 사전 프로토콜](notes/prevalidation_protocol2_2026-10-04.tex)에 따라 검출 POVM·시계 스펙트럼·표지 빈도를 바꿨다. 432행에서 균일 표지 삭제, $P=0$ 또는 전하 twirl이 보장하는 불변성 360행은 모두 유지됐다. 그러나 비균일 표지의 비제약 coherent 검출 72행은 표지를 지워도 공통 위상 정보가 남았다. [정확 격자](results/qiskit_rank_holdout2_grid.json)와 [독립 Statevector/Aer 회로 대조](results/qiskit_rank_holdout2_circuit.json)를 저장했다. 따라서 발전연구 주제는 정하지 않고, 기준 분포·드리프트 법칙의 오가정 및 유한 샷 검정력 등 남은 검증을 먼저 진행한다.

[I01 선검증 3 결과 노트](notes/why_three_prevalidation3.tex)는 [세 번째 사전 프로토콜](notes/prevalidation_protocol3_2026-10-04.tex)의 실제 샷 배분으로 균일/비균일 표지를 대조했다. Aer 336회로의 [raw counts](results/qiskit_rank_holdout3_shots.json)는 보존했지만, 원래 재배분 검정은 표지별 확률까지 같다는 강한 null을 가정해 **합산 확률만 같은가**의 확증 판정에서는 철회했다. [사후 이항·재가중 감사](results/qiskit_rank_holdout3_audit.json)는 구현 진단으로만 사용한다. 경계 보존 약한-null 검정과 독립 seed를 [선검증 4b 프로토콜](notes/prevalidation_protocol4b_2026-10-04.tex)에 고정했다. 발전연구 주제는 여전히 보류한다.

[I01 선검증 4b 결과 노트](notes/why_three_prevalidation4b.tex)는 약한 합산 null에 맞춘 bootstrap을 고정한 뒤 독립 Aer seed로 336회로를 다시 실행한 결과다. [새 raw counts와 주 검정](results/qiskit_rank_holdout4b_weaknull.json)에서 비균일·비제약·$\tau=0$ 조건만 8검정 문턱 $0.00625$를 통과했다(Monte Carlo 해상도 바닥 $p=1/10001\approx0.00010$). 나머지 정확-null 여섯 조건과 약한 양수 하나는 통과하지 않았다. [새 사후 이항 감사](results/qiskit_rank_holdout4b_audit.json)와 [약한-null 방법 calibration](results/qiskit_rank_holdout4b_calibration.json)은 구현·통계 점검이다. 이 정정·재실행은 검정의 논리적 전제를 분명히 한 성과지만, 다른 cutoff와 실제 시계 레지스터까지 확인한 것은 아니다. 발전연구 주제는 고르지 않았다.

[I01 선검증 5 결과 노트](notes/why_three_prevalidation5.tex)는 [다섯 번째 사전 프로토콜](notes/prevalidation_protocol5_2026-10-04.tex)에 따라 과학 위상을 고정하고 표지 수의 무작위 요동과 블록 간 표지 분포 변화를 분리했다. Aer 252회로의 표지 삭제 분석에서는 균일→비균일 변화의 비제약 두 시계 조건만 8검정 문턱을 넘었다. 균일→균일 및 $P=0$ 대조군은 넘지 않았다. 같은 [raw counts](results/qiskit_rank_holdout5_random_labels.json)를 기록된 표지로 균일 목표에 표준화하면 8검정 모두 문턱을 넘지 않았다. [사후 20000패널 감사](results/qiskit_rank_holdout5_allocation_audit.json)는 무작위 표지 수를 고정 배분으로 잘못 취급할 때 명목 오류율이 크게 부풀 수 있음을 확인했다. 실제 시계 후선택, 미지 표지 분포 및 다른 cutoff는 아직 검증하지 않았다.

실행 환경은 conda `quantum`의 Python, NumPy, Matplotlib, Qiskit 및 Qiskit Aer다. 현재 실행·테스트 방법과 폴더별 책임은 [개발 안내](docs/DEVELOPMENT.md)에 정리했다. 프로젝트 루트에서:

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/run.py --list
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/test.py
```


LaTeX 소스는 `notes` 디렉터리에서 `xelatex`으로 컴파일하도록 작성했다. Qiskit Aer 0.17.2의 이 Windows 환경에서는 합성된 `multiplexer` 명령을 그대로 전달하면 접근 위반이 발생해, 실험 코드가 이를 표준 `u3,cx` 게이트로 분해한다. 이는 패키지를 변경하지 않는 우회다. 전체 회귀 테스트 134개는 통과했다.

## I01 선검증 6: 명시적 시계 레지스터

[사전 프로토콜](notes/prevalidation_protocol6_2026-10-04.tex), [독립 반복 프로토콜](notes/prevalidation_protocol6b_2026-10-04.tex), [결과 LaTeX 노트](notes/why_three_prevalidation6.tex)를 참조한다. 6-qubit 계에 2-qubit 에너지 시계를 붙여 시계 위상과 후선택을 실제 회로로 구현했다. Statevector/Aer 밀도행렬의 정확 확률 오차는 최대 `1.41e-14`; 수락률의 예측값은 `1/4`이다. 두 독립 Aer seed에서 각각 336회로·44,040,192 준비를 실행했다. 첫 실행의 8검정 가족 문턱 `0.00625`에서는 비균일 표지·비제약 상태·`tau=0`만 검출됐고(`p=0.0053`), 두 실행 합산에서는 같은 강한 조건이 `p=0.0001`, 약한 `tau=pi/2`가 `p=0.0039`였다. 약한 조건의 단독 결과는 `p=0.2013/0.0058`로 엇갈렸으므로 견고한 독립 재현으로 읽지 않는다. 정확 이항 모형의 사후 200패널 감사에서 그 약한 조건의 가족 문턱 검출률은 단일/합산 예산에 대해 `11/200`, `15/200`이었다. 구조적 rank와 유한 샷 검출을 구별하고, 발전연구 주제 선정은 계속 보류한다.

원자료: [첫 실행](results/qiskit_rank_holdout6_clock.json), [독립 반복](results/qiskit_rank_holdout6_clock_replicate.json), [합산 분석](results/qiskit_rank_holdout6_clock_combined.json), [검정 보정 감사](results/qiskit_rank_holdout6_calibration.json). 구현: [시계 회로와 검정](src/relational_atlas_i01/holdout/qiskit_rank_holdout6_clock.py), [독립 반복](src/relational_atlas_i01/holdout/qiskit_rank_holdout6_replication.py), [보정 감사](src/relational_atlas_i01/holdout/qiskit_rank_holdout6_calibration.py).

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/run.py qiskit_rank_holdout6_clock --shots-per-label 131072 --workers 6 --seed 20261027 --bootstraps 10000
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/run.py qiskit_rank_holdout6_replication
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/run.py qiskit_rank_holdout6_calibration --panels 200 --bootstraps 999 --seed 20261032
```

## I01 선검증 7: 국소 오류 stress test

[사전 프로토콜](notes/prevalidation_protocol7_2026-10-04.tex)과 [결과 LaTeX 노트](notes/why_three_prevalidation7.tex)는 명시적 시계 회로에 준비 amplitude damping, 과학 위상 뒤 phase damping, 대칭 읽기 오류를 각각 단독으로 넣었다. 세 오류를 확률 0.001, 0.01, 0.03에서 Aer density-matrix로 계산한 결과, 균일 표지에서 생기는 영과 P=0 제약에서 생기는 영, 총 네 pooled null은 모두 수치 정밀도에서 유지됐다. p=0.03에서는 강한 비제약 contrast가 이상값의 0.933/0.954/0.849배로 남았다(준비/위상/읽기). 약한 tau=pi/2 contrast는 준비ㆍ위상 damping에서는 0.926/0.975배였지만, 대칭 읽기 오류에서는 1.858배가 됐다. 읽기 채널은 all-zero 사건을 전체 bit-string 확률의 가중합으로 바꾸므로, 이 증가는 위상 정보의 회복이 아니다. 특히 대칭 읽기 오류 아래에는 축별 출력분포가 달라져 세 축을 각각 계산했다. 결과는 [JSON](results/qiskit_rank_holdout7_noise_stress.json), 구현은 [코드](src/relational_atlas_i01/holdout/qiskit_rank_holdout7_noise_stress.py)에 있다.

## I01 선검증 8--11: coherent reference boundary and finite-shot confirmation

한쪽 $R_x$ 오류와 label-dependent readout 오류를 분리한 뒤, 준비 뒤 $R_x(\epsilon)$와 과학 위상 뒤 $R_x(-\epsilon)$를 짝으로 둔 coherent intervention을 시험했다. 이 짝 연산만이 이전에 정확한 영이던 $P=0$ pooled constraint를 누설시켰고, exact Aer scan은 구조적 경계 $epsilon=0.2$ rad 및 이 검정의 검출 경계 $epsilon=1.0$ rad를 사전 고정했다. 이어 workers 6, label당 131072 shots, 10000 bounded weak-null bootstraps, 4-test Bonferroni 기준 0.0125로 총 22,020,096 preparations를 실행했다. $epsilon=1.0$ constraint는 $p=0.00180$로 검출됐고, 균일 표본 음성 대조군은 $p=0.72073$, $epsilon=0.2$ below-power 조건은 $p=0.87751$이었다. 따라서 현 토이 회로에서는 ``두 시점의 coherent reference access가 constraint leakage를 관측 가능하게 만든다''는 메커니즘은 지지되지만, 자발적 차원 선택이나 일반 이론 주장은 여전히 보류한다.

[8단계 결과](notes/why_three_prevalidation8.tex), [9단계 결과](notes/why_three_prevalidation9.tex), [10단계 결과](notes/why_three_prevalidation10.tex), [11단계 결과](notes/why_three_prevalidation11.tex), [finite-shot raw counts 및 분석](results/qiskit_rank_holdout11_paired_shots.json), [구현](src/relational_atlas_i01/holdout/qiskit_rank_holdout11_paired_shots.py)을 참조한다.

## I01 선검증 12--15: support transfer selection rule

paired transverse intervention의 exact Aer 전수 atlas는 한쪽 pulse와 $Z$축 pulse가 영을 보존하고, preparation-side 및 post-science-layer의 $X/Y$ 접근이 필요한 것을 확인했다. 다만 ``같은 qubit이 필요하다''는 12단계의 잠정 해석은 13단계에서 기각됐다. 6x6 support 행렬은 same support, conjugate partner(수치적 영), 다른 momentum pair의 같은 sign bit, 반대 sign bit라는 네 class를 보였다. (n=2,3,4) 일반화에서는 conjugate-partner 영과 same-support/반대-sign leakage가 유지됐지만, 다른-pair 같은-sign leakage는 (n=2)에서 영이고 (n=3,4)에서는 비영이었다. 그러므로 현상은 exact toy circuit에서 dimension-dependent interference selection rule로만 기술하며, 물리적 3차원 선택의 증거로 승격하지 않는다.

[12단계](notes/why_three_prevalidation12.tex), [13단계](notes/why_three_prevalidation13.tex), [14단계](notes/why_three_prevalidation14.tex), [15단계](notes/why_three_prevalidation15.tex) 결과 노트와 [14단계 전수 행렬](results/qiskit_rank_holdout14_transfer_matrix.json), [15단계 차원 비교](results/qiskit_rank_holdout15_dimension_transfer.json)를 함께 보관한다.

## I01 선검증 16--18: angle law and finite-shot support selection

세 nonzero transfer class는 leading (sin^2epsilon) angle response를 보이지만 (epsilon=pi)에서 작고 재현 가능한 residual도 남는다. 그러므로 이를 단일 닫힌 항으로 과장하지 않았다. strongest (epsilon=pi/2)에서 17단계 4-test finite-shot family는 same-support와 다른-pair 반대-sign을 검출하고 conjugate-partner null을 보존했다. 같은-sign class는 예측 (Lambda=9.90)의 경계에서 (p=0.0264)로 familywise cutoff를 넘지 못했지만, 독립 doubled-budget 18단계에서 (Lambda=19.80), (p=0.00010)으로 검출됐다. 이때 17단계와 18단계는 통계적으로 합산하지 않는다.

[16단계](notes/why_three_prevalidation16.tex), [17단계](notes/why_three_prevalidation17.tex), [18단계](notes/why_three_prevalidation18.tex) 결과와 [4-class raw analysis](results/qiskit_rank_holdout17_transfer_shots.json), [same-sign independent replication](results/qiskit_rank_holdout18_same_sign_replication.json)을 참조한다.

## I01 주제 결정: 전하 섹터 경로와 코히런트 기준 접근

19--23단계에서는 매번 별도의 노트를 만들지 않고 실험 코드와 JSON 결과를 먼저 축적했다. [종합 LaTeX 원고](notes/topic_decision_2026-10-06.tex)와 [검증한 PDF](output/pdf/topic_decision_2026-10-06.pdf)에 수식, 선행연구와의 경계, 주제 결정을 모았다.

선택한 작업 주제는 **제약 위상 간섭계에서 코히런트 기준 접근 오류의 전하 섹터 경로 지문**이다. 짝을 이룬 두 $R_x$ 접근의 복귀 진폭은 직접 경로와 두 번의 비트 반전 경로 두 항으로 정확히 분해된다. 이 규칙은 16단계 각도 스캔을 최대 $1.16\times10^{-16}$ 오차로 재현하며, $\epsilon=\pi$의 작은 잔차까지 설명한다. 켤레 비트는 허용 경로가 0개, 다른 축의 같은 부호 비트는 $n=2$에서 0개, $n=3$부터 양수다. 따라서 3은 이 경로의 **최초 개방 차원**이며, 3차원만 선택된다는 뜻은 아니다. 균일 위상 표지에서는 관측 대비가 영으로 돌아가므로 비균일 표지 자원을 가정한다.

정확한 Aer 밀도행렬에서는 오류율 0.03의 준비 amplitude damping과 읽기 오류가 켤레 영을 각각 $7.21\times10^{-6}$, $1.65\times10^{-5}$로 깨지만 phase damping은 계산 정밀도에서 영을 유지했다. 준비 오류의 누출은 접근한 $q_0,q_1$이 아닌 다른 시스템 큐비트에서 왔고, 읽기 오류의 1차 누출은 거의 $q_1$의 한 비트 결과에서 왔다. 현재 유한 샷 예산에 대한 켤레 누출의 근사 noncentrality는 각각 0.0021, 0.0068에 불과하므로 이를 실용적인 오류 위치 추정으로 과장하지 않는다. 관련 원자료는 [잡음 스캔](results/qiskit_rank_holdout19_transfer_noise.json), [정확 경로와 각도 법칙](results/qiskit_rank_holdout20_sector_paths.json), [준비 오류 위치](results/qiskit_rank_holdout21_fault_locality.json), [읽기 오류 경로](results/qiskit_rank_holdout22_readout_paths.json), [개별 원격 큐비트](results/qiskit_rank_holdout23_remote_faults.json)에 있다.

## I01 본연구 24--26: 경로 지문의 인코딩 의존성

주제 결정 뒤의 본연구는 경로 지문 중 어느 부분이 cutoff·코드 밖 위상·접근 연산의 변경에도 남는지를 사전 예측으로 시험했다. 세 단계 모두 정확 계산과 독립 Aer 대조이며, 유한-shots 검출력·잡음 안정성·우주적 차원 선택은 다루지 않았다. 경로 수는 (같은 접근, 켤레, 다른 축·같은 부호, 다른 축·반대 부호) 순서다.

| 단계 | 개입 | 핵심 결과 |
|---|---|---|
| 24 | 축당 cutoff 2, one-hot 부호 비트 | $n=2,3,4$ 경로 수 $(5,0,0,2)$, $(19,0,2,2)$, $(85,0,8,10)$. 같은 부호 첫 개방 $n=3$은 유지되고 $n=4$ 값은 4에서 8로 변함. 168개 복귀확률의 희소 진폭식–Aer 최대차 $4.77\times10^{-18}$ |
| 25 | 펄스 사이에만 방문하는 코드 밖 multi-hot 위상만 변경 | cutoff 1은 완전히 불변. cutoff 2에서는 같은 비트 클래스만 최대 $3.42\times10^{-4}$ 변하고 켤레·다른 축 클래스는 정확히 불변. 끝점 경로 수는 허용 조건이지 신호 크기 결정식이 아님 |
| 26 | 코드 보존 접근(유효 코드의 $0\leftrightarrow\pm1$만 회전) | 경로 수 $(2,0,0,2)$, $(9,0,2,2)$, $(37,0,8,10)$. 다른 축·같은 부호 첫 개방 $n=3$ 유지, 코드 밖 위상 민감도 0. 진폭식이 $\langle r\vert D_bUD_a\vert w\rangle+s^2\langle r\vert H_bUH_a\vert w\rangle$로 바뀜. 사전 판정 전부 통과 |

문서: [24단계 사전 예측](notes/research_protocol24_charge_graph_cutoff2_2026-10-06.tex)([PDF](output/pdf/research_protocol24_charge_graph_cutoff2_2026-10-06.pdf)), [25단계 사전 예측](notes/research_protocol25_invalid_phase_extension_2026-10-06.tex)([PDF](output/pdf/research_protocol25_invalid_phase_extension_2026-10-06.pdf)), [24--25 결과·증명](notes/research_results24_25_charge_graph_2026-10-06.tex)([PDF](output/pdf/research_results24_25_charge_graph_2026-10-06.pdf)), [26단계 사전 예측](notes/research_protocol26_physical_access_2026-10-15.md), [26단계 결과](notes/research_result26_physical_access_2026-10-15.md). 원자료: [24단계](results/qiskit_rank_holdout24_charge_graph.json), [25단계](results/qiskit_rank_holdout25_invalid_extension.json), [26단계](results/qiskit_rank_holdout26_physical_access.json).

## I01 문헌 검토와 본연구 27a (2026-11-13)

[문헌 검토](notes/literature_review_2026-11-13.md)에서 섹터 밖 경유 전이는 게이지 위반의 2차 유효 과정과 코히런트 게이지 드리프트의 디지털 사례이고, 균일 표지 평균은 대칭 보호 문헌의 Z_L twirl임을 확인했다. 이에 따라 기여 범위를 정확한 두 항 공식, 경로 클래스별 구조적 영, 사전등록 유한 샷 검정으로 좁혔다.

이어서 [27a 사전 프로토콜](notes/research_protocol27a_exact_design_2026-11-13.md)을 코드보다 먼저 커밋하고 [샷 없는 검증](notes/research_result27a_exact_design_2026-11-13.md)을 수행했으며, 사전 판정 17개가 모두 통과했다. 결과는 네 가지다.
- 선검증 1의 짧은 표지 실패는 Z_L 에일리어싱이며, cos τ 인자 때문에 τ=0에서만 나타난다.
- 접근 펄스 twirl은 간섭항을 소거해 대비를 4–5자릿수 줄인다.
- 단순 반복은 켤레 영을 깨므로 증폭 수단이 아니다.
- 대각 잡음은 켤레 영을 보존한다.

[코드](src/relational_atlas_i01/holdout/qiskit_rank_holdout27a_exact_design.py), [원자료](results/qiskit_rank_holdout27a_exact_design.json).

이 결과를 반영한 [수정 원고](notes/topic_decision_revised_2026-11-13.tex)([PDF](output/pdf/topic_decision_revised_2026-11-13.pdf))가 현재 주장의 기준이다. 10-06 원고는 기록으로 남긴다.

현재 전체 회귀 테스트는 `scripts/test.py`에서 163개 통과한다(2026-11-13).
