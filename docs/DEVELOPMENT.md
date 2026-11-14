# I01 코드 구조와 실행

실험의 과학적 가설·사전 프로토콜은 `notes/`, 저장된 원자료와 그림은 `results/`에 둔다. 이 리팩터링은 실험의 파라미터, 난수 seed, 판정 규칙, 결과 형식을 변경하지 않는다.

| 위치 | 역할 |
| --- | --- |
| `src/relational_atlas_i01/geometry/` | 초기 기하·활성화 토이 모형 |
| `src/relational_atlas_i01/phase_reference/` | 표지·위상 기준·시계 개입 |
| `src/relational_atlas_i01/holdout/` | 사전 지정 rank 선검증과 통계 감사 |
| `src/relational_atlas_i01/holdout/_clock_postselection.py` | 선검증 6의 재사용 가능한 순수 회로 구성 |
| `src/relational_atlas_i01/paths.py` | 현재 작업 디렉터리와 독립인 프로젝트 경로 |
| `src/relational_atlas_i01/cli.py` | 실험 이름을 모듈에 매핑하는 공통 실행기 |
| `scripts/run.py` | 실험 CLI 진입점 |
| `scripts/test.py` | 전체 회귀 테스트 진입점 |
| `tests/` | 회로·수학·저장 결과·구조 회귀 테스트 |

루트 폴더에서 PowerShell 예시:

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/run.py --list
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/run.py qiskit_rank_holdout6_clock --help
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/run.py qiskit_rank_holdout6_clock --checks-only
& 'C:\Users\user\anaconda3\Library\bin\conda.bat' run --no-capture-output -n quantum python scripts/test.py
```

다른 환경에서는 Python 3.12에서 `python -m pip install -r requirements-research.txt`와 `python -m pip install --no-deps -e .`를 실행한 뒤 `i01-experiment --list` 또는 `python -m unittest discover -s tests -p 'test_*.py'`로 실행할 수 있다. 핀 파일은 현재 `quantum` 환경에서 확인한 버전 기록이며 다른 운영체제에서의 설치까지 검증한 것은 아니다. `scripts/` 진입점은 설치 없이도 `src/`를 찾는다. 과거 `python code/<이름>.py` 호출은 `python scripts/run.py <이름>`으로 바뀌었고 실험별 인자는 그대로 전달된다.

실험 모듈에서는 계산·회로 생성·분석 함수를 import 가능한 함수로 유지하고, 파일 쓰기와 CLI 인자 해석은 `main()` 안에서 한다. 새 실험은 해당 영역에 모듈을 추가하고 `tests/`에 회귀 검사를 붙인다. `PROJECT_ROOT` 또는 `RESULTS_DIR`를 사용하여 경로를 지정한다. 저장된 원자료를 재실행하면 덮어쓸 수 있으므로, 오래 걸리거나 비용이 큰 실험은 먼저 `--help`와 각 모듈의 저비용 확인 옵션을 확인한다. 이 구조에서는 파일명 자체가 실행 이름이므로 기존 이름을 함부로 바꾸지 않는다.

CPU 병렬 실행에서는 검증된 로컬 기본값 workers 6을 유지하고 BLAS 내부 스레드를 1로 제한한다. 더 큰 서버에서는 실험별 `--workers` 인자를 명시한다. 프로토콜과 seed를 먼저 고정한 후 실험을 실행한다.

`notes/`의 기존 LaTeX 문서에서는 과학적 기록을 그대로 두고, 재현 코드와 테스트의 상대 경로만 새 위치로 갱신했다.
