# BeamSearchBot 성능 최적화 가이드

## 개요

`BeamSearchBot`은 빔 서치 기반의 다중 배치(ply) 탐색 봇이다.
이 문서는 구현된 최적화 내용과 설정 방법을 설명한다.

---

## 주요 최적화 내역

### 1. Lean Simulation (핵심 최적화)

기존 이동·회전 행동 시뮬레이션은 env를 **2회** 복제했다:

```
(구) clone1 = deepcopy(env)        # 1번째 복제
    try_move(clone1, -1)
    clone2 = deepcopy(clone1)      # 2번째 복제 (평가용)
    hard_drop(clone2)
    evaluate(clone2)
    node.env = clone1              # 이동 후 상태 저장
```

`simulate_action_lean`은 이동 + 드롭을 **1회** 복제로 처리한다:

```
(최적화) clone = deepcopy(env)     # 1번 복제
         try_move(clone, -1)
         hard_drop(clone)          # 동일 클론에 인플레이스 적용
         evaluate(clone)
         node.env = clone          # 배치 완료 후 상태 저장
```

이동·회전 행동에서 deepcopy 횟수가 **절반**으로 감소한다.

### 2. StateSnapshot 생략

`build_summary_from_env`는 `StateSnapshot`을 거치지 않고
`env.board._grid`를 직접 참조해 `OutcomeSummary`를 생성한다.

- board 40×10 리스트 복사 생략
- snapshot 객체 생성 비용 제거

### 3. SearchNode(slots=True)

`@dataclass(slots=True)` 적용으로:
- 속성 접근 속도 향상 (`__dict__` 조회 생략)
- 인스턴스 메모리 절약

### 4. path_actions 경량화

`track_path=False`(기본값)이면 전체 경로 대신 `[root_action]`만 저장한다.
깊은 탐색에서 리스트 객체 생성·복사 비용을 줄인다.

`track_path=True`로 설정하면 전체 경로를 저장해 디버깅에 사용할 수 있다.

### 5. 2단계 Pruning

각 depth 확장 후 다음 순서로 후보를 정리한다:

1. **즉시 점수(immediate_score) 필터** (O(n)): `best_immediate - futility_delta` 미만 제거
2. **누적 점수(cumulative_score) 정렬 + beam_width 절단**: 최종 빔 구성

두 단계 분리로 불필요한 정렬 비용을 줄인다.

### 6. 시간 예산 우선 탐색

- depth loop **시작 시** 시간 체크
- child expansion loop **내부**에서도 시간 체크 (중간 중단 지원)
- 최소 depth=1은 항상 완료 보장
- 중단 시 현재까지 탐색한 최선 빔 유지

### 7. _fast_clone 전략 분리

```python
def _fast_clone(self, env):
    copy_fn = getattr(env, "copy", None)
    if copy_fn is not None and callable(copy_fn):
        return copy_fn()          # GameState.copy() 구현 시 자동 활용
    return copy.deepcopy(env)     # 현재 기본값
```

향후 `GameState`가 `copy()` 메서드를 제공하면 어댑터 코드 수정 없이 성능이 개선된다.

---

## SearchConfig Preset

| Preset | beam_width | depth | time_budget_ms | root_top_k | 용도 |
|--------|-----------|-------|----------------|------------|------|
| `safe_default()` | 120 | 4 | 35 ms | None | 일반 게임플레이 |
| `fast_play()` | 80 | 3 | 20 ms | 12 | 빠른 응답 필요 시 |
| `deeper_play()` | 180 | 5 | 50 ms | 16 | 높은 정확도 우선 |

```python
from bot.beam_search_bot import BeamSearchBot
from bot.search_config import SearchConfig

# 기본 설정
bot = BeamSearchBot(config=SearchConfig.safe_default())

# 빠른 설정
bot = BeamSearchBot(config=SearchConfig.fast_play())

# 직접 설정
bot = BeamSearchBot(config=SearchConfig(
    beam_width=60,
    depth=3,
    time_budget_ms=25.0,
    track_path=True,   # 경로 추적 (디버깅)
))
```

---

## 벤치마크 실행

```bash
# 모든 preset 측정
python scripts/bench_bot.py

# 특정 preset
python scripts/bench_bot.py --config fast_play

# 더 많은 데이터로
python scripts/bench_bot.py --pieces 20 --games 5

# 상세 출력
python scripts/bench_bot.py --verbose
```

### 참고 측정값 (Windows 11, Python 3.13)

| Config | Mean ms | P95 ms | nodes/call | depth |
|--------|---------|--------|-----------|-------|
| safe_default | ~40 ms | ~44 ms | ~38 | ~2.4 |
| fast_play | ~23 ms | ~27 ms | ~21 | ~2.0 |
| deeper_play | ~53 ms | ~59 ms | ~46 | ~3.1 |
| no_budget (beam=60, d=4) | ~452 ms | ~831 ms | ~428 | 4.0 |

> 시간 예산이 없으면 depth=4 탐색에 수백 ms가 소요된다.
> 실시간 게임에서는 반드시 `time_budget_ms`를 설정하라.

---

## 탐색 결과 확인

```python
result = bot.search(env)

print(result.best_action)       # 선택된 행동
print(result.best_score)        # 최선 누적 점수
print(result.expanded_nodes)    # 탐색한 총 노드 수
print(result.searched_depth)    # 실제 탐색 깊이
print(result.root_scores)       # 루트 행동별 최고 점수
print(result.best_path)         # 행동 경로 (track_path=True일 때 유효)
```

---

## 파일 구조

```
bot/
  search_config.py      SearchConfig dataclass + preset 메서드
  search_node.py        SearchNode(slots=True), SearchResult
  adapter.py            BotEnvAdapter + simulate_action_lean + build_summary_from_env
  beam_search_bot.py    BeamSearchBot (최적화 버전)

scripts/
  bench_bot.py          성능 벤치마크 스크립트

tests/
  test_beam_search_bot.py   기능 테스트
  test_search_perf.py       성능·설정 테스트
```
