"""
bot/eval_features.py — 보드 형태 feature 추출 함수 모음

설계 원칙
----------
* 게임 규칙을 구현하지 않는다. 순수하게 보드 형태를 측정한다.
* 입력은 StateSnapshot.board 형식의 40×10 리스트
  (각 셀 = None(빈 칸) 또는 MinoType(고정 블록/가비지)).
* 환경 원본 객체에 접근하지 않으며, 읽기 전용으로 board만 사용한다.
* 각 feature는 독립된 함수로 분리한다.

좌표 규약
----------
  row 0   = 최상단 버퍼 (숨김)
  row 39  = 최하단 바닥
  높이(height) = 바닥(row 39)에서 가장 높은 블록까지의 셀 수
                 (높이 1 = row 39에만 블록 있음)
"""

from __future__ import annotations

from engine.board import BOARD_COLS, BOARD_ROWS, VISIBLE_ROW_START

# 가시 영역 행 수 (row 20~39 = 20행)
_VISIBLE_ROWS: int = BOARD_ROWS - VISIBLE_ROW_START   # 20

# 보드 높이 초과 패널티 임계값
_UPPER_HALF_THRESHOLD: int = _VISIBLE_ROWS // 2        # 10
_UPPER_QUARTER_THRESHOLD: int = _VISIBLE_ROWS * 3 // 4 # 15


# ── 열 높이 ───────────────────────────────────────────────────────────────────

def column_heights(board: list[list]) -> list[int]:
    """
    각 열의 높이를 반환한다 (10개 정수 리스트).
    높이 = 바닥부터 가장 높은 블록까지의 셀 수.
    빈 열은 0.
    """
    heights: list[int] = []
    for col in range(BOARD_COLS):
        h = 0
        for row in range(BOARD_ROWS):   # 위(row 0)에서 아래(row 39)로
            if board[row][col] is not None:
                h = BOARD_ROWS - row    # 해당 row에서 바닥까지 거리
                break
        heights.append(h)
    return heights


# ── 홀(구멍) ──────────────────────────────────────────────────────────────────

def holes(board: list[list], heights: list[int]) -> int:
    """
    홀 수: 각 열에서 최고 블록보다 아래에 있는 빈 셀의 합.
    """
    count = 0
    for col in range(BOARD_COLS):
        if heights[col] == 0:
            continue
        top_row = BOARD_ROWS - heights[col]
        for row in range(top_row + 1, BOARD_ROWS):
            if board[row][col] is None:
                count += 1
    return count


# ── 셀 커버리지 ───────────────────────────────────────────────────────────────

def cell_coveredness(board: list[list], heights: list[int]) -> int:
    """
    각 홀에 대해 그 위에 쌓인 블록 수의 합(커버리지).
    홀이 깊게 묻혀 있을수록 값이 크다.
    """
    total = 0
    for col in range(BOARD_COLS):
        if heights[col] == 0:
            continue
        top_row = BOARD_ROWS - heights[col]
        blocks_above = 0
        for row in range(top_row, BOARD_ROWS):
            if board[row][col] is not None:
                blocks_above += 1
            else:
                # 이 셀은 홀: 그 위의 블록 수(blocks_above)만큼 커버리지 누적
                total += blocks_above
    return total


# ── 높이 지표 ─────────────────────────────────────────────────────────────────

def max_height(heights: list[int]) -> int:
    """열 높이 목록에서 최대값을 반환한다."""
    return max(heights) if heights else 0


def upper_half_excess(max_h: int) -> int:
    """
    최대 높이가 가시 영역 절반(10)을 초과하는 양.
    초과하지 않으면 0.
    """
    return max(0, max_h - _UPPER_HALF_THRESHOLD)


def upper_quarter_excess(max_h: int) -> int:
    """
    최대 높이가 가시 영역 상위 1/4 경계(15)를 초과하는 양.
    초과하지 않으면 0.
    """
    return max(0, max_h - _UPPER_QUARTER_THRESHOLD)


# ── 울퉁불퉁함(bumpiness) ─────────────────────────────────────────────────────

def bumpiness(
    heights: list[int],
    well_col: int | None = None,
) -> tuple[int, int]:
    """
    인접한 열 높이 차이의 절댓값 합과 제곱합을 반환한다.

    well_col이 지정되면 해당 열이 포함된 인접 쌍은 계산에서 제외한다.
    (웰 열 양쪽의 차이는 웰 형태에서 자연스러운 것이므로 패널티를 줄임)

    반환: (bumpiness_abs, bumpiness_sq)
    """
    abs_sum = 0
    sq_sum = 0
    for i in range(BOARD_COLS - 1):
        if well_col is not None and (i == well_col or i + 1 == well_col):
            continue
        diff = abs(heights[i] - heights[i + 1])
        abs_sum += diff
        sq_sum += diff * diff
    return abs_sum, sq_sum


# ── 행 전환 ───────────────────────────────────────────────────────────────────

def row_transitions(board: list[list], max_h: int) -> int:
    """
    각 행에서 빈 셀↔채워진 셀 전환 횟수의 합.
    보드 좌우 경계(외부)는 채워진 것으로 간주한다.
    max_h에 해당하는 행부터 바닥(row 39)까지만 계산한다.
    """
    if max_h == 0:
        return 0

    transitions = 0
    start_row = BOARD_ROWS - max_h

    for row in range(start_row, BOARD_ROWS):
        # 왼쪽 경계(외부=채워짐) ↔ 첫 번째 셀
        if board[row][0] is None:
            transitions += 1
        # 인접 셀 쌍
        for col in range(BOARD_COLS - 1):
            a = board[row][col] is not None
            b = board[row][col + 1] is not None
            if a != b:
                transitions += 1
        # 마지막 셀 ↔ 오른쪽 경계(외부=채워짐)
        if board[row][BOARD_COLS - 1] is None:
            transitions += 1

    return transitions


# ── 웰(well) 탐지 ─────────────────────────────────────────────────────────────

def find_well(heights: list[int]) -> tuple[int, int]:
    """
    가장 깊은 웰(well)의 열 인덱스와 깊이를 반환한다.

    웰 깊이 = min(왼쪽 이웃 높이, 오른쪽 이웃 높이) − 해당 열 높이.
    경계 열은 이웃이 없는 방향을 매우 높은 값(_VISIBLE_ROWS + 1)으로 처리한다.
    웰이 없으면 (−1, 0)을 반환한다.
    """
    _INF = _VISIBLE_ROWS + 1
    best_col = -1
    best_depth = 0

    for col in range(BOARD_COLS):
        left_h  = heights[col - 1] if col > 0              else _INF
        right_h = heights[col + 1] if col < BOARD_COLS - 1 else _INF
        depth   = min(left_h, right_h) - heights[col]
        if depth > best_depth:
            best_depth = depth
            best_col   = col

    return best_col, best_depth


# ── TSD 오버행 ────────────────────────────────────────────────────────────────

def tsd_overhang_count(board: list[list], heights: list[int]) -> int:
    """
    TSD(T-Spin Double) 형성을 위한 오버행 위치 수. 최대 2를 반환한다.

    각 열 c에 대해 다음 조건을 모두 만족할 때 오버행으로 인정한다:
      1. 오버행 존재: 최상단 셀이 채워져 있고, 바로 아래 셀이 비어 있다.
      2. 벽 존재: 왼쪽 또는 오른쪽 인접 열의 높이가 h 이상이고,
                  그 열의 최상단 2칸(h-1, h-2 위치)이 모두 채워져 있다.
      3. 반대편 개방: 벽이 있는 방향의 반대쪽 h-2 위치가 비어 있어
                      T-피스 진입 공간이 확보되어 있다.
    """
    count = 0

    for c in range(BOARD_COLS):
        h = heights[c]
        if h < 2:
            continue

        top_row = BOARD_ROWS - h  # 최상단 블록의 행 인덱스

        # 조건 1: 오버행 — 최상단 채워짐, 바로 아래 비어 있음
        has_overhang = (
            board[top_row][c] is not None and
            board[top_row + 1][c] is None
        )
        if not has_overhang:
            continue

        # 조건 2: 왼쪽 벽 — 왼쪽 열이 높이 h 이상이고 위 2칸 채워짐
        wall_left = (
            c > 0 and
            heights[c - 1] >= h and
            board[top_row][c - 1] is not None and
            board[top_row + 1][c - 1] is not None
        )

        # 조건 2: 오른쪽 벽 — 오른쪽 열이 높이 h 이상이고 위 2칸 채워짐
        wall_right = (
            c < BOARD_COLS - 1 and
            heights[c + 1] >= h and
            board[top_row][c + 1] is not None and
            board[top_row + 1][c + 1] is not None
        )

        # 조건 3: 벽 반대편이 열려 있어야 T-피스 진입 가능
        if wall_left:
            open_right = (
                (c < BOARD_COLS - 1 and board[top_row + 1][c + 1] is None)
                or c == BOARD_COLS - 1
            )
            if open_right:
                count += 1

        if wall_right:
            open_left = (
                (c > 0 and board[top_row + 1][c - 1] is None)
                or c == 0
            )
            if open_left:
                count += 1

    return min(count, 2)


# ── 4-wide 웰 ─────────────────────────────────────────────────────────────────

def four_wide_well_score(heights: list[int]) -> float:
    """
    4-wide 웰 형태(좌측 또는 우측 4열이 나머지보다 낮게 유지된 구조) 점수.
    fusion/eval.rs four_wide_well_score 포팅.

    좌측 4-wide: 열 0~3 평균 < 열 4~9 평균이고 차이 >= MIN_DEPTH_DIFF(3.0)
    우측 4-wide: 열 6~9 평균 < 열 0~5 평균이고 차이 >= MIN_DEPTH_DIFF(3.0)
    임계값을 초과하는 차이만큼 점수를 부여하며, 형성되지 않았으면 0.
    """
    _MIN_DEPTH_DIFF = 3.0

    left_well_avg  = sum(heights[:4]) / 4.0
    left_rest_avg  = sum(heights[4:]) / 6.0
    right_well_avg = sum(heights[6:]) / 4.0
    right_rest_avg = sum(heights[:6]) / 6.0

    score = 0.0
    left_diff  = left_rest_avg  - left_well_avg
    right_diff = right_rest_avg - right_well_avg
    if left_diff >= _MIN_DEPTH_DIFF:
        score = max(score, left_diff - _MIN_DEPTH_DIFF + 1.0)
    if right_diff >= _MIN_DEPTH_DIFF:
        score = max(score, right_diff - _MIN_DEPTH_DIFF + 1.0)
    return score
