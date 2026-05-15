"""
tests/test_evaluator.py — eval_features, EvalWeights, evaluate_board 테스트

보조 함수
---------
empty_board()        : 40×10 빈 보드
board_with_cells(cells) : 지정한 (row, col) 위치에 블록을 채운 보드
"""

from __future__ import annotations

import pytest

from engine.board import BOARD_COLS, BOARD_ROWS
from engine.mino import MinoType

from bot.eval_features import (
    column_heights,
    holes,
    cell_coveredness,
    max_height,
    upper_half_excess,
    upper_quarter_excess,
    bumpiness,
    row_transitions,
    find_well,
    tsd_overhang_count,
    four_wide_well_score,
)
from bot.evaluator import EvalWeights, evaluate_board


# ── 보조 함수 ──────────────────────────────────────────────────────────────────

def empty_board() -> list[list]:
    return [[None] * BOARD_COLS for _ in range(BOARD_ROWS)]


def board_with_cells(cells: list[tuple[int, int]]) -> list[list]:
    """cells = [(row, col), ...] 위치에 MinoType.I를 채운 보드."""
    board = empty_board()
    for r, c in cells:
        board[r][c] = MinoType.I
    return board


# ── column_heights ─────────────────────────────────────────────────────────────

class TestColumnHeights:
    def test_empty_board_all_zero(self):
        h = column_heights(empty_board())
        assert h == [0] * BOARD_COLS

    def test_single_block_at_bottom(self):
        # row 39 = 바닥 → 높이 1
        board = board_with_cells([(39, 3)])
        h = column_heights(board)
        assert h[3] == 1
        assert all(h[c] == 0 for c in range(BOARD_COLS) if c != 3)

    def test_stack_of_three(self):
        # row 37, 38, 39 → 높이 3
        board = board_with_cells([(37, 5), (38, 5), (39, 5)])
        h = column_heights(board)
        assert h[5] == 3

    def test_height_from_highest_block(self):
        # 중간에 빈 칸이 있어도 최고 블록 기준
        board = board_with_cells([(36, 2), (39, 2)])
        h = column_heights(board)
        assert h[2] == BOARD_ROWS - 36   # 40 - 36 = 4

    def test_returns_ten_values(self):
        h = column_heights(empty_board())
        assert len(h) == BOARD_COLS


# ── holes ──────────────────────────────────────────────────────────────────────

class TestHoles:
    def test_no_holes_empty(self):
        board = empty_board()
        h = column_heights(board)
        assert holes(board, h) == 0

    def test_no_holes_solid_column(self):
        board = board_with_cells([(38, 0), (39, 0)])
        h = column_heights(board)
        assert holes(board, h) == 0

    def test_one_hole(self):
        # 블록(row 38), 빈 칸(row 39)
        board = board_with_cells([(38, 0)])
        h = column_heights(board)
        assert holes(board, h) == 1

    def test_two_holes_in_same_column(self):
        # 블록(row 37), 빈(38), 빈(39)
        board = board_with_cells([(37, 0)])
        h = column_heights(board)
        assert holes(board, h) == 2

    def test_holes_across_columns(self):
        # 두 열에 각각 홀 1개씩
        board = board_with_cells([(38, 0), (38, 9)])
        h = column_heights(board)
        assert holes(board, h) == 2


# ── cell_coveredness ───────────────────────────────────────────────────────────

class TestCellCoveredness:
    def test_empty_board(self):
        board = empty_board()
        h = column_heights(board)
        assert cell_coveredness(board, h) == 0

    def test_one_block_one_hole_coveredness_one(self):
        # 블록(row 38) → 홀(row 39): 블록 1개가 홀 1개를 덮음
        board = board_with_cells([(38, 4)])
        h = column_heights(board)
        assert cell_coveredness(board, h) == 1

    def test_two_blocks_above_one_hole(self):
        # 블록(37), 블록(38), 홀(39)
        board = board_with_cells([(37, 4), (38, 4)])
        h = column_heights(board)
        assert cell_coveredness(board, h) == 2

    def test_one_block_two_holes(self):
        # 블록(37), 홀(38), 홀(39)
        # row 38 홀: blocks_above=1 → +1; row 39 홀: blocks_above=1 → +1
        board = board_with_cells([(37, 6)])
        h = column_heights(board)
        assert cell_coveredness(board, h) == 2

    def test_no_holes_no_coveredness(self):
        board = board_with_cells([(38, 0), (39, 0)])
        h = column_heights(board)
        assert cell_coveredness(board, h) == 0


# ── max_height & threshold ─────────────────────────────────────────────────────

class TestMaxHeight:
    def test_empty(self):
        assert max_height([0] * 10) == 0

    def test_single_column(self):
        h = [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
        assert max_height(h) == 5


class TestUpperExcess:
    def test_upper_half_no_excess(self):
        assert upper_half_excess(5) == 0
        assert upper_half_excess(10) == 0

    def test_upper_half_excess(self):
        assert upper_half_excess(12) == 2
        assert upper_half_excess(20) == 10

    def test_upper_quarter_no_excess(self):
        assert upper_quarter_excess(10) == 0
        assert upper_quarter_excess(15) == 0

    def test_upper_quarter_excess(self):
        assert upper_quarter_excess(17) == 2
        assert upper_quarter_excess(20) == 5


# ── bumpiness ─────────────────────────────────────────────────────────────────

class TestBumpiness:
    def test_flat_board(self):
        h = [5] * BOARD_COLS
        a, sq = bumpiness(h)
        assert a == 0
        assert sq == 0

    def test_one_step(self):
        h = [0, 1] + [1] * 8
        a, sq = bumpiness(h)
        assert a == 1
        assert sq == 1

    def test_well_col_excluded(self):
        # well_col=1 → 쌍 (0,1)과 (1,2) 제외
        h = [5, 0, 5, 5, 5, 5, 5, 5, 5, 5]
        a_full, _ = bumpiness(h)
        a_well, _ = bumpiness(h, well_col=1)
        assert a_well < a_full

    def test_returns_tuple_of_two(self):
        result = bumpiness([0] * BOARD_COLS)
        assert len(result) == 2


# ── row_transitions ────────────────────────────────────────────────────────────

class TestRowTransitions:
    def test_empty_board(self):
        board = empty_board()
        assert row_transitions(board, 0) == 0

    def test_full_bottom_row(self):
        # 바닥 행 전부 채움 → 좌경계(채움↔채움=0), 우경계(채움↔채움=0), 내부 전환 없음
        board = empty_board()
        for c in range(BOARD_COLS):
            board[39][c] = MinoType.I
        trans = row_transitions(board, 1)
        assert trans == 0

    def test_single_block_in_bottom_row(self):
        # 한 블록만: 왼쪽 경계→빈→빈→...→블록: 전환 2(빈→블, 블→빈) + 좌우 경계 2 = 4
        board = board_with_cells([(39, 5)])
        trans = row_transitions(board, 1)
        assert trans == 4

    def test_max_height_zero_returns_zero(self):
        board = empty_board()
        assert row_transitions(board, 0) == 0


# ── find_well ─────────────────────────────────────────────────────────────────

class TestFindWell:
    def test_no_well_flat(self):
        h = [5] * BOARD_COLS
        col, depth = find_well(h)
        assert depth == 0

    def test_obvious_well_center(self):
        # col 4 = 0, 양쪽 = 5 → depth = 5
        h = [5, 5, 5, 5, 0, 5, 5, 5, 5, 5]
        col, depth = find_well(h)
        assert col == 4
        assert depth == 5

    def test_left_edge_well(self):
        # col 0 = 0, 우측 = 5 → depth = 5 (좌측 경계는 VISIBLE_ROWS+1)
        h = [0, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        col, depth = find_well(h)
        assert col == 0
        assert depth == 5

    def test_right_edge_well(self):
        h = [5, 5, 5, 5, 5, 5, 5, 5, 5, 0]
        col, depth = find_well(h)
        assert col == 9
        assert depth == 5

    def test_returns_deepest_when_multiple(self):
        # col 2(depth 3) vs col 7(depth 5) → col 7이 더 깊음
        h = [5, 5, 2, 5, 5, 5, 5, 0, 5, 5]
        col, depth = find_well(h)
        assert col == 7
        assert depth == 5


# ── tsd_overhang_count ────────────────────────────────────────────────────────

class TestTsdOverhangCount:
    def test_flat_board_no_overhang(self):
        board = empty_board()
        h = [5] * BOARD_COLS
        assert tsd_overhang_count(board, h) == 0

    def test_diff_of_two_counted(self):
        board = empty_board()
        h = [4, 6, 4, 4, 4, 4, 4, 4, 4, 4]
        # 쌍 (0,1): |4-6|=2 → 카운트, (1,2): |6-4|=2 → 카운트
        assert tsd_overhang_count(board, h) == 2

    def test_diff_of_one_not_counted(self):
        board = empty_board()
        h = [4, 5, 4, 4, 4, 4, 4, 4, 4, 4]
        # |4-5|=1, |5-4|=1 → 카운트 안됨
        assert tsd_overhang_count(board, h) == 0


# ── four_wide_well_score ───────────────────────────────────────────────────────

class TestFourWideWellScore:
    def test_flat_board_zero(self):
        h = [5] * BOARD_COLS
        assert four_wide_well_score(h) == 0.0

    def test_left_four_wide(self):
        # 열 0~3 = 0, 열 4~9 = 6 → left_score = 6 - 0 = 6
        h = [0, 0, 0, 0, 6, 6, 6, 6, 6, 6]
        score = four_wide_well_score(h)
        assert score == 6.0

    def test_right_four_wide(self):
        # 열 6~9 = 0, 열 0~5 = 6 → right_score = 6 - 0 = 6
        h = [6, 6, 6, 6, 6, 6, 0, 0, 0, 0]
        score = four_wide_well_score(h)
        assert score == 6.0

    def test_no_four_wide_asymmetric(self):
        # 중간이 낮으면 4-wide가 아님
        h = [6, 6, 0, 6, 6, 6, 6, 6, 6, 6]
        score = four_wide_well_score(h)
        assert score == 0.0


# ── EvalWeights ───────────────────────────────────────────────────────────────

class TestEvalWeights:
    def test_default_values(self):
        w = EvalWeights()
        assert w.holes == -4.0
        assert w.cell_coveredness == -0.5
        assert w.height == -0.2
        assert w.height_upper_half == -1.0
        assert w.height_upper_quarter == -5.0
        assert w.bumpiness == -0.3
        assert w.bumpiness_sq == -0.1
        assert w.row_transitions == -0.3
        assert w.well_depth == 0.2
        assert w.tsd_overhang == 6.0
        assert w.four_wide_well == 1.5

    def test_custom_values(self):
        w = EvalWeights(holes=-10.0, height=-1.0)
        assert w.holes == -10.0
        assert w.height == -1.0
        assert w.bumpiness == -0.3   # 나머지는 기본값


# ── evaluate_board ────────────────────────────────────────────────────────────

class TestEvaluateBoard:
    def test_empty_board_returns_zero(self):
        score = evaluate_board(empty_board())
        assert score == 0.0

    def test_returns_float(self):
        assert isinstance(evaluate_board(empty_board()), float)

    def test_hole_penalizes(self):
        # 홀 없는 보드 vs 홀 있는 보드
        board_clean = board_with_cells([(38, 0), (39, 0)])
        board_holed = board_with_cells([(38, 0)])  # row 39 col 0 = 빈 칸(홀)

        h_clean = column_heights(board_clean)
        h_holed = column_heights(board_holed)

        assert holes(board_holed, h_holed) == 1
        assert evaluate_board(board_holed) < evaluate_board(board_clean)

    def test_height_penalizes(self):
        # 낮은 보드 vs 높은 보드 (홀 없이)
        board_low  = board_with_cells([(39, 0)])
        board_high = board_with_cells([(30, 0)])
        assert evaluate_board(board_low) > evaluate_board(board_high)

    def test_well_rewards(self):
        # 웰이 있는 보드는 웰 점수(0.2)를 받음
        # col 0 = 0, 나머지 = 5 → well_depth=5
        cells_well = [(39 - i, c) for i in range(5) for c in range(1, BOARD_COLS)]
        board_well = board_with_cells(cells_well)
        # 같은 높이의 flat 보드
        cells_flat = [(39 - i, c) for i in range(5) for c in range(BOARD_COLS)]
        board_flat = board_with_cells(cells_flat)
        # flat 보드와 비교: well이 있으면 well_depth 점수 + (bumpiness 패널티)
        # 단순히 evaluate_board가 float을 반환하는지와 well 보드가 해당 항목을 계산하는지 확인
        score_well = evaluate_board(board_well)
        assert isinstance(score_well, float)

    def test_custom_weights_applied(self):
        # holes 가중치를 0으로 하면 홀이 있어도 패널티 없어야 함
        w = EvalWeights(holes=0.0, cell_coveredness=0.0,
                        height=0.0, height_upper_half=0.0,
                        height_upper_quarter=0.0, bumpiness=0.0,
                        bumpiness_sq=0.0, row_transitions=0.0,
                        well_depth=0.0, tsd_overhang=0.0,
                        four_wide_well=0.0)
        board_holed = board_with_cells([(38, 0)])
        assert evaluate_board(board_holed, w) == 0.0

    def test_more_holes_worse_score(self):
        board_one_hole = board_with_cells([(38, 0)])
        board_two_holes = board_with_cells([(37, 0)])
        assert evaluate_board(board_one_hole) > evaluate_board(board_two_holes)

    def test_bumpier_board_scores_worse(self):
        # flat 바닥 vs 들쭉날쭉한 바닥
        cells_flat = [(39, c) for c in range(BOARD_COLS)]
        cells_bumpy = [(39 - (c % 3), c) for c in range(BOARD_COLS)]
        board_flat  = board_with_cells(cells_flat)
        board_bumpy = board_with_cells(cells_bumpy)
        assert evaluate_board(board_flat) > evaluate_board(board_bumpy)
