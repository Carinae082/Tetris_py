"""
bot/evaluator.py — 보드 형태 평가기

EvalWeights: 평가 가중치 dataclass (기본값 = fusion 스타일 초기값)
evaluate_board: feature 가중합으로 보드 점수를 계산한다.

설계 원칙
----------
* 공격, 콤보, B2B 가치를 점수에 포함하지 않는다 (순수 board-shape 평가).
* 빈 보드는 0.0을 반환한다.
* StateSnapshot.board (40×10 리스트)를 직접 입력으로 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .eval_features import (
    column_heights,
    holes          as _holes,
    cell_coveredness as _coveredness,
    max_height     as _max_height,
    upper_half_excess,
    upper_quarter_excess,
    bumpiness      as _bumpiness,
    row_transitions as _row_transitions,
    find_well,
    tsd_overhang_count,
    four_wide_well_score,
)


# ── 가중치 ────────────────────────────────────────────────────────────────────

@dataclass
class EvalWeights:
    """
    evaluate_board에 사용하는 feature 가중치.
    양수 = 가산점, 음수 = 패널티.
    """
    holes:               float = -4.0
    cell_coveredness:    float = -0.5
    height:              float = -0.2
    height_upper_half:   float = -1.0
    height_upper_quarter:float = -5.0
    bumpiness:           float = -0.3
    bumpiness_sq:        float = -0.1
    row_transitions:     float = -0.3
    well_depth:          float =  0.2
    tsd_overhang:        float =  6.0
    four_wide_well:      float =  1.5


_DEFAULT_WEIGHTS = EvalWeights()


# ── 평가 함수 ─────────────────────────────────────────────────────────────────

def evaluate_board(
    board: list[list],
    weights: EvalWeights | None = None,
    heights: list[int] | None = None,
) -> float:
    """
    보드 형태를 평가해 점수(float)를 반환한다.

    score =
        holes            * w.holes
      + coveredness       * w.cell_coveredness
      + max_height        * w.height
      + upper_half_excess * w.height_upper_half
      + upper_qtr_excess  * w.height_upper_quarter
      + bumpiness_abs     * w.bumpiness
      + bumpiness_sq      * w.bumpiness_sq
      + row_transitions   * w.row_transitions
      + well_depth        * w.well_depth
      + tsd_overhang      * w.tsd_overhang
      + four_wide_well    * w.four_wide_well

    빈 보드는 0.0을 반환한다.

    매개변수
    --------
    board   : StateSnapshot.board 형식 (40×10, None 또는 MinoType)
    weights : EvalWeights 인스턴스 (None 이면 기본값 사용)
    heights : 미리 계산된 열 높이 목록 (None 이면 내부에서 계산)
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS

    if heights is None:
        heights = column_heights(board)
    max_h   = _max_height(heights)

    if max_h == 0:
        return 0.0

    well_col, well_depth = find_well(heights)
    well_col_arg = well_col if well_col >= 0 else None

    f_holes       = _holes(board, heights)
    f_covered     = _coveredness(board, heights)
    f_bump_a, f_bump_sq = _bumpiness(heights, well_col_arg)
    f_upper_half  = upper_half_excess(max_h)
    f_upper_qtr   = upper_quarter_excess(max_h)
    f_row_trans   = _row_transitions(board, max_h)
    f_tsd         = tsd_overhang_count(board, heights)
    f_fw          = four_wide_well_score(heights)

    score = (
        f_holes      * weights.holes
        + f_covered  * weights.cell_coveredness
        + max_h      * weights.height
        + f_upper_half * weights.height_upper_half
        + f_upper_qtr  * weights.height_upper_quarter
        + f_bump_a   * weights.bumpiness
        + f_bump_sq  * weights.bumpiness_sq
        + f_row_trans * weights.row_transitions
        + well_depth * weights.well_depth
        + f_tsd      * weights.tsd_overhang
        + f_fw       * weights.four_wide_well
    )

    return score
