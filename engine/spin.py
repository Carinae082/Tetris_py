"""
engine/spin.py — 스핀 판정 (All-Mini+ 기준, TETR.IO Season 2)

All-Mini+ 규칙 요약
--------------------
- 공통: 마지막 유효 동작이 회전이어야 판정 대상
- T 이외 미노: lock 직전 위로 1칸 이동 불가(immobile) → Mini Spin
- T 미노:
    1. 대각 4 코너 중 3칸 이상 차 있으면 T-spin 후보
       a. front 2칸 모두 차 있고 back 1칸 이상 차 있으면 → Full T-spin
          (단, 90° 5번째 킥 사용 시 mini→full 승격으로도 Full이 될 수 있음)
       b. 그 외 → Mini T-spin
    2. 3-corner 미충족이지만 immobile(위 1칸 이동 불가) → Mini Spin (All-Mini+ 추가)

상태 표기: 0=North, 1=East, 2=South, 3=West
"""

from __future__ import annotations

from .board import BOARD_ROWS, BOARD_COLS
from .mino import ActivePiece, MinoType
from .state import GameState
from .types import SpinType

# ── T 피스 front/back 코너 인덱스 ─────────────────────────────────────────────
#
# 코너 좌표 (piece.row + dr, piece.col + dc):
#   index 0 = TL  (dr=0, dc=0)
#   index 1 = TR  (dr=0, dc=2)
#   index 2 = BL  (dr=2, dc=0)
#   index 3 = BR  (dr=2, dc=2)
#
# T 피스 중심: (piece.row+1, piece.col+1)
# bump 방향(회전 상태)에 따른 front/back 코너 인덱스:

_CORNER_OFFSETS: list[tuple[int, int]] = [
    (0, 0),   # 0: TL
    (0, 2),   # 1: TR
    (2, 0),   # 2: BL
    (2, 2),   # 3: BR
]

_T_FRONT_BACK: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
    0: ((0, 1), (2, 3)),   # North (bump↑):  front=TL,TR  back=BL,BR
    1: ((1, 3), (0, 2)),   # East  (bump→):  front=TR,BR  back=TL,BL
    2: ((2, 3), (0, 1)),   # South (bump↓):  front=BL,BR  back=TL,TR
    3: ((0, 2), (1, 3)),   # West  (bump←):  front=TL,BL  back=TR,BR
}


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _cell_filled(board, row: int, col: int) -> bool:
    """경계 밖 또는 블록이 있으면 True (벽·바닥·스택 모두 '차 있음')."""
    if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
        return True
    return board.get(row, col) is not None


def _blocked_up(board, piece: ActivePiece) -> bool:
    """위로 1칸 이동 불가(immobile) 이면 True."""
    above = ActivePiece(piece.type, piece.row - 1, piece.col, piece.rotation)
    return not board.is_valid_position(above)


# ── 공개 API ──────────────────────────────────────────────────────────────────

def detect_spin(state: GameState) -> SpinType:
    """
    lock 직전에 호출한다.

    전제 조건 미충족(active 없음, 마지막 동작이 회전 아님)이면 NONE을 반환.
    """
    if state.active is None or not state.last_was_rotation:
        return SpinType.NONE

    piece = state.active

    if piece.type == MinoType.T:
        return _detect_t_spin(state)
    else:
        return _detect_non_t_spin(state)


# ── 비T 미노 ──────────────────────────────────────────────────────────────────

def _detect_non_t_spin(state: GameState) -> SpinType:
    """All-Mini+: immobile(위 1칸 불가) 이면 Mini Spin."""
    if _blocked_up(state.board, state.active):
        return SpinType.MINI
    return SpinType.NONE


# ── T 미노 ────────────────────────────────────────────────────────────────────

def _detect_t_spin(state: GameState) -> SpinType:
    piece = state.active
    board = state.board

    # T 피스 bounding box 좌상단: (piece.row, piece.col)
    pr, pc = piece.row, piece.col

    # 4 코너 채워짐 여부
    filled = [_cell_filled(board, pr + dr, pc + dc) for dr, dc in _CORNER_OFFSETS]
    filled_count = sum(filled)

    # ── 3-corner 규칙 ────────────────────────────────────────────────────────
    if filled_count >= 3:
        front_idx, back_idx = _T_FRONT_BACK[piece.rotation]
        front_filled = filled[front_idx[0]] and filled[front_idx[1]]
        back_filled = sum(filled[i] for i in back_idx)

        if front_filled and back_filled >= 1:
            # front 2칸 + back 1칸 이상 → Full T-spin
            return SpinType.FULL
        else:
            # Mini 이지만 kick upgrade 조건 충족 시 Full로 승격
            if state.last_kick_upgrades:
                return SpinType.FULL
            return SpinType.MINI

    # ── All-Mini+ immobile fallback (3-corner 미충족 시) ─────────────────────
    if _blocked_up(board, piece):
        return SpinType.MINI

    return SpinType.NONE
