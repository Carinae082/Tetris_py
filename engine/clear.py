"""
engine/clear.py — difficult clear 판정, Combo, B2B 갱신

difficult clear 기준 (TETR.IO Season 2 / All-Mini+)
----------------------------------------------------
  Single / Double / Triple    → non-difficult
  Quad (4줄)                  → difficult
  Spin Mini Single 이상       → difficult  (non-T all-spin 포함)
  Spin Single/Double/Triple   → difficult
  All Clear                   → 별도 보너스: B2B +1 (clear 종류 무관)

B2B 상태 의미
  -1  : B2B 체인 없음
   0  : 첫 difficult clear — 체인 시작, 이번엔 B2B 보너스 없음
  ≥1  : 연속 difficult clear 중 — 이 값이 공격력 보너스 계산에 사용됨
"""

from __future__ import annotations

from dataclasses import dataclass

from .board import BOARD_ROWS, BOARD_COLS
from .state import GameState
from .types import SpinType


# ── ClearResult ────────────────────────────────────────────────────────────────

@dataclass
class ClearResult:
    lines: int            # 이번에 클리어된 줄 수
    spin: SpinType        # 스핀 타입
    difficult: bool       # difficult clear 여부 (Quad / any spin with lines)
    all_clear: bool       # 보드가 완전히 비었는가 (All Clear / Perfect Clear)
    combo: int            # 이번 클리어 후의 combo 값 (0 = 첫 연속, -1 = 없음)
    b2b: int              # 이번 클리어 후의 B2B 값
    b2b_delta: int        # 이번 클리어로 B2B가 얼마나 변했는가


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def is_difficult(lines: int, spin: SpinType) -> bool:
    """
    해당 클리어가 B2B 체인을 유지/시작하는 difficult clear인지 판정한다.

    - Quad(4줄)               → True
    - Spin (Mini 포함) + 1줄 이상 → True  (all-spins 포함, Season 2 기준)
    - 그 외 Single/Double/Triple  → False
    - 줄 클리어 없음              → False
    """
    if lines == 0:
        return False
    if lines >= 4:
        return True
    if spin != SpinType.NONE:   # MINI or FULL
        return True
    return False


def _board_is_empty(board) -> bool:
    """보드의 모든 셀이 비어 있으면 True (All Clear 조건)."""
    return all(
        board.get(r, c) is None
        for r in range(BOARD_ROWS)
        for c in range(BOARD_COLS)
    )


# ── 공개 API ──────────────────────────────────────────────────────────────────

def process_clear(state: GameState, lines: int, spin: SpinType) -> ClearResult:
    """
    lock_active() 완료 직후 호출한다.
    combo, B2B 필드를 갱신하고 ClearResult를 반환한다.

    매개변수
    --------
    state : GameState — combo, b2b 필드가 여기서 갱신된다
    lines : int       — 이번 lock으로 클리어된 줄 수
    spin  : SpinType  — detect_spin()으로 얻은 lock 전 스핀 타입
    """
    b2b_before = state.b2b

    # ── 줄 클리어 없음 ────────────────────────────────────────────────────────
    if lines == 0:
        state.combo = -1    # 연속이 끊겼으므로 콤보 초기화
        return ClearResult(
            lines=0, spin=spin,
            difficult=False, all_clear=False,
            combo=state.combo, b2b=state.b2b, b2b_delta=0,
        )

    # ── Combo ─────────────────────────────────────────────────────────────────
    # combo = -1 상태에서 +1 하면 0 (첫 연속), 이후 계속 +1
    state.combo += 1

    # ── Difficult / B2B ───────────────────────────────────────────────────────
    difficult = is_difficult(lines, spin)

    if difficult:
        if state.b2b < 0:
            state.b2b = 0       # 체인 시작 (이번 클리어엔 B2B 공격 보너스 없음)
        else:
            state.b2b += 1      # 체인 계속 (+1)
    else:
        state.b2b = -1          # 체인 끊김

    # ── All Clear 감지 ────────────────────────────────────────────────────────
    # clear_lines()가 이미 완료된 후이므로, 지금 보드가 비어 있으면 All Clear
    # B2B 갱신(+2)은 attack.py의 compute_attack()에서 처리한다.
    ac = _board_is_empty(state.board)

    return ClearResult(
        lines=lines,
        spin=spin,
        difficult=difficult,
        all_clear=ac,
        combo=state.combo,
        b2b=state.b2b,
        b2b_delta=state.b2b - b2b_before,
    )
