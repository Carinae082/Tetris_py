# engine/attack.py
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .clear import ClearResult
from .state import GameState
from .types import SpinType


# ----------------------------------------------------------------------
# 기본 상수 (네 현재 엔진 정책 기준)
# ----------------------------------------------------------------------

ALL_CLEAR_ATTACK = 5
B2B_CHARGING_BONUS = 1

# opener phase: 첫 14피스 동안, 공격 < pending 이면 2배 상쇄
OPENER_PHASE_PIECE_LIMIT = 14

# surge: internal difficult streak 4부터 저장 시작
SURGE_START_STREAK = 4


# ----------------------------------------------------------------------
# 반환 구조체
# ----------------------------------------------------------------------

@dataclass
class AttackResult:
    lines: int
    spin: SpinType

    base: int = 0
    combo_bonus: int = 0
    b2b_bonus: int = 0
    all_clear_bonus: int = 0

    surge_before: int = 0
    surge_after: int = 0
    surge_sent: int = 0
    surge_chunks: list[int] = field(default_factory=list)

    subtotal: int = 0          # cancel 전 총 공격력
    pending_before: int = 0
    canceled: int = 0
    sent: int = 0

    opener_phase: bool = False
    difficult: bool = False
    all_clear: bool = False
    combo: int = -1
    b2b: int = -1


# ----------------------------------------------------------------------
# 내부 헬퍼
# ----------------------------------------------------------------------

def _base_attack(lines: int, spin: SpinType) -> int:
    """
    Season 2 current 구현용 기본 공격표.
    네 엔진 정책과 잘 맞는 실무용 표로 잡아 둔다.
    """
    if spin == SpinType.NONE:
        table = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}
        return table.get(lines, 0)

    if spin == SpinType.MINI:
        table = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}
        return table.get(lines, 0)

    # FULL spin
    table = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
    return table.get(lines, 0)


def _combo_bonus(base: int, combo: int) -> int:
    """
    TETR.IO multiplier combo system에 맞춘 보너스.
    combo는 process_clear() 이후 값이 들어온다.
      combo = 0 : 첫 연속
      combo = 1 : 두 번째 연속
      ...
    """
    if combo < 0:
        return 0

    if base > 0:
        boosted = math.floor(base * (1.0 + 0.25 * combo))
        return boosted - base

    # base == 0일 때도 2-combo부터는 미세한 garbage가 생긴다
    if combo >= 2:
        return math.floor(math.log(1.0 + 1.25 * combo))

    return 0


def _b2b_before_current_clear(clear: ClearResult) -> int:
    """
    현재 clear 전의 B2B 값을 복원한다.
    이유:
    - clear.b2b 는 이미 difficult 반영 + All Clear 보너스까지 먹은 뒤 값
    - B2B +1 공격 보너스는 '이번 clear 전에 이미 체인이 있었는가'로 판단해야 함
    """
    return clear.b2b - clear.b2b_delta


def _b2b_bonus(clear: ClearResult) -> int:
    """
    B2B Charging:
    - 첫 difficult clear는 체인 시작만 하고 공격 보너스는 없음
    - 두 번째 difficult clear부터 +1
    """
    if not clear.difficult:
        return 0

    before = _b2b_before_current_clear(clear)
    return B2B_CHARGING_BONUS if before >= 0 else 0


def _all_clear_bonus(clear: ClearResult) -> int:
    return ALL_CLEAR_ATTACK if clear.all_clear else 0


def _split_front_loaded_3(total: int) -> list[int]:
    """
    surge를 3덩이로 나눌 때 앞쪽 덩이가 나머지를 가져가게 분할.
    예:
      8 -> [3,3,2]
      5 -> [2,2,1]
      2 -> [1,1]
    """
    if total <= 0:
        return []

    q, r = divmod(total, 3)
    out: list[int] = []
    for i in range(3):
        x = q + (1 if i < r else 0)
        if x > 0:
            out.append(x)
    return out


def _internal_difficult_streak(clear: ClearResult) -> int:
    """
    clear.py의 B2B 표현을 '내부 difficult streak 길이'로 되돌린다.

    clear.py 기준:
      첫 difficult clear 후 b2b = 0  (streak 1)
      두 번째 연속 후 b2b = 1        (streak 2)
    즉 internal streak = b2b + 1

    All Clear의 B2B +2 보정은 attack.py(compute_attack)에서만 일어나므로
    clear.b2b는 difficult 판정 결과만 반영한 순수값이다.
    """
    if not clear.difficult:
        return 0

    if clear.b2b < 0:
        return 0
    return clear.b2b + 1


def _update_and_maybe_fire_surge(state: GameState, clear: ClearResult) -> tuple[int, list[int], int, int]:
    """
    반환:
      surge_before, surge_sent, surge_chunks, surge_after
    """
    surge_before = state.surge
    surge_sent = 0
    surge_chunks: list[int] = []

    if clear.difficult:
        streak = _internal_difficult_streak(clear)
        if streak >= SURGE_START_STREAK:
            # 현재 streak 길이만큼 저장된 것으로 간주
            state.surge = streak
    else:
        # non-difficult clear에서 체인이 끊길 때 저장된 surge 발사
        if clear.lines > 0 and state.surge > 0:
            surge_sent = state.surge
            surge_chunks = _split_front_loaded_3(state.surge)
            state.surge = 0

    return surge_before, surge_sent, surge_chunks, state.surge


# ----------------------------------------------------------------------
# 공개 API
# ----------------------------------------------------------------------

def compute_attack(state: GameState, clear: ClearResult) -> AttackResult:
    """
    process_clear() 직후 호출.
    이 함수는 다음까지 처리한다:
      1) base/combo/B2B/AC 계산
      2) surge 저장 또는 발사
      3) incoming garbage 상쇄
      4) 남은 공격을 state.outgoing에 누적
    """
    result = AttackResult(
        lines=clear.lines,
        spin=clear.spin,
        difficult=clear.difficult,
        all_clear=clear.all_clear,
        combo=clear.combo,
        b2b=clear.b2b,
    )

    # 줄 클리어가 없으면 공격 없음
    if clear.lines == 0:
        result.surge_before = state.surge
        result.surge_after = state.surge
        return result

    result.base = _base_attack(clear.lines, clear.spin)
    result.combo_bonus = _combo_bonus(result.base, clear.combo)
    result.b2b_bonus = _b2b_bonus(clear)
    result.all_clear_bonus = _all_clear_bonus(clear)

    # All Clear: B2B streak +2 (clear.py는 AC를 b2b에 반영하지 않으므로 여기서 처리)
    if clear.all_clear:
        state.b2b += 2

    (
        result.surge_before,
        result.surge_sent,
        result.surge_chunks,
        result.surge_after,
    ) = _update_and_maybe_fire_surge(state, clear)

    result.subtotal = (
        result.base
        + result.combo_bonus
        + result.b2b_bonus
        + result.all_clear_bonus
        + result.surge_sent
    )

    result.pending_before = state.incoming.total_lines()

    if result.subtotal <= 0:
        return result

    # opener phase 여부
    result.opener_phase = (
        state.pieces_placed <= OPENER_PHASE_PIECE_LIMIT
        and result.subtotal < result.pending_before
    )

    if result.opener_phase:
        # 2배 상쇄
        cancel_budget = result.subtotal * 2
        remaining_budget = state.incoming.cancel(cancel_budget)
        result.canceled = cancel_budget - remaining_budget

        # 실제 공격 자원은 2줄 상쇄에 1줄을 쓴 셈
        spent_attack = math.ceil(result.canceled / 2)
        result.sent = max(0, result.subtotal - spent_attack)
    else:
        remaining_attack = state.incoming.cancel(result.subtotal)
        result.canceled = result.subtotal - remaining_attack
        result.sent = remaining_attack

    state.outgoing += result.sent
    return result


# ── 공개 헬퍼 (테스트·외부 모듈용) ──────────────────────────────────────────────

def get_base_attack(lines: int, spin: SpinType) -> int:
    """기본 공격력 조회. 테이블에 없는 조합은 0."""
    return _base_attack(lines, spin)


def apply_combo(base: int, combo: int) -> int:
    """
    콤보 배율을 적용한 공격력을 반환한다.

    combo <= 0             -> base 그대로
    base > 0               -> floor(base * (1 + 0.25 * combo))
    base == 0, combo >= 2  -> floor(ln(1 + 1.25 * combo))
    base == 0, combo < 2   -> 0
    """
    if combo <= 0:
        return base
    boosted = base + _combo_bonus(base, combo)
    return boosted


def split_surge(total: int) -> list[int]:
    """
    surge를 3덩어리로 분할. 나누어 떨어지지 않으면 앞쪽이 나머지를 가짐.

    예:  8 -> [3, 3, 2]   5 -> [2, 2, 1]   9 -> [3, 3, 3]
    """
    return _split_front_loaded_3(total)