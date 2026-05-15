"""
engine/physics.py — 중력, Lock Delay, 이동/회전(SRS), 드롭

공개 함수
---------
try_move(state, dc)         좌우 이동  (dc: -1=왼쪽, +1=오른쪽)
try_rotate(state, steps)    회전       (steps: 1=CW, -1=CCW, 2=180°)
hard_drop(state)            즉시 락    → lines_cleared 반환
soft_drop_step(state)       1칸 아래   → 이동 성공 여부 반환
gravity_tick(state, dt)     중력+락 타이머 진행 → TickResult 반환
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mino import ActivePiece, MinoType
from .spin import detect_spin
from .state import GameState, LockResult
from .types import SpinType

# ── 상수 ──────────────────────────────────────────────────────────────────────

LOCK_DELAY: float = 0.500          # 초 (500 ms)
GRAVITY_20G: float = 20.0          # G 이상이면 스폰 즉시 바닥으로
_FPS: float = 60.0                 # G → rows/s 변환 기준 프레임레이트

# ── SRS 킥 테이블 ─────────────────────────────────────────────────────────────
#
# 입력 좌표계: (x, y) — x=오른쪽(+), y=위(+)  (Tetris Wiki / TETR.IO 표기)
# 내부 변환:  (row, col) = (−y, x)  (row↓, col→)
#
# 상태 표기:  0=North, 1=East, 2=South, 3=West
# 딕셔너리 키: (from_rotation, to_rotation)

# JLSTZ 90° — 표준 SRS와 동일
_JLSTZ_90: dict[tuple[int, int], list[tuple[int, int]]] = {
    # 원본(x,y)       →  변환(row,col)
    (0, 1): [( 0, 0), ( 0,-1), (-1,-1), ( 2, 0), ( 2,-1)],
    (1, 0): [( 0, 0), ( 0, 1), ( 1, 1), (-2, 0), (-2, 1)],
    (1, 2): [( 0, 0), ( 0, 1), ( 1, 1), (-2, 0), (-2, 1)],
    (2, 1): [( 0, 0), ( 0,-1), (-1,-1), ( 2, 0), ( 2,-1)],
    (2, 3): [( 0, 0), ( 0, 1), (-1, 1), ( 2, 0), ( 2, 1)],
    (3, 2): [( 0, 0), ( 0,-1), ( 1,-1), (-2, 0), (-2,-1)],
    (3, 0): [( 0, 0), ( 0,-1), ( 1,-1), (-2, 0), (-2,-1)],
    (0, 3): [( 0, 0), ( 0, 1), (-1, 1), ( 2, 0), ( 2, 1)],
}

# I 미노 90° — TETR.IO SRS+ (표준 SRS와 2·3번 테스트 순서가 다름)
_I_90: dict[tuple[int, int], list[tuple[int, int]]] = {
    (0, 1): [( 0, 0), ( 0, 1), ( 0,-2), ( 1,-2), (-2, 1)],
    (1, 0): [( 0, 0), ( 0,-1), ( 0, 2), (-1, 2), ( 2,-1)],
    (1, 2): [( 0, 0), ( 0,-1), ( 0, 2), ( 1, 2), (-2,-1)],
    (2, 1): [( 0, 0), ( 0, 1), ( 0,-2), (-1,-2), ( 2, 1)],
    (2, 3): [( 0, 0), ( 0,-1), ( 0, 2), (-1, 2), ( 2,-1)],
    (3, 2): [( 0, 0), ( 0, 1), ( 0,-2), ( 1,-2), (-2, 1)],
    (3, 0): [( 0, 0), ( 0, 1), ( 0,-2), (-1,-2), ( 2, 1)],
    (0, 3): [( 0, 0), ( 0,-1), ( 0, 2), ( 1, 2), (-2,-1)],
}

# 180° 킥 — 피스 종류 무관하게 공용
# (O는 킥 없음; try_rotate에서 O를 별도 처리)
_KICKS_180: dict[tuple[int, int], list[tuple[int, int]]] = {
    (0, 2): [( 0, 0), (-1, 0), (-1, 1), (-1,-1), ( 0, 1), ( 0,-1)],  # N→S
    (2, 0): [( 0, 0), ( 1, 0), ( 1,-1), ( 1, 1), ( 0,-1), ( 0, 1)],  # S→N
    (1, 3): [( 0, 0), ( 0, 1), (-2, 1), (-1, 1), (-2, 0), (-1, 0)],  # E→W
    (3, 1): [( 0, 0), ( 0,-1), (-2,-1), (-1,-1), (-2, 0), (-1, 0)],  # W→E
}

# O 미노: 킥 없음 (회전 시 위치 그대로, 불가능하면 거부)
_O_NO_KICK: list[tuple[int, int]] = [(0, 0)]

_KICK_TABLE_90: dict[MinoType, dict[tuple[int, int], list[tuple[int, int]]]] = {
    MinoType.I: _I_90,
    MinoType.J: _JLSTZ_90,
    MinoType.L: _JLSTZ_90,
    MinoType.O: {},   # O는 킥 없음
    MinoType.S: _JLSTZ_90,
    MinoType.T: _JLSTZ_90,
    MinoType.Z: _JLSTZ_90,
}

# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────


def _is_grounded(state: GameState) -> bool:
    """활성 미노가 바닥 또는 스택에 닿아 있는가."""
    p = state.active
    if p is None:
        return False
    one_below = ActivePiece(p.type, p.row + 1, p.col, p.rotation)
    return not state.board.is_valid_position(one_below)


def _drop_to_floor(state: GameState) -> None:
    """활성 미노를 가능한 가장 낮은 위치까지 내린다 (lock하지 않음)."""
    p = state.active
    if p is None:
        return
    while True:
        below = ActivePiece(p.type, p.row + 1, p.col, p.rotation)
        if not state.board.is_valid_position(below):
            break
        p = below
    state.active = p


def _sync_lock_timer(state: GameState) -> None:
    """
    이동/회전 성공 후 호출.
    - 착지 상태이면 lock_timer를 LOCK_DELAY로 리셋 (재시작).
    - 공중 상태이면 lock_timer를 None으로 초기화.
    """
    if _is_grounded(state):
        state.lock_timer = LOCK_DELAY
    else:
        state.lock_timer = None


# ── 공개 함수 ─────────────────────────────────────────────────────────────────


def _reset_rotation_tracking(state: GameState) -> None:
    """이동/드롭 등 회전 이외 동작 후 스핀 판정 플래그를 초기화한다."""
    state.last_was_rotation = False
    state.last_kick_index = -1
    state.last_kick_upgrades = False


def try_move(state: GameState, dc: int) -> bool:
    """
    활성 미노를 좌(dc=-1) 또는 우(dc=+1)로 1칸 이동한다.
    성공하면 lock_timer를 리셋하고 True를 반환한다.
    이동은 스핀 판정 플래그를 초기화한다.
    """
    if state.active is None:
        return False
    p = state.active
    candidate = ActivePiece(p.type, p.row, p.col + dc, p.rotation)
    if not state.board.is_valid_position(candidate):
        return False
    state.active = candidate
    _reset_rotation_tracking(state)
    _sync_lock_timer(state)
    return True


def try_rotate(state: GameState, steps: int) -> bool:
    """
    활성 미노를 회전한다.

    steps:  1 = CW(시계방향)
           -1 = CCW(반시계방향)
            2 = 180°

    SRS 킥을 시도하고, 성공하면 lock_timer를 리셋하고 True를 반환한다.
    O 미노는 회전 상태 변경은 허용하지만 킥은 없다.
    """
    if state.active is None:
        return False

    p = state.active

    if steps == 2:
        return _rotate_180(state)

    new_rot = (p.rotation + steps) % 4
    # O 미노는 킥 없음; 그 외 미노는 킥 테이블 조회
    kicks = _KICK_TABLE_90.get(p.type, _JLSTZ_90).get((p.rotation, new_rot), _O_NO_KICK)

    for i, (dr, dc) in enumerate(kicks):
        candidate = ActivePiece(p.type, p.row + dr, p.col + dc, new_rot)
        if state.board.is_valid_position(candidate):
            state.active = candidate
            state.last_was_rotation = True
            state.last_kick_index = i
            # 90° 회전의 5번째 킥(index 4) = mini→full 승격 대상
            state.last_kick_upgrades = (i == 4)
            _sync_lock_timer(state)
            return True
    return False


def _rotate_180(state: GameState) -> bool:
    """180° 회전. O 미노는 킥 없음. mini→full 승격은 없음."""
    p = state.active
    new_rot = (p.rotation + 2) % 4

    if p.type == MinoType.O:
        candidate = ActivePiece(p.type, p.row, p.col, new_rot)
        if state.board.is_valid_position(candidate):
            state.active = candidate
            state.last_was_rotation = True
            state.last_kick_index = 0
            state.last_kick_upgrades = False
            _sync_lock_timer(state)
            return True
        return False

    kicks = _KICKS_180.get((p.rotation, new_rot), [(0, 0)])
    for i, (dr, dc) in enumerate(kicks):
        candidate = ActivePiece(p.type, p.row + dr, p.col + dc, new_rot)
        if state.board.is_valid_position(candidate):
            state.active = candidate
            state.last_was_rotation = True
            state.last_kick_index = i
            state.last_kick_upgrades = False   # 180° 는 승격 없음
            _sync_lock_timer(state)
            return True
    return False


def hard_drop(state: GameState) -> LockResult:
    """
    활성 미노를 바닥까지 즉시 내리고 잠근다.

    hard drop은 이동이 아니므로 last_was_rotation 플래그를 초기화하지 않는다.
    즉, 회전 직후 hard drop을 해도 스핀 판정이 유지된다.
    스핀 감지는 바닥 도달 후 실제 lock 위치에서 수행한다.
    """
    if state.active is None:
        return LockResult()
    _drop_to_floor(state)
    # 스핀 감지: hard drop으로 이동한 최종 위치에서 판정
    spin = detect_spin(state)
    state.lock_timer = None
    lock_result = state.lock_active()
    lock_result.spin = spin
    return lock_result


def soft_drop_step(state: GameState) -> bool:
    """
    활성 미노를 1칸 아래로 내린다 (non-locking: 즉시 lock하지 않음).
    플레이어가 직접 내리는 동작이므로 스핀 판정 플래그를 초기화한다.
    착지 시 lock_timer가 아직 None이면 새로 시작한다.
    반환값: 이동 성공 여부.
    """
    if state.active is None:
        return False
    p = state.active
    candidate = ActivePiece(p.type, p.row + 1, p.col, p.rotation)
    if not state.board.is_valid_position(candidate):
        if state.lock_timer is None:
            state.lock_timer = LOCK_DELAY
        return False
    state.active = candidate
    _reset_rotation_tracking(state)
    if _is_grounded(state) and state.lock_timer is None:
        state.lock_timer = LOCK_DELAY
    return True


# ── TickResult ─────────────────────────────────────────────────────────────────


@dataclass
class TickResult:
    moved_down: bool = False      # 중력으로 아래 이동했는가
    locked: bool = False          # 이 틱에서 piece가 lock됐는가
    lines_cleared: int = 0        # lock 후 클리어된 줄 수
    clutch_clear: bool = False    # vanish zone에서 lock했지만 줄을 지워 생존
    spin: SpinType = SpinType.NONE  # lock 시 감지된 스핀 타입


def gravity_tick(state: GameState, dt: float) -> TickResult:
    """
    중력과 Lock Delay 타이머를 dt초만큼 진행한다.

    흐름:
      1. gravity >= 20 G → 즉시 바닥 낙하 후 타이머 시작
      2. 일반 중력 → 누산기에 rows/s * dt를 더하고 1칸씩 내림
      3. 착지 중이면 lock_timer 카운트다운
      4. lock_timer <= 0 이면 lock

    반환값: TickResult (무슨 일이 일어났는지 요약)
    """
    result = TickResult()

    if state.active is None:
        return result

    rows_per_sec = state.gravity * _FPS  # G → rows/s

    # 1. 20 G: 즉시 바닥
    if state.gravity >= GRAVITY_20G:
        _drop_to_floor(state)
        if state.lock_timer is None:
            state.lock_timer = LOCK_DELAY
    else:
        # 2. 일반 중력 누산
        state.gravity_acc += rows_per_sec * dt
        while state.gravity_acc >= 1.0:
            state.gravity_acc -= 1.0
            p = state.active
            candidate = ActivePiece(p.type, p.row + 1, p.col, p.rotation)
            if state.board.is_valid_position(candidate):
                state.active = candidate
                result.moved_down = True
            else:
                # 바닥에 닿음
                state.gravity_acc = 0.0
                if state.lock_timer is None:
                    state.lock_timer = LOCK_DELAY
                break

        # 중력 이동 없이도 착지 상태일 수 있음 (처음 착지 감지)
        if state.lock_timer is None and _is_grounded(state):
            state.lock_timer = LOCK_DELAY

    # 3 & 4. Lock Delay 카운트다운
    if state.lock_timer is not None:
        state.lock_timer -= dt
        if state.lock_timer <= 0.0:
            state.lock_timer = None
            # 스핀 감지는 lock 전 현재 위치에서 수행
            spin = detect_spin(state)
            lock_result = state.lock_active()
            lock_result.spin = spin
            result.locked = True
            result.lines_cleared = lock_result.lines_cleared
            result.clutch_clear = lock_result.clutch_clear
            result.spin = spin

    return result
