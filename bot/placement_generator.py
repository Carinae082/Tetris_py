"""
bot/placement_generator.py — 도달 가능한 최종 배치 생성기

FinalPlacement            : 단일 최종 배치 (frozen=True, hashable)
list_reachable_placements : BFS 기반 도달 가능 배치 열거

설계 원칙
----------
* 게임 규칙을 직접 구현하지 않는다.
  이동·회전 합법 여부는 engine.physics.try_move / try_rotate 에 위임한다.
  (SRS 킥 테이블, 착지 판정, 회전 추적 플래그 모두 엔진이 처리)
* board 는 BFS 전체에서 공유 참조 (불변). 복제 비용이 없다.
* 방문 집합 키: (row, col, rotation, last_was_rotation, last_kick_index,
                  last_kick_upgrades) → 스핀 판정 컨텍스트까지 구분
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from engine.mino import ActivePiece, MinoType, SPAWN
from engine.physics import try_move, try_rotate
from engine.state import GameState, RoundResult

from .adapter import Action


# ── FinalPlacement ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FinalPlacement:
    """
    단일 최종 배치 정보. frozen=True → hashable → dict 키로 사용 가능.

    piece_type         : 배치되는 미노 타입
    row, col           : 최종 바운딩 박스 위치 (보드 절대 좌표)
    rotation           : 최종 회전 상태 (0=N, 1=E, 2=S, 3=W)
    use_hold           : 홀드를 사용해 이 미노를 꺼냈는가
    last_was_rotation  : 마지막 유효 동작이 회전이었는가 (스핀 판정용)
    last_kick_index    : 마지막 회전의 킥 인덱스 (−1 = 회전 아님)
    last_kick_upgrades : 해당 킥이 mini→full 승격을 유발하는가
    input_sequence     : 이 배치에 도달하는 원시 행동 목록
                         (HOLD? + 이동/회전/SOFT_DROP... + HARD_DROP)
    """
    piece_type:         MinoType
    row:                int
    col:                int
    rotation:           int
    use_hold:           bool
    last_was_rotation:  bool
    last_kick_index:    int
    last_kick_upgrades: bool
    input_sequence:     tuple  # tuple[Action, ...]


# ── 최소 상태 래퍼 ─────────────────────────────────────────────────────────────

class _PS:
    """
    BFS 전용 최소 상태 래퍼.
    try_move / try_rotate 가 접근하는 필드(board, active, last_was_rotation,
    last_kick_index, last_kick_upgrades, lock_timer)만 유지한다.

    board 는 BFS 내내 공유 참조로 유지되며 절대 수정하지 않는다.
    """

    __slots__ = (
        "board", "active",
        "last_was_rotation", "last_kick_index", "last_kick_upgrades",
        "lock_timer",
    )

    def __init__(
        self,
        board,
        active: ActivePiece,
        lwr: bool = False,
        lki: int = -1,
        lku: bool = False,
    ) -> None:
        self.board              = board
        self.active             = active
        self.last_was_rotation  = lwr
        self.last_kick_index    = lki
        self.last_kick_upgrades = lku
        self.lock_timer: float | None = None

    def key(self) -> tuple:
        """BFS 방문 중복 제거 키."""
        p = self.active
        return (
            p.row, p.col, p.rotation,
            self.last_was_rotation,
            self.last_kick_index,
            self.last_kick_upgrades,
        )

    def copy(self) -> _PS:
        """이동 가능한 상태만 복사 (board 는 공유 유지)."""
        p = self.active
        return _PS(
            self.board,
            ActivePiece(p.type, p.row, p.col, p.rotation),
            self.last_was_rotation,
            self.last_kick_index,
            self.last_kick_upgrades,
        )

    def is_grounded(self) -> bool:
        """한 칸 아래가 막혀 있으면 True (착지 상태)."""
        p = self.active
        below = ActivePiece(p.type, p.row + 1, p.col, p.rotation)
        return not self.board.is_valid_position(below)


def _soft_drop_one(ps: _PS) -> bool:
    """
    한 칸 아래로 이동한다. 성공하면 True, 이미 착지해 있으면 False.
    소프트 드롭은 스핀 판정 플래그를 초기화한다.
    """
    p = ps.active
    below = ActivePiece(p.type, p.row + 1, p.col, p.rotation)
    if ps.board.is_valid_position(below):
        ps.active             = below
        ps.last_was_rotation  = False
        ps.last_kick_index    = -1
        ps.last_kick_upgrades = False
        return True
    return False


# ── BFS ───────────────────────────────────────────────────────────────────────

# 전환 목록: (Action, 상태 변환 함수)
# try_move / try_rotate 는 _PS 를 duck-typing 으로 수용한다.
_TRANSITIONS = [
    (Action.LEFT,       lambda ps: try_move(ps, -1)),
    (Action.RIGHT,      lambda ps: try_move(ps, +1)),
    (Action.ROTATE_CW,  lambda ps: try_rotate(ps, 1)),
    (Action.ROTATE_CCW, lambda ps: try_rotate(ps, -1)),
    (Action.ROTATE_180, lambda ps: try_rotate(ps, 2)),
    (Action.SOFT_DROP,  _soft_drop_one),
]


def _bfs(start: _PS, use_hold: bool) -> list[FinalPlacement]:
    """
    start 위치에서 BFS 를 실행해 도달 가능한 모든 최종 배치를 반환한다.

    큐 항목: (_PS 스냅샷, 현재까지의 행동 튜플)
    방문 집합으로 중복 상태를 방지한다.
    착지 상태(한 칸 아래가 막힘)를 발견하면 FinalPlacement 로 기록한다.
    """
    visited: set = {start.key()}
    # (state, actions_so_far)
    queue: deque[tuple[_PS, tuple]] = deque()
    queue.append((start, ()))

    landings: list[FinalPlacement] = []
    landing_keys: set = set()

    while queue:
        cur, cur_actions = queue.popleft()

        # 착지 여부 확인 → FinalPlacement 기록
        if cur.is_grounded():
            lkey = cur.key()
            if lkey not in landing_keys:
                landing_keys.add(lkey)
                p      = cur.active
                prefix = (Action.HOLD,) if use_hold else ()
                seq    = prefix + cur_actions + (Action.HARD_DROP,)
                landings.append(FinalPlacement(
                    piece_type         = p.type,
                    row                = p.row,
                    col                = p.col,
                    rotation           = p.rotation,
                    use_hold           = use_hold,
                    last_was_rotation  = cur.last_was_rotation,
                    last_kick_index    = cur.last_kick_index,
                    last_kick_upgrades = cur.last_kick_upgrades,
                    input_sequence     = seq,
                ))

        # 전환 시도
        for action, fn in _TRANSITIONS:
            nps = cur.copy()
            if fn(nps):
                nkey = nps.key()
                if nkey not in visited:
                    visited.add(nkey)
                    queue.append((nps, cur_actions + (action,)))

    return landings


# ── 공개 API ──────────────────────────────────────────────────────────────────

def list_reachable_placements(env: GameState) -> list[FinalPlacement]:
    """
    현재 환경에서 도달 가능한 모든 최종 배치를 반환한다.

    활성 미노가 없거나 게임이 종료된 경우 빈 목록을 반환한다.
    홀드가 가능한 경우 홀드 후 배치도 포함한다.

    이 함수는 engine.physics.try_move / try_rotate 에 완전히 위임하며
    게임 규칙을 직접 구현하지 않는다.
    """
    if env.active is None or env.round_result != RoundResult.ONGOING:
        return []

    board = env.board
    p     = env.active

    # ── 현재 활성 미노 BFS ──────────────────────────────────────────────────
    start = _PS(board, ActivePiece(p.type, p.row, p.col, p.rotation))
    placements = _bfs(start, use_hold=False)

    # ── 홀드 BFS ────────────────────────────────────────────────────────────
    if not env.hold_used:
        if env.hold is None:
            # hold 슬롯이 비어 있음: 현재 미노 → hold, next_queue[0] 이 스폰됨
            hold_type = env.next_queue[0] if env.next_queue else None
        else:
            # hold 슬롯과 교환: hold 슬롯에 있던 미노가 스폰됨
            hold_type = env.hold

        if hold_type is not None:
            sp_row, sp_col = SPAWN[hold_type]
            hold_piece     = ActivePiece(hold_type, sp_row, sp_col, 0)
            if board.is_valid_position(hold_piece):
                hold_start = _PS(board, hold_piece)
                placements.extend(_bfs(hold_start, use_hold=True))

    return placements
