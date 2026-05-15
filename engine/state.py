from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from .board import Board, VISIBLE_ROW_START
from .garbage import GarbageQueue
from .mino import ActivePiece, MinoType, SHAPES
from .types import SpinType


class RoundResult(Enum):
    ONGOING = auto()
    WIN = auto()
    LOSE = auto()


@dataclass
class LockResult:
    """lock_active() 반환값."""
    lines_cleared: int = 0
    clutch_clear: bool = False    # vanish zone 배치였지만 줄 클리어로 생존
    spin: SpinType = SpinType.NONE  # lock 시점에 감지된 스핀 타입
    clears_garbage: bool = False  # 클리어된 줄 중 가비지 행이 포함됐는지


@dataclass
class GameState:
    """
    한 플레이어의 엔진 내부 상태 전체.

    GameState는 bag를 직접 소유하지 않는다.
    서버(세션)가 공유 SevenBag에서 미노를 꺼내 enqueue()로 양쪽 플레이어에
    같은 순서로 주입하는 것이 TETR.IO 방식의 공용 bag 구현이다.
    """

    # ── 보드 ──────────────────────────────────────────────────────────
    board: Board = field(default_factory=Board)

    # ── 활성 미노 ─────────────────────────────────────────────────────
    active: ActivePiece | None = None

    # ── Hold ──────────────────────────────────────────────────────────
    hold: MinoType | None = None
    hold_used: bool = False   # 이번 미노에서 hold를 이미 썼는가

    # ── Next queue ────────────────────────────────────────────────────
    next_queue: deque[MinoType] = field(default_factory=deque)

    # ── 콤보 / B2B ────────────────────────────────────────────────────
    combo: int = -1   # -1 = 콤보 없음; 0 이상 = 연속 클리어 횟수
    b2b: int = -1     # -1 = B2B 없음; 0 이상 = 연속 B2B 적격 클리어 횟수

    # ── Surge ─────────────────────────────────────────────────────────
    surge: int = 0    # 저장된 surge 라인 수

    # ── 가비지 ────────────────────────────────────────────────────────
    outgoing: int = 0              # 상대에게 보낼 공격 라인 수 (발사 전 누적)
    incoming: GarbageQueue = field(default_factory=GarbageQueue)

    # ── 라운드 통계 ───────────────────────────────────────────────────
    pieces_placed: int = 0
    round_result: RoundResult = RoundResult.ONGOING

    # ── block-out / spawn 검사 ────────────────────────────────────────
    spawn_blocked: bool = False    # 마지막 스폰 시도가 막혔는가

    # ── 스핀 판정용 회전 추적 ────────────────────────────────────────────
    # 마지막 유효 동작이 회전인지 추적 (스핀 판정의 전제 조건)
    last_was_rotation: bool = False
    # 마지막 회전에 사용된 킥 인덱스 (-1 = 회전 아님 또는 무효)
    # 90° 회전: JLSTZ 기준 5개 킥 중 index 4가 "last kick" mini→full 승격 대상
    last_kick_index: int = -1
    # 90° 회전의 last kick(index==4)으로 성공했는가 (180°는 항상 False)
    last_kick_upgrades: bool = False

    # ── 중력 / Lock Delay ─────────────────────────────────────────────
    # gravity: TETR.IO G 단위 (rows/frame at 60 fps).  1G = 60 rows/s.
    # 20G 이상이면 스폰 직후 즉시 바닥으로 낙하한다.
    gravity: float = 0.2           # G 단위, 기본값 = 0.2 G
    gravity_acc: float = 0.0       # 중력 누산기 (내부 전용)
    lock_timer: float | None = None  # None=공중; 양수=남은 lock delay(초)

    # ==================================================================
    # 미노 큐 관리 (외부에서 주입)
    # ==================================================================

    def enqueue(self, mino_type: MinoType) -> None:
        """
        서버(세션)가 bag에서 꺼낸 미노를 이 플레이어의 큐에 추가한다.
        양쪽 플레이어에 같은 미노를 같은 순서로 주입하는 것은 호출자의 책임.
        """
        self.next_queue.append(mino_type)

    # ==================================================================
    # 미노 스폰
    # ==================================================================

    def _activate(self, piece: ActivePiece) -> None:
        """스폰 성공 후 공통 상태 초기화."""
        self.active = piece
        self.gravity_acc = 0.0
        self.lock_timer = None
        self.last_was_rotation = False
        self.last_kick_index = -1
        self.last_kick_upgrades = False
        self.spawn_blocked = False

    def _find_clutch_spawn(self, mino_type: MinoType) -> ActivePiece | None:
        """
        Clutch Clear 시 스폰 위치를 위로 밀어 탐색한다.

        정상 스폰 위치에서 시작해 row를 1씩 줄이며, 바운딩 박스에서
        가장 위쪽 셀(min_dr)이 보드 밖으로 나가기 전까지 탐색한다.
        유효한 위치가 있으면 해당 ActivePiece를 반환하고, 없으면 None.
        """
        piece = ActivePiece.spawn(mino_type)
        min_dr = min(dr for dr, dc in SHAPES[mino_type][0])

        while piece.row + min_dr >= 0:
            if self.board.is_valid_position(piece):
                return piece
            piece = ActivePiece(piece.type, piece.row - 1, piece.col, 0)

        return None

    def spawn_next(self, clutch_clear: bool = False) -> bool:
        """
        next_queue 앞에서 미노를 꺼내 스폰한다.

        clutch_clear=True 이면 정상 스폰이 막혔을 때 Clutch Clear 규칙을 적용한다:
          다음 미노를 스택 위로 밀어 유효한 위치를 찾아 스폰한다.
          유효한 위치가 없으면 패배.

        clutch_clear=False(기본값)이면 정상 스폰이 막히는 순간 block-out → LOSE.

        반환값: 스폰 성공 여부.
        """
        if not self.next_queue:
            raise RuntimeError("next_queue가 비어 있습니다 — 서버가 enqueue()를 호출해야 합니다")

        mino_type = self.next_queue.popleft()
        piece = ActivePiece.spawn(mino_type)

        if self.board.is_valid_position(piece):
            self._activate(piece)
            return True

        # 정상 스폰 실패
        if clutch_clear:
            clutch_piece = self._find_clutch_spawn(mino_type)
            if clutch_piece is not None:
                self._activate(clutch_piece)
                return True

        # block-out (clutch 조건 미충족 or 위로 밀어도 자리 없음)
        self.spawn_blocked = True
        self.round_result = RoundResult.LOSE
        return False

    # ==================================================================
    # Hold
    # ==================================================================

    def try_hold(self) -> bool:
        """
        현재 활성 미노와 hold 슬롯을 교환한다.
        이번 미노에서 이미 hold를 사용했거나 활성 미노가 없으면 False 반환.
        """
        if self.hold_used or self.active is None:
            return False

        current_type = self.active.type

        if self.hold is None:
            # hold 슬롯이 비어 있으면 현재 미노를 hold에 넣고 다음 미노를 스폰
            self.hold = current_type
            self.active = None
            success = self.spawn_next()
        else:
            # hold 슬롯과 현재 미노를 교환
            swapped_in = self.hold
            self.hold = current_type
            self.active = None

            piece = ActivePiece.spawn(swapped_in)
            if not self.board.is_valid_position(piece):
                self.spawn_blocked = True
                self.round_result = RoundResult.LOSE
                return False

            self._activate(piece)
            success = True

        if success:
            self.hold_used = True  # 이번 미노에서 hold 사용 완료
        return success

    # ==================================================================
    # 미노 고정 (lock)
    # ==================================================================

    def lock_active(self) -> LockResult:
        """
        활성 미노를 보드에 고정하고 줄을 지운다.

        게임 오버 규칙:
          - 스폰 불가(spawn blocked):       spawn_next() 에서 이미 처리
          - 가비지 vanish zone 충돌:         apply_incoming_garbage() 에서 처리
          - 단순 lock-out(버퍼 zone 배치):  사망 조건 아님 (rule 3)
          - Clutch Clear (버퍼 zone 배치 + 줄 클리어): 생존 (rule 4)

        반환값: LockResult
        """
        assert self.active is not None, "고정할 활성 미노가 없습니다"

        # lock 전에 vanish zone(버퍼) 여부 확인
        # VISIBLE_ROW_START(20)보다 위쪽 row에 셀이 있으면 vanish zone 배치
        in_vanish = any(r < VISIBLE_ROW_START for r, _ in self.active.cells())

        self.board.lock(self.active)
        self.active = None
        self.hold_used = False
        self.pieces_placed += 1

        lines_cleared, clears_garbage = self.board.clear_lines()

        clutch_clear = False
        if in_vanish:
            if lines_cleared > 0:
                # Clutch Clear: vanish zone 배치였지만 줄을 지워 생존
                clutch_clear = True
            # lines_cleared == 0인 경우도 rule 3에 따라 사망 처리 안 함

        return LockResult(
            lines_cleared=lines_cleared,
            clutch_clear=clutch_clear,
            clears_garbage=clears_garbage,
        )

    # ==================================================================
    # 가비지
    # ==================================================================

    def add_incoming_garbage(self, lines: int, hole_col: int | None = None) -> None:
        """받을 가비지를 큐에 추가한다."""
        self.incoming.add(lines, hole_col)

    def apply_incoming_garbage(self) -> bool:
        """
        대기 중인 가비지를 모두 보드에 적용한다.

        가비지 상승 후 활성 미노 위치가 보드와 충돌하면
        가비지가 미노를 vanish zone으로 밀어올린 것으로 간주 → LOSE.

        반환값: True = vanish zone 충돌 발생(게임 오버), False = 정상.
        """
        for g in self.incoming.pop_all():
            self.board.add_garbage(g.lines, g.hole_col)

        if self.active is not None and not self.board.is_valid_position(self.active):
            # 가비지가 활성 미노 위치까지 올라왔음 → vanish zone 게임 오버
            self.round_result = RoundResult.LOSE
            return True
        return False

    # ==================================================================
    # 고속 복제 (beam search / RL 훈련용)
    # ==================================================================

    def copy(self) -> "GameState":
        """
        deepcopy 대비 ~5× 빠른 수동 복제.
        board._grid를 행 단위 슬라이스로 복사하고,
        ActivePiece·deque 등 불필요한 재귀 복사를 피한다.
        """
        from collections import deque as _deque

        new = GameState.__new__(GameState)

        # Board
        nb = Board.__new__(Board)
        nb._grid          = [row[:] for row in self.board._grid]
        nb._hash          = self.board._hash
        nb._hash_dirty    = self.board._hash_dirty
        nb._col_heights   = self.board._col_heights[:]
        nb._heights_dirty = self.board._heights_dirty
        new.board = nb

        # ActivePiece
        p = self.active
        new.active = (ActivePiece(p.type, p.row, p.col, p.rotation)
                      if p is not None else None)

        new.hold      = self.hold
        new.hold_used = self.hold_used
        new.next_queue = _deque(self.next_queue)

        new.combo = self.combo
        new.b2b   = self.b2b
        new.surge = self.surge

        new.outgoing = self.outgoing

        # GarbageQueue — queue 내용만 복사, rng는 독립 인스턴스
        from .garbage import GarbageQueue as _GQ
        nq = _GQ()
        nq._queue.extend(self.incoming._queue)  # GarbageLine은 frozen dataclass
        new.incoming = nq

        new.pieces_placed = self.pieces_placed
        new.round_result  = self.round_result
        new.spawn_blocked = self.spawn_blocked

        new.last_was_rotation  = self.last_was_rotation
        new.last_kick_index    = self.last_kick_index
        new.last_kick_upgrades = self.last_kick_upgrades

        new.gravity     = self.gravity
        new.gravity_acc = self.gravity_acc
        new.lock_timer  = self.lock_timer

        return new

    # ==================================================================
    # 디버그
    # ==================================================================

    def __repr__(self) -> str:
        hold_str = self.hold.value if self.hold else "None"
        next_str = " ".join(m.value for m in self.next_queue)
        return (
            f"GameState("
            f"active={self.active}, "
            f"hold={hold_str}(used={self.hold_used}), "
            f"next=[{next_str}], "
            f"combo={self.combo}, b2b={self.b2b}, surge={self.surge}, "
            f"outgoing={self.outgoing}, incoming={self.incoming.total_lines()}lines, "
            f"pieces={self.pieces_placed}, result={self.round_result.name}"
            f")"
        )
