from __future__ import annotations

import random
from collections import deque

from .mino import MinoType

# GARBAGE는 플레이어블 미노가 아니므로 제외
PLAYABLE_MINOS: list[MinoType] = [
    MinoType.I, MinoType.J, MinoType.L,
    MinoType.O, MinoType.S, MinoType.T, MinoType.Z,
]

# next queue가 이 값 이하로 떨어지면 새 bag를 보충
_REFILL_THRESHOLD = 6


class SevenBag:
    """
    7-bag 랜덤 생성기.

    TETR.IO Tetra League / Custom Game에서는 양쪽 플레이어가 같은 bag 시퀀스를
    공유한다. 따라서 이 객체를 하나 생성해 두 GameState에 모두 주입하면 된다.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._sequence: deque[MinoType] = deque()
        self._refill()

    # ------------------------------------------------------------------
    # 내부
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """7종 미노를 한 번씩 담아 섞고 시퀀스 뒤에 이어 붙인다."""
        bag = PLAYABLE_MINOS[:]
        self._rng.shuffle(bag)
        self._sequence.extend(bag)

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def peek(self, n: int) -> list[MinoType]:
        """소비하지 않고 다음 n개를 반환한다."""
        while len(self._sequence) < n:
            self._refill()
        return [self._sequence[i] for i in range(n)]

    def pop(self) -> MinoType:
        """다음 미노를 소비해 반환한다. 필요하면 자동으로 bag를 보충한다."""
        if len(self._sequence) <= _REFILL_THRESHOLD:
            self._refill()
        return self._sequence.popleft()

    def __len__(self) -> int:
        return len(self._sequence)

    def __repr__(self) -> str:
        preview = [m.value for m in list(self._sequence)[:7]]
        return f"SevenBag(next={preview}...)"
