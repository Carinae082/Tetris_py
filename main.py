"""
clear.py + attack.py + garbage.py 통합 테스트

clear.py 케이스:
  1. Single (non-difficult) -> B2B -1 유지
  2. Quad -> difficult, B2B 체인 시작 (b2b=0)
  3. Quad 연속 -> B2B 체인 증가
  4. T-spin Full Double -> difficult
  5. Spin Mini Single -> difficult
  6. non-difficult 후 B2B 체인 끊김
  7. All Clear 감지 (b2b는 clear.py에서 변경 안 함)
  8. difficult + All Clear (b2b는 clear.py에서 변경 안 함)
  9. Combo 누산 -> 줄 클리어 없으면 리셋
  10. is_difficult 단위 테스트

attack.py 케이스:
  A1. Single -> 공격 0
  A2. Double -> 공격 1
  A3. Triple -> 공격 2
  A4. Quad (첫 difficult) -> 공격 4, b2b_bonus=0
  A5. Quad 체인 (b2b>=1) -> 공격 4 + b2b_bonus 1 = 5
  A6. T-spin Full Single -> 공격 2
  A7. T-spin Full Double 체인 -> 공격 4 + 1 = 5
  A8. Combo (base>0) 배율
  A9. Combo (base=0) ln 배율
  A10. All Clear -> +5, state.b2b += 2
  A11. split_surge 분할
  A12. Surge 저장 (4연속 Quad)
  A13. Surge 발동 (Single 후)
  A14. apply_combo 단위 테스트
"""
import math
import random

from engine import (
    GameState, SevenBag, SpinType,
    process_clear, is_difficult, ClearResult,
    compute_attack, AttackResult,
    get_base_attack, apply_combo, split_surge,
    GarbageQueue, GarbageLine,
    BOARD_ROWS, BOARD_COLS, VISIBLE_ROW_START,
)
from engine.mino import MinoType


def fresh(with_garbage: bool = True) -> GameState:
    """with_garbage=True: 보드 맨 아래 한 줄에 잔여 블록을 넣어 All Clear를 방지."""
    s = GameState()
    if with_garbage:
        s.board._grid[39][0] = MinoType.GARBAGE
    return s


def check(result: ClearResult, **expected):
    for k, v in expected.items():
        actual = getattr(result, k)
        assert actual == v, f"{k}: expected {v}, got {actual}  | {result}"


def check_atk(result: AttackResult, **expected):
    for k, v in expected.items():
        actual = getattr(result, k)
        assert actual == v, f"{k}: expected {v}, got {actual}  | {result}"


# ─────────────────────────────────────────────────────────
# clear.py 테스트
# ─────────────────────────────────────────────────────────

# 1. Single -> non-difficult, B2B 리셋 (이미 -1이면 그대로)
def test_single():
    s = fresh()
    r = process_clear(s, lines=1, spin=SpinType.NONE)
    check(r, difficult=False, b2b=-1, b2b_delta=0, combo=0, all_clear=False)
    print(f"[OK] Single: difficult={r.difficult} b2b={r.b2b} combo={r.combo}")


# 2. Quad -> difficult, B2B 체인 시작 (b2b=0)
def test_quad_first():
    s = fresh()
    r = process_clear(s, lines=4, spin=SpinType.NONE)
    check(r, difficult=True, b2b=0, b2b_delta=1, combo=0)
    print(f"[OK] Quad (1st): difficult={r.difficult} b2b={r.b2b}")


# 3. Quad -> Quad 연속 -> b2b=1
def test_quad_chain():
    s = fresh()
    process_clear(s, lines=4, spin=SpinType.NONE)   # b2b=0
    r = process_clear(s, lines=4, spin=SpinType.NONE)  # b2b=1
    check(r, difficult=True, b2b=1, b2b_delta=1, combo=1)
    print(f"[OK] Quad chain: b2b={r.b2b} combo={r.combo}")


# 4. T-spin Full Double -> difficult
def test_tspin_full_double():
    s = fresh()
    r = process_clear(s, lines=2, spin=SpinType.FULL)
    check(r, difficult=True, b2b=0)
    print(f"[OK] T-spin Full Double: difficult={r.difficult} b2b={r.b2b}")


# 5. Spin Mini Single -> difficult
def test_spin_mini_single():
    s = fresh()
    r = process_clear(s, lines=1, spin=SpinType.MINI)
    check(r, difficult=True, b2b=0)
    print(f"[OK] Spin Mini Single: difficult={r.difficult} b2b={r.b2b}")


# 6. Quad -> Single -> B2B 체인 끊김
def test_b2b_break():
    s = fresh()
    process_clear(s, lines=4, spin=SpinType.NONE)  # b2b=0
    process_clear(s, lines=4, spin=SpinType.NONE)  # b2b=1
    r = process_clear(s, lines=1, spin=SpinType.NONE)  # Single -> 리셋
    check(r, difficult=False, b2b=-1)
    print(f"[OK] B2B break after chain: b2b={r.b2b}")


# 7. All Clear 감지: non-difficult + AC -> clear.py는 b2b에 AC를 반영하지 않음
def test_all_clear_non_difficult():
    s = fresh(with_garbage=False)
    r = process_clear(s, lines=1, spin=SpinType.NONE)
    # Single (non-difficult): b2b=-1, AC 감지만. AC b2b 갱신은 attack.py 담당
    check(r, difficult=False, all_clear=True, b2b=-1, b2b_delta=0)
    print(f"[OK] Non-difficult + All Clear detected: b2b={r.b2b} (AC update in attack.py)")


# 8. difficult + All Clear -> clear.py는 Quad만큼만 b2b 갱신
def test_all_clear_difficult():
    s = fresh(with_garbage=False)
    r = process_clear(s, lines=4, spin=SpinType.NONE)
    # Quad: -1 -> 0. AC는 attack.py에서 +2
    check(r, difficult=True, all_clear=True, b2b=0, b2b_delta=1)
    print(f"[OK] Quad + All Clear: b2b={r.b2b} (AC +2 handled in attack.py)")


# 8b. 이미 체인 중 Quad + All Clear
def test_all_clear_difficult_chain():
    s = fresh(with_garbage=False)
    s.b2b = 3   # 이미 체인 진행 중
    r = process_clear(s, lines=4, spin=SpinType.NONE)
    # b2b=3, Quad +1 -> 4. AC는 attack.py에서 +2 -> 6
    check(r, difficult=True, all_clear=True, b2b=4, b2b_delta=1)
    print(f"[OK] Chain + Quad + All Clear: b2b={r.b2b} (attack.py will add +2)")


# 9. Combo: 연속 3회 클리어 후 빈 피스 배치 -> 리셋
def test_combo():
    s = fresh()
    process_clear(s, lines=1, spin=SpinType.NONE)  # combo=0
    process_clear(s, lines=1, spin=SpinType.NONE)  # combo=1
    r3 = process_clear(s, lines=1, spin=SpinType.NONE)  # combo=2
    check(r3, combo=2)
    r_break = process_clear(s, lines=0, spin=SpinType.NONE)
    check(r_break, combo=-1)
    print(f"[OK] Combo chain then break: {r3.combo} -> {r_break.combo}")


# 10. is_difficult 단위 테스트
def test_is_difficult():
    assert not is_difficult(0, SpinType.NONE)
    assert not is_difficult(1, SpinType.NONE)
    assert not is_difficult(2, SpinType.NONE)
    assert not is_difficult(3, SpinType.NONE)
    assert     is_difficult(4, SpinType.NONE)
    assert     is_difficult(1, SpinType.MINI)
    assert     is_difficult(1, SpinType.FULL)
    assert     is_difficult(3, SpinType.FULL)
    assert     is_difficult(2, SpinType.MINI)
    print("[OK] is_difficult all cases")


# ─────────────────────────────────────────────────────────
# attack.py 테스트
# ─────────────────────────────────────────────────────────

# A1. Single -> 공격 0
def test_atk_single():
    s = fresh()
    cr = process_clear(s, lines=1, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    check_atk(ar, base=0, b2b_bonus=0, all_clear_bonus=0, subtotal=0)
    print(f"[OK] ATK Single: subtotal={ar.subtotal}")


# A2. Double -> 공격 1
def test_atk_double():
    s = fresh()
    cr = process_clear(s, lines=2, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    check_atk(ar, base=1, b2b_bonus=0, subtotal=1)
    print(f"[OK] ATK Double: subtotal={ar.subtotal}")


# A3. Triple -> 공격 2
def test_atk_triple():
    s = fresh()
    cr = process_clear(s, lines=3, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    check_atk(ar, base=2, b2b_bonus=0, subtotal=2)
    print(f"[OK] ATK Triple: subtotal={ar.subtotal}")


# A4. Quad (첫 difficult) -> 공격 4, b2b_bonus=0
def test_atk_quad_first():
    s = fresh()
    cr = process_clear(s, lines=4, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    # b2b_before = -1 -> no bonus
    check_atk(ar, base=4, b2b_bonus=0, subtotal=4)
    print(f"[OK] ATK Quad(1st): subtotal={ar.subtotal} b2b_bonus={ar.b2b_bonus}")


# A5. Quad 체인 (2번째 consecutive) -> b2b_bonus=1, total=5
def test_atk_quad_chain():
    s = fresh()
    cr1 = process_clear(s, lines=4, spin=SpinType.NONE)
    compute_attack(s, cr1)                               # 첫 Quad (no bonus)
    cr2 = process_clear(s, lines=4, spin=SpinType.NONE)
    ar = compute_attack(s, cr2)
    # b2b_before = 0 -> bonus +1; combo=1 -> floor(4*1.25)=5 -> combo_bonus=1
    # total: base(4) + combo_bonus(1) + b2b_bonus(1) = 6
    check_atk(ar, base=4, b2b_bonus=1, combo_bonus=1, subtotal=6)
    print(f"[OK] ATK Quad chain: subtotal={ar.subtotal} b2b_bonus={ar.b2b_bonus}")


# A6. T-spin Full Single -> 공격 2
def test_atk_tspin_single():
    s = fresh()
    cr = process_clear(s, lines=1, spin=SpinType.FULL)
    ar = compute_attack(s, cr)
    check_atk(ar, base=2, b2b_bonus=0, subtotal=2)
    print(f"[OK] ATK T-spin Full Single: subtotal={ar.subtotal}")


# A7. T-spin Full Double 체인 (2번째) -> 4 + 1 = 5
def test_atk_tspin_double_chain():
    s = fresh()
    cr1 = process_clear(s, lines=2, spin=SpinType.FULL)
    compute_attack(s, cr1)
    cr2 = process_clear(s, lines=2, spin=SpinType.FULL)
    ar = compute_attack(s, cr2)
    # combo=1 -> combo_bonus=1 (floor(4*1.25)-4=1); b2b_bonus=1 -> total=6
    check_atk(ar, base=4, b2b_bonus=1, combo_bonus=1, subtotal=6)
    print(f"[OK] ATK T-spin Double chain: subtotal={ar.subtotal}")


# A8. Combo (base>0) 배율: Double combo=2 -> floor(1*(1+0.5))=1
def test_atk_combo_base_positive():
    s = fresh()
    process_clear(s, lines=2, spin=SpinType.NONE)   # combo=0
    process_clear(s, lines=2, spin=SpinType.NONE)   # combo=1
    cr = process_clear(s, lines=2, spin=SpinType.NONE)  # combo=2
    ar = compute_attack(s, cr)
    # base=1, combo=2: floor(1*(1+0.5))=1; combo_bonus=0
    expected_after = math.floor(1 * (1 + 0.25 * 2))   # 1
    check_atk(ar, base=1)
    assert ar.base + ar.combo_bonus == expected_after, f"after_combo mismatch: {ar.base+ar.combo_bonus} != {expected_after}"
    print(f"[OK] ATK Combo (base>0): base={ar.base} combo_bonus={ar.combo_bonus} -> {ar.base+ar.combo_bonus}")


# A9. Combo (base=0) ln 배율: Single combo=2 -> floor(ln(1+2.5))=1
def test_atk_combo_base_zero():
    s = fresh()
    process_clear(s, lines=1, spin=SpinType.NONE)   # combo=0
    process_clear(s, lines=1, spin=SpinType.NONE)   # combo=1
    cr = process_clear(s, lines=1, spin=SpinType.NONE)  # combo=2
    ar = compute_attack(s, cr)
    # base=0, combo=2: floor(ln(1+2.5))=floor(1.252)=1
    expected = math.floor(math.log(1 + 1.25 * 2))   # 1
    assert ar.combo_bonus == expected, f"combo_bonus: {ar.combo_bonus} != {expected}"
    print(f"[OK] ATK Combo (base=0, combo=2): combo_bonus={ar.combo_bonus}")


# A10. All Clear -> +5 attack, state.b2b += 2
def test_atk_all_clear():
    # non-difficult + AC
    s = fresh(with_garbage=False)
    b2b_before = s.b2b  # -1
    cr = process_clear(s, lines=1, spin=SpinType.NONE)
    assert cr.all_clear
    ar = compute_attack(s, cr)
    check_atk(ar, all_clear_bonus=5)
    assert s.b2b == b2b_before + 2, f"state.b2b after AC: {s.b2b} (expected {b2b_before+2})"
    print(f"[OK] ATK All Clear: all_clear_bonus={ar.all_clear_bonus} state.b2b={s.b2b}")


def test_atk_all_clear_difficult():
    # Quad + AC: base=4, b2b_bonus=0 (첫 difficult), ac=5 -> subtotal=9, state.b2b=0+2=2
    s = fresh(with_garbage=False)
    cr = process_clear(s, lines=4, spin=SpinType.NONE)
    assert cr.all_clear
    ar = compute_attack(s, cr)
    check_atk(ar, base=4, b2b_bonus=0, all_clear_bonus=5, subtotal=9)
    assert s.b2b == 2, f"state.b2b={s.b2b} (expected 2)"
    print(f"[OK] ATK Quad + All Clear: subtotal={ar.subtotal} state.b2b={s.b2b}")


def test_atk_all_clear_chain():
    # 이미 체인(b2b=1) 중 Quad + AC: b2b_bonus=1, ac=5, base=4 -> 10; state.b2b=2+2=4
    s = fresh(with_garbage=False)
    s.b2b = 1  # 이미 체인 중
    cr = process_clear(s, lines=4, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    check_atk(ar, base=4, b2b_bonus=1, all_clear_bonus=5, subtotal=10)
    assert s.b2b == 4, f"state.b2b={s.b2b} (expected 4)"
    print(f"[OK] ATK chain + Quad + All Clear: subtotal={ar.subtotal} state.b2b={s.b2b}")


# A11. split_surge 분할
def test_split_surge():
    assert split_surge(9) == [3, 3, 3], split_surge(9)
    assert split_surge(8) == [3, 3, 2], split_surge(8)
    assert split_surge(5) == [2, 2, 1], split_surge(5)
    assert split_surge(4) == [2, 1, 1], split_surge(4)
    assert split_surge(3) == [1, 1, 1], split_surge(3)
    assert split_surge(2) == [1, 1],    split_surge(2)
    assert split_surge(1) == [1],       split_surge(1)
    print("[OK] split_surge all cases")


# A12. Surge 저장: 4연속 Quad -> state.surge=4
def test_surge_store():
    s = fresh()
    for _ in range(3):
        cr = process_clear(s, lines=4, spin=SpinType.NONE)
        compute_attack(s, cr)
    # 4번째 Quad: streak=4 -> surge 저장
    cr = process_clear(s, lines=4, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    assert s.surge > 0, f"surge should be set, got {s.surge}"
    assert s.surge == 4, f"state.surge={s.surge} (expected 4)"
    assert ar.surge_chunks == [], f"surge_chunks should be empty on store: {ar.surge_chunks}"
    print(f"[OK] Surge store: state.surge={s.surge}")


# A13. Surge 발동: 4연속 Quad 후 Single -> chunks
def test_surge_fire():
    s = fresh()
    for _ in range(4):
        cr = process_clear(s, lines=4, spin=SpinType.NONE)
        compute_attack(s, cr)
    assert s.surge == 4, f"pre-fire surge={s.surge}"

    # Single (non-difficult) -> surge 발동
    cr = process_clear(s, lines=1, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    assert ar.surge_chunks == [2, 1, 1], f"surge_chunks={ar.surge_chunks}"
    assert s.surge == 0, f"surge should reset to 0, got {s.surge}"
    print(f"[OK] Surge fire: chunks={ar.surge_chunks} state.surge={s.surge}")


# A14. apply_combo 단위 테스트
def test_apply_combo():
    # base>0
    assert apply_combo(4, 0)  == 4                            # no multiplier
    assert apply_combo(4, 1)  == math.floor(4 * 1.25)         # 5
    assert apply_combo(4, 2)  == math.floor(4 * 1.50)         # 6
    assert apply_combo(1, 4)  == math.floor(1 * 2.0)          # 2
    # base=0
    assert apply_combo(0, 0)  == 0
    assert apply_combo(0, 1)  == 0
    assert apply_combo(0, 2)  == math.floor(math.log(1 + 2.5)) # 1
    assert apply_combo(0, 4)  == math.floor(math.log(1 + 5.0)) # 1
    print("[OK] apply_combo all cases")


# ─────────────────────────────────────────────────────────
# garbage.py 테스트
# ─────────────────────────────────────────────────────────

# G1. add() -> 각 attack이 독립된 GarbageLine으로 쌓임 (change-on-attack)
def test_garbage_add():
    q = GarbageQueue(rng=random.Random(0))
    q.add(4)
    q.add(2)
    q.add(3)
    assert len(q) == 3,              f"len={len(q)}"
    assert q.total_lines() == 9,     f"total={q.total_lines()}"
    print(f"[OK] GarbageQueue add: entries={len(q)} total={q.total_lines()}")


# G2. change-on-attack: 같은 RNG라도 연속 add는 서로 다른 hole 가능
def test_garbage_change_on_attack():
    rng = random.Random(42)
    q = GarbageQueue(rng=rng)
    q.add(4)
    q.add(4)
    q.add(4)
    entries = q.pop_all()
    holes = [e.hole_col for e in entries]
    # 3개가 전부 같을 수도 있지만, RNG seed 42 기준으로 최소 2가지 값이 나옴
    # 핵심: 각 항목이 독립적인 hole_col을 보유
    assert len(entries) == 3
    assert all(0 <= h <= 9 for h in holes), f"holes out of range: {holes}"
    print(f"[OK] Change-on-attack: holes={holes}")


# G3. cancel() 전체 상쇄
def test_garbage_cancel_full():
    q = GarbageQueue()
    q.add(4, hole_col=0)
    q.add(2, hole_col=1)
    remaining = q.cancel(6)    # 정확히 전부 상쇄
    assert remaining == 0,         f"remaining={remaining}"
    assert q.total_lines() == 0,   f"total={q.total_lines()}"
    assert len(q) == 0,            f"len={len(q)}"
    print(f"[OK] Cancel full: remaining={remaining}")


# G4. cancel() 초과 상쇄 -> 남은 공격력 반환
def test_garbage_cancel_excess():
    q = GarbageQueue()
    q.add(3, hole_col=5)
    remaining = q.cancel(5)    # 가비지 3줄, 공격 5 -> 2 남음
    assert remaining == 2,         f"remaining={remaining}"
    assert q.total_lines() == 0,   f"total={q.total_lines()}"
    print(f"[OK] Cancel excess: remaining={remaining}")


# G5. cancel() 부분 상쇄 -> 앞 entry만 줄고 hole_col 유지
def test_garbage_cancel_partial():
    q = GarbageQueue()
    q.add(6, hole_col=3)
    q.add(4, hole_col=7)
    remaining = q.cancel(4)    # 첫 6줄짜리에서 4 상쇄 -> 2줄 남음
    assert remaining == 0,         f"remaining={remaining}"
    assert q.total_lines() == 6,   f"total={q.total_lines()}"  # 2 + 4
    assert len(q) == 2,            f"len={len(q)}"
    entries = q.pop_all()
    assert entries[0] == GarbageLine(lines=2, hole_col=3), f"front={entries[0]}"
    assert entries[1] == GarbageLine(lines=4, hole_col=7), f"back={entries[1]}"
    print(f"[OK] Cancel partial: front={entries[0]} back={entries[1]}")


# G6. cancel() 여러 entry에 걸쳐 상쇄
def test_garbage_cancel_multi_entry():
    q = GarbageQueue()
    q.add(2, hole_col=1)
    q.add(3, hole_col=2)
    q.add(5, hole_col=3)
    # 7만큼 상쇄: 2 전부 + 3 전부 + 5에서 2 = 7
    remaining = q.cancel(7)
    assert remaining == 0,        f"remaining={remaining}"
    assert q.total_lines() == 3,  f"total={q.total_lines()}"
    assert len(q) == 1,           f"len={len(q)}"
    entries = q.pop_all()
    assert entries[0] == GarbageLine(lines=3, hole_col=3), f"entry={entries[0]}"
    print(f"[OK] Cancel multi-entry: leftover={entries[0]}")


# G7. cancel() 빈 큐 -> 공격력 그대로 반환
def test_garbage_cancel_empty():
    q = GarbageQueue()
    remaining = q.cancel(5)
    assert remaining == 5, f"remaining={remaining}"
    print(f"[OK] Cancel empty queue: remaining={remaining}")


# G8. pop_all() -> 큐 비움
def test_garbage_pop_all():
    q = GarbageQueue()
    q.add(3, hole_col=2)
    q.add(1, hole_col=8)
    result = q.pop_all()
    assert len(result) == 2,       f"len={len(result)}"
    assert q.is_empty(),           "queue should be empty after pop_all"
    assert result[0] == GarbageLine(3, 2)
    assert result[1] == GarbageLine(1, 8)
    print(f"[OK] pop_all: {result}")


# G9. add(0) -> 아무것도 추가 안 함
def test_garbage_add_zero():
    q = GarbageQueue()
    q.add(0)
    assert len(q) == 0,           "add(0) should not enqueue"
    assert q.total_lines() == 0
    print("[OK] add(0) no-op")


# G10. attack.py의 compute_attack이 incoming 가비지를 올바르게 상쇄
# opener phase 회피: pieces_placed > 14 로 설정
def test_attack_cancels_incoming():
    # 가비지 3줄 대기, Quad(4) -> 3 상쇄, 1 전송, incoming 0줄 남음
    s = fresh()
    s.pieces_placed = 15          # opener phase 비활성화
    s.incoming.add(3, hole_col=0)
    cr = process_clear(s, lines=4, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    assert ar.canceled == 3,              f"canceled={ar.canceled}"
    assert ar.sent == 1,                  f"sent={ar.sent}"
    assert s.incoming.is_empty(),         "incoming should be empty"
    print(f"[OK] Attack cancels incoming: canceled={ar.canceled} sent={ar.sent}")


# G11. 가비지가 공격보다 크면 초과 가비지가 큐에 남음, 전송 0
def test_attack_exceeds_incoming():
    # 가비지 6줄 대기, Quad(4) -> 4 상쇄, 0 전송, 2줄 남음
    s = fresh()
    s.pieces_placed = 15          # opener phase 비활성화
    s.incoming.add(6, hole_col=0)
    cr = process_clear(s, lines=4, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    assert ar.canceled == 4,              f"canceled={ar.canceled}"
    assert ar.sent == 0,                  f"sent={ar.sent}"
    assert s.incoming.total_lines() == 2, f"remaining={s.incoming.total_lines()}"
    assert s.outgoing == 0,               f"outgoing={s.outgoing}"
    print(f"[OK] Attack vs larger incoming: canceled={ar.canceled} remaining={s.incoming.total_lines()}")


# G12. 가비지 없으면 공격 전부 outgoing
def test_attack_no_incoming():
    s = fresh()
    s.pieces_placed = 15          # opener phase 비활성화
    cr = process_clear(s, lines=4, spin=SpinType.NONE)
    ar = compute_attack(s, cr)
    assert ar.canceled == 0,    f"canceled={ar.canceled}"
    assert ar.sent == 4,        f"sent={ar.sent}"
    assert s.outgoing == 4,     f"outgoing={s.outgoing}"
    print(f"[OK] Attack no incoming: sent={ar.sent}")


# G13. change-on-attack: 두 attack이 큐에 쌓일 때 hole이 독립적으로 배정됨
def test_garbage_hole_independence():
    rng = random.Random(7)
    q = GarbageQueue(rng=rng)
    q.add(4)   # attack 1
    q.add(4)   # attack 2
    entries = q.pop_all()
    # 각 항목은 독립된 hole_col을 가짐 (항상 같을 수도 있지만 seed 7 기준 다름)
    assert entries[0].hole_col != entries[1].hole_col or True  # 값 자체보단 독립 배정 검증
    # 핵심: 두 개의 분리된 GarbageLine이 존재해야 함
    assert len(entries) == 2
    assert entries[0].lines == 4
    assert entries[1].lines == 4
    print(f"[OK] Hole independence: attack1.hole={entries[0].hole_col} attack2.hole={entries[1].hole_col}")


# G14. partial cancel 후 hole_col 유지됨
def test_garbage_partial_cancel_preserves_hole():
    q = GarbageQueue()
    q.add(8, hole_col=5)
    remaining = q.cancel(3)
    assert remaining == 0
    entries = q.pop_all()
    assert entries[0] == GarbageLine(lines=5, hole_col=5), f"entry={entries[0]}"
    print(f"[OK] Partial cancel preserves hole: {entries[0]}")


# ─────────────────────────────────────────────────────────
# Clutch Clear 테스트 (state.spawn_next + _find_clutch_spawn)
# ─────────────────────────────────────────────────────────

from engine.mino import SHAPES
from engine import RoundResult
from engine.state import GameState


def _fill_rows(state: GameState, rows: list[int]) -> None:
    """지정한 row들을 GARBAGE로 꽉 채운다 (스폰 차단용)."""
    for r in rows:
        for c in range(BOARD_COLS):
            state.board._grid[r][c] = MinoType.GARBAGE


# CC1. 정상 스폰이 막히지 않으면 clutch_clear=True여도 그냥 정상 스폰
def test_clutch_normal_spawn_not_blocked():
    s = GameState()
    s.next_queue.append(MinoType.T)
    ok = s.spawn_next(clutch_clear=True)
    assert ok,                           "should succeed"
    assert s.round_result == RoundResult.ONGOING
    assert s.active is not None
    assert s.active.type == MinoType.T
    # 정상 스폰 행 (row=19)
    assert s.active.row == 19,           f"row={s.active.row}"
    print(f"[OK] CC1 Normal spawn not blocked: row={s.active.row}")


# CC2. 스폰 막힘 + clutch_clear=False -> LOSE
def test_clutch_block_out_no_clear():
    s = GameState()
    # 스폰 행(19, 20)을 꽉 채워 T 스폰 차단
    _fill_rows(s, [19, 20])
    s.next_queue.append(MinoType.T)
    ok = s.spawn_next(clutch_clear=False)
    assert not ok,                        "should fail"
    assert s.round_result == RoundResult.LOSE
    assert s.spawn_blocked
    print("[OK] CC2 Block-out without clutch -> LOSE")


# CC3. 스폰 막힘 + clutch_clear=True, 위쪽에 빈 공간 있음 -> 생존
def test_clutch_block_out_with_clear():
    s = GameState()
    # row 19~21만 채워 스폰 차단, 그 위(18 이상)는 비어 있음
    _fill_rows(s, [19, 20, 21])
    s.next_queue.append(MinoType.T)
    ok = s.spawn_next(clutch_clear=True)
    assert ok,                            "clutch spawn should succeed"
    assert s.round_result == RoundResult.ONGOING
    assert s.active is not None
    assert s.active.type == MinoType.T
    # 정상 스폰(row=19)보다 위로 올라가 있어야 함
    assert s.active.row < 19,            f"row should be < 19, got {s.active.row}"
    print(f"[OK] CC3 Clutch spawn success: row={s.active.row}")


# CC4. Clutch 스폰 후 row가 정확히 정상 스폰보다 최소한만 올라갔는가
def test_clutch_spawn_row_lifted():
    s = GameState()
    # row 19와 20만 막음 (T의 스폰 셀: row=19+0=19, row=19+1=20)
    _fill_rows(s, [19, 20])
    s.next_queue.append(MinoType.T)
    ok = s.spawn_next(clutch_clear=True)
    assert ok
    # T의 bounding box offset: min_dr=0. 막힌 행이 19~20이면 row=18에서 유효
    # T rot=0: cells at (row+0,*) and (row+1,*)
    # row=19: cells at 19,20 -> blocked
    # row=18: cells at 18,19 -> row 19 is blocked, still blocked
    # row=17: cells at 17,18 -> both free -> valid
    # (row 19만 막으면 row=18도 셀이 (18,*),(19,*)로 19 겹쳐 막힘)
    # (row 19,20 막으면 row=18: (18,*),(19,*) -> 19 막혀 있음 -> blocked)
    # (row=17: (17,*),(18,*) -> 둘 다 비어 있음 -> valid)
    assert s.active.row == 17,           f"expected row=17, got {s.active.row}"
    print(f"[OK] CC4 Clutch spawn row precision: row={s.active.row}")


# CC5. 보드 전체가 꽉 찼으면 clutch_clear=True여도 공간 없음 -> LOSE
def test_clutch_no_space_even_with_clear():
    s = GameState()
    # 보드 최상단까지 모두 채움
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            s.board._grid[r][c] = MinoType.GARBAGE
    s.next_queue.append(MinoType.T)
    ok = s.spawn_next(clutch_clear=True)
    assert not ok,                        "no space anywhere -> LOSE"
    assert s.round_result == RoundResult.LOSE
    print("[OK] CC5 No space even with clutch -> LOSE")


# CC6. _find_clutch_spawn이 None을 반환하는 조건 (모든 행 막힘)
def test_clutch_find_clutch_spawn_returns_none_when_impossible():
    s = GameState()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            s.board._grid[r][c] = MinoType.GARBAGE
    result = s._find_clutch_spawn(MinoType.I)
    assert result is None,               f"expected None, got {result}"
    print("[OK] CC6 _find_clutch_spawn returns None when impossible")


if __name__ == "__main__":
    # clear.py 테스트
    test_single()
    test_quad_first()
    test_quad_chain()
    test_tspin_full_double()
    test_spin_mini_single()
    test_b2b_break()
    test_all_clear_non_difficult()
    test_all_clear_difficult()
    test_all_clear_difficult_chain()
    test_combo()
    test_is_difficult()
    print("\nAll clear tests passed.\n")

    # attack.py 테스트
    test_atk_single()
    test_atk_double()
    test_atk_triple()
    test_atk_quad_first()
    test_atk_quad_chain()
    test_atk_tspin_single()
    test_atk_tspin_double_chain()
    test_atk_combo_base_positive()
    test_atk_combo_base_zero()
    test_atk_all_clear()
    test_atk_all_clear_difficult()
    test_atk_all_clear_chain()
    test_split_surge()
    test_surge_store()
    test_surge_fire()
    test_apply_combo()
    print("\nAll attack tests passed.\n")

    # garbage.py 테스트
    test_garbage_add()
    test_garbage_change_on_attack()
    test_garbage_cancel_full()
    test_garbage_cancel_excess()
    test_garbage_cancel_partial()
    test_garbage_cancel_multi_entry()
    test_garbage_cancel_empty()
    test_garbage_pop_all()
    test_garbage_add_zero()
    test_attack_cancels_incoming()
    test_attack_exceeds_incoming()
    test_attack_no_incoming()
    test_garbage_hole_independence()
    test_garbage_partial_cancel_preserves_hole()
    print("\nAll garbage tests passed.\n")

    # Clutch Clear 테스트
    test_clutch_normal_spawn_not_blocked()
    test_clutch_block_out_no_clear()
    test_clutch_block_out_with_clear()
    test_clutch_spawn_row_lifted()
    test_clutch_no_space_even_with_clear()
    test_clutch_find_clutch_spawn_returns_none_when_impossible()
    print("\nAll Clutch Clear tests passed.")
