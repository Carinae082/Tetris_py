# game.py ??Tetris with Home / Settings / Game screens
import json
import os
import pygame
import sys
import threading
from enum import Enum, auto

from engine import (
    GameState, SevenBag, RoundResult,
    process_clear, compute_attack,
    BOARD_COLS, VISIBLE_ROW_START,
    try_move, try_rotate, hard_drop, soft_drop_step,
    gravity_tick, LOCK_DELAY,
)
from engine.mino import MinoType, ActivePiece, SHAPES
from engine.types import SpinType

from bot.beam_search_bot import BeamSearchBot
from bot.search_config import SearchConfig
from bot.adapter import BotEnvAdapter, Action as BotAction
from bot.placement_generator import list_reachable_placements
from bot.rl_trainer import (
    RLTrainer, TrainConfig, GenerationResult,
    WEIGHT_NAMES, WEIGHT_DEFAULTS, WEIGHT_BOUNDS, N_WEIGHTS,
    load_weights, weights_to_objects,
)

# ?? Layout constants ????????????????????????????????????????????????????????????

CELL              = 28
VISIBLE_ROWS      = 22                        # rows shown on screen
DISPLAY_ROW_START = VISIBLE_ROW_START - (VISIBLE_ROWS - 20)  # = 18
BOARD_W           = CELL * BOARD_COLS         # 280
BOARD_H           = CELL * VISIBLE_ROWS       # 672

LEFT_W    = 160    # left sidebar width  (~25% of content area)
RIGHT_W   = 148    # right sidebar width (~23% of content area)
H_GAP     = 16     # horizontal gap between sidebar and board
MARGIN_X  = 18     # outer horizontal margin
MARGIN_Y  = 20     # top/bottom padding

# Derived positions
WINDOW_W  = MARGIN_X + LEFT_W + H_GAP + BOARD_W + H_GAP + RIGHT_W + MARGIN_X  # 660
WINDOW_H  = BOARD_H + MARGIN_Y * 2                                               # 600

BOARD_OX  = MARGIN_X + LEFT_W + H_GAP   # 194
BOARD_OY  = MARGIN_Y                     # 20

LEFT_X    = MARGIN_X                     # 18
RIGHT_X   = BOARD_OX + BOARD_W + H_GAP  # 490

CLEAR_DISPLAY_SECS = 2.0   # seconds to display clear result text

# ?? Sound loading ????????????????????????????????????????????????????????????????

_SND_RATE   = 44100
# Attack subtotal threshold that triggers the power sound layer
_CRUNCH_ATK = 4
# Directory containing the OGG sound effects
_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "plorald_sfx_volume_minus_11db")


def build_sounds() -> dict:
    """
    Load all OGG sound effects from plorald_sfx_volume_minus_11db/.
    Returns empty dict if the directory or files are missing.
    """
    def _load(name: str):
        path = os.path.join(_SFX_DIR, name)
        if os.path.exists(path):
            try:
                return pygame.mixer.Sound(path)
            except Exception:
                return None
        return None

    sounds: dict = {}

    # Combo sounds: combo_1.ogg ??combo_16.ogg
    # combo_N_power.ogg plays when the attack spike is large
    for n in range(1, 17):
        sounds[('clear', n)]       = _load(f"combo_{n}.ogg")
        sounds[('clear_power', n)] = _load(f"combo_{n}_power.ogg")

    # Back-to-back chain sounds: btb_1 / btb_2 / btb_3
    for n in range(1, 4):
        sounds[('btb', n)] = _load(f"btb_{n}.ogg")
    sounds['btb_break'] = _load("btb_break.ogg")

    # One-shot event sounds
    sounds['allclear'] = _load("allclear.ogg")
    sounds['gameover'] = _load("gameover.ogg")
    sounds['warning']  = _load("warning.ogg")
    sounds['clutch']   = _load("clutch.ogg")

    return sounds

# ?? Colors ??????????????????????????????????????????????????????????????????????

COLORS = {
    MinoType.I:       (0, 240, 240),
    MinoType.J:       (30, 100, 240),
    MinoType.L:       (240, 160, 0),
    MinoType.O:       (240, 240, 0),
    MinoType.S:       (0, 220, 0),
    MinoType.T:       (160, 0, 240),
    MinoType.Z:       (240, 40, 40),
    MinoType.GARBAGE: (100, 100, 100),
}

BG           = (0, 0, 0)           # pure black background
GRID_COLOR   = (45, 45, 45)        # subtle grid lines
PANEL_BG     = (8, 8, 8)           # near-black panel fill
BORDER_COLOR = (255, 255, 255)     # white panel/board borders
TEXT_COLOR   = (255, 255, 255)     # primary white text
DIM_COLOR    = (110, 110, 110)     # dimmed labels
STACK_COLOR  = (52, 52, 52)        # dark gray for locked cells
STACK_BORDER = (70, 70, 70)        # border of locked cells
SPIN_COLOR   = (255, 60, 200)      # magenta for spin qualifier
COMBO_COLOR  = (120, 255, 60)      # lime for combo
B2B_COLOR    = (120, 255, 60)      # lime for B2B
ACCENT       = (100, 180, 255)     # blue accent (home/settings)
ACCENT2      = (255, 200, 60)      # yellow accent (home/settings)
LOCK_BAR_BG  = (35, 35, 35)
LOCK_BAR_FG  = (180, 180, 255)
BTN_NORMAL   = (20, 20, 20)
BTN_HOVER    = (40, 40, 40)
BTN_BORDER   = (80, 80, 80)
BTN_ACTIVE   = (50, 80, 140)
SLOT_EMPTY   = (15, 15, 15)

# ?? Screen enum ?????????????????????????????????????????????????????????????????

class Screen(Enum):
    HOME        = auto()
    SETTINGS    = auto()
    MODE_SELECT = auto()
    GAME        = auto()
    BOT         = auto()
    TRAINING    = auto()

# ?? Key-binding schema ??????????????????????????????????????????????????????????

_KEY_ACTIONS = [
    "Move Left", "Move Right",
    "Rotate CW", "Rotate CCW", "Rotate 180",
    "Hard Drop", "Soft Drop", "Hold",
]

_DEFAULT_PRIMARIES = {
    "Move Left":  pygame.K_LEFT,
    "Move Right": pygame.K_RIGHT,
    "Rotate CW":  pygame.K_UP,
    "Rotate CCW": pygame.K_z,
    "Rotate 180": pygame.K_a,
    "Hard Drop":  pygame.K_SPACE,
    "Soft Drop":  pygame.K_DOWN,
    "Hold":       pygame.K_c,
}

def _default_keys() -> dict[str, list]:
    return {a: [_DEFAULT_PRIMARIES[a], None, None] for a in _KEY_ACTIONS}

# ?? Persistent settings ?????????????????????????????????????????????????????????

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

def _load_settings() -> dict:
    defaults = {"das": 133, "arr": 0, "sdf": 1, "keys": _default_keys()}
    if not os.path.exists(SETTINGS_PATH):
        return defaults
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = defaults.copy()
        merged["das"] = int(data.get("das", merged["das"]))
        merged["arr"] = int(data.get("arr", merged["arr"]))
        merged["sdf"] = int(data.get("sdf", merged["sdf"]))
        loaded_keys = data.get("keys", {})
        for action in _KEY_ACTIONS:
            if action in loaded_keys:
                slots = loaded_keys[action]
                padded = [(int(s) if s is not None else None)
                          for s in (slots + [None, None, None])[:3]]
                merged["keys"][action] = padded
        return merged
    except Exception:
        return defaults

def _save_settings(s: dict) -> None:
    try:
        serializable = {"das": s["das"], "arr": s["arr"], "sdf": s["sdf"], "keys": s["keys"]}
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    except Exception:
        pass

settings = _load_settings()

def _key_matches(action: str, k: int) -> bool:
    return any(s == k for s in settings["keys"][action] if s is not None)

# ?? Helpers ?????????????????????????????????????????????????????????????????????

def _darken(color, factor: float = 0.6):
    return tuple(max(0, int(c * factor)) for c in color)

def _ghost_color(color):
    return tuple(max(0, int(c * 0.6)) for c in color)

def _key_name(k: int | None) -> str:
    if k is None:
        return "---"
    name = pygame.key.name(k)
    return name.upper() if len(name) == 1 else name.title()

def _ghost_cells(state: GameState):
    p = state.active
    if p is None:
        return []
    ghost = ActivePiece(p.type, p.row, p.col, p.rotation)
    while True:
        below = ActivePiece(ghost.type, ghost.row + 1, ghost.col, ghost.rotation)
        if not state.board.is_valid_position(below):
            break
        ghost = below
    return ghost.cells()

def _draw_mini_piece(surface, cells, color, box_x, box_y, box_w, box_h):
    if not cells:
        return
    min_r = min(r for r, c in cells)
    max_r = max(r for r, c in cells)
    min_c = min(c for r, c in cells)
    max_c = max(c for r, c in cells)
    shape_h = max_r - min_r + 1
    shape_w = max_c - min_c + 1
    mini = min(box_w // max(shape_w, 1), box_h // max(shape_h, 1), 20)
    ox = box_x + (box_w - shape_w * mini) // 2
    oy = box_y + (box_h - shape_h * mini) // 2
    for r, c in cells:
        px = ox + (c - min_c) * mini
        py = oy + (r - min_r) * mini
        pygame.draw.rect(surface, color, (px, py, mini, mini))
        pygame.draw.rect(surface, _darken(color), (px, py, mini, mini), 1)

# ?? Clear text helper ????????????????????????????????????????????????????????????

_CLEAR_NAMES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "QUAD"}

def _build_clear_labels(lines: int, spin: SpinType, all_clear: bool,
                        mino_type=None) -> tuple[str, str]:
    """Return (spin_label, clear_label). Empty string means don't display."""
    if lines == 0:
        return "", ""
    if all_clear:
        return "", ""   # displayed as board overlay instead
    piece = mino_type.name if mino_type is not None else ""
    if spin == SpinType.FULL:
        spin_lbl  = f"{piece}-SPIN" if piece else "SPIN"
        clear_lbl = _CLEAR_NAMES.get(lines, f"{lines}-LINE")
    elif spin == SpinType.MINI:
        spin_lbl  = f"{piece}-SPIN MINI" if piece else "SPIN MINI"
        clear_lbl = _CLEAR_NAMES.get(lines, f"{lines}-LINE")
    else:
        spin_lbl  = ""
        clear_lbl = _CLEAR_NAMES.get(lines, f"{lines}-LINE")
    return spin_lbl, clear_lbl

# ?? UI widgets ???????????????????????????????????????????????????????????????????

class Button:
    def __init__(self, rect, label, font):
        self.rect  = pygame.Rect(rect)
        self.label = label
        self.font  = font

    def draw(self, surface, mouse_pos, active=False):
        hovered = self.rect.collidepoint(mouse_pos)
        bg = BTN_ACTIVE if active else (BTN_HOVER if hovered else BTN_NORMAL)
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, BTN_BORDER, self.rect, 2, border_radius=6)
        txt = self.font.render(self.label, True, TEXT_COLOR)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))


class Slider:
    def __init__(self, x, y, w, label, lo, hi, value, font_sm, unit="ms",
                 hide_value=False):
        self.X, self.Y, self.W = x, y, w
        self.track = pygame.Rect(x, y, w, 6)
        self.label      = label
        self.lo, self.hi = lo, hi
        self.value      = value
        self.font       = font_sm
        self.unit       = unit
        self.hide_value = hide_value
        self._drag      = False

    def _handle_x(self):
        ratio = (self.value - self.lo) / max(self.hi - self.lo, 1)
        return int(self.X + ratio * self.W)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hx = self._handle_x()
            if pygame.Rect(hx - 9, self.Y - 8, 18, 22).collidepoint(event.pos):
                self._drag = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag = False
        elif event.type == pygame.MOUSEMOTION and self._drag:
            ratio = max(0.0, min(1.0, (event.pos[0] - self.X) / self.W))
            self.value = int(self.lo + ratio * (self.hi - self.lo))

    def draw(self, surface, _mouse=None):
        surface.blit(self.font.render(self.label, True, DIM_COLOR),
                     (self.X, self.Y - 22))
        if not self.hide_value:
            val_str = f"{self.value} {self.unit}" if self.value > 0 else "0 (instant)"
            surface.blit(self.font.render(val_str, True, ACCENT),
                         (self.X + self.W + 12, self.Y - 6))
        pygame.draw.rect(surface, BTN_NORMAL, self.track, border_radius=3)
        filled_w = self._handle_x() - self.X
        if filled_w > 0:
            pygame.draw.rect(surface, ACCENT,
                             pygame.Rect(self.X, self.Y, filled_w, 6), border_radius=3)
        hx = self._handle_x()
        pygame.draw.circle(surface, ACCENT,    (hx, self.Y + 3), 9)
        pygame.draw.circle(surface, TEXT_COLOR, (hx, self.Y + 3), 5)

# ?? Home screen ??????????????????????????????????????????????????????????????????

class HomeScreen:
    def __init__(self, fonts):
        _, self._font_lg, self._font_md, self._font_sm = fonts
        cx = WINDOW_W // 2
        self._btn_play    = Button((cx - 100, 330, 200, 52), "PLAY",     self._font_md)
        self._btn_setting = Button((cx - 100, 400, 200, 52), "SETTINGS", self._font_md)
        self._anim_t = 0.0

    def handle_event(self, event):
        if self._btn_play.is_clicked(event):    return Screen.MODE_SELECT
        if self._btn_setting.is_clicked(event): return Screen.SETTINGS
        return None

    def update(self, dt): self._anim_t += dt

    def draw(self, surface):
        mouse = pygame.mouse.get_pos()
        title = self._font_lg.render("TETRIS", True, ACCENT)
        surface.blit(title, title.get_rect(center=(WINDOW_W // 2, 170)))
        sub = self._font_sm.render("TETR.IO Season 2 rules", True, DIM_COLOR)
        surface.blit(sub, sub.get_rect(center=(WINDOW_W // 2, 215)))
        for i, mino in enumerate([MinoType.T, MinoType.I, MinoType.S, MinoType.Z]):
            cx = int(WINDOW_W // 2 + (i - 1.5) * 60
                     + 7 * pygame.math.Vector2(1, 0).rotate(self._anim_t * 25 + i * 50).x)
            _draw_mini_piece(surface, SHAPES[mino][0], COLORS[mino], cx - 28, 248, 56, 40)
        self._btn_play.draw(surface, mouse)
        self._btn_setting.draw(surface, mouse)
        hint = self._font_sm.render("ESC to quit", True, DIM_COLOR)
        surface.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H - 28)))

# ?? Mode select screen ???????????????????????????????????????????????????????????

class ModeSelectScreen:
    """PLAY瑜??꾨Ⅸ ??SOLO / BOT / TRAINING 以??섎굹瑜?怨좊Ⅴ???붾㈃."""

    def __init__(self, fonts):
        _, self._font_lg, self._font_md, self._font_sm = fonts
        cx = WINDOW_W // 2
        self._btn_back     = Button((18, 14, 90, 32), "< BACK", self._font_sm)
        self._btn_solo     = Button((cx - 120, 210, 240, 56), "SOLO",     self._font_md)
        self._btn_bot      = Button((cx - 120, 300, 240, 56), "BOT",      self._font_md)
        self._btn_training = Button((cx - 120, 390, 240, 56), "TRAINING", self._font_md)

    def handle_event(self, event):
        if self._btn_back.is_clicked(event):     return Screen.HOME
        if self._btn_solo.is_clicked(event):     return Screen.GAME
        if self._btn_bot.is_clicked(event):      return Screen.BOT
        if self._btn_training.is_clicked(event): return Screen.TRAINING
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return Screen.HOME
        return None

    def update(self, dt): pass

    def draw(self, surface):
        mouse = pygame.mouse.get_pos()

        title = self._font_lg.render("SELECT MODE", True, ACCENT)
        surface.blit(title, title.get_rect(center=(WINDOW_W // 2, 130)))

        self._btn_solo.draw(surface, mouse)
        solo_desc = self._font_sm.render("Manual play", True, DIM_COLOR)
        surface.blit(solo_desc, solo_desc.get_rect(center=(WINDOW_W // 2, 272)))

        self._btn_bot.draw(surface, mouse)
        bot_desc = self._font_sm.render("Watch AI play", True, DIM_COLOR)
        surface.blit(bot_desc, bot_desc.get_rect(center=(WINDOW_W // 2, 362)))

        self._btn_training.draw(surface, mouse)
        tr_desc = self._font_sm.render("媛以묒튂 媛뺥솕?숈뒿", True, DIM_COLOR)
        surface.blit(tr_desc, tr_desc.get_rect(center=(WINDOW_W // 2, 452)))

        self._btn_back.draw(surface, mouse)

        hint = self._font_sm.render("ESC: ?ㅻ줈", True, DIM_COLOR)
        surface.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H - 28)))


# ?? Settings screen ??????????????????????????????????????????????????????????????

_SLOT_LABELS = ["1st", "2nd", "3rd"]
_SLOT_W      = 82
_SLOT_GAP    = 4
_ROW_H       = 36
_ROW_TOP     = 258   # pushed down to fit 3 sensitivity sliders
_LABEL_X     = 36
_LABEL_W     = 148
_SLOTS_X     = _LABEL_X + _LABEL_W + 8

class SettingsScreen:
    def __init__(self, fonts):
        _, self._font_lg, self._font_md, self._font_sm = fonts
        self._btn_back  = Button((18, 14, 90, 32), "< BACK", self._font_sm)
        self._btn_reset = Button((WINDOW_W - 128, WINDOW_H - 36, 110, 28),
                                 "RESET", self._font_sm)
        self._slider_das = Slider(180, 110, 200, "DAS", 0, 300,
                                  settings["das"], self._font_sm)
        self._slider_arr = Slider(180, 158, 200, "ARR", 0, 100,
                                  settings["arr"], self._font_sm)
        self._slider_sdf = Slider(180, 206, 200, "SDF", 1, 41,
                                  settings["sdf"], self._font_sm, unit="",
                                  hide_value=True)
        self._rebinding: tuple[str, int] | None = None

    def handle_event(self, event):
        if self._btn_back.is_clicked(event):
            self._rebinding = None
            _save_settings(settings)
            return Screen.HOME

        if event.type == pygame.KEYDOWN:
            if self._rebinding is not None:
                action, slot = self._rebinding
                self._rebinding = None
                if event.key == pygame.K_ESCAPE:
                    pass
                elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and slot > 0:
                    settings["keys"][action][slot] = None
                    _save_settings(settings)
                else:
                    for a in _KEY_ACTIONS:
                        for i in range(3):
                            if settings["keys"][a][i] == event.key:
                                if not (a == action and i == slot):
                                    settings["keys"][a][i] = None
                    settings["keys"][action][slot] = event.key
                    _save_settings(settings)
                return None
            if event.key == pygame.K_ESCAPE:
                _save_settings(settings)
                return Screen.HOME

        if self._rebinding is None:
            self._slider_das.handle_event(event)
            self._slider_arr.handle_event(event)
            self._slider_sdf.handle_event(event)
            settings["das"] = self._slider_das.value
            settings["arr"] = self._slider_arr.value
            settings["sdf"] = self._slider_sdf.value

        if self._btn_reset.is_clicked(event):
            settings["keys"] = _default_keys()
            settings["das"]  = 133
            settings["arr"]  = 0
            settings["sdf"]  = 1
            self._slider_das.value = 133
            self._slider_arr.value = 0
            self._slider_sdf.value = 1
            self._rebinding = None
            _save_settings(settings)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, action in enumerate(_KEY_ACTIONS):
                row_y = _ROW_TOP + i * (_ROW_H + 2)
                for slot in range(3):
                    bx = _SLOTS_X + slot * (_SLOT_W + _SLOT_GAP)
                    btn = pygame.Rect(bx, row_y + 3, _SLOT_W, _ROW_H - 6)
                    if btn.collidepoint(event.pos):
                        self._rebinding = (action, slot)

        return None

    def draw(self, surface):
        mouse = pygame.mouse.get_pos()
        title = self._font_md.render("SETTINGS", True, ACCENT)
        surface.blit(title, title.get_rect(center=(WINDOW_W // 2, 40)))
        self._btn_back.draw(surface, mouse)

        sec = self._font_sm.render("SENSITIVITY", True, DIM_COLOR)
        surface.blit(sec, (_LABEL_X, 90))
        pygame.draw.line(surface, GRID_COLOR, (_LABEL_X, 104), (WINDOW_W - _LABEL_X, 104))

        surface.blit(self._font_sm.render("Delayed Auto Shift", True, DIM_COLOR),
                     (_LABEL_X, 100))
        self._slider_das.draw(surface)

        surface.blit(self._font_sm.render("Auto Repeat Rate", True, DIM_COLOR),
                     (_LABEL_X, 148))
        self._slider_arr.draw(surface)

        surface.blit(self._font_sm.render("Soft Drop Factor", True, DIM_COLOR),
                     (_LABEL_X, 196))
        self._slider_sdf.draw(surface)
        # Override the default unit display with SDF-specific label
        sdf_val = self._slider_sdf.value
        sdf_str = "infinite" if sdf_val >= 41 else str(sdf_val)
        sdf_surf = self._font_sm.render(sdf_str, True, ACCENT)
        surface.blit(sdf_surf, (180 + 200 + 12, 196 + 4))

        kb = self._font_sm.render("KEY BINDINGS", True, DIM_COLOR)
        surface.blit(kb, (_LABEL_X, 232))
        pygame.draw.line(surface, GRID_COLOR, (_LABEL_X, 246), (WINDOW_W - _LABEL_X, 246))

        for si, label in enumerate(_SLOT_LABELS):
            hx = _SLOTS_X + si * (_SLOT_W + _SLOT_GAP) + _SLOT_W // 2
            hdr = self._font_sm.render(label, True, DIM_COLOR)
            surface.blit(hdr, hdr.get_rect(center=(hx, _ROW_TOP - 12)))

        for i, action in enumerate(_KEY_ACTIONS):
            row_y = _ROW_TOP + i * (_ROW_H + 2)
            if i % 2 == 0:
                pygame.draw.rect(surface, (22, 22, 35),
                                 pygame.Rect(_LABEL_X - 4, row_y,
                                             WINDOW_W - 2 * _LABEL_X + 8, _ROW_H),
                                 border_radius=4)
            surface.blit(self._font_sm.render(action, True, TEXT_COLOR),
                         (_LABEL_X, row_y + 10))
            slots = settings["keys"][action]
            for slot, key_val in enumerate(slots):
                bx  = _SLOTS_X + slot * (_SLOT_W + _SLOT_GAP)
                btn = pygame.Rect(bx, row_y + 3, _SLOT_W, _ROW_H - 6)
                is_rebinding = (self._rebinding == (action, slot))
                is_empty     = (key_val is None)
                is_primary   = (slot == 0)
                if is_rebinding:
                    bg, border = BTN_ACTIVE, ACCENT2
                elif is_empty:
                    bg, border = SLOT_EMPTY, GRID_COLOR
                elif btn.collidepoint(mouse):
                    bg, border = BTN_HOVER, BTN_BORDER
                else:
                    bg, border = BTN_NORMAL, BTN_BORDER
                pygame.draw.rect(surface, bg, btn, border_radius=5)
                pygame.draw.rect(surface, border, btn, 1, border_radius=5)
                if is_rebinding:
                    txt = self._font_sm.render("...", True, ACCENT2)
                elif is_empty and not is_primary:
                    txt = self._font_sm.render("+", True, DIM_COLOR)
                else:
                    color = ACCENT if not is_empty else DIM_COLOR
                    txt = self._font_sm.render(_key_name(key_val), True, color)
                surface.blit(txt, txt.get_rect(center=btn.center))

        self._btn_reset.draw(surface, mouse)
        surface.blit(self._font_sm.render("Reset to defaults", True, DIM_COLOR),
                     (WINDOW_W - 258, WINDOW_H - 30))
        if self._rebinding:
            action, slot = self._rebinding
            hint = ("Press a key to bind  |  ESC: cancel" if slot == 0
                    else "Press a key to bind  |  ESC: cancel  |  DEL: clear")
        else:
            hint = "Click a slot to rebind  |  conflicts are removed automatically"
        surface.blit(self._font_sm.render(hint, True, DIM_COLOR), (0, WINDOW_H - 16))

# ?? Game screen ??????????????????????????????????????????????????????????????????

# Vertical rhythm for left panel sections (relative to BOARD_OY)
_HOLD_LABEL_H  = 22    # pixels for the "HOLD" label
_HOLD_BOX_H    = 72    # height of the hold piece box
_HOLD_GAP      = 10    # gap below hold box
_CLEAR_AREA_H  = 100   # reserved height for spin + clear text
_B2B_H         = 42    # line height for B2B row
_COMBO_H       = 44    # line height for COMBO row
_STAT_H        = 62    # height for each PIECES/ATTACK/TIME row

# Precomputed Y offsets inside left panel (from BOARD_OY):
_LEFT_CLEAR_Y  = _HOLD_LABEL_H + _HOLD_BOX_H + _HOLD_GAP           # 104
_LEFT_B2B_Y    = _LEFT_CLEAR_Y + _CLEAR_AREA_H                      # 204
_LEFT_COMBO_Y  = _LEFT_B2B_Y   + _B2B_H                             # 246
_LEFT_STAT0_Y  = _LEFT_COMBO_Y + _COMBO_H                           # 290  PIECES
_LEFT_STAT1_Y  = _LEFT_STAT0_Y + _STAT_H                            # 352  APP
_LEFT_STAT2_Y  = _LEFT_STAT1_Y + _STAT_H                            # 414  APM
_LEFT_STAT3_Y  = _LEFT_STAT2_Y + _STAT_H                            # 476  TIME

# Right panel: 5 next-piece boxes
_NEXT_LABEL_H  = 22
_NEXT_BOX_H    = 62
_NEXT_BOX_GAP  = 4
_NEXT_TOTAL_H  = _NEXT_LABEL_H + 5 * (_NEXT_BOX_H + _NEXT_BOX_GAP) # 352
_RIGHT_SCORE_Y = _NEXT_TOTAL_H + 14                                  # 366


class GameScreen:
    def __init__(self, fonts, gfonts: dict, sounds: dict):
        _, self._font_lg, self._font_md, self._font_sm = fonts
        self._gf  = gfonts
        self._snd = sounds
        self.reset()

    def reset(self):
        self.state = GameState()
        self.state.gravity = 0.02
        self.bag          = SevenBag()
        self.lines_total  = 0
        self.level        = 1
        self.pieces_total = 0
        self.attack_total = 0
        self.game_time    = 0.0
        self.game_over    = False

        self._spin_label     = ""
        self._clear_label    = ""
        self._clear_timer    = 0.0
        self._all_clear_timer = 0.0

        self._das_ms     = settings["das"]
        self._arr_ms     = settings["arr"]
        self._sdf        = settings["sdf"]
        self._held_left  = False
        self._held_right = False
        self._das_timer  = 0.0
        self._arr_timer  = 0.0
        self._das_dir    = 0
        self._soft_held  = False

        self._refill_queue()
        self.state.spawn_next()
        self._refill_queue()

    def _refill_queue(self):
        while len(self.state.next_queue) < 6:
            self.state.enqueue(self.bag.pop())

    def _process_lock(self, lock_result, mino_type=None):
        cr = process_clear(self.state,
                           lines=lock_result.lines_cleared,
                           spin=lock_result.spin)
        ar = compute_attack(self.state, cr)
        self.attack_total += ar.sent
        self.lines_total  += lock_result.lines_cleared
        self.pieces_total += 1

        new_level = self.lines_total // 10 + 1
        if new_level != self.level:
            self.level = new_level
            self.state.gravity = 0.02 * (1.3 ** (self.level - 1))

        if lock_result.lines_cleared > 0:
            self._spin_label, self._clear_label = _build_clear_labels(
                lock_result.lines_cleared, lock_result.spin, cr.all_clear, mino_type)
            self._clear_timer = CLEAR_DISPLAY_SECS
            if cr.all_clear:
                self._all_clear_timer = CLEAR_DISPLAY_SECS
            self._play_clear_sound(cr.combo, ar.subtotal,
                                   cr.b2b, cr.b2b_delta, cr.all_clear)

        spawned = self.state.spawn_next(clutch_clear=lock_result.clutch_clear)
        self._refill_queue()
        if not spawned or self.state.round_result == RoundResult.LOSE:
            self.game_over = True
            go_snd = self._snd.get('gameover')
            if go_snd:
                go_snd.play()

    def _pressed(self, action: str, k: int) -> bool:
        return _key_matches(action, k)

    # ?? Event ????????????????????????????????????????????????????????????????????

    def handle_event(self, event):
        if self.game_over:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:     self.reset()
                elif event.key == pygame.K_ESCAPE: return Screen.HOME
            return None

        if event.type == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_ESCAPE: return Screen.HOME
            if k == pygame.K_r:      self.reset(); return None

            if   self._pressed("Rotate CW",  k): try_rotate(self.state, 1)
            elif self._pressed("Rotate CCW", k): try_rotate(self.state, -1)
            elif self._pressed("Rotate 180", k): try_rotate(self.state, 2)
            elif self._pressed("Hard Drop",  k):
                if self.state.active:
                    mtype = self.state.active.type
                    self._process_lock(hard_drop(self.state), mtype)
            elif self._pressed("Hold", k):
                self.state.try_hold()
                self._refill_queue()
            elif self._pressed("Move Left", k):
                self._held_left = True
                self._das_dir   = -1
                self._das_timer = self._arr_timer = 0.0
                try_move(self.state, -1)
            elif self._pressed("Move Right", k):
                self._held_right = True
                self._das_dir    = 1
                self._das_timer  = self._arr_timer = 0.0
                try_move(self.state, 1)
            elif self._pressed("Soft Drop", k):
                self._soft_held = True

        elif event.type == pygame.KEYUP:
            k = event.key
            if self._pressed("Move Left",  k):
                self._held_left = False
                if self._das_dir == -1: self._das_dir = 0
            if self._pressed("Move Right", k):
                self._held_right = False
                if self._das_dir == 1:  self._das_dir = 0
            if self._pressed("Soft Drop",  k):
                self._soft_held = False

        return None

    # ?? Update ???????????????????????????????????????????????????????????????????

    def update(self, dt: float):
        if self.game_over:
            return
        dt_ms = dt * 1000.0
        self.game_time += dt

        if self._clear_timer > 0:
            self._clear_timer = max(0.0, self._clear_timer - dt)
        if self._all_clear_timer > 0:
            self._all_clear_timer = max(0.0, self._all_clear_timer - dt)

        if self._das_dir != 0 and (self._held_left or self._held_right):
            self._das_timer += dt_ms
            if self._das_timer >= self._das_ms:
                if self._arr_ms == 0:
                    for _ in range(BOARD_COLS):
                        if not try_move(self.state, self._das_dir):
                            break
                else:
                    self._arr_timer += dt_ms
                    while self._arr_timer >= self._arr_ms:
                        self._arr_timer -= self._arr_ms
                        try_move(self.state, self._das_dir)

        if self._soft_held:
            steps = 20 if self._sdf >= 41 else self._sdf
            for _ in range(steps):
                if not soft_drop_step(self.state):
                    break

        if self.state.active is not None:
            mtype = self.state.active.type
            tick = gravity_tick(self.state, dt)
            if tick.locked:
                class _LR:
                    lines_cleared = tick.lines_cleared
                    clutch_clear  = tick.clutch_clear
                    spin          = tick.spin
                self._process_lock(_LR(), mtype)

    # ?? Draw ?????????????????????????????????????????????????????????????????????

    def draw(self, surface: pygame.Surface):
        self._draw_board(surface)
        self._draw_ghost(surface)
        if self.state.active:
            self._draw_active(surface)
        self._draw_hold(surface)
        self._draw_left_stats(surface)
        self._draw_next(surface)
        self._draw_right_stats(surface)
        self._draw_all_clear_overlay(surface)
        self._draw_danger_zone(surface)

        if self.game_over:
            self._draw_game_over(surface)

    def _cell_px(self, r, c):
        return BOARD_OX + c * CELL, BOARD_OY + (r - DISPLAY_ROW_START) * CELL

    # ?? Board ?????????????????????????????????????????????????????????????????????

    def _draw_board(self, surface):
        pygame.draw.rect(surface, PANEL_BG, (BOARD_OX, BOARD_OY, BOARD_W, BOARD_H))
        for vr in range(VISIBLE_ROWS):
            br = vr + DISPLAY_ROW_START
            for c in range(BOARD_COLS):
                x = BOARD_OX + c * CELL
                y = BOARD_OY + vr * CELL
                cell = self.state.board.get(br, c)
                if cell is not None:
                    color = COLORS.get(cell, (128, 128, 128))
                    pygame.draw.rect(surface, color, (x, y, CELL, CELL))
                    pygame.draw.rect(surface, _darken(color), (x, y, CELL, CELL), 1)
                else:
                    pygame.draw.rect(surface, GRID_COLOR, (x, y, CELL, CELL), 1)
        # Thin white border around the board
        pygame.draw.rect(surface, BORDER_COLOR,
                         (BOARD_OX - 2, BOARD_OY - 2, BOARD_W + 4, BOARD_H + 4), 2)

    def _draw_active(self, surface):
        p     = self.state.active
        color = COLORS.get(p.type, (128, 128, 128))
        for r, c in p.cells():
            if r < DISPLAY_ROW_START:
                continue
            x, y = self._cell_px(r, c)
            pygame.draw.rect(surface, color, (x, y, CELL, CELL))
            pygame.draw.rect(surface, _darken(color), (x, y, CELL, CELL), 1)

    def _draw_ghost(self, surface):
        if self.state.active is None:
            return
        color = COLORS.get(self.state.active.type, (128, 128, 128))
        dim   = _ghost_color(color)
        for r, c in _ghost_cells(self.state):
            if r < DISPLAY_ROW_START:
                continue
            x, y = self._cell_px(r, c)
            pygame.draw.rect(surface, dim, (x + 1, y + 1, CELL - 2, CELL - 2), 2)

    # ?? Hold ??????????????????????????????????????????????????????????????????????

    def _draw_hold(self, surface):
        gf = self._gf
        sx = LEFT_X
        sy = BOARD_OY

        lbl = gf['label'].render("HOLD", True, DIM_COLOR)
        surface.blit(lbl, (sx, sy))

        box = pygame.Rect(sx, sy + _HOLD_LABEL_H, LEFT_W, _HOLD_BOX_H)
        pygame.draw.rect(surface, PANEL_BG, box)
        pygame.draw.rect(surface, BORDER_COLOR, box, 2)

        if self.state.hold is not None:
            color = COLORS.get(self.state.hold, (128, 128, 128))
            if self.state.hold_used:
                color = _darken(color, 0.4)
            _draw_mini_piece(surface, SHAPES[self.state.hold][0],
                             color, box.x + 4, box.y + 4, box.w - 8, box.h - 8)

    # ?? Left stats panel ??????????????????????????????????????????????????????????

    def _draw_left_stats(self, surface):
        gf = self._gf
        sx = LEFT_X

        # ?? Spin / Clear result text (fades after CLEAR_DISPLAY_SECS) ????????????
        if self._clear_timer > 0 and self._clear_label:
            cy = BOARD_OY + _LEFT_CLEAR_Y
            if self._spin_label:
                spin_surf = gf['spin'].render(self._spin_label, True, SPIN_COLOR)
                surface.blit(spin_surf, (sx, cy))
                cy += gf['spin'].get_height() + 2
            clear_surf = gf['clear'].render(self._clear_label, True, TEXT_COLOR)
            surface.blit(clear_surf, (sx, cy))

        # ?? B2B ??????????????????????????????????????????????????????????????????
        sy = BOARD_OY + _LEFT_B2B_Y
        if self.state.b2b >= 1:
            b2b_surf = gf['b2b'].render(f"B2B 횞{self.state.b2b}", True, B2B_COLOR)
        elif self.state.b2b == 0:
            b2b_surf = gf['b2b'].render("B2B", True, B2B_COLOR)
        else:
            b2b_surf = gf['b2b'].render("B2B", True, DIM_COLOR)
        surface.blit(b2b_surf, (sx, sy))

        # ?? COMBO ?????????????????????????????????????????????????????????????????
        sy = BOARD_OY + _LEFT_COMBO_Y
        if self.state.combo >= 1:
            combo_surf = gf['combo'].render(f"{self.state.combo} COMBO", True, COMBO_COLOR)
        else:
            combo_surf = gf['combo'].render("COMBO", True, DIM_COLOR)
        surface.blit(combo_surf, (sx, sy))

        # ?? PIECES ????????????????????????????????????????????????????????????????
        sy = BOARD_OY + _LEFT_STAT0_Y
        surface.blit(gf['sm'].render("PIECES", True, DIM_COLOR), (sx, sy))
        surface.blit(gf['stat'].render(str(self.pieces_total), True, TEXT_COLOR),
                     (sx, sy + 16))

        # ?? APP (Attack Per Piece) ????????????????????????????????????????????????
        sy = BOARD_OY + _LEFT_STAT1_Y
        app = self.attack_total / self.pieces_total if self.pieces_total > 0 else 0.0
        surface.blit(gf['sm'].render("APP", True, DIM_COLOR), (sx, sy))
        app_int  = int(app)
        app_frac = f".{int((app - app_int) * 100):02d}"
        int_surf  = gf['stat'].render(str(app_int), True, TEXT_COLOR)
        frac_surf = gf['stat_sm'].render(app_frac, True, DIM_COLOR)
        surface.blit(int_surf, (sx, sy + 16))
        frac_y = sy + 16 + int_surf.get_height() - frac_surf.get_height()
        surface.blit(frac_surf, (sx + int_surf.get_width(), frac_y))

        # ?? APM (Attack Per Minute) ???????????????????????????????????????????????
        sy = BOARD_OY + _LEFT_STAT2_Y
        apm = self.attack_total / (self.game_time / 60.0) if self.game_time > 0 else 0.0
        surface.blit(gf['sm'].render("APM", True, DIM_COLOR), (sx, sy))
        apm_int  = int(apm)
        apm_frac = f".{int((apm - apm_int) * 10)}"
        apm_int_surf  = gf['stat'].render(str(apm_int), True, TEXT_COLOR)
        apm_frac_surf = gf['stat_sm'].render(apm_frac, True, DIM_COLOR)
        surface.blit(apm_int_surf, (sx, sy + 16))
        apm_frac_y = sy + 16 + apm_int_surf.get_height() - apm_frac_surf.get_height()
        surface.blit(apm_frac_surf, (sx + apm_int_surf.get_width(), apm_frac_y))

        # ?? TIME ??????????????????????????????????????????????????????????????????
        sy = BOARD_OY + _LEFT_STAT3_Y
        surface.blit(gf['sm'].render("TIME", True, DIM_COLOR), (sx, sy))
        total_s   = int(self.game_time)
        mins      = total_s // 60
        secs      = total_s % 60
        tenths    = int((self.game_time - total_s) * 10)
        time_main = f"{mins}:{secs:02d}"
        main_surf  = gf['stat'].render(time_main, True, TEXT_COLOR)
        tenth_surf = gf['stat_sm'].render(f".{tenths}", True, DIM_COLOR)
        surface.blit(main_surf, (sx, sy + 16))
        # Tenths in smaller font, baseline-aligned with main number
        tenth_y = sy + 16 + main_surf.get_height() - tenth_surf.get_height()
        surface.blit(tenth_surf, (sx + main_surf.get_width(), tenth_y))

        # ?? ESC hint ??????????????????????????????????????????????????????????????
        esc = gf['sm'].render("ESC: menu   R: restart", True, DIM_COLOR)
        surface.blit(esc, (sx, BOARD_OY + BOARD_H - 16))

    # ?? Next queue panel ??????????????????????????????????????????????????????????

    def _draw_next(self, surface):
        gf = self._gf
        sx = RIGHT_X
        sy = BOARD_OY

        lbl = gf['label'].render("NEXT", True, DIM_COLOR)
        surface.blit(lbl, (sx, sy))

        bw = RIGHT_W
        bh = _NEXT_BOX_H
        for i, mino in enumerate(list(self.state.next_queue)[:5]):
            by  = sy + _NEXT_LABEL_H + i * (bh + _NEXT_BOX_GAP)
            box = pygame.Rect(sx, by, bw, bh)
            pygame.draw.rect(surface, PANEL_BG, box)
            pygame.draw.rect(surface, BORDER_COLOR, box, 2)
            _draw_mini_piece(surface, SHAPES[mino][0],
                             COLORS.get(mino, (128, 128, 128)),
                             box.x + 4, box.y + 4, box.w - 8, box.h - 8)

    # ?? Right stats (VS score) ????????????????????????????????????????????????????

    def _draw_right_stats(self, surface):
        gf = self._gf
        sx = RIGHT_X
        sy = BOARD_OY + _RIGHT_SCORE_Y

        surface.blit(gf['sm'].render("VS SCORE", True, DIM_COLOR), (sx, sy))
        score_surf = gf['stat'].render(str(self.state.outgoing), True, TEXT_COLOR)
        surface.blit(score_surf, (sx, sy + 16))

    # ?? All-clear board overlay ???????????????????????????????????????????????????

    def _draw_all_clear_overlay(self, surface):
        if self._all_clear_timer <= 0:
            return
        ratio = self._all_clear_timer / CLEAR_DISPLAY_SECS
        alpha = int(210 * ratio)   # fades from ~210 ??0
        txt = self._gf['clear'].render("ALL CLEAR", True, (255, 255, 255))
        txt.set_alpha(alpha)
        cx = BOARD_OX + BOARD_W // 2
        cy = BOARD_OY + BOARD_H // 2
        surface.blit(txt, txt.get_rect(center=(cx, cy)))

    # ?? Danger zone overlay ???????????????????????????????????????????????????????

    def _play_clear_sound(self, combo: int, attack_subtotal: int,
                          b2b: int, b2b_delta: int, all_clear: bool) -> None:
        """
        Play OGG-based clear sounds.

        combo index mapping (1-based, capped at 16):
          combo = 0  (first consecutive clear) ??combo_1
          combo = 1  (second)                  ??combo_2
          ...
          combo ??15                           ??combo_16

        Power variant plays instead of the base sound when the attack
        spike is large (subtotal ??_CRUNCH_ATK).

        B2B sounds:
          b2b = 0  ??btb_1  (chain started)
          b2b = 1  ??btb_2  (second consecutive difficult clear)
          b2b ??2  ??btb_3  (long chain)
          b2b broke (b2b == -1 and b2b_delta < 0) ??btb_break
        """
        # Combo sound index (1-based, 1??6)
        idx = min(max(combo, 0) + 1, 16) if combo >= 0 else 1

        if attack_subtotal >= _CRUNCH_ATK:
            snd = self._snd.get(('clear_power', idx)) or self._snd.get(('clear', idx))
        else:
            snd = self._snd.get(('clear', idx))
        if snd:
            snd.play()

        # All-clear overlay sound
        if all_clear:
            ac = self._snd.get('allclear')
            if ac:
                ac.play()

        # B2B chain or break
        if b2b >= 0:
            btb_idx = min(b2b + 1, 3)          # 1, 2, or 3
            btb_snd = self._snd.get(('btb', btb_idx))
            if btb_snd:
                btb_snd.play()
        elif b2b == -1 and b2b_delta < 0:      # chain just broke
            brk = self._snd.get('btb_break')
            if brk:
                brk.play()

    def _draw_danger_zone(self, surface):
        """
        Draw red X marks on the next piece's spawn cells when any locked
        piece exists within 3 rows below any of those spawn cells.
        """
        if not self.state.next_queue:
            return
        next_type = self.state.next_queue[0]
        spawn_cells = ActivePiece.spawn(next_type).cells()

        # Trigger: any locked cell within 3 rows of any spawn cell (same column)
        triggered = any(
            self.state.board.get(r, c) is not None
            for sr, sc in spawn_cells
            for c in [sc]
            for r in range(sr, sr + 4)
        )
        if not triggered:
            return

        pad = 5
        for r, c in spawn_cells:
            x, y = self._cell_px(r, c)
            pygame.draw.line(surface, (220, 30, 30),
                             (x + pad, y + pad),
                             (x + CELL - pad, y + CELL - pad), 2)
            pygame.draw.line(surface, (220, 30, 30),
                             (x + CELL - pad, y + pad),
                             (x + pad, y + CELL - pad), 2)

    # ?? Game over overlay ?????????????????????????????????????????????????????????

    def _draw_game_over(self, surface):
        ov = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 170))
        surface.blit(ov, (0, 0))
        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        go = self._font_lg.render("GAME OVER", True, (240, 60, 60))
        surface.blit(go, go.get_rect(center=(cx, cy - 50)))
        info = self._font_md.render(
            f"Lines: {self.lines_total}   Level: {self.level}", True, TEXT_COLOR)
        surface.blit(info, info.get_rect(center=(cx, cy + 5)))
        hint = self._font_sm.render("R: restart   |   ESC: main menu", True, DIM_COLOR)
        surface.blit(hint, hint.get_rect(center=(cx, cy + 48)))

# ?? Bot game screen ??????????????????????????????????????????????????????????????

class BotGameScreen(GameScreen):
    """
    GameScreen??洹몃?濡?怨꾩듅?섎릺 ?ㅻ낫??議곗옉 ???BeamSearchBot???쇱뒪瑜?諛곗튂?쒕떎.

    ?숈옉 諛⑹떇
    ----------
    * 遊뉗씠 FinalPlacement 瑜??좏깮?섎㈃ input_sequence 瑜?_pending_actions ???ｋ뒗??
    * _ACTION_DELAY 留덈떎 ?먯뿉???됰룞 ?섎굹??爰쇰궡 ?ㅽ뻾?쒕떎.
      - SOFT_DROP : soft_drop_step
      - ROTATE_*  : try_rotate
      - LEFT/RIGHT: try_move
      - HOLD      : try_hold + refill
      - HARD_DROP : hard_drop + _process_lock ???ㅼ쓬 ?쇱뒪 以鍮???_PLACE_DELAY ?湲?    * 愿?꾩옄媛 ?쇱뒪 ?대룞??蹂????덈룄濡??됰룞 ?ъ씠??吏㏃? ?쒕젅?대? ?붾떎.
    """

    _ACTION_DELAY = 0.06   # ?됰룞 ?섎굹 ?ㅽ뻾 ???ㅼ쓬 ?됰룞源뚯? ?湲?(珥?
    _PLACE_DELAY  = 0.35   # HARD_DROP ???ㅼ쓬 ?쇱뒪 泥섎━源뚯? ?湲?(珥?

    def __init__(self, fonts, gfonts, sounds):
        super().__init__(fonts, gfonts, sounds)
        self._bot = self._make_bot()
        self._pending_actions: list = []   # list[BotAction]
        self._action_t        = 0.0        # ?ㅼ쓬 ?됰룞源뚯? ?⑥? ?쒓컙

    # ?? reset ????????????????????????????????????????????????????????????????

    def _make_bot(self) -> BeamSearchBot:
        trained_weights = load_weights()
        if trained_weights is None:
            return BeamSearchBot(config=SearchConfig.safe_default())

        eval_weights, trained_config = weights_to_objects(trained_weights)
        trained_config.time_budget_ms = SearchConfig.safe_default().time_budget_ms
        return BeamSearchBot(config=trained_config, eval_weights=eval_weights)

    def reset(self):
        super().reset()
        self._bot = self._make_bot()
        self._pending_actions = []
        self._action_t        = 0.0

    # ?? ?대깽?? ESC 쨌 R 留?泥섎━ ???????????????????????????????????????????????

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return Screen.HOME
            if event.key == pygame.K_r:
                self.reset()
        return None

    # ?? ?낅뜲?댄듃: 遊뉗씠 ?쇱뒪瑜????됰룞???ㅽ뻾 ????????????????????????????????

    def update(self, dt):
        if self.game_over:
            return

        self.game_time += dt
        if self._clear_timer > 0:
            self._clear_timer = max(0.0, self._clear_timer - dt)
        if self._all_clear_timer > 0:
            self._all_clear_timer = max(0.0, self._all_clear_timer - dt)

        # ??대㉧ ?湲?以?        if self._action_t > 0:
            self._action_t = max(0.0, self._action_t - dt)
            return

        if self.state.active is None:
            return

        # ?먭? 鍮꾩뼱 ?덉쑝硫?遊뉗뿉寃??ㅼ쓬 諛곗튂瑜?臾쇱뼱蹂몃떎
        if not self._pending_actions:
            placement = self._bot.select_placement(self.state)
            if placement is None:
                self.game_over = True
                return
            self._pending_actions = list(placement.input_sequence)

        # ?됰룞 ?섎굹 爰쇰궡 ?ㅽ뻾
        action = self._pending_actions.pop(0)
        self._execute_one_action(action)

    def _execute_one_action(self, action: BotAction):
        """input_sequence ?먯꽌 爰쇰궦 ?됰룞 ?섎굹瑜??ㅼ젣 GameState ???곸슜?쒕떎."""
        if self.state.active is None:
            return

        if action == BotAction.HOLD:
            self.state.try_hold()
            self._refill_queue()
            self._action_t = self._ACTION_DELAY

        elif action == BotAction.LEFT:
            try_move(self.state, -1)
            self._action_t = self._ACTION_DELAY

        elif action == BotAction.RIGHT:
            try_move(self.state, +1)
            self._action_t = self._ACTION_DELAY

        elif action == BotAction.ROTATE_CW:
            try_rotate(self.state, 1)
            self._action_t = self._ACTION_DELAY

        elif action == BotAction.ROTATE_CCW:
            try_rotate(self.state, -1)
            self._action_t = self._ACTION_DELAY

        elif action == BotAction.ROTATE_180:
            try_rotate(self.state, 2)
            self._action_t = self._ACTION_DELAY

        elif action == BotAction.SOFT_DROP:
            soft_drop_step(self.state)
            self._action_t = self._ACTION_DELAY

        elif action == BotAction.HARD_DROP:
            mtype = self.state.active.type
            lr    = hard_drop(self.state)
            self._process_lock(lr, mtype)
            # 諛곗튂 ?꾨즺 ??湲??쒕젅?대줈 愿?꾩옄媛 寃곌낵瑜?蹂????덇쾶 ??            self._action_t = self._PLACE_DELAY

    # ?? 洹몃━湲? GameScreen怨??숈씪 + BOT 諭껋? ?????????????????????????????????

    def draw(self, surface):
        super().draw(surface)
        badge = self._gf['label'].render("??BOT", True, ACCENT2)
        surface.blit(badge, badge.get_rect(topright=(BOARD_OX - H_GAP + LEFT_W, BOARD_OY)))


# ?? Training screen ???????????????????????????????????????????????????????????????

_RL_GREEN  = (50, 220, 100)
_RL_YELLOW = (220, 190, 50)
_RL_RED    = (220, 80,  60)

_W_SHORT = [
    "holes", "cell_cov", "height",
    "h_upper", "h_qtr",
    "bump", "bump_sq", "row_tr",
    "well", "tsd", "4wide",
    "atk_w", "chain_w", "ctx_w",
]

# 洹몃옒???곸뿭
_GR_X = MARGIN_X
_GR_Y = 78
_GR_W = WINDOW_W - MARGIN_X * 2
_GR_H = 148

# 媛以묒튂 ?⑤꼸 (醫? / 而⑦듃濡?(??
_WP_X  = MARGIN_X
_WP_Y  = _GR_Y + _GR_H + 8
_WP_W  = 300
_ROW_H = 18

_CP_X  = _WP_X + _WP_W + 18
_CP_W  = WINDOW_W - _CP_X - MARGIN_X


class TrainingScreen:
    """RL 媛以묒튂 ?덈젴 ?붾㈃ ??ES(Evolution Strategy) 諛깃렇?쇱슫???ㅻ젅??"""

    def __init__(self, fonts):
        _, self._font_lg, self._font_md, self._font_sm = fonts

        self._btn_back  = Button((18, 14, 90, 32), "< BACK",       self._font_sm)
        self._btn_start = Button((_CP_X, _WP_Y,      _CP_W, 36), "START",        self._font_sm)
        self._btn_stop  = Button((_CP_X, _WP_Y + 44, _CP_W, 36), "STOP",         self._font_sm)
        self._btn_save  = Button((_CP_X, _WP_Y + 88, _CP_W, 36), "SAVE WEIGHTS", self._font_sm)

        self._trainer: RLTrainer | None = None
        self._history_best: list[float] = []
        self._history_mean: list[float] = []
        self._last_gen: GenerationResult | None = None
        self._status   = "?덈젴 以鍮꾨맖"
        self._save_msg = ""
        self._save_t   = 0.0

        # ?ㅻ젅??媛?result ?꾨떖??        self._pending: GenerationResult | None = None
        self._lock = threading.Lock()

    # ?? 肄쒕갚 (諛깃렇?쇱슫???ㅻ젅?쒖뿉???몄텧?? ??????????????????????????????????

    def _on_generation(self, result: GenerationResult) -> None:
        with self._lock:
            self._pending = result

    def _flush(self) -> None:
        with self._lock:
            result, self._pending = self._pending, None
        if result is None:
            return
        self._last_gen = result
        self._history_best.append(result.best_fitness)
        self._history_mean.append(result.mean_fitness)
        if len(self._history_best) > 120:
            self._history_best = self._history_best[-120:]
            self._history_mean = self._history_mean[-120:]

    # ?? ?대깽?????????????????????????????????????????????????????????????????

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._stop_trainer()
            return Screen.HOME

        if self._btn_back.is_clicked(event):
            self._stop_trainer()
            return Screen.HOME

        if self._btn_start.is_clicked(event) and not self._is_running():
            init = load_weights() or WEIGHT_DEFAULTS
            self._trainer = RLTrainer(
                initial_weights=init,
                on_generation=self._on_generation,
            )
            self._history_best.clear()
            self._history_mean.clear()
            self._last_gen = None
            self._trainer.start()
            self._status = "?덈젴 以?.."

        if self._btn_stop.is_clicked(event) and self._is_running():
            self._stop_trainer()
            self._status = "Training stopped"

        if self._btn_save.is_clicked(event) and self._trainer is not None:
            self._trainer.save()
            self._save_msg = "saved!"
            self._save_t   = 2.0

        return None

    def _stop_trainer(self):
        if self._trainer and self._trainer.is_running():
            self._trainer.stop()

    def _is_running(self) -> bool:
        return self._trainer is not None and self._trainer.is_running()

    # ?? ?낅뜲?댄듃 ?????????????????????????????????????????????????????????????

    def update(self, dt: float):
        self._flush()
        if self._save_t > 0:
            self._save_t = max(0.0, self._save_t - dt)
            if self._save_t == 0.0:
                self._save_msg = ""
        if not self._is_running() and self._status == "?덈젴 以?..":
            self._status = "?덈젴 ?꾨즺"

    # ?? 洹몃━湲????????????????????????????????????????????????????????????????

    def draw(self, surface: pygame.Surface):
        mouse = pygame.mouse.get_pos()

        # ?? Title + back
        title = self._font_md.render("RL WEIGHT TRAINING", True, ACCENT)
        surface.blit(title, title.get_rect(center=(WINDOW_W // 2, 30)))
        self._btn_back.draw(surface, mouse)

        # ?? Stats bar
        gen   = self._last_gen.generation   if self._last_gen else 0
        sigma = self._last_gen.sigma        if self._last_gen else 0.150
        best  = self._last_gen.best_fitness if self._last_gen else None
        mean  = self._last_gen.mean_fitness if self._last_gen else None

        def _fmt(v):
            return f"{v:.3f}" if v is not None else "---"

        stats_items = [
            (f"GEN {gen}", TEXT_COLOR),
            (f"BEST {_fmt(best)}", _RL_GREEN),
            (f"AVG  {_fmt(mean)}",  _RL_YELLOW),
            (f"? {sigma:.3f}",      DIM_COLOR),
        ]
        sx = _GR_X
        for txt, col in stats_items:
            s = self._font_sm.render(txt, True, col)
            surface.blit(s, (sx, 56))
            sx += s.get_width() + 28

        # ?? Fitness graph
        self._draw_graph(surface)

        # ?? Weight bars (left)
        weights = self._trainer.weights if self._trainer else (load_weights() or WEIGHT_DEFAULTS)
        self._draw_weights(surface, weights)

        # ?? Controls (right)
        running = self._is_running()
        self._btn_start.draw(surface, mouse, active=not running)
        self._btn_stop.draw( surface, mouse, active=running)
        self._btn_save.draw( surface, mouse)

        # status
        st_col = _RL_GREEN if running else (_RL_RED if "?뺤?" in self._status else DIM_COLOR)
        surface.blit(self._font_sm.render(self._status, True, st_col),
                     (_CP_X, _WP_Y + 136))

        if self._trainer:
            info = (f"?몃? {self._trainer.generation} / "
                    f"{self._trainer.cfg.max_generations}")
            surface.blit(self._font_sm.render(info, True, DIM_COLOR),
                         (_CP_X, _WP_Y + 154))

        if self._save_msg:
            surface.blit(
                self._font_sm.render(self._save_msg, True, ACCENT2),
                (_CP_X, _WP_Y + 172),
            )

        # config hint
        tc = self._trainer.cfg if self._trainer else TrainConfig()
        hint_lines = [
            f"pop={tc.population_size}  sigma={tc.sigma:.2f}",
            f"games={tc.n_eval_games}  pieces={tc.n_pieces}",
            f"greedy depth-1 eval",
        ]
        hy = _WP_Y + 200
        for line in hint_lines:
            surface.blit(self._font_sm.render(line, True, DIM_COLOR), (_CP_X, hy))
            hy += 16

    # ?? Graph ????????????????????????????????????????????????????????????????

    def _draw_graph(self, surface: pygame.Surface):
        x, y, w, h = _GR_X, _GR_Y, _GR_W, _GR_H
        pygame.draw.rect(surface, PANEL_BG,   (x, y, w, h))
        pygame.draw.rect(surface, GRID_COLOR, (x, y, w, h), 1)

        # axis label
        surface.blit(
            self._font_sm.render("APP (attack per piece)", True, DIM_COLOR),
            (x + 4, y + 4),
        )

        if not self._history_best:
            msg = self._font_sm.render("Press START to begin training", True, DIM_COLOR)
            surface.blit(msg, msg.get_rect(center=(x + w // 2, y + h // 2)))
            return

        # y-range with padding
        all_v  = self._history_best + self._history_mean
        v_min  = min(all_v) - 0.05
        v_max  = max(all_v) + 0.05
        v_span = max(v_max - v_min, 0.01)

        PAD = 20  # bottom padding for x-axis

        def _py(v: float) -> int:
            ratio = (v - v_min) / v_span
            return y + h - PAD - int(ratio * (h - PAD - 6))

        def _px(i: int, n: int) -> int:
            return x + 2 + int(i / max(n - 1, 1) * (w - 4))

        def _draw_series(vals: list[float], color):
            n = len(vals)
            if n < 2:
                return
            pts = [(_px(i, n), _py(v)) for i, v in enumerate(vals)]
            pygame.draw.lines(surface, color, False, pts, 2)

        # grid lines (3 horizontal)
        for frac in (0.25, 0.5, 0.75):
            gy = y + PAD + int((1 - frac) * (h - PAD - 6))
            pygame.draw.line(surface, GRID_COLOR, (x + 1, gy), (x + w - 2, gy))
            gv = v_min + frac * v_span
            surface.blit(
                self._font_sm.render(f"{gv:.2f}", True, GRID_COLOR),
                (x + 2, gy - 8),
            )

        _draw_series(self._history_best, _RL_GREEN)
        _draw_series(self._history_mean, _RL_YELLOW)

        # legend (top-right)
        n = len(self._history_best)
        bx = x + w - 120
        surface.blit(self._font_sm.render(f"best {self._history_best[-1]:.3f}", True, _RL_GREEN),  (bx, y + 4))
        surface.blit(self._font_sm.render(f"avg  {self._history_mean[-1]:.3f}", True, _RL_YELLOW), (bx, y + 18))
        surface.blit(self._font_sm.render(f"n={n}", True, DIM_COLOR), (bx, y + 32))

    # ?? Weight bars ??????????????????????????????????????????????????????????

    def _draw_weights(self, surface: pygame.Surface, weights: list[float]):
        x, y = _WP_X, _WP_Y
        BAR_X   = x + 80
        BAR_W   = _WP_W - 80 - 52
        VAL_X   = BAR_X + BAR_W + 4

        for i, (short, val, (lo, hi)) in enumerate(
            zip(_W_SHORT, weights, WEIGHT_BOUNDS)
        ):
            ry = y + i * _ROW_H

            if i % 2 == 0:
                pygame.draw.rect(surface, (14, 14, 24), (x, ry, _WP_W, _ROW_H))

            # label
            surface.blit(
                self._font_sm.render(short, True, DIM_COLOR),
                (x + 2, ry + 2),
            )

            # bar background
            pygame.draw.rect(surface, (30, 30, 30), (BAR_X, ry + 4, BAR_W, _ROW_H - 8))

            # filled bar
            span  = hi - lo
            ratio = (val - lo) / span if span != 0 else 0
            fill  = max(1, int(ratio * BAR_W))
            color = _RL_GREEN if val >= 0 else _RL_RED
            pygame.draw.rect(surface, color, (BAR_X, ry + 4, fill, _ROW_H - 8))

            # default value marker (yellow tick)
            def_ratio = (WEIGHT_DEFAULTS[i] - lo) / span if span != 0 else 0
            def_px    = BAR_X + int(def_ratio * BAR_W)
            pygame.draw.line(surface, ACCENT2,
                             (def_px, ry + 3), (def_px, ry + _ROW_H - 4), 1)

            # value text
            surface.blit(
                self._font_sm.render(f"{val:+.2f}", True, TEXT_COLOR),
                (VAL_X, ry + 2),
            )


# ?? Main ?????????????????????????????????????????????????????????????????????????

def main():
    pygame.mixer.pre_init(_SND_RATE, -16, 2, 512)
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Tetris")
    clock  = pygame.time.Clock()
    sounds = build_sounds()

    # Font name priority: condensed esports-style ??bold fallbacks
    _FN = "bahnschrift,agencyfb,impact,arial narrow,consolas,arial"

    # Legacy font tuple expected by HomeScreen and SettingsScreen
    fonts = (
        pygame.font.SysFont(_FN, 58, bold=True),   # [0] unused by Home/Settings
        pygame.font.SysFont(_FN, 36, bold=True),   # [1] font_lg
        pygame.font.SysFont(_FN, 22, bold=True),   # [2] font_md
        pygame.font.SysFont(_FN, 15),              # [3] font_sm
    )

    # Tournament HUD fonts for GameScreen
    gfonts = {
        'sm':      pygame.font.SysFont(_FN, 14),             # small labels
        'label':   pygame.font.SysFont(_FN, 18, bold=True),  # HOLD / NEXT titles
        'spin':    pygame.font.SysFont(_FN, 30, bold=True),  # spin qualifier
        'clear':   pygame.font.SysFont(_FN, 52, bold=True),  # QUAD / TRIPLE
        'b2b':     pygame.font.SysFont(_FN, 34, bold=True),  # B2B chain
        'combo':   pygame.font.SysFont(_FN, 32, bold=True),  # COMBO
        'stat':    pygame.font.SysFont(_FN, 46, bold=True),  # stat numbers
        'stat_sm': pygame.font.SysFont(_FN, 20),             # decimal/suffix
    }

    home      = HomeScreen(fonts)
    mode_sel  = ModeSelectScreen(fonts)
    cfg       = SettingsScreen(fonts)
    game      = GameScreen(fonts, gfonts, sounds)
    bot_game  = BotGameScreen(fonts, gfonts, sounds)
    training  = TrainingScreen(fonts)
    current   = Screen.HOME

    while True:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _save_settings(settings)
                pygame.quit(); sys.exit()

            if current == Screen.HOME:
                result = home.handle_event(event)
                if result == Screen.MODE_SELECT:
                    current = Screen.MODE_SELECT
                elif result == Screen.SETTINGS:
                    current = Screen.SETTINGS
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    _save_settings(settings)
                    pygame.quit(); sys.exit()

            elif current == Screen.MODE_SELECT:
                result = mode_sel.handle_event(event)
                if result == Screen.GAME:
                    game.reset(); current = Screen.GAME
                elif result == Screen.BOT:
                    bot_game.reset(); current = Screen.BOT
                elif result == Screen.TRAINING:
                    current = Screen.TRAINING
                elif result == Screen.HOME:
                    current = Screen.HOME

            elif current == Screen.SETTINGS:
                result = cfg.handle_event(event)
                if result == Screen.HOME:
                    current = Screen.HOME

            elif current == Screen.GAME:
                result = game.handle_event(event)
                if result == Screen.HOME:
                    current = Screen.HOME

            elif current == Screen.BOT:
                result = bot_game.handle_event(event)
                if result == Screen.HOME:
                    current = Screen.HOME

            elif current == Screen.TRAINING:
                result = training.handle_event(event)
                if result == Screen.HOME:
                    current = Screen.HOME

        if current == Screen.HOME:
            home.update(dt)
        elif current == Screen.MODE_SELECT:
            mode_sel.update(dt)
        elif current == Screen.GAME:
            game.update(dt)
        elif current == Screen.BOT:
            bot_game.update(dt)
        elif current == Screen.TRAINING:
            training.update(dt)

        screen.fill(BG)
        if current == Screen.HOME:          home.draw(screen)
        elif current == Screen.MODE_SELECT: mode_sel.draw(screen)
        elif current == Screen.SETTINGS:    cfg.draw(screen)
        elif current == Screen.GAME:        game.draw(screen)
        elif current == Screen.BOT:         bot_game.draw(screen)
        elif current == Screen.TRAINING:    training.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()

