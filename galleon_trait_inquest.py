#!/usr/bin/env python3
"""
Galleon Trait Inquest

A first-person PyOpenGL/Pygame social deduction game aboard a ship at sea.

Controls:
  WASD / mouse       Move and look
  Shift              Move faster
  E                  Talk to nearby crew, or use the helm
  1-5                Pick dialogue options while talking
  C                  Cycle the selected crew member's personality verdict
  R                  Cycle the selected crew member's best ship role
  Tab                Open or close the crew ledger
  Up / Down          Select crew in the ledger
  V                  Convene the final council at the helm
  Q                  Close dialogue, ledger, or council
  F5                 Restart
  Esc                Quit

Dependencies:
  pip install pygame PyOpenGL PyOpenGL_accelerate
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
except ImportError as exc:
    raise SystemExit("This game requires pygame. Install it with: pip install pygame") from exc

try:
    from OpenGL.GL import *
    from OpenGL.GLU import gluPerspective
except ImportError as exc:
    raise SystemExit(
        "This game requires PyOpenGL. Install it with: pip install PyOpenGL PyOpenGL_accelerate"
    ) from exc


WIDTH, HEIGHT = 1180, 720
FPS = 60
TITLE = "Galleon Trait Inquest"

PLAYER_HEIGHT = 1.67
PLAYER_RADIUS = 0.36
WALK_SPEED = 4.15
SPRINT_MULT = 1.55
MOUSE_SENS = 0.105
MAX_PITCH = 74.0

DECK_Z_MIN = -17.8
DECK_Z_MAX = 17.0
HELM_POS = (0.0, 15.0)

TRAIT_CHOICES = [
    "Unknown",
    "Treasonous",
    "Mutinous",
    "Trustworthy",
    "Lovable",
    "Likeable",
]

ROLE_CHOICES = [
    "Unassigned",
    "Navigator",
    "Gunnery",
    "Quartermaster",
    "Surgeon",
    "Morale",
    "Deck Watch",
    "Secure Brig",
    "Signals Watch",
]

DIALOGUE_OPTIONS = [
    ("loyalty", "Test loyalty"),
    ("crisis", "Pose a crisis"),
    ("rumor", "Ask about rumors"),
    ("kindness", "Offer confidence"),
    ("role", "Discuss best duty"),
]

DIGIT_KEYS = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]

WHITE = (238, 241, 235)
MIST = (164, 199, 211)
GOLD = (242, 189, 83)
INK = (22, 27, 31)
RED = (223, 74, 62)
GREEN = (94, 204, 130)
BLUE = (92, 158, 218)
PARCHMENT = (232, 220, 182)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def angle_delta(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def length2(x: float, z: float) -> float:
    return math.sqrt(x * x + z * z)


def direction_from_yaw(yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    return math.sin(yaw), -math.cos(yaw)


def yaw_toward(dx: float, dz: float) -> float:
    return math.degrees(math.atan2(dx, dz))


def deck_half_width(z: float) -> float:
    az = abs(z)
    if az < 10.5:
        return 5.75
    if z < 0.0:
        return clamp(5.75 - (az - 10.5) * 0.42, 2.25, 5.75)
    return clamp(5.75 - (az - 10.5) * 0.50, 1.75, 5.75)


def wrap_text(font: pygame.font.Font, text: str, width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.size(trial)[0] <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def color4(color: Sequence[float], alpha: float = 1.0) -> Tuple[float, float, float, float]:
    if max(color) > 1.0:
        return color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha
    return color[0], color[1], color[2], alpha


def set_color(color: Sequence[float], alpha: float = 1.0) -> None:
    glColor4f(*color4(color, alpha))


def draw_box(
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    color: Sequence[float],
    alpha: float = 1.0,
) -> None:
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0
    set_color(color, alpha)
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)
    glVertex3f(x0, y1, z0)
    glVertex3f(x1, y1, z0)
    glVertex3f(x1, y1, z1)
    glVertex3f(x0, y1, z1)
    glNormal3f(0, -1, 0)
    glVertex3f(x0, y0, z1)
    glVertex3f(x1, y0, z1)
    glVertex3f(x1, y0, z0)
    glVertex3f(x0, y0, z0)
    glNormal3f(0, 0, 1)
    glVertex3f(x0, y0, z1)
    glVertex3f(x0, y1, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y0, z1)
    glNormal3f(0, 0, -1)
    glVertex3f(x1, y0, z0)
    glVertex3f(x1, y1, z0)
    glVertex3f(x0, y1, z0)
    glVertex3f(x0, y0, z0)
    glNormal3f(1, 0, 0)
    glVertex3f(x1, y0, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z0)
    glVertex3f(x1, y0, z0)
    glNormal3f(-1, 0, 0)
    glVertex3f(x0, y0, z0)
    glVertex3f(x0, y1, z0)
    glVertex3f(x0, y1, z1)
    glVertex3f(x0, y0, z1)
    glEnd()


def draw_cylinder(
    radius: float,
    height: float,
    color: Sequence[float],
    segments: int = 14,
    cap: bool = True,
) -> None:
    y0 = -height / 2.0
    y1 = height / 2.0
    set_color(color)
    glBegin(GL_QUADS)
    for i in range(segments):
        a0 = math.tau * i / segments
        a1 = math.tau * (i + 1) / segments
        x0, z0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
        glNormal3f(math.cos(a0), 0, math.sin(a0))
        glVertex3f(x0, y0, z0)
        glVertex3f(x0, y1, z0)
        glNormal3f(math.cos(a1), 0, math.sin(a1))
        glVertex3f(x1, y1, z1)
        glVertex3f(x1, y0, z1)
    glEnd()
    if cap:
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 1, 0)
        glVertex3f(0, y1, 0)
        for i in range(segments + 1):
            a = math.tau * i / segments
            glVertex3f(math.cos(a) * radius, y1, math.sin(a) * radius)
        glEnd()
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, -1, 0)
        glVertex3f(0, y0, 0)
        for i in range(segments, -1, -1):
            a = math.tau * i / segments
            glVertex3f(math.cos(a) * radius, y0, math.sin(a) * radius)
        glEnd()


def draw_sphere(radius: float, color: Sequence[float], rings: int = 7, segments: int = 14) -> None:
    set_color(color)
    for r in range(rings):
        t0 = math.pi * r / rings
        t1 = math.pi * (r + 1) / rings
        y0 = math.cos(t0) * radius
        y1 = math.cos(t1) * radius
        rr0 = math.sin(t0) * radius
        rr1 = math.sin(t1) * radius
        glBegin(GL_QUAD_STRIP)
        for s in range(segments + 1):
            a = math.tau * s / segments
            x0, z0 = math.cos(a) * rr0, math.sin(a) * rr0
            x1, z1 = math.cos(a) * rr1, math.sin(a) * rr1
            glNormal3f(x0 / radius, y0 / radius, z0 / radius)
            glVertex3f(x0, y0, z0)
            glNormal3f(x1 / radius, y1 / radius, z1 / radius)
            glVertex3f(x1, y1, z1)
        glEnd()


def draw_2d_rect(x: float, y: float, w: float, h: float, color: Sequence[int], alpha: float = 1.0) -> None:
    set_color(color, alpha)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_2d_line(x0: float, y0: float, x1: float, y1: float, color: Sequence[int], width: float = 1.0) -> None:
    glLineWidth(width)
    set_color(color)
    glBegin(GL_LINES)
    glVertex2f(x0, y0)
    glVertex2f(x1, y1)
    glEnd()
    glLineWidth(1.0)


class GLText:
    def __init__(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        self.tiny = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 34, bold=True)
        self.title = pygame.font.SysFont("georgia", 42, bold=True)
        self.cache: Dict[Tuple[str, Tuple[int, int, int], str], Tuple[int, int, int]] = {}

    def font_for(self, size: str) -> pygame.font.Font:
        if size == "tiny":
            return self.tiny
        if size == "small":
            return self.small
        if size == "big":
            return self.big
        if size == "title":
            return self.title
        return self.font

    def _texture(self, text: str, color: Tuple[int, int, int], size: str) -> Tuple[int, int, int]:
        key = (text, color, size)
        if key in self.cache:
            return self.cache[key]
        font = self.font_for(size)
        surf = font.render(text, True, color)
        data = pygame.image.tostring(surf, "RGBA", True)
        w, h = surf.get_width(), surf.get_height()
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        self.cache[key] = (tex, w, h)
        return self.cache[key]

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int] = WHITE,
        size: str = "normal",
        center: bool = False,
        shadow: bool = True,
    ) -> None:
        if shadow:
            self._draw_raw(text, x + 2, y + 2, (2, 5, 8), size, center, 0.72)
        self._draw_raw(text, x, y, color, size, center, 1.0)

    def _draw_raw(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int],
        size: str,
        center: bool,
        alpha: float,
    ) -> None:
        tex, w, h = self._texture(text, color, size)
        if center:
            x -= w / 2.0
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor4f(1.0, 1.0, 1.0, alpha)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + w, y)
        glTexCoord2f(1, 1)
        glVertex2f(x + w, y + h)
        glTexCoord2f(0, 1)
        glVertex2f(x, y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def wrapped(
        self,
        text: str,
        x: float,
        y: float,
        width: int,
        color: Tuple[int, int, int] = WHITE,
        size: str = "normal",
        line_gap: int = 4,
    ) -> float:
        font = self.font_for(size)
        lines = wrap_text(font, text, width)
        line_h = font.get_linesize()
        for i, line in enumerate(lines):
            self.draw(line, x, y + i * (line_h + line_gap), color, size)
        return y + len(lines) * (line_h + line_gap)


@dataclass(frozen=True)
class DialogueBeat:
    line: str
    clue: str
    expression: str


@dataclass
class CrewMember:
    name: str
    title: str
    pos: Tuple[float, float]
    trait: str
    best_role: str
    coat: Tuple[float, float, float]
    skin: Tuple[float, float, float]
    hair: Tuple[float, float, float]
    default_expression: str
    summary: str
    dialogue: Dict[str, DialogueBeat]
    verdict: str = "Unknown"
    role: str = "Unassigned"
    asked: set = field(default_factory=set)
    clues: List[str] = field(default_factory=list)
    expression: str = ""
    bob_seed: float = 0.0

    def __post_init__(self) -> None:
        if not self.expression:
            self.expression = self.default_expression
        self.bob_seed = random.random() * math.tau

    @property
    def evidence_count(self) -> int:
        return len(self.asked)


def build_crew() -> List[CrewMember]:
    return [
        CrewMember(
            name="Mira Vale",
            title="Star Navigator",
            pos=(-2.7, -12.0),
            trait="Trustworthy",
            best_role="Navigator",
            coat=(0.18, 0.34, 0.58),
            skin=(0.74, 0.55, 0.40),
            hair=(0.10, 0.07, 0.05),
            default_expression="calm",
            summary="Precise, patient, and hated by anyone who wants the ship lost.",
            dialogue={
                "loyalty": DialogueBeat(
                    "The compass is a promise. Bend it once for coin and the whole crew sails blind.",
                    "Frames loyalty as a practical duty, not a performance.",
                    "steady",
                ),
                "crisis": DialogueBeat(
                    "Reef to port, panic on deck: shorten sail, mark wind, then speak plainly to the helm.",
                    "Gives a calm ordered response under pressure.",
                    "calm",
                ),
                "rumor": DialogueBeat(
                    "Brine counts powder keys when he thinks no one watches. Mara seals letters with merchant wax.",
                    "Shares specific observations without asking for reward.",
                    "guarded",
                ),
                "kindness": DialogueBeat(
                    "Kind words help, but give me clean stars and an honest wheel. That is how you keep people alive.",
                    "Values competence and transparent navigation.",
                    "warm",
                ),
                "role": DialogueBeat(
                    "Put me by the charts. I will keep bearings public so no private plot can move us.",
                    "Open navigation is her best contribution.",
                    "steady",
                ),
            },
        ),
        CrewMember(
            name="Tomas Brine",
            title="Gun Deck Master",
            pos=(3.1, -5.5),
            trait="Mutinous",
            best_role="Gunnery",
            coat=(0.50, 0.16, 0.12),
            skin=(0.64, 0.45, 0.31),
            hair=(0.05, 0.05, 0.05),
            default_expression="guarded",
            summary="Brave in battle, bitter about rank, dangerous if humiliated.",
            dialogue={
                "loyalty": DialogueBeat(
                    "I am loyal to a captain who listens before the cannon speaks. Ignore the deck and the deck answers.",
                    "Loyalty is conditional on officers yielding authority.",
                    "guarded",
                ),
                "crisis": DialogueBeat(
                    "If officers freeze, someone strong takes the wheel. The articles matter less than survival.",
                    "Openly justifies taking command during fear.",
                    "angry",
                ),
                "rumor": DialogueBeat(
                    "Mara is too polished. Rowan buys smiles with extra rum. The lower deck sees more than officers think.",
                    "Deflects suspicion while invoking lower deck resentment.",
                    "deceit",
                ),
                "kindness": DialogueBeat(
                    "Do not soften me. Give me powder, a clean firing line, and no fool shouting orders from lace sleeves.",
                    "Responds to kindness with contempt for command.",
                    "angry",
                ),
                "role": DialogueBeat(
                    "Put me near the guns. Keep me useful and you will never hear my temper pointed inward.",
                    "Best used at gunnery, but he needs oversight.",
                    "guarded",
                ),
            },
        ),
        CrewMember(
            name="Mara Quill",
            title="Passenger Scribe",
            pos=(-3.3, 0.6),
            trait="Treasonous",
            best_role="Secure Brig",
            coat=(0.48, 0.24, 0.54),
            skin=(0.79, 0.61, 0.47),
            hair=(0.18, 0.09, 0.04),
            default_expression="deceit",
            summary="Elegant, observant, and always near signal flags when ships appear.",
            dialogue={
                "loyalty": DialogueBeat(
                    "Loyalty? Such a large word for a wet deck. I prefer agreements with signatures.",
                    "Treats loyalty as a contract instead of allegiance.",
                    "deceit",
                ),
                "crisis": DialogueBeat(
                    "In crisis I would secure the captain's papers first. Routes are the soul of a voyage.",
                    "Prioritizes route documents over crew safety.",
                    "guarded",
                ),
                "rumor": DialogueBeat(
                    "Rumors are coins. Spend them carefully. I heard the eastward convoy pays for advance weather.",
                    "Knows the market value of route intelligence.",
                    "deceit",
                ),
                "kindness": DialogueBeat(
                    "How generous. Generosity opens locks faster than steel if the hand is educated.",
                    "Studies favors as tools for access.",
                    "deceit",
                ),
                "role": DialogueBeat(
                    "Signals, ledgers, correspondence. I have a gift for making distant people understand.",
                    "Her gift with signals is exactly what makes her unsafe.",
                    "anxious",
                ),
            },
        ),
        CrewMember(
            name="Pip Tallow",
            title="Cabin Cook",
            pos=(2.3, 6.0),
            trait="Lovable",
            best_role="Morale",
            coat=(0.88, 0.53, 0.22),
            skin=(0.84, 0.67, 0.48),
            hair=(0.56, 0.32, 0.13),
            default_expression="warm",
            summary="No poker face, huge heart, and the only person who can make storm biscuits edible.",
            dialogue={
                "loyalty": DialogueBeat(
                    "I would not sell the ship. Who would eat my biscuits if everyone drowned?",
                    "Answers simply and with obvious communal attachment.",
                    "warm",
                ),
                "crisis": DialogueBeat(
                    "I would get water, bandages, and sing loud enough that scared hands remember they are hands.",
                    "Instinct is comfort and practical aid.",
                    "warm",
                ),
                "rumor": DialogueBeat(
                    "Mara smells like ink after midnight. Brine kicks buckets when the captain speaks. I am bad at whispering.",
                    "Not subtle, but observant and unguarded.",
                    "anxious",
                ),
                "kindness": DialogueBeat(
                    "If you are kind to the crew, they stop hiding their hunger. That tells you more than shouting.",
                    "Understands morale as information.",
                    "warm",
                ),
                "role": DialogueBeat(
                    "Let me feed the watch and carry messages. People tell soup things they never tell officers.",
                    "Best used to lift morale and surface quiet fears.",
                    "warm",
                ),
            },
        ),
        CrewMember(
            name="Rowan Finch",
            title="Quartermaster",
            pos=(-2.4, 9.2),
            trait="Likeable",
            best_role="Quartermaster",
            coat=(0.18, 0.48, 0.34),
            skin=(0.68, 0.48, 0.34),
            hair=(0.36, 0.20, 0.08),
            default_expression="warm",
            summary="Charming, fair with stores, and clever enough to smooth conflict before it sparks.",
            dialogue={
                "loyalty": DialogueBeat(
                    "Trust is counted like water. Everyone sees the barrel, everyone knows the measure.",
                    "Links loyalty to fair public accounting.",
                    "steady",
                ),
                "crisis": DialogueBeat(
                    "I would ration first, flatter second, threaten last. Panic hates a ledger it can inspect.",
                    "Uses charm and process to hold people together.",
                    "warm",
                ),
                "rumor": DialogueBeat(
                    "Brine wants applause more than blood. Mara wants privacy more than sleep. Pip just wants jam.",
                    "Reads people accurately without malice.",
                    "warm",
                ),
                "kindness": DialogueBeat(
                    "Kindness is a tool, not a weakness. Use it honestly and the crew will forgive bad weather.",
                    "Understands social trust as ship infrastructure.",
                    "warm",
                ),
                "role": DialogueBeat(
                    "Give me stores and disputes. I can turn a near-fight into a queue if I have numbers.",
                    "Best used managing resources and conflict.",
                    "steady",
                ),
            },
        ),
        CrewMember(
            name="Ione Grey",
            title="Ship Surgeon",
            pos=(2.8, 13.0),
            trait="Trustworthy",
            best_role="Surgeon",
            coat=(0.68, 0.68, 0.62),
            skin=(0.55, 0.40, 0.30),
            hair=(0.67, 0.66, 0.58),
            default_expression="calm",
            summary="Austere, steady, and more interested in pulse than politics.",
            dialogue={
                "loyalty": DialogueBeat(
                    "I have stitched rebels and officers. My loyalty is to the living body of this ship.",
                    "Defines duty by preserving the crew, not gaining influence.",
                    "calm",
                ),
                "crisis": DialogueBeat(
                    "Triage first. Stop bleeding, isolate fever, then ask who caused the wound.",
                    "Orders action by harm reduction.",
                    "steady",
                ),
                "rumor": DialogueBeat(
                    "Brine has stress in the jaw. Mara has ink on the wrong fingers. Rowan sleeps when the stores balance.",
                    "Observes physical tells with clinical precision.",
                    "guarded",
                ),
                "kindness": DialogueBeat(
                    "Kindness is dosage. Too little and people break. Too much and they hide the fracture.",
                    "Compassion is measured and practical.",
                    "calm",
                ),
                "role": DialogueBeat(
                    "Keep me in the surgery and at council. Fear becomes quieter when wounds are named correctly.",
                    "Best used as surgeon and clear-eyed advisor.",
                    "steady",
                ),
            },
        ),
    ]


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(TITLE)
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.text = GLText()
        self.running = True
        self.random = random.Random(8)
        self.init_gl()
        self.reset()

    def init_gl(self) -> None:
        glViewport(0, 0, WIDTH, HEIGHT)
        glClearColor(0.36, 0.56, 0.66, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.28, 0.32, 0.35, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.90, 0.86, 0.72, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.35, 0.38, 0.42, 1.0))
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogfv(GL_FOG_COLOR, (0.36, 0.56, 0.66, 1.0))
        glFogf(GL_FOG_START, 58.0)
        glFogf(GL_FOG_END, 170.0)

    def reset(self) -> None:
        self.player_x = 0.0
        self.player_z = 12.2
        self.player_yaw = 180.0
        self.player_pitch = -4.0
        self.walk_phase = 0.0
        self.sea_time = 0.0
        self.near_npc: Optional[CrewMember] = None
        self.active_npc: Optional[CrewMember] = None
        self.crew = build_crew()
        self.dialogue_line = "Question the crew. Read faces. Assign each person a trait and the duty that best protects the ship."
        self.dialogue_clue = ""
        self.ledger_open = False
        self.ledger_index = 0
        self.final_open = False
        self.final_results: List[Tuple[CrewMember, bool, bool]] = []
        self.final_score = 0
        self.final_summary = ""
        self.message_timer = 4.5
        self.message = "Find each sailor, question them, and classify the crew before the watch changes."
        self.grab_mouse(True)

    def grab_mouse(self, enabled: bool) -> None:
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        if enabled:
            pygame.mouse.get_rel()

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.033)
            self.handle_events()
            self.update(dt)
            self.render()
            pygame.display.flip()
        pygame.quit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.MOUSEMOTION and self.mouse_look_enabled:
                dx, dy = event.rel
                self.player_yaw = (self.player_yaw + dx * MOUSE_SENS) % 360.0
                self.player_pitch = clamp(self.player_pitch + dy * MOUSE_SENS, -MAX_PITCH, MAX_PITCH)
            if event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)

    @property
    def mouse_look_enabled(self) -> bool:
        return not self.ledger_open and not self.final_open

    def handle_keydown(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
            return
        if key == pygame.K_F5:
            self.reset()
            return
        if key == pygame.K_q:
            if self.final_open:
                self.final_open = False
                self.grab_mouse(True)
            elif self.ledger_open:
                self.ledger_open = False
                self.grab_mouse(True)
            elif self.active_npc:
                self.close_dialogue()
            return
        if key == pygame.K_TAB:
            if not self.final_open:
                self.ledger_open = not self.ledger_open
                if self.ledger_open:
                    self.active_npc = None
                self.grab_mouse(not self.ledger_open)
            return
        if self.final_open:
            return
        if self.ledger_open:
            self.handle_ledger_key(key)
            return
        if key == pygame.K_e:
            if self.near_npc:
                self.begin_dialogue(self.near_npc)
            elif self.distance_to_helm() < 2.2:
                self.show_message("The helm bell waits. Press V here to convene the final council.", 3.0)
            return
        if key == pygame.K_v:
            if self.distance_to_helm() < 2.5:
                self.convene_council()
            else:
                self.show_message("Stand at the helm to convene the final council.", 2.8)
            return
        if self.active_npc:
            if key in DIGIT_KEYS:
                self.pick_dialogue(DIGIT_KEYS.index(key))
            elif key == pygame.K_c:
                self.cycle_verdict(self.active_npc)
            elif key == pygame.K_r:
                self.cycle_role(self.active_npc)

    def handle_ledger_key(self, key: int) -> None:
        if key == pygame.K_UP:
            self.ledger_index = (self.ledger_index - 1) % len(self.crew)
        elif key == pygame.K_DOWN:
            self.ledger_index = (self.ledger_index + 1) % len(self.crew)
        elif key == pygame.K_c:
            self.cycle_verdict(self.crew[self.ledger_index])
        elif key == pygame.K_r:
            self.cycle_role(self.crew[self.ledger_index])
        elif key == pygame.K_v:
            self.convene_council()

    def begin_dialogue(self, npc: CrewMember) -> None:
        self.active_npc = npc
        self.ledger_open = False
        self.dialogue_line = f"{npc.name} watches your face as carefully as you watch theirs."
        self.dialogue_clue = npc.summary
        npc.expression = npc.default_expression
        self.show_message(f"Interviewing {npc.name}. Use 1-5, then C/R to classify.", 2.0)

    def close_dialogue(self) -> None:
        if self.active_npc:
            self.active_npc.expression = self.active_npc.default_expression
        self.active_npc = None
        self.dialogue_line = "The interview ends. The sea keeps listening."
        self.dialogue_clue = ""

    def pick_dialogue(self, index: int) -> None:
        if not self.active_npc or index >= len(DIALOGUE_OPTIONS):
            return
        key, _label = DIALOGUE_OPTIONS[index]
        npc = self.active_npc
        beat = npc.dialogue[key]
        if key in npc.asked:
            self.dialogue_line = f"{npc.name} gives you the same answer with less patience: {beat.line}"
            self.dialogue_clue = "Already recorded."
        else:
            npc.asked.add(key)
            npc.clues.append(beat.clue)
            self.dialogue_line = beat.line
            self.dialogue_clue = beat.clue
            self.show_message(f"New clue logged for {npc.name}.", 1.5)
        npc.expression = beat.expression

    def cycle_verdict(self, npc: CrewMember) -> None:
        idx = TRAIT_CHOICES.index(npc.verdict)
        npc.verdict = TRAIT_CHOICES[(idx + 1) % len(TRAIT_CHOICES)]
        self.show_message(f"{npc.name}: verdict set to {npc.verdict}.", 1.6)

    def cycle_role(self, npc: CrewMember) -> None:
        idx = ROLE_CHOICES.index(npc.role)
        npc.role = ROLE_CHOICES[(idx + 1) % len(ROLE_CHOICES)]
        self.show_message(f"{npc.name}: role set to {npc.role}.", 1.6)

    def show_message(self, text: str, seconds: float = 2.0) -> None:
        self.message = text
        self.message_timer = seconds

    def distance_to_helm(self) -> float:
        return math.hypot(self.player_x - HELM_POS[0], self.player_z - HELM_POS[1])

    def convene_council(self) -> None:
        self.final_results.clear()
        score = 0
        danger_missed = 0
        morale_wronged = 0
        for npc in self.crew:
            trait_ok = npc.verdict == npc.trait
            role_ok = npc.role == npc.best_role
            if trait_ok:
                score += 12
            if role_ok:
                score += 7
            score += min(npc.evidence_count, 5)
            if npc.trait in ("Treasonous", "Mutinous") and not trait_ok:
                danger_missed += 1
            if npc.trait in ("Trustworthy", "Lovable", "Likeable") and npc.verdict in ("Treasonous", "Mutinous"):
                morale_wronged += 1
            self.final_results.append((npc, trait_ok, role_ok))
        max_score = len(self.crew) * (12 + 7 + 5)
        self.final_score = score
        if danger_missed == 0 and morale_wronged == 0 and score >= int(max_score * 0.82):
            self.final_summary = "The watch holds. The route stays secret, the guns stay loyal, and the crew believes it has been seen clearly."
        elif danger_missed:
            self.final_summary = "A dangerous heart slipped through the council. The voyage survives the night, but the tide now carries a knife."
        elif morale_wronged:
            self.final_summary = "You protected the hull and bruised the community. Innocent crew obey, but they do not sing."
        else:
            self.final_summary = "The ship sails on with an uneasy but workable order. Better evidence would make the next dawn kinder."
        self.final_open = True
        self.ledger_open = False
        self.active_npc = None
        self.grab_mouse(False)

    def update(self, dt: float) -> None:
        self.sea_time += dt
        self.message_timer = max(0.0, self.message_timer - dt)
        self.update_near_npc()
        if not self.ledger_open and not self.final_open and not self.active_npc:
            self.move_player(dt)

    def update_near_npc(self) -> None:
        best: Optional[CrewMember] = None
        best_dist = 2.35
        for npc in self.crew:
            dist = math.hypot(npc.pos[0] - self.player_x, npc.pos[1] - self.player_z)
            if dist < best_dist:
                best = npc
                best_dist = dist
        self.near_npc = best

    def move_player(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        dx = 0.0
        dz = 0.0
        forward_x, forward_z = direction_from_yaw(self.player_yaw)
        right_x, right_z = direction_from_yaw(self.player_yaw + 90.0)
        if keys[pygame.K_w]:
            dx += forward_x
            dz += forward_z
        if keys[pygame.K_s]:
            dx -= forward_x
            dz -= forward_z
        if keys[pygame.K_d]:
            dx += right_x
            dz += right_z
        if keys[pygame.K_a]:
            dx -= right_x
            dz -= right_z
        mag = length2(dx, dz)
        if mag > 0.001:
            speed = WALK_SPEED * (SPRINT_MULT if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
            dx = dx / mag * speed * dt
            dz = dz / mag * speed * dt
            self.walk_phase += dt * speed * 2.2
            self.player_x += dx
            self.player_z += dz
            self.constrain_to_deck()

    def constrain_to_deck(self) -> None:
        self.player_z = clamp(self.player_z, DECK_Z_MIN + PLAYER_RADIUS, DECK_Z_MAX - PLAYER_RADIUS)
        half = deck_half_width(self.player_z) - PLAYER_RADIUS
        self.player_x = clamp(self.player_x, -half, half)

    def render(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_3d()
        self.draw_sky()
        self.draw_ocean()
        self.draw_ship()
        self.draw_crew()
        self.draw_helm_marker()
        self.setup_2d()
        self.draw_ui()
        self.end_2d()

    def setup_3d(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_FOG)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, WIDTH / HEIGHT, 0.05, 240.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        sway = math.sin(self.sea_time * 0.65) * 0.42
        bob = math.sin(self.sea_time * 1.15) * 0.035
        glRotatef(self.player_pitch + sway, 1, 0, 0)
        glRotatef(-self.player_yaw, 0, 1, 0)
        glTranslatef(-self.player_x, -(PLAYER_HEIGHT + bob), -self.player_z)
        glLightfv(GL_LIGHT0, GL_POSITION, (38.0, 42.0, -50.0, 1.0))

    def setup_2d(self) -> None:
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def end_2d(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_sky(self) -> None:
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glPushMatrix()
        glLoadIdentity()
        glBegin(GL_QUADS)
        glColor4f(0.34, 0.55, 0.70, 1.0)
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glColor4f(0.72, 0.80, 0.78, 1.0)
        glVertex3f(1, 0.12, -1)
        glVertex3f(-1, 0.12, -1)
        glColor4f(0.72, 0.80, 0.78, 1.0)
        glVertex3f(-1, 0.12, -1)
        glVertex3f(1, 0.12, -1)
        glColor4f(0.15, 0.31, 0.50, 1.0)
        glVertex3f(1, 1, -1)
        glVertex3f(-1, 1, -1)
        glEnd()
        self.draw_sun_disc(0.67, -0.42, 0.085)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_sun_disc(self, x: float, y: float, r: float) -> None:
        glColor4f(1.0, 0.77, 0.36, 0.95)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, y, -1)
        for i in range(33):
            a = math.tau * i / 32
            glVertex3f(x + math.cos(a) * r, y + math.sin(a) * r, -1)
        glEnd()

    def draw_ocean(self) -> None:
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        size = 150
        step = 8
        t = self.sea_time
        for ix in range(-size, size, step):
            glBegin(GL_QUAD_STRIP)
            for iz in range(-size, size + step, step):
                for x in (ix, ix + step):
                    z = iz
                    wave = math.sin((x * 0.06) + t * 1.1) * 0.18 + math.cos((z * 0.08) - t * 1.3) * 0.12
                    shade = 0.48 + 0.15 * math.sin((x + z) * 0.03 + t)
                    glColor4f(0.03, 0.22 + shade * 0.12, 0.34 + shade * 0.20, 0.88)
                    glVertex3f(x, -1.25 + wave, z)
            glEnd()
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)

    def draw_ship(self) -> None:
        self.draw_hull()
        self.draw_deck()
        self.draw_rails()
        self.draw_masts_and_sails()
        self.draw_props()

    def draw_hull(self) -> None:
        dark_wood = (0.24, 0.12, 0.06)
        warm_wood = (0.38, 0.20, 0.09)
        draw_box(0, -0.46, -1.0, 11.8, 0.55, 31.0, dark_wood)
        draw_box(-5.85, -0.05, -1.0, 0.55, 1.1, 30.0, warm_wood)
        draw_box(5.85, -0.05, -1.0, 0.55, 1.1, 30.0, warm_wood)
        draw_box(0, -0.20, -17.0, 4.7, 0.95, 1.8, dark_wood)
        draw_box(0, -0.12, 16.1, 3.9, 1.05, 2.2, dark_wood)
        draw_box(0, 0.58, 15.8, 5.5, 1.25, 2.9, (0.30, 0.15, 0.08))

    def draw_deck(self) -> None:
        plank_a = (0.45, 0.29, 0.14)
        plank_b = (0.53, 0.34, 0.16)
        gap = (0.09, 0.06, 0.04)
        z = -16.5
        while z < 16.2:
            half = deck_half_width(z) - 0.1
            draw_box(0, 0.05, z, half * 2.0, 0.08, 0.92, plank_a if int((z + 20) * 2) % 2 else plank_b)
            draw_box(0, 0.102, z + 0.46, half * 2.0, 0.025, 0.025, gap)
            z += 0.92
        for x in [-3.8, -1.9, 0.0, 1.9, 3.8]:
            draw_box(x, 0.13, -1.0, 0.045, 0.035, 31.0, (0.14, 0.09, 0.05))
        draw_box(0, 0.22, 15.0, 2.8, 0.28, 1.2, (0.24, 0.15, 0.08))
        draw_box(0, 0.53, 15.0, 0.18, 0.45, 0.18, (0.10, 0.08, 0.06))
        self.draw_wheel(0, 0.92, 14.55)

    def draw_wheel(self, x: float, y: float, z: float) -> None:
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(90, 1, 0, 0)
        set_color((0.48, 0.27, 0.10))
        glLineWidth(5.0)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            a = math.tau * i / 32
            glVertex3f(math.cos(a) * 0.62, math.sin(a) * 0.62, 0)
        glEnd()
        glLineWidth(3.0)
        glBegin(GL_LINES)
        for i in range(8):
            a = math.tau * i / 8
            glVertex3f(0, 0, 0)
            glVertex3f(math.cos(a) * 0.84, math.sin(a) * 0.84, 0)
        glEnd()
        glLineWidth(1.0)
        glPopMatrix()

    def draw_rails(self) -> None:
        rail = (0.35, 0.19, 0.08)
        for side in (-1, 1):
            for z in [v * 2.15 - 15.4 for v in range(15)]:
                half = deck_half_width(z)
                draw_box(side * half, 0.75, z, 0.22, 1.15, 0.22, rail)
            for z in (-10.0, 0.0, 9.5):
                half = deck_half_width(z)
                draw_box(side * half, 1.26, z, 0.26, 0.18, 12.0, rail)
                draw_box(side * half, 0.74, z, 0.20, 0.15, 11.6, (0.27, 0.14, 0.06))
        draw_box(0, 1.02, -17.0, 4.2, 0.25, 0.36, rail)
        draw_box(0, 1.12, 16.6, 3.3, 0.25, 0.36, rail)

    def draw_masts_and_sails(self) -> None:
        for x, z, h, sail_w, sail_h in [(-0.4, -8.0, 8.8, 6.4, 4.4), (0.35, 5.1, 7.3, 5.4, 3.7)]:
            glPushMatrix()
            glTranslatef(x, 3.7, z)
            draw_cylinder(0.16, h, (0.30, 0.16, 0.07), 16)
            glPopMatrix()
            draw_box(x, 5.4, z, sail_w + 0.7, 0.12, 0.12, (0.27, 0.15, 0.07))
            draw_box(x, 2.8, z, sail_w + 0.3, 0.12, 0.12, (0.27, 0.15, 0.07))
            self.draw_sail(x, z + 0.08, 3.05, sail_w, sail_h)
            self.draw_rope(x, z, sail_w)
        for x, z in [(-0.4, -8.0), (0.35, 5.1)]:
            draw_box(x, 8.15, z, 1.6, 0.08, 0.08, (0.26, 0.13, 0.05))
            draw_box(x + 0.55, 8.42, z, 0.9, 0.45, 0.035, (0.65, 0.08, 0.10))

    def draw_sail(self, x: float, z: float, y: float, w: float, h: float) -> None:
        glDisable(GL_CULL_FACE)
        glBegin(GL_QUADS)
        for i in range(7):
            t0 = i / 7.0
            t1 = (i + 1) / 7.0
            shade0 = 0.83 + 0.08 * math.sin(i)
            shade1 = 0.83 + 0.08 * math.sin(i + 1)
            glColor4f(shade0, shade0 * 0.96, shade0 * 0.82, 0.96)
            glNormal3f(0, 0, 1)
            glVertex3f(x - w / 2 + w * t0, y, z)
            glVertex3f(x - w / 2 + w * t1, y, z)
            glColor4f(shade1, shade1 * 0.96, shade1 * 0.82, 0.96)
            glVertex3f(x - w / 2 + w * t1 - 0.25 * math.sin(t1 * math.pi), y + h, z)
            glVertex3f(x - w / 2 + w * t0 - 0.25 * math.sin(t0 * math.pi), y + h, z)
        glEnd()
        glBegin(GL_LINES)
        set_color((0.58, 0.49, 0.34))
        for i in range(8):
            xx = x - w / 2 + w * i / 7.0
            glVertex3f(xx, y, z + 0.01)
            glVertex3f(xx - 0.20, y + h, z + 0.01)
        glEnd()

    def draw_rope(self, x: float, z: float, w: float) -> None:
        glDisable(GL_LIGHTING)
        set_color((0.18, 0.13, 0.09))
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for side in (-1, 1):
            glVertex3f(x, 7.8, z)
            glVertex3f(x + side * w / 2.0, 0.7, z + side * 1.1)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_props(self) -> None:
        for x, z in [(-4.1, -3.2), (4.0, 1.8), (-4.0, 5.8), (4.2, -10.0)]:
            glPushMatrix()
            glTranslatef(x, 0.55, z)
            draw_cylinder(0.45, 0.85, (0.37, 0.19, 0.08), 14)
            draw_box(0, 0.34, 0, 0.96, 0.08, 0.96, (0.12, 0.09, 0.07))
            draw_box(0, -0.34, 0, 0.96, 0.08, 0.96, (0.12, 0.09, 0.07))
            glPopMatrix()
        for x, z in [(-4.5, -7.6), (4.5, -7.6), (-4.5, -1.4), (4.5, -1.4)]:
            glPushMatrix()
            glTranslatef(x, 0.58, z)
            glRotatef(90, 0, 0, 1)
            draw_cylinder(0.22, 1.45, (0.06, 0.06, 0.065), 12)
            glPopMatrix()
            draw_box(x * 0.99, 0.35, z, 0.8, 0.26, 0.6, (0.21, 0.12, 0.07))
        for x, z in [(0.0, -1.8), (3.8, 8.6), (-3.6, 12.4)]:
            draw_box(x, 0.38, z, 1.1, 0.65, 0.9, (0.37, 0.22, 0.10))
            draw_box(x, 0.74, z, 1.16, 0.08, 0.96, (0.14, 0.09, 0.05))

    def draw_crew(self) -> None:
        for npc in self.crew:
            self.draw_npc(npc)

    def draw_npc(self, npc: CrewMember) -> None:
        x, z = npc.pos
        dist = math.hypot(self.player_x - x, self.player_z - z)
        yaw = yaw_toward(self.player_x - x, self.player_z - z)
        bob = math.sin(self.sea_time * 1.8 + npc.bob_seed) * 0.025
        glPushMatrix()
        glTranslatef(x, 0.14 + bob, z)
        glRotatef(yaw, 0, 1, 0)
        selected = npc is self.near_npc or npc is self.active_npc
        if selected:
            self.draw_selection_ring(dist)
        draw_box(0, 0.88, 0, 0.68, 1.18, 0.38, npc.coat)
        draw_box(0, 1.50, 0.04, 0.52, 0.12, 0.34, (0.92, 0.82, 0.62))
        draw_box(-0.48, 0.94, 0, 0.18, 0.88, 0.22, npc.coat)
        draw_box(0.48, 0.94, 0, 0.18, 0.88, 0.22, npc.coat)
        draw_box(-0.18, 0.20, 0, 0.22, 0.55, 0.24, (0.12, 0.10, 0.09))
        draw_box(0.18, 0.20, 0, 0.22, 0.55, 0.24, (0.12, 0.10, 0.09))
        glPushMatrix()
        glTranslatef(0, 1.74, 0.02)
        draw_sphere(0.285, npc.skin, 7, 16)
        glPushMatrix()
        glTranslatef(0, 0.12, -0.03)
        draw_sphere(0.295, npc.hair, 4, 16)
        glPopMatrix()
        self.draw_face(npc.expression)
        glPopMatrix()
        glPopMatrix()

    def draw_selection_ring(self, dist: float) -> None:
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        pulse = 0.55 + 0.45 * math.sin(self.sea_time * 4.5)
        color = (GOLD[0] / 255.0, GOLD[1] / 255.0, GOLD[2] / 255.0, 0.50 + pulse * 0.35)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for i in range(36):
            a = math.tau * i / 36
            glVertex3f(math.cos(a) * 0.72, 0.035, math.sin(a) * 0.72)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_face(self, expression: str) -> None:
        glDisable(GL_LIGHTING)
        feature = (0.025, 0.020, 0.018)
        cheek = (0.92, 0.34, 0.27)
        if expression == "angry":
            eyes = [(-0.095, 0.055, 0.027), (0.095, 0.055, 0.027)]
            brows = [(-0.16, 0.12, -0.03), (-0.04, 0.08, 0.03), (0.04, 0.08, -0.03), (0.16, 0.12, 0.03)]
            mouth = -0.055
            anxiety = False
        elif expression == "anxious":
            eyes = [(-0.105, 0.06, 0.040), (0.105, 0.06, 0.040)]
            brows = [(-0.15, 0.13, 0.03), (-0.04, 0.15, -0.01), (0.04, 0.15, 0.01), (0.15, 0.13, -0.03)]
            mouth = 0.035
            anxiety = True
        elif expression == "deceit":
            eyes = [(-0.12, 0.055, 0.025), (0.08, 0.065, 0.025)]
            brows = [(-0.16, 0.12, 0.00), (-0.04, 0.12, 0.00), (0.04, 0.13, -0.02), (0.16, 0.10, 0.02)]
            mouth = -0.02
            anxiety = False
        elif expression == "guarded":
            eyes = [(-0.095, 0.052, 0.021), (0.095, 0.052, 0.021)]
            brows = [(-0.16, 0.11, 0.00), (-0.04, 0.11, 0.00), (0.04, 0.11, 0.00), (0.16, 0.11, 0.00)]
            mouth = 0.02
            anxiety = False
        elif expression == "warm":
            eyes = [(-0.095, 0.06, 0.024), (0.095, 0.06, 0.024)]
            brows = [(-0.15, 0.13, 0.01), (-0.04, 0.125, -0.01), (0.04, 0.125, -0.01), (0.15, 0.13, 0.01)]
            mouth = 0.09
            anxiety = False
            self.draw_face_dot(-0.16, -0.03, 0.035, cheek, 0.35)
            self.draw_face_dot(0.16, -0.03, 0.035, cheek, 0.35)
        elif expression == "steady":
            eyes = [(-0.095, 0.055, 0.023), (0.095, 0.055, 0.023)]
            brows = [(-0.15, 0.12, 0.0), (-0.04, 0.12, 0.0), (0.04, 0.12, 0.0), (0.15, 0.12, 0.0)]
            mouth = 0.0
            anxiety = False
        else:
            eyes = [(-0.095, 0.055, 0.025), (0.095, 0.055, 0.025)]
            brows = [(-0.15, 0.12, 0.0), (-0.04, 0.12, 0.0), (0.04, 0.12, 0.0), (0.15, 0.12, 0.0)]
            mouth = 0.01
            anxiety = False
        for ex, ey, er in eyes:
            self.draw_face_dot(ex, ey, er, feature, 1.0)
            if anxiety:
                self.draw_face_dot(ex + 0.006, ey + 0.004, er * 0.38, (0.90, 0.92, 0.86), 1.0)
        glLineWidth(2.0)
        set_color(feature)
        glBegin(GL_LINES)
        glVertex3f(brows[0][0], brows[0][1], 0.262)
        glVertex3f(brows[1][0], brows[1][1], 0.262)
        glVertex3f(brows[2][0], brows[2][1], 0.262)
        glVertex3f(brows[3][0], brows[3][1], 0.262)
        glEnd()
        self.draw_mouth(mouth, feature)
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_face_dot(
        self,
        x: float,
        y: float,
        r: float,
        color: Sequence[float],
        alpha: float,
    ) -> None:
        set_color(color, alpha)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, y, 0.268)
        for i in range(17):
            a = math.tau * i / 16
            glVertex3f(x + math.cos(a) * r, y + math.sin(a) * r, 0.269)
        glEnd()

    def draw_mouth(self, mood: float, color: Sequence[float]) -> None:
        set_color(color)
        glLineWidth(2.4)
        glBegin(GL_LINE_STRIP)
        for i in range(11):
            t = i / 10.0
            x = -0.13 + t * 0.26
            curve = math.sin(t * math.pi)
            y = -0.105 - mood * curve
            if mood < 0.0:
                y = -0.115 - mood * curve
            glVertex3f(x, y, 0.272)
        glEnd()

    def draw_helm_marker(self) -> None:
        if self.distance_to_helm() > 6.0:
            return
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        set_color((0.95, 0.78, 0.34))
        glBegin(GL_LINE_LOOP)
        for i in range(48):
            a = math.tau * i / 48
            glVertex3f(HELM_POS[0] + math.cos(a) * 1.2, 0.18, HELM_POS[1] + math.sin(a) * 1.2)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_ui(self) -> None:
        self.draw_crosshair()
        self.draw_top_hud()
        if self.message_timer > 0.0:
            self.draw_toast()
        if self.active_npc:
            self.draw_dialogue_panel()
        if self.ledger_open:
            self.draw_ledger()
        if self.final_open:
            self.draw_final_council()

    def draw_crosshair(self) -> None:
        if self.ledger_open or self.final_open:
            return
        cx, cy = WIDTH / 2, HEIGHT / 2
        draw_2d_line(cx - 8, cy, cx - 3, cy, WHITE, 1.4)
        draw_2d_line(cx + 3, cy, cx + 8, cy, WHITE, 1.4)
        draw_2d_line(cx, cy - 8, cx, cy - 3, WHITE, 1.4)
        draw_2d_line(cx, cy + 3, cx, cy + 8, WHITE, 1.4)

    def draw_top_hud(self) -> None:
        draw_2d_rect(18, 16, 418, 78, (9, 17, 22), 0.62)
        self.text.draw("Galleon Trait Inquest", 32, 24, GOLD, "big")
        done = sum(1 for npc in self.crew if npc.verdict != "Unknown" and npc.role != "Unassigned")
        self.text.draw(f"Classified {done}/{len(self.crew)} | Helm distance {self.distance_to_helm():.1f}", 34, 62, MIST, "small")
        if self.near_npc and not self.active_npc and not self.ledger_open:
            draw_2d_rect(WIDTH / 2 - 245, HEIGHT - 94, 490, 50, (8, 13, 16), 0.72)
            self.text.draw(f"E: interview {self.near_npc.name}, {self.near_npc.title}", WIDTH / 2, HEIGHT - 82, WHITE, "normal", True)
            self.text.draw("Watch the face. Record a verdict with C and a ship role with R.", WIDTH / 2, HEIGHT - 58, MIST, "small", True)
        elif self.distance_to_helm() < 2.4 and not self.ledger_open:
            draw_2d_rect(WIDTH / 2 - 255, HEIGHT - 88, 510, 44, (8, 13, 16), 0.72)
            self.text.draw("V: convene final council at the helm", WIDTH / 2, HEIGHT - 77, GOLD, "normal", True)
            self.text.draw("Tab opens the crew ledger before you decide.", WIDTH / 2, HEIGHT - 55, MIST, "small", True)

    def draw_toast(self) -> None:
        w = min(760, max(360, self.text.font.size(self.message)[0] + 34))
        x = WIDTH / 2 - w / 2
        y = 106
        draw_2d_rect(x, y, w, 38, (13, 20, 24), 0.70)
        self.text.draw(self.message, WIDTH / 2, y + 9, PARCHMENT, "small", True)

    def draw_dialogue_panel(self) -> None:
        npc = self.active_npc
        if not npc:
            return
        panel_h = 255
        y = HEIGHT - panel_h - 18
        draw_2d_rect(28, y, WIDTH - 56, panel_h, (8, 12, 16), 0.82)
        draw_2d_rect(28, y, 7, panel_h, GOLD, 0.95)
        self.text.draw(f"{npc.name} - {npc.title}", 52, y + 18, GOLD, "big")
        self.text.draw(f"Expression: {npc.expression.title()}   Evidence: {npc.evidence_count}/5", 54, y + 57, MIST, "small")
        self.text.wrapped(self.dialogue_line, 54, y + 82, WIDTH - 420, WHITE, "normal")
        if self.dialogue_clue:
            self.text.wrapped(f"Clue: {self.dialogue_clue}", 54, y + 143, WIDTH - 420, (210, 224, 190), "small")
        side_x = WIDTH - 342
        draw_2d_rect(side_x, y + 20, 286, 90, (22, 31, 34), 0.84)
        self.text.draw("Your read", side_x + 16, y + 32, PARCHMENT, "small")
        self.text.draw(f"C Trait: {npc.verdict}", side_x + 16, y + 55, WHITE, "normal")
        self.text.draw(f"R Role:  {npc.role}", side_x + 16, y + 80, WHITE, "normal")
        opt_y = y + 126
        for i, (_key, label) in enumerate(DIALOGUE_OPTIONS):
            color = (178, 196, 202) if _key in npc.asked else WHITE
            self.text.draw(f"{i + 1}. {label}", side_x + 14, opt_y + i * 24, color, "small")
        self.text.draw("Q: close", side_x + 14, y + panel_h - 28, MIST, "small")

    def draw_ledger(self) -> None:
        draw_2d_rect(76, 58, WIDTH - 152, HEIGHT - 116, (7, 11, 15), 0.91)
        draw_2d_rect(76, 58, WIDTH - 152, 54, (52, 37, 22), 0.85)
        self.text.draw("Crew Ledger", 102, 71, PARCHMENT, "big")
        self.text.draw("Up/Down select | C trait | R role | V final council | Q close", WIDTH - 555, 83, MIST, "small")
        header_y = 128
        self.text.draw("Name", 112, header_y, GOLD, "small")
        self.text.draw("Evidence", 342, header_y, GOLD, "small")
        self.text.draw("Verdict", 470, header_y, GOLD, "small")
        self.text.draw("Best Role", 655, header_y, GOLD, "small")
        self.text.draw("Latest Clues", 840, header_y, GOLD, "small")
        row_y = header_y + 28
        for i, npc in enumerate(self.crew):
            selected = i == self.ledger_index
            if selected:
                draw_2d_rect(96, row_y - 7, WIDTH - 192, 72, (51, 69, 73), 0.72)
            elif i % 2 == 0:
                draw_2d_rect(96, row_y - 7, WIDTH - 192, 72, (19, 27, 31), 0.38)
            name_color = PARCHMENT if selected else WHITE
            self.text.draw(npc.name, 112, row_y, name_color, "normal")
            self.text.draw(npc.title, 112, row_y + 22, MIST, "tiny")
            self.text.draw(f"{npc.evidence_count}/5", 360, row_y + 10, WHITE, "normal")
            verdict_color = self.color_for_verdict(npc.verdict)
            self.text.draw(npc.verdict, 470, row_y + 10, verdict_color, "normal")
            self.text.draw(npc.role, 655, row_y + 10, WHITE, "normal")
            clue_text = npc.clues[-1] if npc.clues else npc.summary
            self.text.wrapped(clue_text, 840, row_y, WIDTH - 960, MIST, "tiny", 1)
            row_y += 78

    def color_for_verdict(self, verdict: str) -> Tuple[int, int, int]:
        if verdict == "Treasonous":
            return (255, 112, 96)
        if verdict == "Mutinous":
            return (255, 160, 86)
        if verdict == "Trustworthy":
            return (126, 218, 152)
        if verdict == "Lovable":
            return (255, 198, 214)
        if verdict == "Likeable":
            return (164, 213, 255)
        return (190, 198, 198)

    def draw_final_council(self) -> None:
        draw_2d_rect(82, 52, WIDTH - 164, HEIGHT - 104, (8, 11, 14), 0.94)
        self.text.draw("Final Council", WIDTH / 2, 70, GOLD, "title", True)
        self.text.draw(f"Score: {self.final_score}", WIDTH / 2, 116, PARCHMENT, "big", True)
        self.text.wrapped(self.final_summary, 132, 156, WIDTH - 264, WHITE, "normal")
        y = 224
        for npc, trait_ok, role_ok in self.final_results:
            row_color = (27, 43, 35) if trait_ok and role_ok else (49, 30, 28)
            draw_2d_rect(130, y - 8, WIDTH - 260, 50, row_color, 0.58)
            self.text.draw(npc.name, 150, y, WHITE, "normal")
            trait_status = "OK" if trait_ok else f"Truth: {npc.trait}"
            role_status = "OK" if role_ok else f"Best: {npc.best_role}"
            self.text.draw(f"Trait {trait_status}", 380, y, GREEN if trait_ok else RED, "small")
            self.text.draw(f"Role {role_status}", 610, y, GREEN if role_ok else RED, "small")
            self.text.draw(f"Your read: {npc.verdict} / {npc.role}", 825, y, MIST, "small")
            y += 60
        self.text.draw("F5 restart | Q close council | Esc quit", WIDTH / 2, HEIGHT - 82, MIST, "small", True)


def run_checks() -> None:
    crew = build_crew()
    assert len(crew) >= 6
    names = {npc.name for npc in crew}
    assert len(names) == len(crew)
    for npc in crew:
        assert npc.trait in TRAIT_CHOICES
        assert npc.best_role in ROLE_CHOICES
        assert set(npc.dialogue) == {key for key, _label in DIALOGUE_OPTIONS}
        for beat in npc.dialogue.values():
            assert beat.expression in {"calm", "warm", "guarded", "angry", "anxious", "deceit", "steady"}
    print(f"OK: {len(crew)} crew profiles, {len(DIALOGUE_OPTIONS)} dialogue tests each.")


def main() -> None:
    if "--check" in sys.argv:
        run_checks()
        return
    Game().run()


if __name__ == "__main__":
    main()
