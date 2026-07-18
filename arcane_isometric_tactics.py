import math
import random
import sys
from dataclasses import dataclass, field

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
except Exception as exc:
    raise SystemExit("This game requires pygame. Install it with: pip install pygame") from exc

try:
    from OpenGL.GL import *
    from OpenGL.GLU import gluLookAt
except Exception as exc:
    raise SystemExit(
        "This game requires PyOpenGL. Install it with: pip install PyOpenGL PyOpenGL_accelerate"
    ) from exc


WIDTH = 1100
HEIGHT = 720
FPS = 60
BOARD_W = 9
BOARD_H = 7
MAX_PARTICLES = 1300

BG = (0.035, 0.045, 0.060, 1.0)
PANEL = (0.055, 0.064, 0.078, 0.90)
PANEL_2 = (0.080, 0.090, 0.108, 0.92)
PANEL_HI = (0.145, 0.170, 0.200, 0.96)
WHITE = (0.92, 0.95, 0.97, 1.0)
MUTED = (0.58, 0.64, 0.69, 1.0)
HP_RED = (0.90, 0.18, 0.18, 1.0)
MP_BLUE = (0.24, 0.54, 0.96, 1.0)
MORALE_GOLD = (0.98, 0.74, 0.26, 1.0)
GREEN = (0.26, 0.86, 0.50, 1.0)
BAD = (1.00, 0.28, 0.24, 1.0)

OBSTACLES = {(4, 3), (2, 1), (6, 2)}
LEY_VENTS = {
    (1, 5): ((0.20, 0.90, 0.72, 1.0), (0.72, 1.00, 0.88, 1.0)),
    (7, 1): ((0.98, 0.45, 0.25, 1.0), (1.00, 0.86, 0.34, 1.0)),
    (5, 5): ((0.52, 0.40, 1.00, 1.0), (0.92, 0.70, 1.00, 1.0)),
}


def clamp(value, low, high):
    return max(low, min(high, value))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(a, b, t):
    return (
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    )


def shade(color, amount, alpha=None):
    return (
        clamp(color[0] * amount, 0.0, 1.0),
        clamp(color[1] * amount, 0.0, 1.0),
        clamp(color[2] * amount, 0.0, 1.0),
        color[3] if alpha is None else alpha,
    )


def dist_tiles(ax, az, bx, bz):
    return math.hypot(ax - bx, az - bz)


def tile_center(x, z):
    return x + 0.5, z + 0.5


@dataclass(frozen=True)
class Spell:
    name: str
    mode: str
    cost: int
    range: float
    radius: float
    hp_delta: int
    mp_delta: int
    morale_delta: int
    caster_hp: int
    caster_mp: int
    caster_morale: int
    colors: tuple
    composition: str
    summary: str


SPELLS = [
    Spell(
        "Sunlance",
        "direct",
        12,
        6.8,
        0.0,
        -34,
        0,
        -9,
        0,
        0,
        0,
        (
            (1.00, 0.72, 0.18, 1.0),
            (1.00, 0.28, 0.10, 1.0),
            (1.00, 0.96, 0.62, 1.0),
        ),
        "beam",
        "Direct HP and morale damage",
    ),
    Spell(
        "Mana Siphon",
        "direct",
        8,
        6.2,
        0.0,
        -8,
        -30,
        -4,
        0,
        18,
        4,
        (
            (0.20, 0.86, 1.00, 1.0),
            (0.52, 0.24, 1.00, 1.0),
            (0.82, 1.00, 1.00, 1.0),
        ),
        "spiral",
        "Direct MP drain, refunds caster",
    ),
    Spell(
        "Dread Bell",
        "aoe",
        14,
        5.8,
        1.65,
        -10,
        0,
        -34,
        0,
        0,
        0,
        (
            (0.85, 0.18, 0.78, 1.0),
            (0.28, 0.08, 0.48, 1.0),
            (1.00, 0.62, 0.95, 1.0),
        ),
        "rings",
        "Area morale collapse",
    ),
    Spell(
        "Prism Storm",
        "aoe",
        20,
        5.2,
        2.05,
        -23,
        -16,
        -13,
        0,
        0,
        0,
        (
            (1.00, 0.22, 0.32, 1.0),
            (0.22, 0.94, 1.00, 1.0),
            (0.42, 1.00, 0.36, 1.0),
            (0.92, 0.42, 1.00, 1.0),
            (1.00, 0.88, 0.20, 1.0),
        ),
        "shards",
        "Area HP, MP, and morale damage",
    ),
    Spell(
        "Rally Sigil",
        "self",
        0,
        0.0,
        0.0,
        18,
        18,
        26,
        0,
        0,
        0,
        (
            (0.28, 0.94, 0.56, 1.0),
            (0.84, 1.00, 0.56, 1.0),
            (0.20, 0.68, 0.94, 1.0),
        ),
        "aura",
        "Restore HP, MP, and morale",
    ),
    Spell(
        "Phase Step",
        "move",
        6,
        3.2,
        0.0,
        0,
        0,
        0,
        0,
        0,
        5,
        (
            (0.32, 0.72, 1.00, 1.0),
            (0.86, 0.92, 1.00, 1.0),
            (0.46, 0.32, 1.00, 1.0),
        ),
        "blink",
        "Reposition and steady morale",
    ),
]

ENEMY_BOLT = Spell(
    "Cinder Hex",
    "direct",
    8,
    5.9,
    0.0,
    -13,
    0,
    -7,
    0,
    0,
    0,
    ((1.00, 0.32, 0.16, 1.0), (0.70, 0.10, 0.08, 1.0), (1.00, 0.78, 0.32, 1.0)),
    "ember",
    "Enemy direct pressure",
)
ENEMY_LEECH = Spell(
    "Static Leech",
    "direct",
    10,
    5.6,
    0.0,
    -6,
    -15,
    -5,
    0,
    8,
    0,
    ((0.18, 0.70, 1.00, 1.0), (0.62, 0.22, 0.90, 1.0), (0.86, 1.00, 1.00, 1.0)),
    "spiral",
    "Enemy MP drain",
)
ENEMY_CHORUS = Spell(
    "Panic Chorus",
    "aoe",
    12,
    5.3,
    1.2,
    -4,
    -5,
    -18,
    0,
    0,
    0,
    ((0.90, 0.18, 0.70, 1.0), (0.22, 0.04, 0.28, 1.0), (1.00, 0.62, 0.92, 1.0)),
    "rings",
    "Enemy morale strike",
)
ENEMY_WEAK = Spell(
    "Rust Claw",
    "direct",
    0,
    1.8,
    0.0,
    -8,
    0,
    -4,
    0,
    0,
    0,
    ((0.72, 0.62, 0.40, 1.0), (0.38, 0.32, 0.22, 1.0), (0.90, 0.78, 0.50, 1.0)),
    "dust",
    "Weak close attack",
)


@dataclass
class Entity:
    name: str
    team: str
    x: int
    z: int
    hp: int
    max_hp: int
    mp: int
    max_mp: int
    morale: int
    max_morale: int
    color: tuple
    accent: tuple
    phase: float = field(default_factory=lambda: random.random() * math.tau)

    @property
    def defeated(self):
        return self.hp <= 0 or self.morale <= 0

    @property
    def center(self):
        return tile_center(self.x, self.z)


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    life: float
    size: float
    c0: tuple
    c1: tuple
    kind: str
    gravity: float = 0.0
    drag: float = 0.0
    age: float = 0.0


@dataclass
class Projectile:
    sx: float
    sy: float
    sz: float
    ex: float
    ey: float
    ez: float
    spell: Spell
    owner: Entity
    target: object
    mode: str
    duration: float
    arc: float
    age: float = 0.0
    resolved: bool = False


class GLText:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        self.tiny = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 38, bold=True)
        self.cache = {}

    def _font(self, size):
        if size == "tiny":
            return self.tiny
        if size == "small":
            return self.small
        if size == "big":
            return self.big
        return self.font

    def _texture(self, text, color, font):
        text = str(text)
        key = (text, color, id(font))
        if key in self.cache:
            return self.cache[key]

        surf = font.render(text, True, color)
        data = pygame.image.tostring(surf, "RGBA", True)
        w, h = surf.get_width(), surf.get_height()
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        self.cache[key] = (tex, w, h)
        return tex, w, h

    def draw(self, text, x, y, color=(240, 244, 248), size="normal", center=False):
        font = self._font(size)
        tex, w, h = self._texture(text, color, font)
        if center:
            x -= w / 2
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x, y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x + w, y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x + w, y + h)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x, y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def cleanup(self):
        for tex, _, _ in self.cache.values():
            glDeleteTextures(int(tex))
        self.cache.clear()


def color4(color):
    glColor4f(color[0], color[1], color[2], color[3])


def draw_rect(x, y, w, h, color):
    color4(color)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_line_2d(x1, y1, x2, y2, color, width=1.0):
    glLineWidth(width)
    color4(color)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()
    glLineWidth(1.0)


def draw_box_center(cx, cy, cz, sx, sy, sz, color):
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2

    glBegin(GL_QUADS)
    color4(shade(color, 1.18))
    glVertex3f(x0, y1, z0)
    glVertex3f(x1, y1, z0)
    glVertex3f(x1, y1, z1)
    glVertex3f(x0, y1, z1)

    color4(shade(color, 0.72))
    glVertex3f(x0, y0, z1)
    glVertex3f(x1, y0, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x0, y1, z1)

    color4(shade(color, 0.84))
    glVertex3f(x1, y0, z0)
    glVertex3f(x1, y0, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z0)

    color4(shade(color, 0.56))
    glVertex3f(x0, y0, z0)
    glVertex3f(x0, y1, z0)
    glVertex3f(x0, y1, z1)
    glVertex3f(x0, y0, z1)

    color4(shade(color, 0.64))
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y0, z0)
    glVertex3f(x1, y1, z0)
    glVertex3f(x0, y1, z0)

    color4(shade(color, 0.46))
    glVertex3f(x0, y0, z0)
    glVertex3f(x0, y0, z1)
    glVertex3f(x1, y0, z1)
    glVertex3f(x1, y0, z0)
    glEnd()


def draw_octahedron(cx, cy, cz, radius, color):
    top = (cx, cy + radius, cz)
    bottom = (cx, cy - radius, cz)
    points = [
        (cx + radius, cy, cz),
        (cx, cy, cz + radius),
        (cx - radius, cy, cz),
        (cx, cy, cz - radius),
    ]

    glBegin(GL_TRIANGLES)
    for i, p0 in enumerate(points):
        p1 = points[(i + 1) % len(points)]
        color4(shade(color, 1.25))
        glVertex3f(*top)
        color4(color)
        glVertex3f(*p0)
        glVertex3f(*p1)

        color4(shade(color, 0.55))
        glVertex3f(*bottom)
        color4(shade(color, 0.86))
        glVertex3f(*p1)
        glVertex3f(*p0)
    glEnd()


def draw_disc3d(cx, cz, radius, color, y=0.026, segments=64):
    color4(color)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(cx, y, cz)
    for i in range(segments + 1):
        a = math.tau * i / segments
        glVertex3f(cx + math.cos(a) * radius, y, cz + math.sin(a) * radius)
    glEnd()


def draw_ring3d(cx, cz, radius, color, y=0.035, segments=80, width=2.0):
    glLineWidth(width)
    color4(color)
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        a = math.tau * i / segments
        glVertex3f(cx + math.cos(a) * radius, y, cz + math.sin(a) * radius)
    glEnd()
    glLineWidth(1.0)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Arcane Isometric Tactics - PyOpenGL")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.text = GLText()
        self.reset()
        self.init_gl()

    def init_gl(self):
        glViewport(0, 0, WIDTH, HEIGHT)
        glClearColor(*BG)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glLineWidth(1.0)

    def reset(self):
        self.player = Entity(
            "The Votary",
            "player",
            1,
            3,
            110,
            110,
            78,
            88,
            92,
            100,
            (0.22, 0.62, 0.95, 1.0),
            (0.88, 0.96, 1.00, 1.0),
        )
        self.enemies = [
            Entity("Glass Heretic", "enemy", 7, 1, 74, 74, 48, 58, 62, 62, (0.76, 0.20, 0.30, 1.0), (1.00, 0.60, 0.62, 1.0)),
            Entity("Mire Abbot", "enemy", 6, 5, 92, 92, 68, 68, 78, 78, (0.32, 0.70, 0.42, 1.0), (0.78, 1.00, 0.54, 1.0)),
            Entity("Choir Warden", "enemy", 3, 5, 68, 68, 82, 82, 85, 85, (0.52, 0.30, 0.88, 1.0), (0.90, 0.66, 1.00, 1.0)),
            Entity("Ash Templar", "enemy", 7, 4, 104, 104, 42, 50, 68, 68, (0.82, 0.44, 0.20, 1.0), (1.00, 0.82, 0.46, 1.0)),
        ]
        self.selected_spell = 0
        self.target_index = 0
        self.cursor_x = 7
        self.cursor_z = 3
        self.phase = "player"
        self.phase_timer = 0.0
        self.enemy_index = 0
        self.enemy_timer = 0.0
        self.turn_number = 1
        self.time = 0.0
        self.message = "Round 1: choose a spell."
        self.particles = []
        self.projectiles = []
        self.pulse = 0.0
        self.ambient_timer = 0.0
        self.camera_scale = 6.15

    def run(self):
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.033)
            self.handle_events()
            self.update(dt)
            self.render()
            pygame.display.flip()

    def quit(self):
        self.text.cleanup()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                self.quit()
            if event.key == pygame.K_r:
                self.reset()
                continue

            if self.phase in ("victory", "defeat"):
                continue

            if self.phase != "player":
                continue

            if pygame.K_1 <= event.key <= pygame.K_6:
                self.selected_spell = event.key - pygame.K_1
                self.align_cursor_to_spell()
                continue

            if event.key == pygame.K_TAB:
                self.cycle_target(1)
                continue

            if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                self.try_cast_selected()
                continue

            if event.key == pygame.K_f:
                self.focus_action()
                continue

            dx, dz = 0, 0
            if event.key in (pygame.K_LEFT, pygame.K_a):
                dx = -1
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                dx = 1
            elif event.key in (pygame.K_UP, pygame.K_w):
                dz = -1
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                dz = 1

            if dx or dz:
                if SPELLS[self.selected_spell].mode == "direct":
                    self.cycle_target(1 if dx + dz > 0 else -1)
                else:
                    self.cursor_x = int(clamp(self.cursor_x + dx, 0, BOARD_W - 1))
                    self.cursor_z = int(clamp(self.cursor_z + dz, 0, BOARD_H - 1))

    def align_cursor_to_spell(self):
        spell = SPELLS[self.selected_spell]
        if spell.mode == "direct":
            target = self.current_target()
            if target:
                self.cursor_x, self.cursor_z = target.x, target.z
        elif spell.mode == "aoe":
            target = self.current_target()
            if target:
                self.cursor_x, self.cursor_z = target.x, target.z
        elif spell.mode == "move":
            self.cursor_x, self.cursor_z = self.player.x, self.player.z

    def active_enemies(self):
        return [enemy for enemy in self.enemies if not enemy.defeated]

    def current_target(self):
        enemies = self.active_enemies()
        if not enemies:
            return None
        self.target_index %= len(enemies)
        return enemies[self.target_index]

    def cycle_target(self, amount):
        enemies = self.active_enemies()
        if not enemies:
            return
        self.target_index = (self.target_index + amount) % len(enemies)
        target = enemies[self.target_index]
        self.cursor_x, self.cursor_z = target.x, target.z
        self.message = f"Targeting {target.name}."

    def update(self, dt):
        self.time += dt
        self.pulse = (math.sin(self.time * 4.0) + 1.0) * 0.5
        self.update_particles(dt)
        self.update_projectiles(dt)
        self.spawn_ambient(dt)

        if self.phase == "animating":
            self.phase_timer -= dt
            if self.phase_timer <= 0.0 and not self.projectiles:
                self.resolve_outcome_or_enemy_turn()
        elif self.phase == "enemy":
            self.update_enemy_turn(dt)

    def update_particles(self, dt):
        alive = []
        for p in self.particles:
            p.age += dt
            if p.age >= p.life:
                continue
            p.vy += p.gravity * dt
            if p.drag:
                drag = max(0.0, 1.0 - p.drag * dt)
                p.vx *= drag
                p.vy *= drag
                p.vz *= drag
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt
            alive.append(p)
        if len(alive) > MAX_PARTICLES:
            alive = alive[-MAX_PARTICLES:]
        self.particles = alive

    def update_projectiles(self, dt):
        active = []
        for projectile in self.projectiles:
            projectile.age += dt
            x, y, z = self.projectile_position(projectile)
            self.spawn_trail(projectile.spell, x, y, z, 3)
            if projectile.age >= projectile.duration and not projectile.resolved:
                projectile.resolved = True
                self.resolve_projectile(projectile)
            if projectile.age < projectile.duration + 0.05:
                active.append(projectile)
        self.projectiles = active

    def spawn_ambient(self, dt):
        self.ambient_timer += dt
        while self.ambient_timer >= 0.035:
            self.ambient_timer -= 0.035
            if random.random() < 0.65:
                (tx, tz), colors = random.choice(list(LEY_VENTS.items()))
                cx, cz = tile_center(tx, tz)
                angle = random.random() * math.tau
                radius = random.random() * 0.28
                self.add_particle(
                    cx + math.cos(angle) * radius,
                    0.08,
                    cz + math.sin(angle) * radius,
                    random.uniform(-0.08, 0.08),
                    random.uniform(0.25, 0.70),
                    random.uniform(-0.08, 0.08),
                    random.uniform(0.55, 1.15),
                    random.uniform(0.035, 0.075),
                    colors[0],
                    colors[1],
                    random.choice(("mote", "spark", "smoke")),
                    gravity=-0.02,
                    drag=0.20,
                )

            if random.random() < 0.40 and self.phase == "player":
                px, pz = self.player.center
                colors = SPELLS[self.selected_spell].colors
                self.add_particle(
                    px + random.uniform(-0.30, 0.30),
                    random.uniform(0.18, 1.05),
                    pz + random.uniform(-0.30, 0.30),
                    random.uniform(-0.05, 0.05),
                    random.uniform(0.08, 0.26),
                    random.uniform(-0.05, 0.05),
                    random.uniform(0.35, 0.70),
                    random.uniform(0.026, 0.055),
                    random.choice(colors),
                    random.choice(colors),
                    "mote",
                    gravity=-0.02,
                    drag=0.35,
                )

    def try_cast_selected(self):
        spell = SPELLS[self.selected_spell]
        if spell.mode == "direct":
            target = self.current_target()
            if not target:
                self.message = "No active target remains."
                return
            distance = dist_tiles(self.player.x, self.player.z, target.x, target.z)
            if distance > spell.range:
                self.message = f"{target.name} is out of range."
                self.warn_at_tile(target.x, target.z)
                return
            if not self.spend_mana(self.player, spell.cost):
                return
            self.cast_projectile(self.player, target, spell, "direct")
            self.consume_player_turn(1.05, f"{spell.name} streaks toward {target.name}.")
            return

        if spell.mode == "aoe":
            distance = dist_tiles(self.player.x, self.player.z, self.cursor_x, self.cursor_z)
            if distance > spell.range:
                self.message = "The chosen sigil is beyond reach."
                self.warn_at_tile(self.cursor_x, self.cursor_z)
                return
            if not self.spend_mana(self.player, spell.cost):
                return
            self.cast_projectile(self.player, (self.cursor_x, self.cursor_z), spell, "aoe")
            self.consume_player_turn(1.20, f"{spell.name} blooms over the grid.")
            return

        if spell.mode == "self":
            if not self.spend_mana(self.player, spell.cost):
                return
            px, pz = self.player.center
            self.spawn_aura(px, 0.18, pz, spell.colors, 130, 1.35)
            self.apply_spell_effect(self.player, spell, self.player)
            self.consume_player_turn(1.00, f"{spell.name} restores the Votary.")
            return

        if spell.mode == "move":
            self.try_phase_step(spell)

    def try_phase_step(self, spell):
        distance = dist_tiles(self.player.x, self.player.z, self.cursor_x, self.cursor_z)
        if distance > spell.range:
            self.message = "Phase Step cannot reach that tile."
            self.warn_at_tile(self.cursor_x, self.cursor_z)
            return
        if not self.is_tile_free(self.cursor_x, self.cursor_z, ignore_player=True):
            self.message = "That tile is occupied or blocked."
            self.warn_at_tile(self.cursor_x, self.cursor_z)
            return
        if not self.spend_mana(self.player, spell.cost):
            return

        old_x, old_z = self.player.center
        self.spawn_burst(old_x, 0.25, old_z, spell.colors, 80, 1.0, ("mote", "smoke", "shard"))
        self.player.x = self.cursor_x
        self.player.z = self.cursor_z
        new_x, new_z = self.player.center
        self.spawn_burst(new_x, 0.25, new_z, spell.colors, 110, 1.2, ("mote", "spark", "shard"))
        self.apply_delta(self.player, 0, 0, spell.caster_morale)
        self.consume_player_turn(0.72, "Phase Step folds the board.")

    def focus_action(self):
        self.apply_delta(self.player, 0, 16, 14)
        px, pz = self.player.center
        colors = ((0.28, 0.92, 0.70, 1.0), (0.38, 0.66, 1.00, 1.0), (0.95, 0.92, 0.50, 1.0))
        self.spawn_aura(px, 0.18, pz, colors, 100, 1.0)
        self.consume_player_turn(0.72, "The Votary focuses and recovers resolve.")

    def spend_mana(self, caster, amount):
        if caster.mp < amount:
            self.message = "Not enough mana."
            cx, cz = caster.center
            self.spawn_burst(cx, 0.55, cz, (BAD, (0.55, 0.06, 0.12, 1.0)), 35, 0.45, ("smoke", "spark"))
            return False
        caster.mp = max(0, caster.mp - amount)
        return True

    def consume_player_turn(self, delay, message):
        self.phase = "animating"
        self.phase_timer = delay
        self.message = message

    def cast_projectile(self, caster, target, spell, mode):
        sx, sz = caster.center
        sy = 0.95
        if mode == "direct":
            ex, ez = target.center
            ey = 0.90
            duration = 0.42 if caster.team == "player" else 0.50
            arc = 0.50
        else:
            tx, tz = target
            ex, ez = tile_center(tx, tz)
            ey = 0.16
            duration = 0.55 if caster.team == "player" else 0.48
            arc = 1.35

        self.projectiles.append(Projectile(sx, sy, sz, ex, ey, ez, spell, caster, target, mode, duration, arc))
        self.spawn_cast_flare(caster, spell)

    def projectile_position(self, projectile):
        t = clamp(projectile.age / projectile.duration, 0.0, 1.0)
        eased = t * t * (3.0 - 2.0 * t)
        x = lerp(projectile.sx, projectile.ex, eased)
        z = lerp(projectile.sz, projectile.ez, eased)
        y = lerp(projectile.sy, projectile.ey, eased) + math.sin(t * math.pi) * projectile.arc
        return x, y, z

    def resolve_projectile(self, projectile):
        spell = projectile.spell
        x, y, z = projectile.ex, projectile.ey, projectile.ez
        if projectile.mode == "direct":
            target = projectile.target
            if isinstance(target, Entity) and not target.defeated:
                self.apply_spell_effect(target, spell, projectile.owner)
                x, z = target.center
                y = 0.72
        elif projectile.mode == "aoe":
            tx, tz = projectile.target
            x, z = tile_center(tx, tz)
            y = 0.12
            if projectile.owner.team == "player":
                for enemy in self.active_enemies():
                    ex, ez = enemy.center
                    if math.hypot(ex - x, ez - z) <= spell.radius:
                        self.apply_spell_effect(enemy, spell, projectile.owner)
            else:
                px, pz = self.player.center
                if math.hypot(px - x, pz - z) <= spell.radius:
                    self.apply_spell_effect(self.player, spell, projectile.owner)

        self.spawn_impact(spell, x, y, z)
        self.check_outcome()

    def apply_spell_effect(self, target, spell, caster):
        self.apply_delta(target, spell.hp_delta, spell.mp_delta, spell.morale_delta)
        if spell.caster_hp or spell.caster_mp or spell.caster_morale:
            self.apply_delta(caster, spell.caster_hp, spell.caster_mp, spell.caster_morale)

    def apply_delta(self, entity, hp_delta, mp_delta, morale_delta):
        was_defeated = entity.defeated
        if hp_delta:
            entity.hp = int(clamp(entity.hp + hp_delta, 0, entity.max_hp))
        if mp_delta:
            entity.mp = int(clamp(entity.mp + mp_delta, 0, entity.max_mp))
        if morale_delta:
            entity.morale = int(clamp(entity.morale + morale_delta, 0, entity.max_morale))

        if not was_defeated and entity.defeated:
            cx, cz = entity.center
            colors = (entity.color, entity.accent, (0.90, 0.90, 0.92, 1.0))
            self.spawn_burst(cx, 0.55, cz, colors, 150, 1.7, ("shard", "spark", "smoke"))
            if entity.team == "enemy":
                self.message = f"{entity.name} breaks."
            else:
                self.message = "The Votary collapses."

    def resolve_outcome_or_enemy_turn(self):
        if self.check_outcome():
            return
        self.start_enemy_turn()

    def check_outcome(self):
        if self.player.defeated:
            self.phase = "defeat"
            self.message = "Defeat."
            return True
        if not self.active_enemies():
            self.phase = "victory"
            self.message = "Victory."
            return True
        return False

    def start_enemy_turn(self):
        self.phase = "enemy"
        self.enemy_index = 0
        self.enemy_timer = 0.18
        self.message = "The hostile chorus answers."

    def update_enemy_turn(self, dt):
        self.enemy_timer -= dt
        if self.enemy_timer > 0.0 or self.projectiles:
            return

        while self.enemy_index < len(self.enemies):
            enemy = self.enemies[self.enemy_index]
            self.enemy_index += 1
            if enemy.defeated:
                continue
            self.enemy_act(enemy)
            self.enemy_timer = 0.82
            return

        self.start_player_turn()

    def start_player_turn(self):
        self.turn_number += 1
        self.phase = "player"
        self.phase_timer = 0.0
        self.player.mp = int(clamp(self.player.mp + 7, 0, self.player.max_mp))
        self.player.morale = int(clamp(self.player.morale + 3, 0, self.player.max_morale))
        for enemy in self.enemies:
            if not enemy.defeated:
                enemy.mp = int(clamp(enemy.mp + 4, 0, enemy.max_mp))
                enemy.morale = int(clamp(enemy.morale + 1, 0, enemy.max_morale))
        self.align_cursor_to_spell()
        self.message = f"Round {self.turn_number}: the board is yours."

    def enemy_act(self, enemy):
        if enemy.morale <= 16 and random.random() < 0.34:
            enemy.morale = max(0, enemy.morale - 4)
            cx, cz = enemy.center
            self.spawn_burst(cx, 0.45, cz, ((0.75, 0.12, 0.18, 1.0), enemy.accent), 45, 0.65, ("smoke", "spark"))
            self.message = f"{enemy.name} hesitates."
            return

        px, pz = self.player.center
        ex, ez = enemy.center
        distance = math.hypot(px - ex, pz - ez)
        spell = self.choose_enemy_spell(enemy, distance)

        if spell is None:
            self.enemy_move_toward_player(enemy)
            return

        if spell.mode == "aoe":
            enemy.mp = max(0, enemy.mp - spell.cost)
            self.cast_projectile(enemy, (self.player.x, self.player.z), spell, "aoe")
        else:
            if spell.cost:
                enemy.mp = max(0, enemy.mp - spell.cost)
            self.cast_projectile(enemy, self.player, spell, "direct")
        self.message = f"{enemy.name} casts {spell.name}."

    def choose_enemy_spell(self, enemy, distance):
        if distance > 5.8:
            return None
        if distance <= 1.9 and enemy.mp < 8:
            return ENEMY_WEAK
        choices = []
        if enemy.mp >= ENEMY_BOLT.cost and distance <= ENEMY_BOLT.range:
            choices.append((ENEMY_BOLT, 4))
        if enemy.mp >= ENEMY_LEECH.cost and self.player.mp > 15 and distance <= ENEMY_LEECH.range:
            choices.append((ENEMY_LEECH, 3))
        if enemy.mp >= ENEMY_CHORUS.cost and self.player.morale > 22 and distance <= ENEMY_CHORUS.range:
            choices.append((ENEMY_CHORUS, 2))
        if not choices:
            if distance <= ENEMY_WEAK.range:
                return ENEMY_WEAK
            return None

        total = sum(weight for _, weight in choices)
        roll = random.uniform(0, total)
        acc = 0
        for spell, weight in choices:
            acc += weight
            if roll <= acc:
                return spell
        return choices[0][0]

    def enemy_move_toward_player(self, enemy):
        dx = self.player.x - enemy.x
        dz = self.player.z - enemy.z
        options = []
        if abs(dx) >= abs(dz):
            options.append((1 if dx > 0 else -1 if dx < 0 else 0, 0))
            options.append((0, 1 if dz > 0 else -1 if dz < 0 else 0))
        else:
            options.append((0, 1 if dz > 0 else -1 if dz < 0 else 0))
            options.append((1 if dx > 0 else -1 if dx < 0 else 0, 0))
        options.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])

        for ox, oz in options:
            if ox == 0 and oz == 0:
                continue
            nx, nz = enemy.x + ox, enemy.z + oz
            if self.is_tile_free(nx, nz):
                old_x, old_z = enemy.center
                enemy.x, enemy.z = nx, nz
                new_x, new_z = enemy.center
                self.spawn_burst(old_x, 0.15, old_z, (enemy.color, enemy.accent), 28, 0.45, ("smoke", "mote"))
                self.spawn_burst(new_x, 0.15, new_z, (enemy.color, enemy.accent), 38, 0.55, ("spark", "mote"))
                self.message = f"{enemy.name} advances."
                return

        self.message = f"{enemy.name} holds position."

    def is_tile_free(self, x, z, ignore_player=False):
        if x < 0 or z < 0 or x >= BOARD_W or z >= BOARD_H:
            return False
        if (x, z) in OBSTACLES:
            return False
        if not ignore_player and self.player.x == x and self.player.z == z and not self.player.defeated:
            return False
        for enemy in self.active_enemies():
            if enemy.x == x and enemy.z == z:
                return False
        return True

    def add_particle(self, x, y, z, vx, vy, vz, life, size, c0, c1, kind, gravity=0.0, drag=0.0):
        if len(self.particles) >= MAX_PARTICLES:
            self.particles.pop(0)
        self.particles.append(Particle(x, y, z, vx, vy, vz, life, size, c0, c1, kind, gravity, drag))

    def spawn_cast_flare(self, caster, spell):
        cx, cz = caster.center
        self.spawn_ring_particles(cx, 0.12, cz, 0.52, spell.colors, 42, 0.72)
        self.spawn_burst(cx, 0.68, cz, spell.colors, 56, 0.85, ("mote", "spark", "shard"))

    def spawn_trail(self, spell, x, y, z, count):
        for _ in range(count):
            angle = random.random() * math.tau
            radial = random.uniform(0.02, 0.10)
            speed = random.uniform(0.05, 0.26)
            kind = random.choice(("mote", "spark", "smoke") if spell.composition != "shards" else ("shard", "spark", "mote"))
            self.add_particle(
                x + math.cos(angle) * radial,
                y + random.uniform(-0.04, 0.04),
                z + math.sin(angle) * radial,
                math.cos(angle) * speed,
                random.uniform(-0.05, 0.10),
                math.sin(angle) * speed,
                random.uniform(0.28, 0.62),
                random.uniform(0.025, 0.070),
                random.choice(spell.colors),
                random.choice(spell.colors),
                kind,
                gravity=random.uniform(-0.10, 0.06),
                drag=0.85,
            )

    def spawn_impact(self, spell, x, y, z):
        if spell.mode == "aoe":
            self.spawn_ring_particles(x, 0.08, z, max(0.8, spell.radius), spell.colors, 120, 1.15)
            self.spawn_burst(x, y + 0.18, z, spell.colors, 170, 1.85, ("spark", "shard", "mote", "smoke"))
        elif spell.composition == "spiral":
            self.spawn_ring_particles(x, 0.45, z, 0.75, spell.colors, 80, 1.0)
            self.spawn_burst(x, y, z, spell.colors, 90, 1.25, ("mote", "spark", "smoke"))
        else:
            self.spawn_burst(x, y, z, spell.colors, 120, 1.45, ("spark", "shard", "mote"))

    def spawn_burst(self, x, y, z, colors, count, speed, kinds):
        for _ in range(count):
            angle = random.random() * math.tau
            lift = random.uniform(0.10, 1.25)
            horizontal = random.uniform(0.05, speed)
            c0 = random.choice(colors)
            c1 = random.choice(colors)
            self.add_particle(
                x + random.uniform(-0.05, 0.05),
                y + random.uniform(-0.04, 0.09),
                z + random.uniform(-0.05, 0.05),
                math.cos(angle) * horizontal,
                lift,
                math.sin(angle) * horizontal,
                random.uniform(0.42, 1.25),
                random.uniform(0.030, 0.105),
                c0,
                c1,
                random.choice(kinds),
                gravity=random.uniform(-1.40, -0.45),
                drag=random.uniform(0.18, 0.85),
            )

    def spawn_aura(self, x, y, z, colors, count, radius):
        for i in range(count):
            angle = math.tau * i / max(1, count) + random.uniform(-0.10, 0.10)
            ring = random.uniform(radius * 0.30, radius)
            tangent = angle + math.pi / 2
            self.add_particle(
                x + math.cos(angle) * ring,
                y + random.uniform(0.05, 1.1),
                z + math.sin(angle) * ring,
                math.cos(tangent) * random.uniform(0.15, 0.65),
                random.uniform(0.05, 0.42),
                math.sin(tangent) * random.uniform(0.15, 0.65),
                random.uniform(0.65, 1.30),
                random.uniform(0.030, 0.080),
                random.choice(colors),
                random.choice(colors),
                random.choice(("mote", "spark", "smoke")),
                gravity=random.uniform(-0.32, -0.05),
                drag=0.36,
            )

    def spawn_ring_particles(self, x, y, z, radius, colors, count, life):
        for i in range(count):
            angle = math.tau * i / max(1, count)
            speed = random.uniform(0.20, 0.72)
            self.add_particle(
                x + math.cos(angle) * radius * random.uniform(0.75, 1.05),
                y + random.uniform(0.00, 0.16),
                z + math.sin(angle) * radius * random.uniform(0.75, 1.05),
                math.cos(angle) * speed,
                random.uniform(0.02, 0.30),
                math.sin(angle) * speed,
                random.uniform(life * 0.60, life * 1.25),
                random.uniform(0.025, 0.070),
                random.choice(colors),
                random.choice(colors),
                random.choice(("spark", "mote", "smoke")),
                gravity=random.uniform(-0.28, 0.04),
                drag=0.22,
            )

    def warn_at_tile(self, x, z):
        cx, cz = tile_center(x, z)
        self.spawn_ring_particles(cx, 0.08, cz, 0.48, (BAD, (1.00, 0.70, 0.48, 1.0)), 45, 0.55)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_3d()
        self.render_world()
        self.render_particles()
        self.setup_2d()
        self.render_ui()
        self.end_2d()

    def setup_3d(self):
        glViewport(0, 0, WIDTH, HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = WIDTH / HEIGHT
        scale = self.camera_scale
        glOrtho(-scale * aspect, scale * aspect, -scale, scale, -40.0, 40.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        cx, cz = BOARD_W / 2, BOARD_H / 2
        gluLookAt(cx + 5.4, 7.6, cz + 6.4, cx, 0.0, cz, 0.0, 1.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def setup_2d(self):
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def end_2d(self):
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)

    def render_world(self):
        self.draw_board()
        self.draw_targeting()
        self.draw_obstacles()
        self.draw_entity(self.player)
        for enemy in self.enemies:
            self.draw_entity(enemy)
        self.draw_projectiles()

    def draw_board(self):
        for z in range(BOARD_H):
            for x in range(BOARD_W):
                base = (0.118, 0.135, 0.147, 1.0) if (x + z) % 2 == 0 else (0.095, 0.108, 0.122, 1.0)
                if (x, z) in LEY_VENTS:
                    base = lerp_color(base, LEY_VENTS[(x, z)][0], 0.22)
                self.draw_tile(x, z, base)

        glLineWidth(1.0)
        color4((0.20, 0.24, 0.27, 0.92))
        glBegin(GL_LINES)
        for x in range(BOARD_W + 1):
            glVertex3f(x, 0.032, 0)
            glVertex3f(x, 0.032, BOARD_H)
        for z in range(BOARD_H + 1):
            glVertex3f(0, 0.032, z)
            glVertex3f(BOARD_W, 0.032, z)
        glEnd()

        for (x, z), colors in LEY_VENTS.items():
            cx, cz = tile_center(x, z)
            draw_disc3d(cx, cz, 0.28 + self.pulse * 0.07, shade(colors[0], 1.0, 0.20))
            draw_ring3d(cx, cz, 0.35 + self.pulse * 0.08, shade(colors[1], 1.0, 0.72), width=1.6)

    def draw_tile(self, x, z, color):
        y = -0.07
        glBegin(GL_QUADS)
        color4(shade(color, 1.10))
        glVertex3f(x, 0.0, z)
        glVertex3f(x + 1, 0.0, z)
        glVertex3f(x + 1, 0.0, z + 1)
        glVertex3f(x, 0.0, z + 1)

        color4(shade(color, 0.48))
        glVertex3f(x, y, z + 1)
        glVertex3f(x + 1, y, z + 1)
        glVertex3f(x + 1, 0.0, z + 1)
        glVertex3f(x, 0.0, z + 1)

        color4(shade(color, 0.58))
        glVertex3f(x + 1, y, z)
        glVertex3f(x + 1, y, z + 1)
        glVertex3f(x + 1, 0.0, z + 1)
        glVertex3f(x + 1, 0.0, z)
        glEnd()

    def draw_targeting(self):
        if self.phase not in ("player", "animating"):
            return

        spell = SPELLS[self.selected_spell]
        px, pz = self.player.center

        if spell.mode == "direct":
            target = self.current_target()
            if target:
                tx, tz = target.center
                draw_ring3d(tx, tz, 0.48 + self.pulse * 0.10, shade(spell.colors[0], 1.0, 0.90), y=0.055, width=2.6)
                self.draw_arcane_line(px, 0.10, pz, tx, 0.10, tz, shade(spell.colors[1], 1.0, 0.46), 2.0)
        elif spell.mode == "aoe":
            cx, cz = tile_center(self.cursor_x, self.cursor_z)
            draw_disc3d(cx, cz, spell.radius, shade(spell.colors[1], 1.0, 0.16), y=0.045)
            draw_ring3d(cx, cz, spell.radius + self.pulse * 0.08, shade(spell.colors[0], 1.0, 0.78), y=0.060, width=2.5)
            self.draw_arcane_line(px, 0.10, pz, cx, 0.10, cz, shade(spell.colors[2 % len(spell.colors)], 1.0, 0.36), 1.8)
        elif spell.mode == "move":
            cx, cz = tile_center(self.cursor_x, self.cursor_z)
            draw_disc3d(cx, cz, 0.45, shade(spell.colors[0], 1.0, 0.20), y=0.050)
            draw_ring3d(cx, cz, 0.48 + self.pulse * 0.10, shade(spell.colors[1], 1.0, 0.82), y=0.065, width=2.4)

    def draw_arcane_line(self, x1, y1, z1, x2, y2, z2, color, width):
        glLineWidth(width)
        color4(color)
        glBegin(GL_LINES)
        glVertex3f(x1, y1, z1)
        glVertex3f(x2, y2, z2)
        glEnd()
        glLineWidth(1.0)

    def draw_obstacles(self):
        for x, z in OBSTACLES:
            cx, cz = tile_center(x, z)
            draw_box_center(cx, 0.22, cz, 0.72, 0.42, 0.72, (0.13, 0.16, 0.18, 1.0))
            draw_octahedron(cx - 0.10, 0.72, cz + 0.02, 0.36, (0.28, 0.88, 0.78, 0.86))
            draw_octahedron(cx + 0.18, 0.50, cz - 0.14, 0.25, (0.62, 0.45, 1.00, 0.82))

    def draw_entity(self, entity):
        cx, cz = entity.center
        defeated = entity.defeated
        bob = 0.035 * math.sin(self.time * 3.0 + entity.phase) if not defeated else 0.0
        body_color = entity.color if not defeated else (0.25, 0.27, 0.29, 0.72)
        accent = entity.accent if not defeated else (0.44, 0.46, 0.48, 0.65)

        if defeated:
            draw_disc3d(cx, cz, 0.42, (0.03, 0.035, 0.04, 0.35), y=0.042)
            draw_box_center(cx, 0.10, cz, 0.58, 0.18, 0.70, body_color)
            return

        draw_ring3d(cx, cz, 0.44 + 0.035 * math.sin(self.time * 2.0 + entity.phase), shade(accent, 1.0, 0.56), y=0.052, width=1.6)
        if entity.team == "player":
            draw_box_center(cx, 0.36 + bob, cz, 0.48, 0.72, 0.42, body_color)
            draw_box_center(cx, 0.82 + bob, cz - 0.02, 0.35, 0.22, 0.35, shade(body_color, 1.16))
            draw_octahedron(cx, 1.12 + bob, cz, 0.18, accent)
            self.draw_staff(cx, cz, accent)
        else:
            draw_box_center(cx, 0.32 + bob, cz, 0.52, 0.62, 0.52, body_color)
            draw_octahedron(cx, 0.86 + bob, cz, 0.26, accent)
            draw_box_center(cx, 0.30 + bob, cz + 0.32, 0.18, 0.48, 0.20, shade(body_color, 0.82))

        self.draw_entity_bars(entity, cx, cz)

    def draw_staff(self, cx, cz, color):
        glLineWidth(3.0)
        color4(shade(color, 0.85))
        glBegin(GL_LINES)
        glVertex3f(cx + 0.34, 0.18, cz - 0.16)
        glVertex3f(cx + 0.44, 1.26, cz - 0.26)
        glEnd()
        glLineWidth(1.0)
        draw_octahedron(cx + 0.45, 1.34, cz - 0.27, 0.10 + self.pulse * 0.025, shade(color, 1.2))

    def draw_entity_bars(self, entity, cx, cz):
        width = 0.78
        y = 1.28
        z = cz - 0.36
        self.draw_bar3d(cx, y, z, width, 0.045, entity.hp / entity.max_hp, HP_RED)
        self.draw_bar3d(cx, y + 0.07, z, width, 0.038, entity.mp / entity.max_mp, MP_BLUE)
        self.draw_bar3d(cx, y + 0.132, z, width, 0.038, entity.morale / entity.max_morale, MORALE_GOLD)

    def draw_bar3d(self, cx, y, z, width, height, fraction, color):
        fraction = clamp(fraction, 0.0, 1.0)
        draw_box_center(cx, y, z, width, height, 0.035, (0.025, 0.028, 0.032, 0.82))
        if fraction > 0:
            filled = width * fraction
            draw_box_center(cx - width / 2 + filled / 2, y + 0.003, z - 0.004, filled, height * 0.90, 0.040, color)

    def draw_projectiles(self):
        for projectile in self.projectiles:
            x, y, z = self.projectile_position(projectile)
            color = projectile.spell.colors[0]
            draw_octahedron(x, y, z, 0.13 + 0.05 * self.pulse, shade(color, 1.15, 0.95))
            draw_ring3d(x, z, 0.18 + self.pulse * 0.05, shade(projectile.spell.colors[-1], 1.0, 0.45), y=max(0.04, y - 0.02), width=1.4)

    def render_particles(self):
        glDepthMask(GL_FALSE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for p in self.particles:
            t = clamp(p.age / p.life, 0.0, 1.0)
            color = lerp_color(p.c0, p.c1, t)
            alpha = color[3] * (1.0 - t) * (1.0 - t)
            color = (color[0], color[1], color[2], alpha)
            size = p.size * (1.0 + 0.6 * (1.0 - t))
            if p.kind == "spark":
                self.draw_particle_spark(p, color)
            elif p.kind == "shard":
                draw_octahedron(p.x, p.y, p.z, size * 1.5, color)
            elif p.kind == "smoke":
                draw_box_center(p.x, p.y, p.z, size * 2.0, size * 1.5, size * 2.0, shade(color, 0.8, alpha * 0.50))
            else:
                self.draw_particle_cross(p.x, p.y, p.z, size, color)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_TRUE)

    def draw_particle_spark(self, p, color):
        glLineWidth(1.6)
        color4(color)
        glBegin(GL_LINES)
        glVertex3f(p.x, p.y, p.z)
        glVertex3f(p.x - p.vx * 0.055, p.y - p.vy * 0.055, p.z - p.vz * 0.055)
        glEnd()
        glLineWidth(1.0)

    def draw_particle_cross(self, x, y, z, size, color):
        color4(color)
        glBegin(GL_QUADS)
        glVertex3f(x - size, y - size, z)
        glVertex3f(x + size, y - size, z)
        glVertex3f(x + size, y + size, z)
        glVertex3f(x - size, y + size, z)

        glVertex3f(x, y - size, z - size)
        glVertex3f(x, y - size, z + size)
        glVertex3f(x, y + size, z + size)
        glVertex3f(x, y + size, z - size)
        glEnd()

    def render_ui(self):
        draw_rect(14, 14, 284, 122, PANEL)
        self.text.draw("THE VOTARY", 28, 25, (235, 244, 250), "normal")
        self.draw_stat_row(28, 54, "HP", self.player.hp, self.player.max_hp, HP_RED)
        self.draw_stat_row(28, 78, "MP", self.player.mp, self.player.max_mp, MP_BLUE)
        self.draw_stat_row(28, 102, "MR", self.player.morale, self.player.max_morale, MORALE_GOLD)

        draw_rect(314, 14, 474, 48, PANEL)
        self.text.draw(self.message, 330, 29, (230, 236, 240), "normal")

        self.render_target_panel()
        self.render_action_bar()

        if self.phase in ("victory", "defeat"):
            self.render_end_overlay()

    def draw_stat_row(self, x, y, label, value, maximum, color):
        self.text.draw(label, x, y - 2, (205, 212, 218), "small")
        self.draw_bar2d(x + 35, y, 168, 12, value / maximum, color)
        self.text.draw(f"{value:3d}/{maximum}", x + 212, y - 3, (230, 234, 238), "small")

    def draw_bar2d(self, x, y, w, h, fraction, color):
        fraction = clamp(fraction, 0.0, 1.0)
        draw_rect(x, y, w, h, (0.018, 0.022, 0.028, 0.94))
        draw_rect(x + 1, y + 1, max(0, (w - 2) * fraction), h - 2, color)

    def render_target_panel(self):
        x = WIDTH - 286
        draw_rect(x, 14, 272, 294, PANEL)
        self.text.draw("HOSTILE CHORUS", x + 16, 26, (235, 240, 244), "normal")

        y = 58
        active = self.active_enemies()
        for enemy in self.enemies:
            selected = enemy in active and active.index(enemy) == self.target_index % max(1, len(active))
            row_color = PANEL_HI if selected else PANEL_2
            if enemy.defeated:
                row_color = (0.040, 0.044, 0.050, 0.78)
            draw_rect(x + 12, y - 6, 248, 52, row_color)
            name_color = (238, 242, 245) if not enemy.defeated else (110, 118, 125)
            self.text.draw(enemy.name, x + 22, y - 1, name_color, "small")
            status = "ROUTED" if enemy.morale <= 0 else "DOWN" if enemy.hp <= 0 else f"{enemy.x},{enemy.z}"
            self.text.draw(status, x + 210, y - 1, name_color, "tiny")
            self.draw_bar2d(x + 22, y + 19, 68, 7, enemy.hp / enemy.max_hp, HP_RED)
            self.draw_bar2d(x + 98, y + 19, 68, 7, enemy.mp / enemy.max_mp, MP_BLUE)
            self.draw_bar2d(x + 174, y + 19, 68, 7, enemy.morale / enemy.max_morale, MORALE_GOLD)
            y += 58

        spell = SPELLS[self.selected_spell]
        y = 260
        self.text.draw(f"Mode: {spell.mode.upper()}", x + 16, y, (210, 218, 224), "small")
        if spell.mode == "aoe":
            self.text.draw(f"Cursor: {self.cursor_x},{self.cursor_z}  Radius {spell.radius:.1f}", x + 16, y + 19, (210, 218, 224), "small")
        elif spell.mode == "move":
            self.text.draw(f"Destination: {self.cursor_x},{self.cursor_z}", x + 16, y + 19, (210, 218, 224), "small")
        else:
            target = self.current_target()
            target_name = target.name if target else "None"
            self.text.draw(f"Target: {target_name}", x + 16, y + 19, (210, 218, 224), "small")

    def render_action_bar(self):
        bar_h = 156
        y0 = HEIGHT - bar_h - 14
        draw_rect(14, y0, WIDTH - 28, bar_h, PANEL)
        self.text.draw(f"ROUND {self.turn_number}", 30, y0 + 14, (228, 234, 238), "normal")
        self.text.draw(f"PHASE {self.phase.upper()}", 154, y0 + 14, (190, 202, 214), "small")

        cell_w = (WIDTH - 68) / 6
        for i, spell in enumerate(SPELLS):
            x = 30 + i * cell_w
            selected = i == self.selected_spell
            bg = (0.165, 0.205, 0.245, 0.98) if selected else PANEL_2
            draw_rect(x, y0 + 46, cell_w - 10, 88, bg)
            draw_rect(x, y0 + 46, 4, 88, spell.colors[0])
            name_color = (245, 248, 250) if selected else (202, 211, 219)
            self.text.draw(f"{i + 1} {spell.name}", x + 13, y0 + 57, name_color, "small")
            self.text.draw(f"{spell.cost} MP", x + 13, y0 + 80, (190, 202, 214), "tiny")
            if spell.mode == "aoe":
                mode = f"AOE r{spell.radius:.1f}"
            elif spell.mode == "direct":
                mode = "DIRECT"
            else:
                mode = spell.mode.upper()
            self.text.draw(mode, x + 13, y0 + 98, (190, 202, 214), "tiny")
            self.text.draw(spell.summary[:22], x + 13, y0 + 116, (150, 162, 174), "tiny")

    def render_end_overlay(self):
        draw_rect(0, 0, WIDTH, HEIGHT, (0.0, 0.0, 0.0, 0.50))
        title = "VICTORY" if self.phase == "victory" else "DEFEAT"
        color = (235, 246, 238) if self.phase == "victory" else (255, 205, 205)
        self.text.draw(title, WIDTH / 2, HEIGHT / 2 - 52, color, "big", center=True)
        self.text.draw("Press R to restart", WIDTH / 2, HEIGHT / 2 + 6, (230, 234, 238), "normal", center=True)


def main():
    Game().run()


if __name__ == "__main__":
    main()
