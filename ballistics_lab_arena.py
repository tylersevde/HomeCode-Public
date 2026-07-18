import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
except ImportError:
    print("This game requires pygame. Install it with: pip install pygame")
    raise

try:
    from OpenGL.GL import *
    from OpenGL.GLU import gluNewQuadric, gluPerspective, gluSphere
except ImportError:
    print("This game requires PyOpenGL. Install it with: pip install PyOpenGL")
    raise


WIDTH, HEIGHT = 1024, 640
FPS = 60
ARENA_W = 28.0
ARENA_D = 20.0
CEILING_H = 4.2
PLAYER_HEIGHT = 1.55
PLAYER_RADIUS = 0.32
MOVE_SPEED = 5.0
SPRINT_MULT = 1.45
STRAFE_MULT = 0.92
MOUSE_SENS = 0.105
MAX_VIEW_DISTANCE = 70.0
WHITE = (0.94, 0.96, 0.98)
MUTED = (0.63, 0.70, 0.76)
AMBER = (1.00, 0.73, 0.26)
CYAN = (0.55, 0.88, 1.00)
GREEN = (0.34, 0.92, 0.48)
RED = (0.96, 0.22, 0.24)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def color_lerp(a: Tuple[float, float, float], b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


@dataclass(frozen=True)
class Weapon:
    name: str
    short_name: str
    damage: int
    pellet_count: int
    spread_deg: float
    fire_delay: float
    reload_time: float
    magazine_size: int
    reserve: int
    max_range: float
    recoil: float
    tint: Tuple[float, float, float]


WEAPONS = [
    Weapon(
        "Revolver",
        "REV",
        damage=62,
        pellet_count=1,
        spread_deg=1.05,
        fire_delay=0.38,
        reload_time=1.55,
        magazine_size=6,
        reserve=54,
        max_range=38.0,
        recoil=0.14,
        tint=(0.72, 0.68, 0.60),
    ),
    Weapon(
        "Bolt Action Rifle",
        "RIFLE",
        damage=118,
        pellet_count=1,
        spread_deg=0.16,
        fire_delay=1.12,
        reload_time=2.05,
        magazine_size=5,
        reserve=35,
        max_range=66.0,
        recoil=0.18,
        tint=(0.54, 0.40, 0.26),
    ),
    Weapon(
        "Pump Shotgun",
        "SHOT",
        damage=18,
        pellet_count=9,
        spread_deg=7.5,
        fire_delay=0.92,
        reload_time=1.85,
        magazine_size=5,
        reserve=45,
        max_range=24.0,
        recoil=0.24,
        tint=(0.34, 0.32, 0.28),
    ),
]


@dataclass
class RectObstacle:
    x1: float
    z1: float
    x2: float
    z2: float
    height: float
    color: Tuple[float, float, float]
    blocks_player: bool = True

    def contains(self, x: float, z: float, pad: float = 0.0) -> bool:
        return self.x1 - pad <= x <= self.x2 + pad and self.z1 - pad <= z <= self.z2 + pad


@dataclass
class Balloon:
    anchor_x: float
    anchor_z: float
    base_y: float
    radius: float
    color: Tuple[float, float, float]
    hp: int
    value: int
    rail_axis: int
    rail_amp: float
    bob_amp: float
    phase: float
    speed: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    alive: bool = True
    hit_flash: float = 0.0

    def update(self, dt: float) -> None:
        self.phase += dt * self.speed
        drift = math.sin(self.phase) * self.rail_amp
        side = math.sin(self.phase * 0.73 + 1.6) * 0.28
        if self.rail_axis == 0:
            self.x = self.anchor_x + drift
            self.z = self.anchor_z + side
        else:
            self.x = self.anchor_x + side
            self.z = self.anchor_z + drift
        self.y = self.base_y + math.sin(self.phase * 1.37) * self.bob_amp
        self.hit_flash = max(0.0, self.hit_flash - dt * 6.0)


@dataclass
class PopEffect:
    x: float
    y: float
    z: float
    color: Tuple[float, float, float]
    timer: float = 0.34
    seed: float = 0.0


@dataclass
class Tracer:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    hit: bool
    timer: float


class BallisticsLabArena:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Ballistics Lab Arena")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("consolas", 20, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 15, bold=True)
        self.big_font = pygame.font.SysFont("consolas", 42, bold=True)
        self._text_cache: Dict[Tuple[str, int, Tuple[int, int, int]], Tuple[int, int, bytes]] = {}
        self.quadric = gluNewQuadric()
        self.rng = random.Random()
        self.init_gl()
        self.build_lab()
        self.reset(full=True)

    def init_gl(self) -> None:
        glViewport(0, 0, WIDTH, HEIGHT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        glClearColor(0.035, 0.045, 0.055, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(72.0, WIDTH / HEIGHT, 0.05, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def build_lab(self) -> None:
        bench = (0.20, 0.25, 0.29)
        console = (0.26, 0.31, 0.35)
        shield = (0.30, 0.40, 0.45)
        crate = (0.34, 0.30, 0.24)
        self.obstacles: List[RectObstacle] = [
            RectObstacle(1.0, 2.6, 3.1, 17.2, 0.82, bench),
            RectObstacle(24.9, 2.6, 27.0, 17.2, 0.82, bench),
            RectObstacle(7.1, 1.15, 20.9, 1.75, 0.92, console),
            RectObstacle(12.8, 9.2, 15.2, 10.8, 0.24, shield, blocks_player=False),
            RectObstacle(5.2, 14.0, 7.6, 16.7, 0.98, crate),
            RectObstacle(20.4, 14.0, 22.8, 16.7, 0.98, crate),
        ]

    def reset(self, full: bool = False) -> None:
        self.player_x = ARENA_W * 0.5
        self.player_z = ARENA_D - 2.4
        self.player_yaw = 0.0
        self.player_pitch = -4.0
        self.weapon_index = 0
        self.clips = {i: weapon.magazine_size for i, weapon in enumerate(WEAPONS)}
        self.reserves = {i: weapon.reserve for i, weapon in enumerate(WEAPONS)}
        self.shoot_timer = 0.0
        self.reload_timer = 0.0
        self.pending_reload_weapon: Optional[int] = None
        self.muzzle_flash = 0.0
        self.hit_flash = 0.0
        self.crosshair_pulse = 0.0
        self.recoil_offset = 0.0
        self.weapon_bob = 0.0
        self.walk_time = 0.0
        self.show_mouse = False
        self.message = "TEST RANGE ARMED"
        self.message_timer = 2.0
        self.score = 0
        self.popped = 0
        self.wave = 1
        self.wave_clear_timer = 0.0
        self.balloons: List[Balloon] = []
        self.pop_effects: List[PopEffect] = []
        self.tracers: List[Tracer] = []
        self.grab_mouse(True)
        self.spawn_wave()

    def grab_mouse(self, enabled: bool) -> None:
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        if enabled:
            pygame.mouse.get_rel()

    @property
    def weapon(self) -> Weapon:
        return WEAPONS[self.weapon_index]

    def spawn_wave(self) -> None:
        self.balloons.clear()
        count = min(8 + self.wave * 2, 18)
        palette = [
            (0.96, 0.16, 0.25),
            (0.98, 0.82, 0.22),
            (0.24, 0.76, 1.00),
            (0.50, 0.94, 0.36),
            (0.90, 0.28, 0.92),
            (1.00, 0.48, 0.18),
            (0.30, 0.42, 1.00),
        ]
        tries = 0
        while len(self.balloons) < count and tries < 600:
            tries += 1
            x = self.rng.uniform(5.0, 23.0)
            z = self.rng.uniform(3.1, 13.8)
            if math.hypot(x - self.player_x, z - self.player_z) < 5.0:
                continue
            if not self.space_clear(x, z, 0.55):
                continue
            radius = self.rng.uniform(0.27, 0.42)
            color = palette[len(self.balloons) % len(palette)]
            hp = int(42 + radius * 60 + self.wave * 4)
            value = int(65 + (0.45 - radius) * 80)
            balloon = Balloon(
                anchor_x=x,
                anchor_z=z,
                base_y=self.rng.uniform(1.45, 2.95),
                radius=radius,
                color=color,
                hp=hp,
                value=value,
                rail_axis=self.rng.randrange(2),
                rail_amp=self.rng.uniform(0.35, 1.20),
                bob_amp=self.rng.uniform(0.10, 0.27),
                phase=self.rng.uniform(0.0, math.tau),
                speed=self.rng.uniform(0.55, 1.25) + self.wave * 0.025,
            )
            balloon.update(0.0)
            self.balloons.append(balloon)
        self.message = f"WAVE {self.wave} TARGETS LIVE"
        self.message_timer = 1.6

    def run(self) -> None:
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.033)
            self.handle_events()
            self.update(dt)
            self.render()
            pygame.display.flip()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_TAB:
                    self.show_mouse = not self.show_mouse
                    self.grab_mouse(not self.show_mouse)
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.switch_weapon(event.key - pygame.K_1)
                if event.key == pygame.K_r:
                    self.try_reload()
                if event.key == pygame.K_F5:
                    self.reset(full=True)
                if event.key == pygame.K_SPACE:
                    self.try_fire()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.try_fire()

    def switch_weapon(self, index: int) -> None:
        if index == self.weapon_index or not 0 <= index < len(WEAPONS):
            return
        self.weapon_index = index
        self.reload_timer = 0.0
        self.pending_reload_weapon = None
        self.message = self.weapon.name.upper()
        self.message_timer = 0.9

    def try_reload(self) -> None:
        if self.reload_timer > 0.0:
            return
        weapon = self.weapon
        clip = self.clips[self.weapon_index]
        reserve = self.reserves[self.weapon_index]
        if clip >= weapon.magazine_size or reserve <= 0:
            return
        self.reload_timer = weapon.reload_time
        self.pending_reload_weapon = self.weapon_index
        self.message = f"RELOADING {weapon.short_name}"
        self.message_timer = 0.8

    def finish_reload(self) -> None:
        if self.pending_reload_weapon is None:
            return
        index = self.pending_reload_weapon
        weapon = WEAPONS[index]
        needed = weapon.magazine_size - self.clips[index]
        loaded = min(needed, self.reserves[index])
        self.clips[index] += loaded
        self.reserves[index] -= loaded
        self.pending_reload_weapon = None

    def update(self, dt: float) -> None:
        self.shoot_timer = max(0.0, self.shoot_timer - dt)
        self.muzzle_flash = max(0.0, self.muzzle_flash - dt * 8.0)
        self.hit_flash = max(0.0, self.hit_flash - dt * 4.5)
        self.crosshair_pulse = max(0.0, self.crosshair_pulse - dt * 4.2)
        self.recoil_offset = max(0.0, self.recoil_offset - dt * 2.6)
        self.message_timer = max(0.0, self.message_timer - dt)

        if self.reload_timer > 0.0:
            self.reload_timer -= dt
            if self.reload_timer <= 0.0:
                self.reload_timer = 0.0
                self.finish_reload()

        if not self.show_mouse:
            mx, my = pygame.mouse.get_rel()
            self.player_yaw += mx * MOUSE_SENS
            self.player_pitch = clamp(self.player_pitch - my * MOUSE_SENS * 0.76, -62.0, 64.0)

        self.move_player(dt)
        for balloon in self.balloons:
            if balloon.alive:
                balloon.update(dt)
        for effect in self.pop_effects:
            effect.timer -= dt
        self.pop_effects = [effect for effect in self.pop_effects if effect.timer > 0.0]
        for tracer in self.tracers:
            tracer.timer -= dt
        self.tracers = [tracer for tracer in self.tracers if tracer.timer > 0.0]

        if not any(balloon.alive for balloon in self.balloons):
            self.wave_clear_timer += dt
            if self.wave_clear_timer > 1.1:
                self.wave += 1
                self.wave_clear_timer = 0.0
                self.spawn_wave()
        else:
            self.wave_clear_timer = 0.0

    def move_player(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        forward = 0.0
        strafe = 0.0
        if keys[pygame.K_w]:
            forward += 1.0
        if keys[pygame.K_s]:
            forward -= 1.0
        if keys[pygame.K_a]:
            strafe -= 1.0
        if keys[pygame.K_d]:
            strafe += 1.0

        speed = MOVE_SPEED * (SPRINT_MULT if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
        if abs(strafe) > 0.0 and abs(forward) == 0.0:
            speed *= STRAFE_MULT

        yaw = math.radians(self.player_yaw)
        dir_x = math.sin(yaw)
        dir_z = -math.cos(yaw)
        right_x = math.cos(yaw)
        right_z = math.sin(yaw)
        length = math.hypot(forward, strafe)
        if length > 1.0:
            forward /= length
            strafe /= length

        dx = (dir_x * forward + right_x * strafe) * speed * dt
        dz = (dir_z * forward + right_z * strafe) * speed * dt
        if dx or dz:
            self.walk_time += dt * speed * 0.65
            self.weapon_bob = min(1.0, self.weapon_bob + dt * 5.0)
        else:
            self.weapon_bob = max(0.0, self.weapon_bob - dt * 4.0)
        self.slide_move(dx, dz)

    def slide_move(self, dx: float, dz: float) -> None:
        nx = self.player_x + dx
        nz = self.player_z + dz
        if not self.collides(nx, self.player_z, PLAYER_RADIUS):
            self.player_x = nx
        if not self.collides(self.player_x, nz, PLAYER_RADIUS):
            self.player_z = nz

    def collides(self, x: float, z: float, radius: float) -> bool:
        if x - radius < 0.55 or x + radius > ARENA_W - 0.55:
            return True
        if z - radius < 0.55 or z + radius > ARENA_D - 0.55:
            return True
        for obstacle in self.obstacles:
            if obstacle.blocks_player and obstacle.contains(x, z, radius):
                return True
        return False

    def space_clear(self, x: float, z: float, radius: float) -> bool:
        if x - radius < 1.0 or x + radius > ARENA_W - 1.0:
            return False
        if z - radius < 1.0 or z + radius > ARENA_D - 1.0:
            return False
        return not any(obstacle.contains(x, z, radius) for obstacle in self.obstacles)

    def try_fire(self) -> None:
        if self.reload_timer > 0.0 or self.shoot_timer > 0.0:
            return
        weapon = self.weapon
        if self.clips[self.weapon_index] <= 0:
            self.message = "EMPTY"
            self.message_timer = 0.45
            self.try_reload()
            return

        self.clips[self.weapon_index] -= 1
        self.shoot_timer = weapon.fire_delay
        self.muzzle_flash = 1.0
        self.crosshair_pulse = 1.0
        self.recoil_offset = min(0.65, self.recoil_offset + weapon.recoil)

        hit_any = False
        origin = (self.player_x, PLAYER_HEIGHT - 0.05, self.player_z)
        for pellet in range(weapon.pellet_count):
            spread = weapon.spread_deg
            yaw_offset = self.rng.uniform(-spread, spread) * 0.5
            pitch_offset = self.rng.uniform(-spread, spread) * 0.5
            direction = self.direction_from_angles(self.player_yaw + yaw_offset, self.player_pitch + pitch_offset)
            hit, end_pos = self.cast_shot(origin, direction, weapon.damage, weapon.max_range)
            hit_any = hit_any or hit
            timer = 0.12 if weapon.pellet_count == 1 else 0.08
            self.tracers.append(Tracer(origin, end_pos, hit, timer))

        if hit_any:
            self.hit_flash = 0.75
        if self.clips[self.weapon_index] == 0:
            self.message = "EMPTY"
            self.message_timer = 0.55

    def direction_from_angles(self, yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        return (math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch))

    def wall_distance_along_ray(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_dist: float,
    ) -> float:
        ox, oy, oz = origin
        dx, dy, dz = direction
        dist = 0.0
        step = 0.06
        while dist < max_dist:
            px = ox + dx * dist
            py = oy + dy * dist
            pz = oz + dz * dist
            if py < 0.04 or py > CEILING_H - 0.04:
                return dist
            if px < 0.42 or px > ARENA_W - 0.42 or pz < 0.42 or pz > ARENA_D - 0.42:
                return dist
            for obstacle in self.obstacles:
                if obstacle.contains(px, pz, 0.0) and py <= obstacle.height:
                    return dist
            dist += step
        return max_dist

    def cast_shot(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        damage: int,
        max_range: float,
    ) -> Tuple[bool, Tuple[float, float, float]]:
        wall_dist = self.wall_distance_along_ray(origin, direction, max_range)
        best_balloon: Optional[Balloon] = None
        best_t = float("inf")
        for balloon in self.balloons:
            if not balloon.alive:
                continue
            center = (balloon.x, balloon.y, balloon.z)
            t = self.ray_sphere_intersection(origin, direction, center, balloon.radius)
            if t is not None and 0.0 < t < best_t and t < wall_dist and t <= max_range:
                best_t = t
                best_balloon = balloon
        if best_balloon is not None:
            best_balloon.hp -= damage
            best_balloon.hit_flash = 1.0
            hit_pos = (
                origin[0] + direction[0] * best_t,
                origin[1] + direction[1] * best_t,
                origin[2] + direction[2] * best_t,
            )
            if best_balloon.hp <= 0:
                self.pop_balloon(best_balloon, best_t)
            return True, hit_pos

        end_dist = min(max_range, wall_dist)
        return False, (
            origin[0] + direction[0] * end_dist,
            origin[1] + direction[1] * end_dist,
            origin[2] + direction[2] * end_dist,
        )

    def pop_balloon(self, balloon: Balloon, distance: float) -> None:
        if not balloon.alive:
            return
        balloon.alive = False
        weapon = self.weapon
        distance_bonus = int(distance * 2.2)
        weapon_bonus = 0
        if weapon.short_name == "RIFLE" and distance > 12.0:
            weapon_bonus = 35
        elif weapon.short_name == "SHOT" and distance < 8.0:
            weapon_bonus = 20
        points = balloon.value + distance_bonus + weapon_bonus
        self.score += points
        self.popped += 1
        self.pop_effects.append(PopEffect(balloon.x, balloon.y, balloon.z, balloon.color, seed=self.rng.random() * 100.0))
        self.message = f"+{points}"
        self.message_timer = 0.5

    def ray_sphere_intersection(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        center: Tuple[float, float, float],
        radius: float,
    ) -> Optional[float]:
        ox, oy, oz = origin
        dx, dy, dz = direction
        cx, cy, cz = center
        lx = ox - cx
        ly = oy - cy
        lz = oz - cz
        a = dx * dx + dy * dy + dz * dz
        b = 2.0 * (dx * lx + dy * ly + dz * lz)
        c = lx * lx + ly * ly + lz * lz - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        root = math.sqrt(disc)
        t1 = (-b - root) / (2.0 * a)
        t2 = (-b + root) / (2.0 * a)
        if t1 > 0.0:
            return t1
        if t2 > 0.0:
            return t2
        return None

    def set_3d(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(72.0, WIDTH / HEIGHT, 0.05, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(-(self.player_pitch + self.recoil_offset * 8.0), 1.0, 0.0, 0.0)
        glRotatef(-self.player_yaw, 0.0, 1.0, 0.0)
        glTranslatef(-self.player_x, -PLAYER_HEIGHT, -self.player_z)

    def set_2d(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def render(self) -> None:
        glDisable(GL_TEXTURE_2D)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.set_2d()
        self.draw_background_gradient()
        self.set_3d()
        self.draw_floor_and_ceiling()
        self.draw_walls()
        self.draw_lab_fixtures()
        self.draw_balloons()
        self.draw_pop_effects()
        self.draw_tracers()
        self.set_2d()
        self.draw_weapon_overlay()
        self.draw_crosshair()
        self.draw_hud()

    def draw_background_gradient(self) -> None:
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_QUADS)
        glColor3f(0.045, 0.060, 0.075)
        glVertex2f(0, 0)
        glVertex2f(WIDTH, 0)
        glColor3f(0.12, 0.16, 0.19)
        glVertex2f(WIDTH, HEIGHT * 0.53)
        glVertex2f(0, HEIGHT * 0.53)
        glColor3f(0.030, 0.034, 0.038)
        glVertex2f(0, HEIGHT)
        glVertex2f(WIDTH, HEIGHT)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_floor_and_ceiling(self) -> None:
        glDisable(GL_CULL_FACE)
        glBegin(GL_QUADS)
        glColor3f(0.12, 0.13, 0.135)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(ARENA_W, 0.0, 0.0)
        glColor3f(0.060, 0.065, 0.070)
        glVertex3f(ARENA_W, 0.0, ARENA_D)
        glVertex3f(0.0, 0.0, ARENA_D)

        glColor3f(0.055, 0.064, 0.073)
        glVertex3f(0.0, CEILING_H, 0.0)
        glVertex3f(0.0, CEILING_H, ARENA_D)
        glColor3f(0.090, 0.105, 0.115)
        glVertex3f(ARENA_W, CEILING_H, ARENA_D)
        glVertex3f(ARENA_W, CEILING_H, 0.0)
        glEnd()

        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(0.22, 0.25, 0.27, 0.35)
        for x in range(int(ARENA_W) + 1):
            glVertex3f(x, 0.012, 0.0)
            glVertex3f(x, 0.012, ARENA_D)
        for z in range(int(ARENA_D) + 1):
            glVertex3f(0.0, 0.012, z)
            glVertex3f(ARENA_W, 0.012, z)

        glColor4f(1.0, 0.78, 0.22, 0.70)
        for x in (9.5, 18.5):
            glVertex3f(x, 0.025, 2.0)
            glVertex3f(x, 0.025, 18.8)
        glColor4f(0.78, 0.90, 1.0, 0.48)
        glVertex3f(4.0, 0.026, 6.0)
        glVertex3f(24.0, 0.026, 6.0)
        glVertex3f(4.0, 0.026, 10.0)
        glVertex3f(24.0, 0.026, 10.0)
        glVertex3f(4.0, 0.026, 14.0)
        glVertex3f(24.0, 0.026, 14.0)
        glEnd()
        glEnable(GL_CULL_FACE)

    def draw_walls(self) -> None:
        wall_color = (0.18, 0.22, 0.25)
        self.draw_box(ARENA_W * 0.5, CEILING_H * 0.5, 0.18, ARENA_W, CEILING_H, 0.36, wall_color)
        self.draw_box(ARENA_W * 0.5, CEILING_H * 0.5, ARENA_D - 0.18, ARENA_W, CEILING_H, 0.36, wall_color)
        self.draw_box(0.18, CEILING_H * 0.5, ARENA_D * 0.5, 0.36, CEILING_H, ARENA_D, wall_color)
        self.draw_box(ARENA_W - 0.18, CEILING_H * 0.5, ARENA_D * 0.5, 0.36, CEILING_H, ARENA_D, wall_color)

        glass = (0.18, 0.34, 0.42)
        self.draw_box(ARENA_W * 0.5, 2.35, ARENA_D - 0.42, 9.6, 1.45, 0.08, glass)
        self.draw_box(0.44, 2.30, ARENA_D * 0.5, 0.08, 1.35, 7.8, glass)
        self.draw_box(ARENA_W - 0.44, 2.30, ARENA_D * 0.5, 0.08, 1.35, 7.8, glass)

        glDisable(GL_CULL_FACE)
        glBegin(GL_QUADS)
        glColor4f(0.72, 0.82, 0.86, 0.16)
        glVertex3f(4.0, 1.15, 0.38)
        glVertex3f(24.0, 1.15, 0.38)
        glVertex3f(24.0, 3.45, 0.38)
        glVertex3f(4.0, 3.45, 0.38)
        glEnd()
        glEnable(GL_CULL_FACE)

    def draw_lab_fixtures(self) -> None:
        for obstacle in self.obstacles:
            y = obstacle.height * 0.5
            self.draw_box(
                (obstacle.x1 + obstacle.x2) * 0.5,
                y,
                (obstacle.z1 + obstacle.z2) * 0.5,
                obstacle.x2 - obstacle.x1,
                obstacle.height,
                obstacle.z2 - obstacle.z1,
                obstacle.color,
            )

        rail_color = (0.42, 0.47, 0.50)
        post_color = (0.32, 0.37, 0.40)
        for z in (4.0, 8.0, 12.0):
            self.draw_box(ARENA_W * 0.5, 3.36, z, 20.0, 0.08, 0.08, rail_color)
            for x in (4.2, 10.8, 17.2, 23.8):
                self.draw_box(x, 1.72, z, 0.08, 3.28, 0.08, post_color)

        for z in (5.7, 10.2):
            self.draw_chronograph_gate(8.6, z)
            self.draw_chronograph_gate(19.4, z)

        self.draw_box(ARENA_W * 0.5, 0.12, 9.95, 3.1, 0.24, 2.25, (0.18, 0.26, 0.29))
        self.draw_box(ARENA_W * 0.5, 0.34, 9.95, 2.3, 0.14, 1.45, (0.36, 0.48, 0.52))

        for x in (2.0, 26.0):
            for z in (3.4, 6.2, 9.0, 11.8, 14.6):
                self.draw_box(x, 1.03, z, 1.1, 0.08, 0.62, (0.08, 0.11, 0.13))
                self.draw_box(x, 1.05, z, 0.76, 0.05, 0.38, (0.12, 0.36, 0.42))

        glDisable(GL_CULL_FACE)
        glBegin(GL_LINES)
        glColor4f(0.60, 0.75, 0.80, 0.30)
        for x in (6.0, 10.0, 14.0, 18.0, 22.0):
            glVertex3f(x, 3.88, 0.6)
            glVertex3f(x, 3.88, 18.0)
        glEnd()
        glEnable(GL_CULL_FACE)

    def draw_chronograph_gate(self, x: float, z: float) -> None:
        color = (0.68, 0.74, 0.75)
        self.draw_box(x - 0.7, 1.15, z, 0.07, 2.1, 0.10, color)
        self.draw_box(x + 0.7, 1.15, z, 0.07, 2.1, 0.10, color)
        self.draw_box(x, 2.18, z, 1.47, 0.07, 0.10, color)
        glDisable(GL_CULL_FACE)
        glBegin(GL_LINES)
        glColor4f(0.34, 0.95, 1.0, 0.38)
        glVertex3f(x - 0.58, 0.48, z)
        glVertex3f(x + 0.58, 1.82, z)
        glVertex3f(x + 0.58, 0.48, z)
        glVertex3f(x - 0.58, 1.82, z)
        glEnd()
        glEnable(GL_CULL_FACE)

    def draw_balloons(self) -> None:
        for balloon in self.balloons:
            if not balloon.alive:
                continue
            dist = math.hypot(balloon.x - self.player_x, balloon.z - self.player_z)
            shade = clamp(1.18 - dist / 35.0, 0.55, 1.0)
            color = tuple(component * shade for component in balloon.color)
            if balloon.hit_flash > 0.0:
                color = color_lerp(color, (1.0, 1.0, 1.0), balloon.hit_flash * 0.65)

            glPushMatrix()
            glTranslatef(balloon.x, balloon.y, balloon.z)
            glScalef(0.86, 1.14, 0.86)
            glColor3f(*color)
            gluSphere(self.quadric, balloon.radius, 22, 14)
            glPopMatrix()

            glDisable(GL_CULL_FACE)
            glBegin(GL_LINES)
            glColor4f(0.82, 0.88, 0.92, 0.55)
            glVertex3f(balloon.x, balloon.y - balloon.radius * 1.05, balloon.z)
            glVertex3f(balloon.x, balloon.y - balloon.radius * 2.2, balloon.z)
            glEnd()
            glEnable(GL_CULL_FACE)

            self.draw_box(balloon.x, balloon.y - balloon.radius * 1.18, balloon.z, 0.08, 0.08, 0.08, color)

    def draw_pop_effects(self) -> None:
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        for effect in self.pop_effects:
            alpha = clamp(effect.timer / 0.34, 0.0, 1.0)
            spread = (1.0 - alpha) * 0.82 + 0.15
            glBegin(GL_LINES)
            glColor4f(effect.color[0], effect.color[1], effect.color[2], alpha)
            for i in range(12):
                ang = effect.seed + i * math.tau / 12.0
                up = math.sin(effect.seed * 0.7 + i) * 0.55
                glVertex3f(effect.x, effect.y, effect.z)
                glVertex3f(
                    effect.x + math.cos(ang) * spread,
                    effect.y + up * spread,
                    effect.z + math.sin(ang) * spread,
                )
            glEnd()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

    def draw_tracers(self) -> None:
        if not self.tracers:
            return
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for tracer in self.tracers:
            alpha = clamp(tracer.timer * 5.0, 0.0, 0.70)
            if tracer.hit:
                glColor4f(1.0, 0.83, 0.34, alpha)
            else:
                glColor4f(0.62, 0.82, 1.0, alpha * 0.75)
            glVertex3f(*tracer.start)
            glVertex3f(*tracer.end)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_box(
        self,
        x: float,
        y: float,
        z: float,
        sx: float,
        sy: float,
        sz: float,
        color: Tuple[float, float, float],
    ) -> None:
        glPushMatrix()
        glTranslatef(x, y, z)
        self.draw_box_local(0.0, 0.0, 0.0, sx, sy, sz, color)
        glPopMatrix()

    def draw_box_local(
        self,
        x: float,
        y: float,
        z: float,
        sx: float,
        sy: float,
        sz: float,
        color: Tuple[float, float, float],
    ) -> None:
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        verts = [
            (x - hx, y - hy, z - hz),
            (x + hx, y - hy, z - hz),
            (x + hx, y + hy, z - hz),
            (x - hx, y + hy, z - hz),
            (x - hx, y - hy, z + hz),
            (x + hx, y - hy, z + hz),
            (x + hx, y + hy, z + hz),
            (x - hx, y + hy, z + hz),
        ]
        faces = [
            (0, 1, 2, 3, 0.78),
            (4, 5, 6, 7, 0.98),
            (3, 2, 6, 7, 1.12),
            (0, 1, 5, 4, 0.62),
            (1, 2, 6, 5, 0.88),
            (0, 3, 7, 4, 0.72),
        ]
        glBegin(GL_QUADS)
        for a, b, c, d, shade in faces:
            glColor3f(color[0] * shade, color[1] * shade, color[2] * shade)
            glVertex3f(*verts[a])
            glVertex3f(*verts[b])
            glVertex3f(*verts[c])
            glVertex3f(*verts[d])
        glEnd()

    def draw_weapon_overlay(self) -> None:
        weapon = self.weapon
        glDisable(GL_DEPTH_TEST)
        bob = math.sin(self.walk_time * 9.0) * 8.0 * self.weapon_bob
        kick = self.muzzle_flash * (18.0 + weapon.recoil * 34.0)
        x = WIDTH - 260 - kick * 0.45
        y = HEIGHT - 118 + bob + kick * 0.18

        glBegin(GL_QUADS)
        glColor4f(0.01, 0.01, 0.012, 0.45)
        glVertex2f(x - 20, y + 22)
        glVertex2f(x + 188, y + 8)
        glVertex2f(x + 244, y + 118)
        glVertex2f(x + 30, y + 130)
        glEnd()

        if weapon.short_name == "REV":
            self.draw_revolver_overlay(x + 28, y + 8, weapon.tint)
            muzzle_x, muzzle_y = x + 203, y + 28
        elif weapon.short_name == "RIFLE":
            self.draw_rifle_overlay(x - 16, y + 4, weapon.tint)
            muzzle_x, muzzle_y = x + 246, y + 18
        else:
            self.draw_shotgun_overlay(x - 4, y + 10, weapon.tint)
            muzzle_x, muzzle_y = x + 238, y + 24

        if self.muzzle_flash > 0.0:
            alpha = clamp(self.muzzle_flash, 0.0, 1.0) * 0.70
            glBegin(GL_TRIANGLES)
            glColor4f(1.0, 0.92, 0.52, alpha)
            glVertex2f(muzzle_x, muzzle_y)
            glColor4f(1.0, 0.55, 0.10, 0.0)
            glVertex2f(muzzle_x + 54, muzzle_y - 22)
            glVertex2f(muzzle_x + 54, muzzle_y + 22)
            glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_revolver_overlay(self, x: float, y: float, tint: Tuple[float, float, float]) -> None:
        metal = tint
        dark = (0.12, 0.13, 0.14)
        grip = (0.30, 0.18, 0.10)
        glBegin(GL_QUADS)
        glColor3f(*dark)
        glVertex2f(x, y + 22)
        glVertex2f(x + 104, y + 14)
        glVertex2f(x + 120, y + 34)
        glVertex2f(x + 14, y + 44)
        glColor3f(*metal)
        glVertex2f(x + 96, y + 17)
        glVertex2f(x + 176, y + 14)
        glVertex2f(x + 180, y + 27)
        glVertex2f(x + 102, y + 33)
        glColor3f(*grip)
        glVertex2f(x + 24, y + 48)
        glVertex2f(x + 70, y + 42)
        glVertex2f(x + 95, y + 112)
        glVertex2f(x + 54, y + 118)
        glEnd()
        glBegin(GL_POLYGON)
        glColor3f(0.45, 0.46, 0.46)
        for i in range(18):
            ang = i * math.tau / 18
            glVertex2f(x + 76 + math.cos(ang) * 28, y + 44 + math.sin(ang) * 28)
        glEnd()
        glBegin(GL_LINE_LOOP)
        glColor4f(0.08, 0.08, 0.08, 0.85)
        for i in range(18):
            ang = i * math.tau / 18
            glVertex2f(x + 76 + math.cos(ang) * 28, y + 44 + math.sin(ang) * 28)
        glEnd()

    def draw_rifle_overlay(self, x: float, y: float, tint: Tuple[float, float, float]) -> None:
        wood = tint
        metal = (0.13, 0.15, 0.16)
        glBegin(GL_QUADS)
        glColor3f(*wood)
        glVertex2f(x + 8, y + 42)
        glVertex2f(x + 126, y + 24)
        glVertex2f(x + 164, y + 42)
        glVertex2f(x + 42, y + 76)
        glColor3f(*wood)
        glVertex2f(x + 18, y + 72)
        glVertex2f(x + 76, y + 61)
        glVertex2f(x + 91, y + 116)
        glVertex2f(x + 38, y + 122)
        glColor3f(*metal)
        glVertex2f(x + 116, y + 24)
        glVertex2f(x + 260, y + 12)
        glVertex2f(x + 262, y + 22)
        glVertex2f(x + 124, y + 35)
        glColor3f(0.18, 0.20, 0.21)
        glVertex2f(x + 118, y + 8)
        glVertex2f(x + 182, y + 4)
        glVertex2f(x + 186, y + 16)
        glVertex2f(x + 122, y + 20)
        glEnd()
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor4f(0.76, 0.80, 0.82, 0.55)
        glVertex2f(x + 150, y + 28)
        glVertex2f(x + 170, y + 51)
        glEnd()

    def draw_shotgun_overlay(self, x: float, y: float, tint: Tuple[float, float, float]) -> None:
        stock = (0.28, 0.16, 0.08)
        pump = tint
        metal = (0.12, 0.13, 0.14)
        glBegin(GL_QUADS)
        glColor3f(*stock)
        glVertex2f(x + 6, y + 42)
        glVertex2f(x + 92, y + 24)
        glVertex2f(x + 125, y + 45)
        glVertex2f(x + 34, y + 79)
        glColor3f(*metal)
        glVertex2f(x + 86, y + 19)
        glVertex2f(x + 248, y + 10)
        glVertex2f(x + 252, y + 22)
        glVertex2f(x + 92, y + 33)
        glColor3f(0.18, 0.19, 0.20)
        glVertex2f(x + 96, y + 34)
        glVertex2f(x + 246, y + 28)
        glVertex2f(x + 248, y + 38)
        glVertex2f(x + 102, y + 47)
        glColor3f(*pump)
        glVertex2f(x + 142, y + 47)
        glVertex2f(x + 202, y + 43)
        glVertex2f(x + 210, y + 66)
        glVertex2f(x + 148, y + 72)
        glColor3f(*stock)
        glVertex2f(x + 32, y + 75)
        glVertex2f(x + 80, y + 62)
        glVertex2f(x + 96, y + 116)
        glVertex2f(x + 50, y + 124)
        glEnd()

    def draw_crosshair(self) -> None:
        cx = WIDTH * 0.5
        cy = HEIGHT * 0.5
        spread_gap = 6.0 + self.weapon.spread_deg * 1.3 + self.crosshair_pulse * 5.0
        size = 9.0 + self.crosshair_pulse * 3.0
        color = AMBER if self.hit_flash > 0.0 else CYAN
        alpha = 0.95
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(*color, alpha)
        glVertex2f(cx - spread_gap - size, cy)
        glVertex2f(cx - spread_gap, cy)
        glVertex2f(cx + spread_gap, cy)
        glVertex2f(cx + spread_gap + size, cy)
        glVertex2f(cx, cy - spread_gap - size)
        glVertex2f(cx, cy - spread_gap)
        glVertex2f(cx, cy + spread_gap)
        glVertex2f(cx, cy + spread_gap + size)
        glEnd()
        if self.hit_flash > 0.0:
            glBegin(GL_LINE_LOOP)
            glColor4f(1.0, 0.84, 0.25, self.hit_flash)
            for i in range(20):
                ang = i * math.tau / 20
                glVertex2f(cx + math.cos(ang) * 20.0, cy + math.sin(ang) * 20.0)
            glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_hud(self) -> None:
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_QUADS)
        glColor4f(0.025, 0.030, 0.034, 0.68)
        glVertex2f(14, HEIGHT - 78)
        glVertex2f(412, HEIGHT - 78)
        glVertex2f(412, HEIGHT - 16)
        glVertex2f(14, HEIGHT - 16)
        glColor4f(0.025, 0.030, 0.034, 0.56)
        glVertex2f(WIDTH - 332, HEIGHT - 78)
        glVertex2f(WIDTH - 14, HEIGHT - 78)
        glVertex2f(WIDTH - 14, HEIGHT - 16)
        glVertex2f(WIDTH - 332, HEIGHT - 16)
        glEnd()

        live_count = sum(1 for balloon in self.balloons if balloon.alive)
        self.draw_text(24, HEIGHT - 68, f"SCORE {self.score}", self.ui_font, WHITE)
        self.draw_text(24, HEIGHT - 40, f"WAVE {self.wave}   TARGETS {live_count:02d}   POPS {self.popped}", self.small_font, MUTED)

        ammo_color = AMBER if self.clips[self.weapon_index] > 0 else RED
        self.draw_text(WIDTH - 318, HEIGHT - 68, self.weapon.name.upper(), self.ui_font, WHITE)
        self.draw_text(
            WIDTH - 318,
            HEIGHT - 40,
            f"AMMO {self.clips[self.weapon_index]:02d}/{self.reserves[self.weapon_index]:02d}",
            self.small_font,
            ammo_color,
        )

        for i, weapon in enumerate(WEAPONS):
            x = 24 + i * 116
            y = 20
            selected = i == self.weapon_index
            glBegin(GL_QUADS)
            if selected:
                glColor4f(0.18, 0.30, 0.34, 0.72)
            else:
                glColor4f(0.04, 0.05, 0.06, 0.48)
            glVertex2f(x, y)
            glVertex2f(x + 100, y)
            glVertex2f(x + 100, y + 34)
            glVertex2f(x, y + 34)
            glEnd()
            self.draw_text(x + 10, y + 8, f"{i + 1} {weapon.short_name}", self.small_font, WHITE if selected else MUTED)

        if self.reload_timer > 0.0:
            ratio = 1.0 - self.reload_timer / self.weapon.reload_time
            bar_w = 240
            x = WIDTH * 0.5 - bar_w * 0.5
            y = HEIGHT - 54
            glBegin(GL_QUADS)
            glColor4f(0.05, 0.06, 0.07, 0.82)
            glVertex2f(x, y)
            glVertex2f(x + bar_w, y)
            glVertex2f(x + bar_w, y + 12)
            glVertex2f(x, y + 12)
            glColor4f(0.95, 0.72, 0.24, 0.92)
            glVertex2f(x, y)
            glVertex2f(x + bar_w * ratio, y)
            glVertex2f(x + bar_w * ratio, y + 12)
            glVertex2f(x, y + 12)
            glEnd()
        elif self.message_timer > 0.0:
            self.draw_text(WIDTH * 0.5 - 88, HEIGHT - 54, self.message, self.ui_font, AMBER)

        glEnable(GL_DEPTH_TEST)

    def draw_text(
        self,
        x: float,
        y: float,
        text: str,
        font: pygame.font.Font,
        color: Tuple[float, float, float],
    ) -> None:
        color255 = tuple(int(clamp(c, 0.0, 1.0) * 255) for c in color)
        key = (text, font.get_height(), color255)
        if key not in self._text_cache:
            surf = font.render(text, True, color255)
            data = pygame.image.tostring(surf, "RGBA", True)
            self._text_cache[key] = (surf.get_width(), surf.get_height(), data)
        w, h, data = self._text_cache[key]
        glWindowPos2d(int(x), int(HEIGHT - y - h))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)


def main() -> None:
    try:
        BallisticsLabArena().run()
    except Exception:
        pygame.quit()
        raise


if __name__ == "__main__":
    main()
