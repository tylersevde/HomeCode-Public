import math
import random
import sys
from dataclasses import dataclass

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective


WIDTH, HEIGHT = 800, 480
FPS = 60
MOUSE_SENS = 0.11
PLAYER_HEIGHT = 1.0
MOVE_SPEED = 6.0
SPRINT_MULT = 1.45
STRAFE_MULT = 0.92
SHOOT_COOLDOWN = 0.085
RELOAD_TIME = 1.15
MAX_AMMO = 30
MAX_HP = 100
ENEMY_DAMAGE = 12
ENEMY_ATTACK_COOLDOWN = 0.95
ENEMY_SPEED = 2.1
BULLET_DAMAGE = 34
MAX_VIEW_DISTANCE = 42.0

MAP_STR = [
    "1111111111111111",
    "1....2.....3...1",
    "1.11.1.111.1.1.1",
    "1....1...1...1.1",
    "1.11111.1.1111.1",
    "1.1....1.1......1",
    "1.1.11.1.11111..1",
    "1...11...2...1..1",
    "111.1111111.1.111",
    "1...1....1...1..1",
    "1.2.1.11.111.1..1",
    "1...1.11.....1..1",
    "1.111.11111111..1",
    "1...3..........31",
    "1.....2.........1",
    "1111111111111111",
]

WORLD_MAP = {}
FREE_CELLS = []
for z, row in enumerate(MAP_STR):
    for x, ch in enumerate(row):
        if ch == ".":
            FREE_CELLS.append((x + 0.5, z + 0.5))
        else:
            WORLD_MAP[(x, z)] = int(ch)

WALL_COLORS = {
    1: (0.32, 0.37, 0.45),
    2: (0.40, 0.36, 0.31),
    3: (0.26, 0.41, 0.34),
}

WHITE = (0.94, 0.96, 0.98)
ACCENT = (0.99, 0.77, 0.27)
DANGER = (0.94, 0.22, 0.22)
CYAN = (0.68, 0.90, 1.00)


def clamp(value, low, high):
    return max(low, min(high, value))


def length2(x, z):
    return math.sqrt(x * x + z * z)


@dataclass
class Enemy:
    x: float
    z: float
    hp: int = 100
    radius: float = 0.42
    cooldown: float = 0.0
    bob: float = 0.0
    strafe_phase: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 0


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("OpenGL Tactical FPS Prototype")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("consolas", 20, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 14, bold=True)
        self.big_font = pygame.font.SysFont("consolas", 44, bold=True)
        self._text_cache = {}

        self.init_gl()
        self.reset(full=True)

    def reset(self, full=False):
        self.player_x = 2.5
        self.player_z = 2.5
        self.player_yaw = 0.0
        self.player_pitch = -3.5
        self.hp = MAX_HP
        self.ammo = MAX_AMMO
        self.score = 0 if full else self.score
        self.kills = 0 if full else self.kills
        self.wave = 1 if full else self.wave
        self.shoot_timer = 0.0
        self.reload_timer = 0.0
        self.damage_flash = 0.0
        self.hit_flash = 0.0
        self.muzzle_flash = 0.0
        self.crosshair_pulse = 0.0
        self.weapon_bob = 0.0
        self.walk_time = 0.0
        self.tracer_timer = 0.0
        self.last_shot_hit = False
        self.last_hit_pos = None
        self.game_over = False
        self.show_mouse = False
        self.grab_mouse(True)
        self.enemies = []
        self.spawn_wave(self.wave)

    def init_gl(self):
        glViewport(0, 0, WIDTH, HEIGHT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.05, 0.07, 0.10, 1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(72.0, WIDTH / HEIGHT, 0.05, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def grab_mouse(self, enabled=True):
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        if enabled:
            pygame.mouse.get_rel()

    def spawn_wave(self, wave):
        self.enemies.clear()
        needed = 4 + wave * 2
        choices = FREE_CELLS[:]
        random.shuffle(choices)
        for ex, ez in choices:
            if math.hypot(ex - self.player_x, ez - self.player_z) < 3.6:
                continue
            self.enemies.append(
                Enemy(
                    x=ex,
                    z=ez,
                    hp=85 + wave * 10,
                    bob=random.random() * math.tau,
                    strafe_phase=random.random() * math.tau,
                )
            )
            if len(self.enemies) >= needed:
                break

    def run(self):
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.03)
            self.handle_events()
            if not self.game_over:
                self.update(dt)
            self.render()
            pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_TAB:
                    self.show_mouse = not self.show_mouse
                    self.grab_mouse(not self.show_mouse)
                elif event.key == pygame.K_r:
                    if self.game_over:
                        self.score = 0
                        self.kills = 0
                        self.wave = 1
                        self.reset()
                    elif self.reload_timer <= 0.0 and self.ammo < MAX_AMMO:
                        self.reload_timer = RELOAD_TIME
                elif event.key == pygame.K_RETURN and self.game_over:
                    self.score = 0
                    self.kills = 0
                    self.wave = 1
                    self.reset()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.game_over:
                self.try_shoot()

    def update(self, dt):
        self.shoot_timer = max(0.0, self.shoot_timer - dt)
        self.reload_timer = max(0.0, self.reload_timer - dt)
        self.damage_flash = max(0.0, self.damage_flash - dt * 1.8)
        self.hit_flash = max(0.0, self.hit_flash - dt * 2.2)
        self.muzzle_flash = max(0.0, self.muzzle_flash - dt * 10.0)
        self.crosshair_pulse = max(0.0, self.crosshair_pulse - dt * 4.2)
        self.tracer_timer = max(0.0, self.tracer_timer - dt * 8.0)

        if self.reload_timer == 0.0 and self.ammo < MAX_AMMO:
            self.ammo = MAX_AMMO

        if not self.show_mouse:
            mx, my = pygame.mouse.get_rel()
            self.player_yaw += mx * MOUSE_SENS
            self.player_pitch = clamp(self.player_pitch - my * MOUSE_SENS * 0.7, -48.0, 48.0)

        self.move_player(dt)
        self.update_enemies(dt)

        if not any(enemy.alive for enemy in self.enemies):
            self.wave += 1
            self.hp = min(MAX_HP, self.hp + 12)
            self.spawn_wave(self.wave)

        if self.hp <= 0:
            self.game_over = True
            self.grab_mouse(False)

    def move_player(self, dt):
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
        if abs(strafe) > 0 and abs(forward) == 0:
            speed *= STRAFE_MULT

        yaw_rad = math.radians(self.player_yaw)
        dir_x = math.sin(yaw_rad)
        dir_z = -math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_z = math.sin(yaw_rad)

        move_x = (dir_x * forward + right_x * strafe) * speed * dt
        move_z = (dir_z * forward + right_z * strafe) * speed * dt

        if move_x or move_z:
            self.walk_time += dt * speed * 0.6
            self.weapon_bob = 1.0
        else:
            self.weapon_bob = max(0.0, self.weapon_bob - dt * 3.0)

        self.slide_move(move_x, move_z, radius=0.24)

    def slide_move(self, dx, dz, radius=0.24):
        nx = self.player_x + dx
        nz = self.player_z + dz
        if not self.collides(nx, self.player_z, radius):
            self.player_x = nx
        if not self.collides(self.player_x, nz, radius):
            self.player_z = nz

    def collides(self, x, z, radius):
        min_x = int(math.floor(x - radius))
        max_x = int(math.floor(x + radius))
        min_z = int(math.floor(z - radius))
        max_z = int(math.floor(z + radius))
        for cz in range(min_z, max_z + 1):
            for cx in range(min_x, max_x + 1):
                if (cx, cz) not in WORLD_MAP:
                    continue
                if x + radius > cx and x - radius < cx + 1 and z + radius > cz and z - radius < cz + 1:
                    return True
        return False

    def update_enemies(self, dt):
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            enemy.cooldown = max(0.0, enemy.cooldown - dt)
            enemy.bob += dt * 3.0
            enemy.strafe_phase += dt * 2.2

            to_x = self.player_x - enemy.x
            to_z = self.player_z - enemy.z
            dist = length2(to_x, to_z)
            if dist < 0.001:
                continue

            attack_range = 1.05
            nx = to_x / dist
            nz = to_z / dist
            side_x = -nz
            side_z = nx

            if dist > attack_range:
                chase_speed = ENEMY_SPEED + min(1.2, self.wave * 0.06)
                strafe = math.sin(enemy.strafe_phase) * 0.55
                move_x = (nx + side_x * strafe * 0.35) * chase_speed * dt
                move_z = (nz + side_z * strafe * 0.35) * chase_speed * dt
                self.move_enemy(enemy, move_x, move_z)
            elif enemy.cooldown <= 0.0:
                self.hp = max(0, self.hp - ENEMY_DAMAGE)
                self.damage_flash = 0.85
                enemy.cooldown = ENEMY_ATTACK_COOLDOWN

    def move_enemy(self, enemy, dx, dz):
        nx = enemy.x + dx
        nz = enemy.z + dz
        if not self.collides(nx, enemy.z, enemy.radius):
            enemy.x = nx
        if not self.collides(enemy.x, nz, enemy.radius):
            enemy.z = nz

    def get_camera_vectors(self):
        yaw = math.radians(self.player_yaw)
        pitch = math.radians(self.player_pitch)
        forward_x = math.sin(yaw) * math.cos(pitch)
        forward_y = math.sin(pitch)
        forward_z = -math.cos(yaw) * math.cos(pitch)
        return forward_x, forward_y, forward_z

    def wall_distance_along_ray(self, origin, direction, max_dist=MAX_VIEW_DISTANCE):
        ox, oy, oz = origin
        dx, dy, dz = direction
        dist = 0.0
        step = 0.04
        while dist < max_dist:
            px = ox + dx * dist
            pz = oz + dz * dist
            if (int(px), int(pz)) in WORLD_MAP:
                return dist
            dist += step
        return max_dist

    def try_shoot(self):
        if self.reload_timer > 0.0 or self.shoot_timer > 0.0:
            return
        if self.ammo <= 0:
            self.reload_timer = RELOAD_TIME
            return

        self.ammo -= 1
        self.shoot_timer = SHOOT_COOLDOWN
        self.muzzle_flash = 1.0
        self.crosshair_pulse = 1.0
        self.tracer_timer = 0.16
        self.last_shot_hit = False
        self.last_hit_pos = None

        origin = (self.player_x, PLAYER_HEIGHT, self.player_z)
        direction = self.get_camera_vectors()
        wall_dist = self.wall_distance_along_ray(origin, direction)

        best_enemy = None
        best_t = float("inf")
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            center = (enemy.x, 0.95 + math.sin(enemy.bob) * 0.03, enemy.z)
            t = self.ray_sphere_intersection(origin, direction, center, enemy.radius)
            if t is not None and 0.0 < t < best_t and t < wall_dist:
                best_t = t
                best_enemy = enemy

        if best_enemy is not None:
            best_enemy.hp -= BULLET_DAMAGE
            self.hit_flash = 0.7
            self.last_shot_hit = True
            hit_x = origin[0] + direction[0] * best_t
            hit_y = origin[1] + direction[1] * best_t
            hit_z = origin[2] + direction[2] * best_t
            self.last_hit_pos = (hit_x, hit_y, hit_z)
            if best_enemy.hp <= 0:
                self.score += 100 + self.wave * 8
                self.kills += 1
        else:
            self.last_hit_pos = (
                origin[0] + direction[0] * min(18.0, wall_dist),
                origin[1] + direction[1] * min(18.0, wall_dist),
                origin[2] + direction[2] * min(18.0, wall_dist),
            )

    def ray_sphere_intersection(self, origin, direction, center, radius):
        ox, oy, oz = origin
        dx, dy, dz = direction
        cx, cy, cz = center
        lx = ox - cx
        ly = oy - cy
        lz = oz - cz

        a = dx * dx + dy * dy + dz * dz
        b = 2.0 * (dx * lx + dy * ly + dz * lz)
        c = lx * lx + ly * ly + lz * lz - radius * radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        s = math.sqrt(disc)
        t1 = (-b - s) / (2 * a)
        t2 = (-b + s) / (2 * a)
        if t1 > 0:
            return t1
        if t2 > 0:
            return t2
        return None

    def set_3d(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(72.0, WIDTH / HEIGHT, 0.05, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(-self.player_pitch, 1.0, 0.0, 0.0)
        glRotatef(-self.player_yaw, 0.0, 1.0, 0.0)
        glTranslatef(-self.player_x, -PLAYER_HEIGHT, -self.player_z)

    def set_2d(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def render(self):
        glDisable(GL_TEXTURE_2D)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.set_2d()
        self.draw_background_gradient()
        self.set_3d()
        self.draw_floor_and_ceiling()
        self.draw_walls()
        self.draw_enemies()
        self.draw_tracer()
        self.set_2d()
        self.draw_weapon_overlay()
        self.draw_crosshair()
        self.draw_hud()
        if self.game_over:
            self.draw_game_over()

    def draw_background_gradient(self):
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_QUADS)
        glColor3f(0.08, 0.10, 0.14)
        glVertex2f(0, 0)
        glVertex2f(WIDTH, 0)
        glColor3f(0.22, 0.28, 0.37)
        glVertex2f(WIDTH, HEIGHT * 0.52)
        glVertex2f(0, HEIGHT * 0.52)
        glColor3f(0.05, 0.05, 0.06)
        glVertex2f(0, HEIGHT)
        glVertex2f(WIDTH, HEIGHT)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_floor_and_ceiling(self):
        glDisable(GL_CULL_FACE)
        map_w = len(MAP_STR[0])
        map_h = len(MAP_STR)

        glBegin(GL_QUADS)
        glColor3f(0.15, 0.16, 0.17)
        glVertex3f(0, 0, 0)
        glVertex3f(map_w, 0, 0)
        glColor3f(0.08, 0.08, 0.09)
        glVertex3f(map_w, 0, map_h)
        glVertex3f(0, 0, map_h)

        glColor3f(0.06, 0.07, 0.08)
        glVertex3f(0, 3.2, 0)
        glVertex3f(0, 3.2, map_h)
        glColor3f(0.10, 0.11, 0.12)
        glVertex3f(map_w, 3.2, map_h)
        glVertex3f(map_w, 3.2, 0)
        glEnd()

        # floor panel lines
        glBegin(GL_LINES)
        glColor4f(0.22, 0.24, 0.26, 0.25)
        for i in range(map_w + 1):
            glVertex3f(i, 0.01, 0)
            glVertex3f(i, 0.01, map_h)
        for j in range(map_h + 1):
            glVertex3f(0, 0.01, j)
            glVertex3f(map_w, 0.01, j)
        glEnd()
        glEnable(GL_CULL_FACE)

    def draw_walls(self):
        for (x, z), kind in WORLD_MAP.items():
            base = WALL_COLORS.get(kind, (0.35, 0.35, 0.35))
            self.draw_box(
                x + 0.5,
                0.5,
                z + 0.5,
                1.0,
                1.0,
                1.0,
                base,
            )

    def draw_enemies(self):
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dist = math.hypot(enemy.x - self.player_x, enemy.z - self.player_z)
            shade = clamp(1.35 - dist / 20.0, 0.35, 1.0)
            bob = math.sin(enemy.bob) * 0.045
            glPushMatrix()
            glTranslatef(enemy.x, 0.0 + bob, enemy.z)
            yaw = math.degrees(math.atan2(self.player_x - enemy.x, -(self.player_z - enemy.z)))
            glRotatef(yaw, 0.0, 1.0, 0.0)

            self.draw_box_local(0.0, 0.58, 0.0, 0.36, 0.72, 0.24, (0.20 * shade, 0.22 * shade, 0.26 * shade))
            self.draw_box_local(0.0, 1.04, 0.0, 0.24, 0.24, 0.24, (0.86 * shade, 0.18 * shade, 0.16 * shade))
            self.draw_box_local(-0.26, 0.56, 0.0, 0.09, 0.52, 0.09, (0.16 * shade, 0.17 * shade, 0.18 * shade))
            self.draw_box_local(0.26, 0.56, 0.0, 0.09, 0.52, 0.09, (0.16 * shade, 0.17 * shade, 0.18 * shade))
            self.draw_box_local(-0.12, 0.14, 0.0, 0.10, 0.28, 0.10, (0.12 * shade, 0.12 * shade, 0.12 * shade))
            self.draw_box_local(0.12, 0.14, 0.0, 0.10, 0.28, 0.10, (0.12 * shade, 0.12 * shade, 0.12 * shade))

            # health bar billboard-ish strip
            hp_ratio = clamp(enemy.hp / (85 + self.wave * 10), 0.0, 1.0)
            glDisable(GL_CULL_FACE)
            glTranslatef(0.0, 1.38, 0.0)
            glRotatef(-yaw, 0.0, 1.0, 0.0)
            glBegin(GL_QUADS)
            glColor4f(0.12, 0.12, 0.12, 0.9)
            glVertex3f(-0.35, 0.00, 0.0)
            glVertex3f(0.35, 0.00, 0.0)
            glVertex3f(0.35, 0.05, 0.0)
            glVertex3f(-0.35, 0.05, 0.0)
            glColor4f(0.95, 0.15, 0.15, 0.95)
            glVertex3f(-0.35, 0.00, 0.001)
            glVertex3f(-0.35 + 0.70 * hp_ratio, 0.00, 0.001)
            glVertex3f(-0.35 + 0.70 * hp_ratio, 0.05, 0.001)
            glVertex3f(-0.35, 0.05, 0.001)
            glEnd()
            glEnable(GL_CULL_FACE)
            glPopMatrix()

    def draw_tracer(self):
        if self.tracer_timer <= 0.0 or self.last_hit_pos is None:
            return
        origin = (self.player_x, PLAYER_HEIGHT - 0.07, self.player_z)
        alpha = clamp(self.tracer_timer * 3.0, 0.0, 0.75)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        if self.last_shot_hit:
            glColor4f(1.0, 0.82, 0.38, alpha)
        else:
            glColor4f(0.75, 0.85, 1.0, alpha * 0.85)
        glVertex3f(*origin)
        glVertex3f(*self.last_hit_pos)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_box(self, x, y, z, sx, sy, sz, color):
        glPushMatrix()
        glTranslatef(x, y, z)
        self.draw_box_local(0, 0, 0, sx, sy, sz, color)
        glPopMatrix()

    def draw_box_local(self, x, y, z, sx, sy, sz, color):
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
            (3, 2, 6, 7, 1.10),
            (0, 1, 5, 4, 0.62),
            (1, 2, 6, 5, 0.86),
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

    def draw_weapon_overlay(self):
        glDisable(GL_DEPTH_TEST)
        bob = math.sin(self.walk_time * 9.2) * 7.0 * self.weapon_bob
        kick = self.muzzle_flash * 18.0
        x = WIDTH - 182 - kick * 0.4
        y = HEIGHT - 110 + bob + kick * 0.25

        # shadow
        glBegin(GL_QUADS)
        glColor4f(0.02, 0.02, 0.02, 0.45)
        glVertex2f(x - 18, y + 18)
        glVertex2f(x + 136, y + 6)
        glVertex2f(x + 188, y + 112)
        glVertex2f(x + 28, y + 120)
        glEnd()

        # gun silhouette
        glBegin(GL_QUADS)
        glColor3f(0.11, 0.12, 0.14)
        glVertex2f(x, y)
        glVertex2f(x + 118, y - 8)
        glVertex2f(x + 150, y + 22)
        glVertex2f(x + 22, y + 36)

        glColor3f(0.18, 0.19, 0.22)
        glVertex2f(x + 82, y + 14)
        glVertex2f(x + 190, y + 6)
        glVertex2f(x + 194, y + 18)
        glVertex2f(x + 90, y + 28)

        glColor3f(0.15, 0.16, 0.18)
        glVertex2f(x + 26, y + 32)
        glVertex2f(x + 78, y + 28)
        glVertex2f(x + 102, y + 108)
        glVertex2f(x + 60, y + 112)
        glEnd()

        # sight line / highlight
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(0.75, 0.82, 0.90, 0.55)
        glVertex2f(x + 12, y + 8)
        glVertex2f(x + 142, y + 10)
        glEnd()

        if self.muzzle_flash > 0.0:
            alpha = clamp(self.muzzle_flash, 0.0, 1.0) * 0.6
            mx = x + 194
            my = y + 13
            glBegin(GL_TRIANGLES)
            glColor4f(1.0, 0.90, 0.55, alpha)
            glVertex2f(mx, my)
            glColor4f(1.0, 0.65, 0.12, 0.0)
            glVertex2f(mx + 42, my - 18)
            glVertex2f(mx + 42, my + 18)
            glEnd()

        glEnable(GL_DEPTH_TEST)

    def draw_crosshair(self):
        cx = WIDTH / 2
        cy = HEIGHT / 2
        size = 8 + self.crosshair_pulse * 3
        gap = 6 + self.crosshair_pulse * 2
        hit = self.hit_flash > 0.0
        color = DANGER if hit else CYAN
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(*color, 0.95)
        glVertex2f(cx - gap - size, cy)
        glVertex2f(cx - gap, cy)
        glVertex2f(cx + gap, cy)
        glVertex2f(cx + gap + size, cy)
        glVertex2f(cx, cy - gap - size)
        glVertex2f(cx, cy - gap)
        glVertex2f(cx, cy + gap)
        glVertex2f(cx, cy + gap + size)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_hud(self):
        glDisable(GL_DEPTH_TEST)

        # damage vignette
        if self.damage_flash > 0.0:
            a = clamp(self.damage_flash * 0.28, 0.0, 0.28)
            glBegin(GL_QUADS)
            glColor4f(0.9, 0.05, 0.05, a)
            glVertex2f(0, 0)
            glVertex2f(WIDTH, 0)
            glVertex2f(WIDTH, HEIGHT)
            glVertex2f(0, HEIGHT)
            glEnd()

        # lower bar frame
        glBegin(GL_QUADS)
        glColor4f(0.03, 0.04, 0.05, 0.64)
        glVertex2f(14, HEIGHT - 72)
        glVertex2f(340, HEIGHT - 72)
        glVertex2f(340, HEIGHT - 18)
        glVertex2f(14, HEIGHT - 18)
        glEnd()

        # health bar
        hp_ratio = clamp(self.hp / MAX_HP, 0.0, 1.0)
        glBegin(GL_QUADS)
        glColor4f(0.15, 0.15, 0.17, 0.88)
        glVertex2f(24, HEIGHT - 34)
        glVertex2f(204, HEIGHT - 34)
        glVertex2f(204, HEIGHT - 22)
        glVertex2f(24, HEIGHT - 22)
        glColor4f(0.20, 0.82, 0.34, 0.92)
        glVertex2f(24, HEIGHT - 34)
        glVertex2f(24 + 180 * hp_ratio, HEIGHT - 34)
        glVertex2f(24 + 180 * hp_ratio, HEIGHT - 22)
        glVertex2f(24, HEIGHT - 22)
        glEnd()

        ammo_color = ACCENT if self.ammo > 6 else (1.0, 0.38, 0.22)
        self.draw_text(24, HEIGHT - 64, f"HP {self.hp:03d}", self.ui_font, WHITE)
        self.draw_text(24, HEIGHT - 92, f"SCORE {self.score}", self.small_font, (0.80, 0.90, 0.96))
        self.draw_text(WIDTH - 200, HEIGHT - 66, f"AMMO {self.ammo:02d}/{MAX_AMMO}", self.ui_font, ammo_color)
        self.draw_text(WIDTH - 200, HEIGHT - 92, f"WAVE {self.wave}   KILLS {self.kills}", self.small_font, (0.80, 0.90, 0.96))

        if self.reload_timer > 0.0:
            self.draw_text(WIDTH / 2 - 48, HEIGHT - 62, "RELOADING", self.ui_font, ACCENT)

        self.draw_text(18, 16, "WASD move  Shift sprint  Mouse aim  Click fire  R reload  Tab mouse", self.small_font, (0.72, 0.84, 0.92))
        glEnable(GL_DEPTH_TEST)

    def draw_game_over(self):
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_QUADS)
        glColor4f(0.01, 0.01, 0.01, 0.72)
        glVertex2f(0, 0)
        glVertex2f(WIDTH, 0)
        glVertex2f(WIDTH, HEIGHT)
        glVertex2f(0, HEIGHT)
        glEnd()
        self.draw_text(WIDTH / 2 - 150, HEIGHT / 2 - 44, "MISSION FAILED", self.big_font, (1.0, 0.44, 0.36))
        self.draw_text(WIDTH / 2 - 100, HEIGHT / 2 + 10, f"Final Score: {self.score}", self.ui_font, WHITE)
        self.draw_text(WIDTH / 2 - 132, HEIGHT / 2 + 42, "Press R or Enter to restart", self.ui_font, ACCENT)
        glEnable(GL_DEPTH_TEST)

    def draw_text(self, x, y, text, font, color):
        key = (text, font.get_height(), tuple(int(c * 255) if isinstance(c, float) else int(c) for c in color))
        if key not in self._text_cache:
            color255 = tuple(int(clamp(c, 0, 1) * 255) if isinstance(c, float) else int(c) for c in color)
            surf = font.render(text, True, color255)
            data = pygame.image.tostring(surf, "RGBA", True)
            self._text_cache[key] = (surf.get_width(), surf.get_height(), data)
        w, h, data = self._text_cache[key]
        glWindowPos2d(int(x), int(HEIGHT - y - h))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)


def main():
    try:
        Game().run()
    except Exception as exc:
        pygame.quit()
        raise exc


if __name__ == "__main__":
    main()
