#!/usr/bin/env python3
"""
Elf and Demon Arena FPS

A true 3D Pygame + PyOpenGL first-person shooter prototype.

Controls:
  WASD              Move
  Mouse             Look
  Left mouse        Attack
  1 / 2 / 3         Shotgun / Revolver / Knife
  R                 Reload current firearm
  Shift             Sprint
  Tab               Toggle mouse grab
  F5                Restart
  Esc               Quit

Dependencies:
  pip install pygame PyOpenGL PyOpenGL_accelerate
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


WIDTH, HEIGHT = 1024, 640
FPS = 60
ARENA_HALF = 92.0
PLAYER_HEIGHT = 1.72
PLAYER_RADIUS = 0.42
PLAYER_SPEED = 7.2
SPRINT_MULT = 1.55
MOUSE_SENS = 0.11
MAX_HP = 125
MAX_PITCH = 78.0
FOG_END = 145.0

WHITE = (0.92, 0.95, 0.96)
HUD = (0.68, 0.82, 0.88)
GOLD = (1.00, 0.74, 0.28)
RED = (0.95, 0.20, 0.15)
GREEN = (0.30, 0.88, 0.42)
BLUE = (0.42, 0.72, 1.00)
DARK = (0.035, 0.045, 0.055)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_delta(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def length3(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize3(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    mag = length3(v)
    if mag <= 0.000001:
        return 0.0, 0.0, -1.0
    return v[0] / mag, v[1] / mag, v[2] / mag


def direction_from_angles(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    return math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    damage: float
    pellets: int
    spread_deg: float
    max_range: float
    fire_delay: float
    clip_size: int
    reload_time: float
    reserve_start: int
    kick: float
    color: Tuple[float, float, float]
    melee: bool = False
    melee_range: float = 0.0
    melee_arc_deg: float = 0.0


WEAPONS: List[WeaponSpec] = [
    WeaponSpec("Shotgun", 15.0, 10, 8.0, 28.0, 0.92, 8, 2.15, 48, 1.0, (0.78, 0.54, 0.28)),
    WeaponSpec("Revolver", 58.0, 1, 1.2, 72.0, 0.42, 6, 1.55, 54, 0.55, (0.82, 0.76, 0.62)),
    WeaponSpec("Knife", 46.0, 1, 0.0, 2.25, 0.38, 0, 0.0, 0, 0.36, (0.78, 0.86, 0.92), True, 2.35, 62.0),
]


@dataclass(frozen=True)
class EnemyType:
    name: str
    hp: float
    speed: float
    radius: float
    height: float
    damage: int
    cooldown: float
    score: int
    color: Tuple[float, float, float]
    ranged: bool = False
    projectile_speed: float = 0.0
    attack_range: float = 0.0


ENEMY_TYPES: Dict[str, EnemyType] = {
    "elf": EnemyType(
        "Elf",
        hp=74,
        speed=4.25,
        radius=0.34,
        height=1.72,
        damage=10,
        cooldown=1.25,
        score=90,
        color=(0.30, 0.58, 0.33),
        ranged=True,
        projectile_speed=22.0,
        attack_range=42.0,
    ),
    "demon": EnemyType(
        "Demon",
        hp=148,
        speed=2.85,
        radius=0.55,
        height=2.22,
        damage=22,
        cooldown=0.95,
        score=150,
        color=(0.72, 0.16, 0.12),
        ranged=False,
        projectile_speed=11.0,
        attack_range=22.0,
    ),
}


@dataclass
class Player:
    x: float = 0.0
    z: float = 18.0
    yaw: float = 180.0
    pitch: float = -2.0
    hp: int = MAX_HP
    weapon_index: int = 0
    clip: Dict[int, int] = field(default_factory=lambda: {i: w.clip_size for i, w in enumerate(WEAPONS)})
    reserve: Dict[int, int] = field(default_factory=lambda: {i: w.reserve_start for i, w in enumerate(WEAPONS)})
    fire_timer: float = 0.0
    reload_timer: float = 0.0
    reloading: bool = False
    kick: float = 0.0
    bob: float = 0.0
    hurt_flash: float = 0.0
    hit_flash: float = 0.0
    score: int = 0
    kills: int = 0

    def weapon(self) -> WeaponSpec:
        return WEAPONS[self.weapon_index]


@dataclass
class Enemy:
    x: float
    z: float
    kind_key: str
    hp: float
    cooldown: float = 0.0
    phase: float = 0.0
    hit_flash: float = 0.0
    alive: bool = True
    strafe_dir: float = 1.0

    @property
    def kind(self) -> EnemyType:
        return ENEMY_TYPES[self.kind_key]


@dataclass
class Projectile:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    speed: float
    damage: int
    radius: float
    ttl: float
    color: Tuple[float, float, float]
    name: str


@dataclass
class Obstacle:
    x: float
    z: float
    w: float
    d: float
    h: float
    color: Tuple[float, float, float]
    solid: bool = True

    @property
    def min_x(self) -> float:
        return self.x - self.w * 0.5

    @property
    def max_x(self) -> float:
        return self.x + self.w * 0.5

    @property
    def min_z(self) -> float:
        return self.z - self.d * 0.5

    @property
    def max_z(self) -> float:
        return self.z + self.d * 0.5


class ArenaFPS:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Elf and Demon Arena FPS")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 19, bold=True)
        self.small = pygame.font.SysFont("consolas", 14, bold=True)
        self.big = pygame.font.SysFont("consolas", 44, bold=True)
        self.text_cache: Dict[Tuple[str, int, Tuple[int, int, int]], Tuple[int, int, bytes]] = {}
        self.rng = random.Random()
        self.setup_gl()
        self.reset()

    def setup_gl(self) -> None:
        glViewport(0, 0, WIDTH, HEIGHT)
        glClearColor(0.10, 0.13, 0.17, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        glEnable(GL_FOG)
        fog_color = (0.11, 0.13, 0.16, 1.0)
        glFogfv(GL_FOG_COLOR, fog_color)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogf(GL_FOG_START, 48.0)
        glFogf(GL_FOG_END, FOG_END)

    def reset(self) -> None:
        self.player = Player()
        self.wave = 1
        self.enemies: List[Enemy] = []
        self.projectiles: List[Projectile] = []
        self.tracers: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float, bool]] = []
        self.obstacles = self.build_obstacles()
        self.message = "Survive the arena"
        self.message_timer = 3.0
        self.wave_flash = 0.0
        self.game_over = False
        self.grab_mouse(True)
        self.spawn_wave()

    def build_obstacles(self) -> List[Obstacle]:
        obstacles: List[Obstacle] = []
        stone = (0.32, 0.35, 0.37)
        dark_stone = (0.23, 0.25, 0.28)
        bronze = (0.50, 0.34, 0.18)

        # Four broad ruined platforms leave the center open while adding cover.
        for x, z, w, d in [
            (-34, -28, 18, 4),
            (34, 28, 18, 4),
            (-34, 30, 4, 18),
            (34, -30, 4, 18),
        ]:
            obstacles.append(Obstacle(x, z, w, d, 2.2, stone))

        # Ring of pillars gives the large arena landmarks and line-of-sight breaks.
        for i in range(16):
            ang = math.tau * i / 16.0
            radius = 54.0 if i % 2 == 0 else 67.0
            px = math.cos(ang) * radius
            pz = math.sin(ang) * radius
            height = 5.6 if i % 3 else 7.2
            obstacles.append(Obstacle(px, pz, 3.4, 3.4, height, dark_stone))

        # Low broken barricades near the center.
        for x, z, w, d in [
            (-12, -8, 12, 2.8),
            (13, 9, 12, 2.8),
            (-8, 14, 2.8, 12),
            (9, -14, 2.8, 12),
        ]:
            obstacles.append(Obstacle(x, z, w, d, 1.25, bronze))
        return obstacles

    def grab_mouse(self, enabled: bool) -> None:
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        if enabled:
            pygame.mouse.get_rel()

    def spawn_wave(self) -> None:
        count = 5 + self.wave * 3
        demon_count = max(1, self.wave // 2)
        elf_count = count - demon_count
        spawn_plan = ["elf"] * elf_count + ["demon"] * demon_count
        self.rng.shuffle(spawn_plan)

        for kind_key in spawn_plan:
            for _ in range(80):
                angle = self.rng.random() * math.tau
                radius = self.rng.uniform(42.0, ARENA_HALF - 12.0)
                x = math.cos(angle) * radius
                z = math.sin(angle) * radius
                if math.hypot(x - self.player.x, z - self.player.z) < 24.0:
                    continue
                if self.circle_collides_world(x, z, ENEMY_TYPES[kind_key].radius + 0.1):
                    continue
                self.enemies.append(
                    Enemy(
                        x=x,
                        z=z,
                        kind_key=kind_key,
                        hp=ENEMY_TYPES[kind_key].hp + self.wave * (8 if kind_key == "elf" else 14),
                        phase=self.rng.random() * math.tau,
                        cooldown=self.rng.random() * 1.2,
                        strafe_dir=-1.0 if self.rng.random() < 0.5 else 1.0,
                    )
                )
                break
        self.message = f"Wave {self.wave}: elves and demons enter"
        self.message_timer = 2.5

    def run(self) -> None:
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.04)
            self.handle_events()
            if not self.game_over:
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
                    self.grab_mouse(not pygame.event.get_grab())
                if event.key == pygame.K_F5:
                    self.reset()
                if event.key == pygame.K_r:
                    if self.game_over:
                        self.reset()
                    else:
                        self.try_reload()
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.switch_weapon(event.key - pygame.K_1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.game_over:
                self.try_attack()

    def switch_weapon(self, index: int) -> None:
        if index == self.player.weapon_index:
            return
        self.player.weapon_index = index
        self.player.reload_timer = 0.0
        self.player.reloading = False
        self.message = WEAPONS[index].name
        self.message_timer = 0.8

    def try_reload(self) -> None:
        weapon = self.player.weapon()
        if weapon.melee or self.player.reload_timer > 0:
            return
        idx = self.player.weapon_index
        if self.player.clip[idx] >= weapon.clip_size:
            return
        if self.player.reserve[idx] <= 0:
            self.message = "No reserve ammo"
            self.message_timer = 0.8
            return
        self.player.reload_timer = weapon.reload_time
        self.player.reloading = True
        self.message = f"Reloading {weapon.name}"
        self.message_timer = 0.7

    def update(self, dt: float) -> None:
        self.player.fire_timer = max(0.0, self.player.fire_timer - dt)
        self.player.hit_flash = max(0.0, self.player.hit_flash - dt * 4.0)
        self.player.hurt_flash = max(0.0, self.player.hurt_flash - dt * 1.8)
        self.player.kick = max(0.0, self.player.kick - dt * 3.6)
        self.message_timer = max(0.0, self.message_timer - dt)
        self.wave_flash = max(0.0, self.wave_flash - dt * 1.8)
        self.tracers = [(a, b, timer - dt * 4.8, hit) for a, b, timer, hit in self.tracers if timer - dt * 4.8 > 0.0]

        if self.player.reload_timer > 0:
            self.player.reload_timer = max(0.0, self.player.reload_timer - dt)
            if self.player.reloading and self.player.reload_timer <= 0:
                self.finish_reload()

        self.update_mouse_look()
        self.update_player_movement(dt)

        if pygame.mouse.get_pressed()[0]:
            self.try_attack()

        self.update_enemies(dt)
        self.update_projectiles(dt)

        if not any(enemy.alive for enemy in self.enemies):
            self.wave += 1
            self.wave_flash = 0.7
            self.player.hp = min(MAX_HP, self.player.hp + 20)
            self.player.reserve[0] += 12
            self.player.reserve[1] += 10
            self.spawn_wave()

        if self.player.hp <= 0:
            self.game_over = True
            self.grab_mouse(False)

    def finish_reload(self) -> None:
        idx = self.player.weapon_index
        weapon = self.player.weapon()
        need = weapon.clip_size - self.player.clip[idx]
        load = min(need, self.player.reserve[idx])
        self.player.clip[idx] += load
        self.player.reserve[idx] -= load
        self.player.reloading = False

    def update_mouse_look(self) -> None:
        if pygame.event.get_grab():
            mx, my = pygame.mouse.get_rel()
            self.player.yaw = (self.player.yaw + mx * MOUSE_SENS) % 360.0
            self.player.pitch = clamp(self.player.pitch - my * MOUSE_SENS, -MAX_PITCH, MAX_PITCH)
        else:
            pygame.mouse.get_rel()

    def update_player_movement(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        turn = 0.0
        if keys[pygame.K_LEFT]:
            turn -= 1.0
        if keys[pygame.K_RIGHT]:
            turn += 1.0
        self.player.yaw = (self.player.yaw + turn * 115.0 * dt) % 360.0

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

        if not forward and not strafe:
            return

        mag = math.hypot(forward, strafe)
        forward /= mag
        strafe /= mag
        speed = PLAYER_SPEED * (SPRINT_MULT if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
        yaw = math.radians(self.player.yaw)
        dir_x = math.sin(yaw)
        dir_z = -math.cos(yaw)
        right_x = math.cos(yaw)
        right_z = math.sin(yaw)
        dx = (dir_x * forward + right_x * strafe) * speed * dt
        dz = (dir_z * forward + right_z * strafe) * speed * dt
        self.player.bob += dt * speed * 1.25
        self.move_player(dx, dz)

    def move_player(self, dx: float, dz: float) -> None:
        nx = self.player.x + dx
        if not self.circle_collides_world(nx, self.player.z, PLAYER_RADIUS):
            self.player.x = nx
        nz = self.player.z + dz
        if not self.circle_collides_world(self.player.x, nz, PLAYER_RADIUS):
            self.player.z = nz

    def circle_collides_world(self, x: float, z: float, radius: float) -> bool:
        if x - radius < -ARENA_HALF or x + radius > ARENA_HALF:
            return True
        if z - radius < -ARENA_HALF or z + radius > ARENA_HALF:
            return True
        for obstacle in self.obstacles:
            if not obstacle.solid:
                continue
            nearest_x = clamp(x, obstacle.min_x, obstacle.max_x)
            nearest_z = clamp(z, obstacle.min_z, obstacle.max_z)
            if (x - nearest_x) ** 2 + (z - nearest_z) ** 2 < radius * radius:
                return True
        return False

    def try_attack(self) -> None:
        weapon = self.player.weapon()
        if self.player.fire_timer > 0.0 or self.player.reload_timer > 0.0:
            return
        self.player.fire_timer = weapon.fire_delay
        self.player.kick = weapon.kick
        if weapon.melee:
            self.swing_knife(weapon)
            return

        idx = self.player.weapon_index
        if self.player.clip[idx] <= 0:
            self.try_reload()
            return
        self.player.clip[idx] -= 1

        hit_any = False
        origin = (self.player.x, PLAYER_HEIGHT - 0.05, self.player.z)
        for _ in range(weapon.pellets):
            yaw = self.player.yaw + self.rng.uniform(-weapon.spread_deg * 0.5, weapon.spread_deg * 0.5)
            pitch = self.player.pitch + self.rng.uniform(-weapon.spread_deg * 0.35, weapon.spread_deg * 0.35)
            direction = direction_from_angles(yaw, pitch)
            hit_any = self.fire_hitscan(origin, direction, weapon.damage, weapon.max_range) or hit_any

        if hit_any:
            self.player.hit_flash = 1.0
        if self.player.clip[idx] <= 0:
            self.try_reload()

    def fire_hitscan(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        damage: float,
        max_range: float,
    ) -> bool:
        world_hit = self.ray_world_distance(origin, direction, max_range)
        best_enemy: Optional[Enemy] = None
        best_t = world_hit
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            center = (enemy.x, enemy.kind.height * 0.62, enemy.z)
            radius = max(0.58, enemy.kind.radius * (1.70 if enemy.kind_key == "demon" else 1.85))
            t = self.ray_sphere(origin, direction, center, radius)
            if t is not None and 0.0 < t < best_t:
                best_t = t
                best_enemy = enemy

        end = (
            origin[0] + direction[0] * min(best_t, max_range),
            origin[1] + direction[1] * min(best_t, max_range),
            origin[2] + direction[2] * min(best_t, max_range),
        )

        if best_enemy is None:
            self.tracers.append((origin, end, 0.30, False))
            return False

        best_enemy.hp -= damage
        best_enemy.hit_flash = 0.22
        self.tracers.append((origin, end, 0.34, True))
        if best_enemy.hp <= 0 and best_enemy.alive:
            best_enemy.alive = False
            self.player.kills += 1
            self.player.score += best_enemy.kind.score + self.wave * 12
            self.message = f"{best_enemy.kind.name} down"
            self.message_timer = 0.6
        return True

    def swing_knife(self, weapon: WeaponSpec) -> None:
        origin = (self.player.x, PLAYER_HEIGHT - 0.05, self.player.z)
        hit_any = False
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - self.player.x
            dz = enemy.z - self.player.z
            distance = math.hypot(dx, dz)
            if distance > weapon.melee_range + enemy.kind.radius:
                continue
            enemy_yaw = math.degrees(math.atan2(dx, -dz))
            if abs(angle_delta(enemy_yaw, self.player.yaw)) > weapon.melee_arc_deg * 0.5:
                continue
            target = (enemy.x, enemy.kind.height * 0.55, enemy.z)
            if self.ray_world_distance(origin, normalize3((target[0] - origin[0], target[1] - origin[1], target[2] - origin[2])), distance) < distance - 0.2:
                continue
            enemy.hp -= weapon.damage
            enemy.hit_flash = 0.25
            hit_any = True
            if enemy.hp <= 0 and enemy.alive:
                enemy.alive = False
                self.player.kills += 1
                self.player.score += enemy.kind.score + self.wave * 10
        self.player.hit_flash = 1.0 if hit_any else 0.0
        if hit_any:
            self.message = "Knife hit"
            self.message_timer = 0.45

    def update_enemies(self, dt: float) -> None:
        live_enemies = [enemy for enemy in self.enemies if enemy.alive]
        for enemy in live_enemies:
            enemy.phase += dt
            enemy.cooldown = max(0.0, enemy.cooldown - dt)
            enemy.hit_flash = max(0.0, enemy.hit_flash - dt)

            dx = self.player.x - enemy.x
            dz = self.player.z - enemy.z
            distance = math.hypot(dx, dz)
            if distance <= 0.001:
                continue
            nx = dx / distance
            nz = dz / distance
            side_x = -nz * enemy.strafe_dir
            side_z = nx * enemy.strafe_dir

            if enemy.kind_key == "elf":
                self.update_elf(enemy, dt, distance, nx, nz, side_x, side_z)
            else:
                self.update_demon(enemy, dt, distance, nx, nz, side_x, side_z)

        self.resolve_enemy_separation(live_enemies, dt)

    def update_elf(
        self,
        enemy: Enemy,
        dt: float,
        distance: float,
        nx: float,
        nz: float,
        side_x: float,
        side_z: float,
    ) -> None:
        desired = 19.0
        if distance < 9.0:
            move_x = (-nx * 1.1 + side_x * 0.55) * enemy.kind.speed * dt
            move_z = (-nz * 1.1 + side_z * 0.55) * enemy.kind.speed * dt
            self.move_enemy(enemy, move_x, move_z)
        elif distance > desired:
            move_x = (nx * 0.72 + side_x * 0.38) * enemy.kind.speed * dt
            move_z = (nz * 0.72 + side_z * 0.38) * enemy.kind.speed * dt
            self.move_enemy(enemy, move_x, move_z)
        else:
            move_x = side_x * enemy.kind.speed * 0.55 * dt
            move_z = side_z * enemy.kind.speed * 0.55 * dt
            self.move_enemy(enemy, move_x, move_z)

        if self.rng.random() < dt * 0.35:
            enemy.strafe_dir *= -1.0

        if distance <= enemy.kind.attack_range and enemy.cooldown <= 0.0 and self.enemy_has_los(enemy):
            enemy.cooldown = enemy.kind.cooldown + self.rng.uniform(0.0, 0.25)
            self.spawn_projectile_from_enemy(enemy, "arrow")

    def update_demon(
        self,
        enemy: Enemy,
        dt: float,
        distance: float,
        nx: float,
        nz: float,
        side_x: float,
        side_z: float,
    ) -> None:
        melee_range = 1.55 + enemy.kind.radius
        if distance <= melee_range:
            if enemy.cooldown <= 0.0:
                enemy.cooldown = enemy.kind.cooldown
                self.damage_player(enemy.kind.damage)
            return

        charge = 1.0 + (0.35 if distance > 16.0 else 0.0)
        wobble = math.sin(enemy.phase * 3.1) * 0.24
        move_x = (nx * charge + side_x * wobble) * enemy.kind.speed * dt
        move_z = (nz * charge + side_z * wobble) * enemy.kind.speed * dt
        self.move_enemy(enemy, move_x, move_z)

        if 8.0 < distance < enemy.kind.attack_range and enemy.cooldown <= 0.0 and self.enemy_has_los(enemy):
            enemy.cooldown = 2.15 + self.rng.uniform(0.0, 0.5)
            self.spawn_projectile_from_enemy(enemy, "fireball")

    def move_enemy(self, enemy: Enemy, dx: float, dz: float) -> None:
        nx = enemy.x + dx
        if not self.circle_collides_world(nx, enemy.z, enemy.kind.radius):
            enemy.x = nx
        else:
            enemy.strafe_dir *= -1.0
        nz = enemy.z + dz
        if not self.circle_collides_world(enemy.x, nz, enemy.kind.radius):
            enemy.z = nz
        else:
            enemy.strafe_dir *= -1.0

    def resolve_enemy_separation(self, live_enemies: List[Enemy], dt: float) -> None:
        for i, a in enumerate(live_enemies):
            for b in live_enemies[i + 1 :]:
                dx = b.x - a.x
                dz = b.z - a.z
                d = math.hypot(dx, dz)
                minimum = a.kind.radius + b.kind.radius + 0.08
                if 0.001 < d < minimum:
                    push = (minimum - d) * 0.5
                    nx = dx / d
                    nz = dz / d
                    if not self.circle_collides_world(a.x - nx * push, a.z - nz * push, a.kind.radius):
                        a.x -= nx * push
                        a.z -= nz * push
                    if not self.circle_collides_world(b.x + nx * push, b.z + nz * push, b.kind.radius):
                        b.x += nx * push
                        b.z += nz * push

    def enemy_has_los(self, enemy: Enemy) -> bool:
        origin = (enemy.x, enemy.kind.height * 0.72, enemy.z)
        target = (self.player.x, PLAYER_HEIGHT * 0.86, self.player.z)
        direction = normalize3((target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]))
        distance = math.dist(origin, target)
        return self.ray_world_distance(origin, direction, distance) >= distance - 0.2

    def spawn_projectile_from_enemy(self, enemy: Enemy, name: str) -> None:
        start = (enemy.x, enemy.kind.height * 0.72, enemy.z)
        target = (
            self.player.x + self.rng.uniform(-0.25, 0.25),
            PLAYER_HEIGHT * 0.82 + self.rng.uniform(-0.10, 0.10),
            self.player.z + self.rng.uniform(-0.25, 0.25),
        )
        direction = normalize3((target[0] - start[0], target[1] - start[1], target[2] - start[2]))
        if name == "arrow":
            projectile = Projectile(
                *start,
                *direction,
                speed=enemy.kind.projectile_speed,
                damage=enemy.kind.damage,
                radius=0.10,
                ttl=3.0,
                color=(0.72, 0.94, 0.48),
                name="arrow",
            )
        else:
            projectile = Projectile(
                *start,
                *direction,
                speed=enemy.kind.projectile_speed,
                damage=16,
                radius=0.22,
                ttl=4.2,
                color=(1.0, 0.28, 0.05),
                name="fireball",
            )
        self.projectiles.append(projectile)

    def update_projectiles(self, dt: float) -> None:
        next_projectiles: List[Projectile] = []
        player_center = (self.player.x, PLAYER_HEIGHT * 0.82, self.player.z)
        for projectile in self.projectiles:
            projectile.ttl -= dt
            if projectile.ttl <= 0.0:
                continue
            projectile.x += projectile.dx * projectile.speed * dt
            projectile.y += projectile.dy * projectile.speed * dt
            projectile.z += projectile.dz * projectile.speed * dt

            if projectile.y < 0.05 or projectile.y > 7.5:
                continue
            if self.point_hits_world(projectile.x, projectile.y, projectile.z, projectile.radius):
                continue
            if math.dist((projectile.x, projectile.y, projectile.z), player_center) <= projectile.radius + 0.52:
                self.damage_player(projectile.damage)
                continue
            next_projectiles.append(projectile)
        self.projectiles = next_projectiles

    def damage_player(self, amount: int) -> None:
        self.player.hp = max(0, self.player.hp - amount)
        self.player.hurt_flash = 1.0

    def point_hits_world(self, x: float, y: float, z: float, radius: float) -> bool:
        if x - radius < -ARENA_HALF or x + radius > ARENA_HALF:
            return True
        if z - radius < -ARENA_HALF or z + radius > ARENA_HALF:
            return True
        for obstacle in self.obstacles:
            if y > obstacle.h + radius:
                continue
            if x + radius >= obstacle.min_x and x - radius <= obstacle.max_x and z + radius >= obstacle.min_z and z - radius <= obstacle.max_z:
                return True
        return False

    def ray_world_distance(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_range: float,
    ) -> float:
        best = max_range
        ox, oy, oz = origin
        dx, dy, dz = direction

        if abs(dx) > 0.000001:
            for plane_x in (-ARENA_HALF, ARENA_HALF):
                t = (plane_x - ox) / dx
                if 0.0 < t < best:
                    z = oz + dz * t
                    y = oy + dy * t
                    if -ARENA_HALF <= z <= ARENA_HALF and 0.0 <= y <= 9.0:
                        best = t
        if abs(dz) > 0.000001:
            for plane_z in (-ARENA_HALF, ARENA_HALF):
                t = (plane_z - oz) / dz
                if 0.0 < t < best:
                    x = ox + dx * t
                    y = oy + dy * t
                    if -ARENA_HALF <= x <= ARENA_HALF and 0.0 <= y <= 9.0:
                        best = t

        for obstacle in self.obstacles:
            hit = self.ray_aabb(
                origin,
                direction,
                (obstacle.min_x, 0.0, obstacle.min_z),
                (obstacle.max_x, obstacle.h, obstacle.max_z),
            )
            if hit is not None and 0.0 < hit < best:
                best = hit
        return best

    @staticmethod
    def ray_sphere(
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
        b = 2.0 * (dx * lx + dy * ly + dz * lz)
        c = lx * lx + ly * ly + lz * lz - radius * radius
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        root = math.sqrt(disc)
        t1 = (-b - root) * 0.5
        t2 = (-b + root) * 0.5
        if t1 > 0.0:
            return t1
        if t2 > 0.0:
            return t2
        return None

    @staticmethod
    def ray_aabb(
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        min_corner: Tuple[float, float, float],
        max_corner: Tuple[float, float, float],
    ) -> Optional[float]:
        t_min = -float("inf")
        t_max = float("inf")
        for i in range(3):
            o = origin[i]
            d = direction[i]
            mn = min_corner[i]
            mx = max_corner[i]
            if abs(d) < 0.000001:
                if o < mn or o > mx:
                    return None
                continue
            inv = 1.0 / d
            t1 = (mn - o) * inv
            t2 = (mx - o) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None
        if t_max < 0.0:
            return None
        return t_min if t_min > 0.0 else t_max

    # -------------------------- Rendering --------------------------

    def set_3d(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(74.0, WIDTH / HEIGHT, 0.05, 260.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(-self.player.pitch, 1.0, 0.0, 0.0)
        glRotatef(-self.player.yaw, 0.0, 1.0, 0.0)
        glTranslatef(-self.player.x, -PLAYER_HEIGHT, -self.player.z)

    def set_2d(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def render(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.set_3d()
        self.draw_arena()
        self.draw_projectiles()
        self.draw_enemies()
        self.draw_tracers()
        self.set_2d()
        self.draw_weapon_overlay()
        self.draw_crosshair()
        self.draw_hud()
        if self.game_over:
            self.draw_game_over()

    def draw_arena(self) -> None:
        glDisable(GL_CULL_FACE)
        glBegin(GL_QUADS)
        glColor3f(0.17, 0.16, 0.13)
        glVertex3f(-ARENA_HALF, 0.0, -ARENA_HALF)
        glVertex3f(ARENA_HALF, 0.0, -ARENA_HALF)
        glColor3f(0.11, 0.12, 0.10)
        glVertex3f(ARENA_HALF, 0.0, ARENA_HALF)
        glVertex3f(-ARENA_HALF, 0.0, ARENA_HALF)
        glEnd()

        glBegin(GL_LINES)
        glColor4f(0.35, 0.32, 0.24, 0.30)
        for value in range(int(-ARENA_HALF), int(ARENA_HALF) + 1, 8):
            glVertex3f(value, 0.015, -ARENA_HALF)
            glVertex3f(value, 0.015, ARENA_HALF)
            glVertex3f(-ARENA_HALF, 0.015, value)
            glVertex3f(ARENA_HALF, 0.015, value)
        glEnd()
        glEnable(GL_CULL_FACE)

        # Massive boundary walls.
        wall_color = (0.27, 0.28, 0.31)
        self.draw_box(0.0, 3.0, -ARENA_HALF - 1.5, ARENA_HALF * 2 + 6, 6.0, 3.0, wall_color)
        self.draw_box(0.0, 3.0, ARENA_HALF + 1.5, ARENA_HALF * 2 + 6, 6.0, 3.0, wall_color)
        self.draw_box(-ARENA_HALF - 1.5, 3.0, 0.0, 3.0, 6.0, ARENA_HALF * 2 + 6, wall_color)
        self.draw_box(ARENA_HALF + 1.5, 3.0, 0.0, 3.0, 6.0, ARENA_HALF * 2 + 6, wall_color)

        # Distant gate shapes on the walls.
        gate_color = (0.42, 0.29, 0.16)
        self.draw_box(0.0, 3.6, -ARENA_HALF - 3.2, 16.0, 7.2, 1.1, gate_color)
        self.draw_box(0.0, 3.6, ARENA_HALF + 3.2, 16.0, 7.2, 1.1, gate_color)
        self.draw_box(-ARENA_HALF - 3.2, 3.6, 0.0, 1.1, 7.2, 16.0, gate_color)
        self.draw_box(ARENA_HALF + 3.2, 3.6, 0.0, 1.1, 7.2, 16.0, gate_color)

        for obstacle in self.obstacles:
            self.draw_box(obstacle.x, obstacle.h * 0.5, obstacle.z, obstacle.w, obstacle.h, obstacle.d, obstacle.color)
            if obstacle.h > 3.0:
                self.draw_box(obstacle.x, obstacle.h + 0.18, obstacle.z, obstacle.w * 1.35, 0.36, obstacle.d * 1.35, (0.38, 0.40, 0.42))

    def draw_enemies(self) -> None:
        for enemy in sorted((e for e in self.enemies if e.alive), key=lambda e: math.hypot(e.x - self.player.x, e.z - self.player.z), reverse=True):
            if enemy.kind_key == "elf":
                self.draw_elf(enemy)
            else:
                self.draw_demon(enemy)
            self.draw_enemy_health(enemy)

    def face_player_yaw(self, x: float, z: float) -> float:
        return math.degrees(math.atan2(self.player.x - x, self.player.z - z))

    def draw_elf(self, enemy: Enemy) -> None:
        shade = 1.0 + (0.45 if enemy.hit_flash > 0 else 0.0)
        bob = math.sin(enemy.phase * 7.0) * 0.045
        glPushMatrix()
        glTranslatef(enemy.x, bob, enemy.z)
        glRotatef(self.face_player_yaw(enemy.x, enemy.z), 0.0, 1.0, 0.0)
        tunic = tuple(clamp(c * shade, 0.0, 1.0) for c in enemy.kind.color)
        skin = tuple(clamp(c * shade, 0.0, 1.0) for c in (0.82, 0.69, 0.50))
        leather = tuple(clamp(c * shade, 0.0, 1.0) for c in (0.25, 0.16, 0.09))

        self.draw_box_local(0.0, 0.78, 0.0, 0.46, 0.94, 0.24, tunic)
        self.draw_box_local(-0.16, 0.25, 0.0, 0.13, 0.50, 0.13, leather)
        self.draw_box_local(0.16, 0.25, 0.0, 0.13, 0.50, 0.13, leather)
        self.draw_box_local(-0.36, 0.86, 0.0, 0.10, 0.70, 0.10, skin)
        self.draw_box_local(0.36, 0.86, 0.0, 0.10, 0.70, 0.10, skin)
        self.draw_box_local(0.0, 1.42, 0.0, 0.32, 0.32, 0.27, skin)
        self.draw_elf_ears(skin)
        self.draw_elf_bow(leather)
        glPopMatrix()

    def draw_elf_ears(self, color: Tuple[float, float, float]) -> None:
        glBegin(GL_TRIANGLES)
        glColor3f(*color)
        glVertex3f(-0.18, 1.45, -0.02)
        glVertex3f(-0.45, 1.52, -0.03)
        glVertex3f(-0.18, 1.55, -0.02)
        glVertex3f(0.18, 1.45, -0.02)
        glVertex3f(0.45, 1.52, -0.03)
        glVertex3f(0.18, 1.55, -0.02)
        glEnd()

    def draw_elf_bow(self, color: Tuple[float, float, float]) -> None:
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(*color)
        glVertex3f(0.48, 1.20, -0.18)
        glVertex3f(0.60, 0.74, -0.18)
        glVertex3f(0.60, 0.74, -0.18)
        glVertex3f(0.48, 0.28, -0.18)
        glColor3f(0.82, 0.82, 0.72)
        glVertex3f(0.48, 1.20, -0.18)
        glVertex3f(0.48, 0.28, -0.18)
        glEnd()
        glLineWidth(1.0)

    def draw_demon(self, enemy: Enemy) -> None:
        shade = 1.0 + (0.40 if enemy.hit_flash > 0 else 0.0)
        bob = math.sin(enemy.phase * 4.0) * 0.035
        glPushMatrix()
        glTranslatef(enemy.x, bob, enemy.z)
        glRotatef(self.face_player_yaw(enemy.x, enemy.z), 0.0, 1.0, 0.0)
        red = tuple(clamp(c * shade, 0.0, 1.0) for c in enemy.kind.color)
        dark_red = tuple(clamp(c * shade, 0.0, 1.0) for c in (0.42, 0.05, 0.05))
        horn = tuple(clamp(c * shade, 0.0, 1.0) for c in (0.78, 0.70, 0.48))

        self.draw_box_local(0.0, 0.90, 0.0, 0.78, 1.18, 0.42, red)
        self.draw_box_local(-0.26, 0.26, 0.0, 0.20, 0.52, 0.20, dark_red)
        self.draw_box_local(0.26, 0.26, 0.0, 0.20, 0.52, 0.20, dark_red)
        self.draw_box_local(-0.55, 0.92, 0.0, 0.18, 0.88, 0.18, dark_red)
        self.draw_box_local(0.55, 0.92, 0.0, 0.18, 0.88, 0.18, dark_red)
        self.draw_box_local(0.0, 1.68, 0.0, 0.52, 0.42, 0.40, red)
        self.draw_demon_horns(horn)
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.75, 0.12)
        glVertex3f(-0.15, 1.72, -0.22)
        glVertex3f(-0.04, 1.72, -0.22)
        glVertex3f(-0.04, 1.78, -0.22)
        glVertex3f(-0.15, 1.78, -0.22)
        glVertex3f(0.04, 1.72, -0.22)
        glVertex3f(0.15, 1.72, -0.22)
        glVertex3f(0.15, 1.78, -0.22)
        glVertex3f(0.04, 1.78, -0.22)
        glEnd()
        glPopMatrix()

    def draw_demon_horns(self, color: Tuple[float, float, float]) -> None:
        glBegin(GL_TRIANGLES)
        glColor3f(*color)
        glVertex3f(-0.22, 1.88, -0.04)
        glVertex3f(-0.48, 2.25, -0.03)
        glVertex3f(-0.08, 1.88, -0.04)
        glVertex3f(0.22, 1.88, -0.04)
        glVertex3f(0.48, 2.25, -0.03)
        glVertex3f(0.08, 1.88, -0.04)
        glEnd()

    def draw_enemy_health(self, enemy: Enemy) -> None:
        hp_ratio = clamp(enemy.hp / (enemy.kind.hp + self.wave * (8 if enemy.kind_key == "elf" else 14)), 0.0, 1.0)
        if hp_ratio >= 0.995:
            return
        glDisable(GL_CULL_FACE)
        glPushMatrix()
        glTranslatef(enemy.x, enemy.kind.height + 0.45, enemy.z)
        glRotatef(-self.player.yaw, 0.0, 1.0, 0.0)
        w = 0.92
        glBegin(GL_QUADS)
        glColor4f(0.02, 0.02, 0.02, 0.86)
        glVertex3f(-w * 0.5, 0.0, 0.0)
        glVertex3f(w * 0.5, 0.0, 0.0)
        glVertex3f(w * 0.5, 0.08, 0.0)
        glVertex3f(-w * 0.5, 0.08, 0.0)
        glColor4f(0.95, 0.12, 0.10, 0.96)
        glVertex3f(-w * 0.5, 0.0, 0.01)
        glVertex3f(-w * 0.5 + w * hp_ratio, 0.0, 0.01)
        glVertex3f(-w * 0.5 + w * hp_ratio, 0.08, 0.01)
        glVertex3f(-w * 0.5, 0.08, 0.01)
        glEnd()
        glPopMatrix()
        glEnable(GL_CULL_FACE)

    def draw_projectiles(self) -> None:
        for projectile in self.projectiles:
            glPushMatrix()
            glTranslatef(projectile.x, projectile.y, projectile.z)
            if projectile.name == "arrow":
                glRotatef(self.player.yaw, 0.0, 1.0, 0.0)
                glBegin(GL_QUADS)
                glColor3f(*projectile.color)
                glVertex3f(-0.04, -0.04, -0.35)
                glVertex3f(0.04, -0.04, -0.35)
                glVertex3f(0.04, 0.04, 0.35)
                glVertex3f(-0.04, 0.04, 0.35)
                glEnd()
            else:
                self.draw_box_local(0.0, 0.0, 0.0, 0.34, 0.34, 0.34, projectile.color)
                glBegin(GL_LINES)
                glColor4f(1.0, 0.65, 0.12, 0.8)
                for i in range(8):
                    ang = math.tau * i / 8
                    glVertex3f(0.0, 0.0, 0.0)
                    glVertex3f(math.cos(ang) * 0.42, math.sin(ang) * 0.12, math.sin(ang) * 0.42)
                glEnd()
            glPopMatrix()

    def draw_tracers(self) -> None:
        if not self.tracers:
            return
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for origin, end, timer, hit in self.tracers:
            alpha = clamp(timer, 0.0, 0.75)
            if hit:
                glColor4f(1.0, 0.84, 0.32, alpha)
            else:
                glColor4f(0.74, 0.84, 1.0, alpha * 0.75)
            glVertex3f(*origin)
            glVertex3f(*end)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)

    def draw_box(self, x: float, y: float, z: float, sx: float, sy: float, sz: float, color: Tuple[float, float, float]) -> None:
        glPushMatrix()
        glTranslatef(x, y, z)
        self.draw_box_local(0.0, 0.0, 0.0, sx, sy, sz, color)
        glPopMatrix()

    def draw_box_local(self, x: float, y: float, z: float, sx: float, sy: float, sz: float, color: Tuple[float, float, float]) -> None:
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
            (0, 1, 2, 3, 0.74),
            (4, 5, 6, 7, 0.98),
            (3, 2, 6, 7, 1.10),
            (0, 1, 5, 4, 0.60),
            (1, 2, 6, 5, 0.84),
            (0, 3, 7, 4, 0.70),
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
        glDisable(GL_DEPTH_TEST)
        weapon = self.player.weapon()
        bob = math.sin(self.player.bob * 7.0) * 7.0
        kick = self.player.kick * 18.0
        x = WIDTH - 255 - kick * 0.6
        y = HEIGHT - 126 + bob + kick * 0.25

        if weapon.name == "Shotgun":
            self.draw_shotgun_overlay(x, y)
        elif weapon.name == "Revolver":
            self.draw_revolver_overlay(x + 24, y + 4)
        else:
            self.draw_knife_overlay(x + 76, y - 16)
        glEnable(GL_DEPTH_TEST)

    def draw_shotgun_overlay(self, x: float, y: float) -> None:
        glBegin(GL_QUADS)
        glColor4f(0.03, 0.025, 0.02, 0.50)
        glVertex2f(x - 18, y + 92)
        glVertex2f(x + 250, y + 72)
        glVertex2f(x + 244, y + 122)
        glVertex2f(x - 26, y + 140)

        glColor3f(0.16, 0.11, 0.07)
        glVertex2f(x, y + 92)
        glVertex2f(x + 112, y + 75)
        glVertex2f(x + 124, y + 106)
        glVertex2f(x + 12, y + 126)

        glColor3f(0.12, 0.12, 0.13)
        glVertex2f(x + 80, y + 58)
        glVertex2f(x + 248, y + 48)
        glVertex2f(x + 250, y + 61)
        glVertex2f(x + 84, y + 74)

        glColor3f(0.28, 0.19, 0.10)
        glVertex2f(x + 74, y + 78)
        glVertex2f(x + 168, y + 71)
        glVertex2f(x + 174, y + 88)
        glVertex2f(x + 78, y + 98)
        glEnd()
        if self.player.fire_timer > self.player.weapon().fire_delay * 0.62:
            self.draw_muzzle_flash(x + 250, y + 55, 60)

    def draw_revolver_overlay(self, x: float, y: float) -> None:
        glBegin(GL_QUADS)
        glColor4f(0.02, 0.02, 0.02, 0.48)
        glVertex2f(x - 18, y + 76)
        glVertex2f(x + 188, y + 64)
        glVertex2f(x + 200, y + 124)
        glVertex2f(x - 14, y + 134)

        glColor3f(0.37, 0.34, 0.29)
        glVertex2f(x + 54, y + 42)
        glVertex2f(x + 188, y + 38)
        glVertex2f(x + 188, y + 52)
        glVertex2f(x + 54, y + 64)

        glColor3f(0.24, 0.23, 0.22)
        glVertex2f(x + 26, y + 56)
        glVertex2f(x + 82, y + 52)
        glVertex2f(x + 96, y + 88)
        glVertex2f(x + 30, y + 96)

        glColor3f(0.21, 0.14, 0.09)
        glVertex2f(x + 44, y + 88)
        glVertex2f(x + 88, y + 85)
        glVertex2f(x + 102, y + 134)
        glVertex2f(x + 66, y + 142)
        glEnd()
        self.draw_disc(x + 62, y + 72, 25, (0.52, 0.48, 0.42))
        if self.player.fire_timer > self.player.weapon().fire_delay * 0.58:
            self.draw_muzzle_flash(x + 188, y + 45, 46)

    def draw_knife_overlay(self, x: float, y: float) -> None:
        slash = 1.0 if self.player.fire_timer > self.player.weapon().fire_delay * 0.45 else 0.0
        glBegin(GL_QUADS)
        glColor4f(0.03, 0.03, 0.03, 0.45)
        glVertex2f(x + 30, y + 98)
        glVertex2f(x + 126, y + 54 - slash * 22)
        glVertex2f(x + 156, y + 88 - slash * 20)
        glVertex2f(x + 62, y + 138)

        glColor3f(0.16, 0.10, 0.06)
        glVertex2f(x + 34, y + 98)
        glVertex2f(x + 76, y + 78)
        glVertex2f(x + 92, y + 110)
        glVertex2f(x + 52, y + 132)

        glColor3f(0.66, 0.73, 0.78)
        glVertex2f(x + 74, y + 76)
        glVertex2f(x + 184, y + 18 - slash * 34)
        glVertex2f(x + 156, y + 88 - slash * 20)
        glVertex2f(x + 90, y + 110)
        glEnd()
        glBegin(GL_TRIANGLES)
        glColor3f(0.88, 0.95, 1.0)
        glVertex2f(x + 184, y + 18 - slash * 34)
        glVertex2f(x + 206, y + 48 - slash * 29)
        glVertex2f(x + 156, y + 88 - slash * 20)
        glEnd()

    def draw_muzzle_flash(self, x: float, y: float, length: float) -> None:
        glBegin(GL_TRIANGLES)
        glColor4f(1.0, 0.90, 0.45, 0.86)
        glVertex2f(x, y)
        glColor4f(1.0, 0.35, 0.05, 0.0)
        glVertex2f(x + length, y - 18)
        glVertex2f(x + length, y + 18)
        glEnd()

    def draw_disc(self, x: float, y: float, radius: float, color: Tuple[float, float, float]) -> None:
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(*color)
        glVertex2f(x, y)
        for i in range(22):
            ang = math.tau * i / 21
            glVertex2f(x + math.cos(ang) * radius, y + math.sin(ang) * radius)
        glEnd()

    def draw_crosshair(self) -> None:
        glDisable(GL_DEPTH_TEST)
        cx = WIDTH * 0.5
        cy = HEIGHT * 0.5
        gap = 7 + self.player.kick * 5.0
        length = 13
        color = GOLD if self.player.hit_flash > 0 else (0.74, 0.92, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(color[0], color[1], color[2], 0.96)
        glVertex2f(cx - gap - length, cy)
        glVertex2f(cx - gap, cy)
        glVertex2f(cx + gap, cy)
        glVertex2f(cx + gap + length, cy)
        glVertex2f(cx, cy - gap - length)
        glVertex2f(cx, cy - gap)
        glVertex2f(cx, cy + gap)
        glVertex2f(cx, cy + gap + length)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)

    def draw_hud(self) -> None:
        glDisable(GL_DEPTH_TEST)

        if self.player.hurt_flash > 0:
            alpha = clamp(self.player.hurt_flash * 0.33, 0.0, 0.33)
            glBegin(GL_QUADS)
            glColor4f(0.86, 0.03, 0.02, alpha)
            glVertex2f(0, 0)
            glVertex2f(WIDTH, 0)
            glVertex2f(WIDTH, HEIGHT)
            glVertex2f(0, HEIGHT)
            glEnd()

        if self.wave_flash > 0:
            alpha = clamp(self.wave_flash * 0.18, 0.0, 0.18)
            glBegin(GL_QUADS)
            glColor4f(1.0, 0.82, 0.30, alpha)
            glVertex2f(0, 0)
            glVertex2f(WIDTH, 0)
            glVertex2f(WIDTH, HEIGHT)
            glVertex2f(0, HEIGHT)
            glEnd()

        hp_ratio = clamp(self.player.hp / MAX_HP, 0.0, 1.0)
        glBegin(GL_QUADS)
        glColor4f(0.02, 0.025, 0.03, 0.72)
        glVertex2f(16, HEIGHT - 82)
        glVertex2f(342, HEIGHT - 82)
        glVertex2f(342, HEIGHT - 18)
        glVertex2f(16, HEIGHT - 18)

        glColor4f(0.10, 0.11, 0.12, 0.95)
        glVertex2f(28, HEIGHT - 38)
        glVertex2f(244, HEIGHT - 38)
        glVertex2f(244, HEIGHT - 24)
        glVertex2f(28, HEIGHT - 24)

        hp_col = GREEN if hp_ratio > 0.35 else RED
        glColor4f(hp_col[0], hp_col[1], hp_col[2], 0.95)
        glVertex2f(28, HEIGHT - 38)
        glVertex2f(28 + 216 * hp_ratio, HEIGHT - 38)
        glVertex2f(28 + 216 * hp_ratio, HEIGHT - 24)
        glVertex2f(28, HEIGHT - 24)
        glEnd()

        weapon = self.player.weapon()
        idx = self.player.weapon_index
        ammo = "UNLIMITED" if weapon.melee else f"{self.player.clip[idx]} / {self.player.reserve[idx]}"
        alive = sum(1 for enemy in self.enemies if enemy.alive)
        self.draw_text(26, HEIGHT - 74, f"HP {self.player.hp:03d}", self.font, WHITE)
        self.draw_text(26, HEIGHT - 104, f"SCORE {self.player.score:05d}", self.small, HUD)
        self.draw_text(WIDTH - 265, HEIGHT - 78, f"{weapon.name.upper()}  {ammo}", self.font, GOLD if not weapon.melee else WHITE)
        self.draw_text(WIDTH - 265, HEIGHT - 104, f"WAVE {self.wave}   ENEMIES {alive}   KILLS {self.player.kills}", self.small, HUD)
        self.draw_text(18, 18, "WASD move  Shift sprint  Mouse aim  1 Shotgun  2 Revolver  3 Knife  R reload  Tab mouse", self.small, HUD)

        if self.player.reload_timer > 0 and not weapon.melee:
            ratio = 1.0 - self.player.reload_timer / max(0.001, weapon.reload_time)
            glBegin(GL_QUADS)
            glColor4f(0.07, 0.08, 0.09, 0.88)
            glVertex2f(WIDTH * 0.5 - 118, HEIGHT - 52)
            glVertex2f(WIDTH * 0.5 + 118, HEIGHT - 52)
            glVertex2f(WIDTH * 0.5 + 118, HEIGHT - 40)
            glVertex2f(WIDTH * 0.5 - 118, HEIGHT - 40)
            glColor4f(0.94, 0.72, 0.28, 0.94)
            glVertex2f(WIDTH * 0.5 - 118, HEIGHT - 52)
            glVertex2f(WIDTH * 0.5 - 118 + 236 * ratio, HEIGHT - 52)
            glVertex2f(WIDTH * 0.5 - 118 + 236 * ratio, HEIGHT - 40)
            glVertex2f(WIDTH * 0.5 - 118, HEIGHT - 40)
            glEnd()
            self.draw_text(WIDTH * 0.5 - 58, HEIGHT - 76, "RELOADING", self.small, GOLD)

        if self.message_timer > 0 and self.message:
            self.draw_text(WIDTH * 0.5, 52, self.message, self.font, GOLD, center=True)

        glEnable(GL_DEPTH_TEST)

    def draw_game_over(self) -> None:
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_QUADS)
        glColor4f(0.0, 0.0, 0.0, 0.72)
        glVertex2f(0, 0)
        glVertex2f(WIDTH, 0)
        glVertex2f(WIDTH, HEIGHT)
        glVertex2f(0, HEIGHT)
        glEnd()
        self.draw_text(WIDTH * 0.5, HEIGHT * 0.5 - 58, "ARENA CLAIMED YOU", self.big, (1.0, 0.32, 0.25), center=True)
        self.draw_text(WIDTH * 0.5, HEIGHT * 0.5 + 2, f"Score {self.player.score}   Kills {self.player.kills}   Wave {self.wave}", self.font, WHITE, center=True)
        self.draw_text(WIDTH * 0.5, HEIGHT * 0.5 + 38, "Press R or F5 to restart", self.font, GOLD, center=True)
        glEnable(GL_DEPTH_TEST)

    def draw_text(
        self,
        x: float,
        y: float,
        text: str,
        font: pygame.font.Font,
        color: Tuple[float, float, float],
        center: bool = False,
    ) -> None:
        color255 = tuple(int(clamp(c, 0.0, 1.0) * 255) for c in color)
        key = (text, font.get_height(), color255)
        if key not in self.text_cache:
            surface = font.render(text, True, color255)
            data = pygame.image.tostring(surface, "RGBA", True)
            self.text_cache[key] = (surface.get_width(), surface.get_height(), data)
        w, h, data = self.text_cache[key]
        if center:
            x -= w * 0.5
        glWindowPos2d(int(x), int(HEIGHT - y - h))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)


def main() -> None:
    try:
        ArenaFPS().run()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
