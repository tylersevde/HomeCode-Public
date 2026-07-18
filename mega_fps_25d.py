import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pygame
except ImportError:
    print("This game requires pygame. Install it with: pip install pygame")
    raise


# ---------------------------
# Config
# ---------------------------
WIDTH, HEIGHT = 720, 360
HALF_W, HALF_H = WIDTH // 2, HEIGHT // 2
FPS = 60
FOV = math.radians(70)
HALF_FOV = FOV / 2
RAY_COUNT = 320
RAY_STEP = WIDTH / RAY_COUNT
MAX_DEPTH = 32
SCREEN_DIST = HALF_W / math.tan(HALF_FOV)
MOUSE_SENS = 0.0032
MOVE_SPEED = 3.4
SPRINT_MULT = 1.5
TURN_SPEED = 2.5
PLAYER_RADIUS = 0.22
MAP_SIZE = 61

SKY = (20, 24, 38)
FLOOR = (28, 24, 22)
WHITE = (240, 240, 240)
BLACK = (0, 0, 0)
RED = (220, 70, 70)
GREEN = (90, 220, 120)
CYAN = (90, 220, 220)
YELLOW = (235, 210, 90)
ORANGE = (235, 145, 60)
PURPLE = (180, 90, 220)

WALL_COLORS = {
    1: (95, 110, 145),
    2: (130, 90, 75),
    3: (70, 130, 100),
    4: (130, 70, 115),
}


@dataclass
class WeaponSpec:
    name: str
    damage: float
    pellet_count: int
    spread: float
    max_range: float
    fire_delay: float
    clip_size: int
    reload_time: float
    ammo_type: str
    burst_count: int = 1
    projectile_speed: float = 0.0
    splash: float = 0.0
    kick: float = 0.0
    color: Tuple[int, int, int] = WHITE


WEAPONS: List[WeaponSpec] = [
    WeaponSpec("Pistol", 26, 1, math.radians(1.0), 18, 0.30, 14, 1.1, "light", kick=0.015, color=(230, 230, 210)),
    WeaponSpec("Scattergun", 11, 8, math.radians(8.0), 11, 0.8, 6, 1.65, "shells", kick=0.05, color=(255, 205, 120)),
    WeaponSpec("Pulse Rifle", 15, 1, math.radians(2.2), 20, 0.09, 34, 1.55, "cells", kick=0.009, color=(150, 235, 255)),
    WeaponSpec("Nova Launcher", 65, 1, 0.0, 22, 0.95, 4, 2.1, "rockets", projectile_speed=11.0, splash=2.6, kick=0.07, color=(255, 120, 120)),
]


@dataclass
class EnemyType:
    name: str
    hp: int
    speed: float
    radius: float
    damage: int
    melee_range: float
    projectile_speed: float
    attack_range: float
    cooldown: float
    color: Tuple[int, int, int]
    ranged: bool = False
    stationary: bool = False
    boss: bool = False
    score: int = 0


ENEMY_TYPES: Dict[str, EnemyType] = {
    "raider": EnemyType("Raider", 50, 2.2, 0.20, 9, 0.9, 0, 0, 0.85, (220, 80, 80), score=50),
    "hound": EnemyType("Hound", 34, 3.3, 0.17, 7, 0.7, 0, 0, 0.55, (255, 170, 60), score=45),
    "soldier": EnemyType("Soldier", 78, 1.85, 0.22, 8, 0.0, 7.5, 11.5, 1.2, (90, 190, 220), ranged=True, score=80),
    "brute": EnemyType("Brute", 145, 1.35, 0.30, 18, 1.1, 0, 0, 1.25, (180, 70, 170), score=120),
    "turret": EnemyType("Turret", 105, 0.0, 0.24, 7, 0.0, 10.5, 15.0, 0.75, (200, 210, 80), ranged=True, stationary=True, score=95),
    "specter": EnemyType("Specter", 62, 2.75, 0.19, 11, 0.85, 0, 0, 0.65, (120, 255, 170), score=100),
    "overmind": EnemyType("Overmind", 950, 1.05, 0.46, 14, 1.2, 9.8, 18.0, 0.33, (255, 70, 140), ranged=True, boss=True, score=2000),
}


@dataclass
class Player:
    x: float
    y: float
    angle: float
    health: int = 1000
    armor: int = 1000
    score: int = 0
    weapon_index: int = 0
    owned_weapons: set = field(default_factory=lambda: {0})
    reserve: Dict[str, int] = field(default_factory=lambda: {"light": 120, "shells": 24, "cells": 70, "rockets": 8})
    clip: Dict[int, int] = field(default_factory=lambda: {0: WEAPONS[0].clip_size, 1: 0, 2: 0, 3: 0})
    fire_timer: float = 0.0
    reload_timer: float = 0.0
    reloading: bool = False
    hurt_flash: float = 0.0
    bob_phase: float = 0.0
    weapon_sway: float = 0.0

    def active_weapon(self) -> WeaponSpec:
        return WEAPONS[self.weapon_index]


@dataclass
class Enemy:
    x: float
    y: float
    kind: EnemyType
    hp: float
    cooldown: float = 0.0
    phase: float = 0.0
    alive: bool = True


@dataclass
class Projectile:
    x: float
    y: float
    dx: float
    dy: float
    speed: float
    damage: int
    owner: str
    radius: float
    ttl: float
    color: Tuple[int, int, int]
    splash: float = 0.0


@dataclass
class Pickup:
    x: float
    y: float
    kind: str
    amount: int
    color: Tuple[int, int, int]
    weapon_unlock: Optional[int] = None
    alive: bool = True


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Mega Rift 2.5D")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.big_font = pygame.font.SysFont("consolas", 34, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 14)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.seed = 23
        self.reset_world()

    def reset_world(self) -> None:
        rng = random.Random(self.seed)
        self.world, start, room_centers, boss_room = generate_level(rng)
        self.player = Player(start[0] + 0.5, start[1] + 0.5, rng.random() * math.tau)
        self.enemies: List[Enemy] = []
        self.projectiles: List[Projectile] = []
        self.pickups: List[Pickup] = []
        self.show_map = False
        self.win = False
        self.dead = False
        self.time_alive = 0.0
        self.spawn_entities(rng, room_centers, boss_room)
        self.depth_buffer = [MAX_DEPTH for _ in range(RAY_COUNT)]

    def spawn_entities(self, rng: random.Random, room_centers: Sequence[Tuple[int, int]], boss_room: Tuple[int, int]) -> None:
        open_tiles = [(x, y) for y in range(1, MAP_SIZE - 1) for x in range(1, MAP_SIZE - 1) if self.world[y][x] == 0]
        used: set = set()

        def add_enemy(kind_name: str, x: float, y: float) -> None:
            self.enemies.append(Enemy(x + 0.5, y + 0.5, ENEMY_TYPES[kind_name], ENEMY_TYPES[kind_name].hp, phase=rng.random() * math.tau))
            used.add((int(x), int(y)))

        room_samples = room_centers[2:]
        for cx, cy in room_samples:
            for _ in range(rng.randint(2, 5)):
                px, py = find_nearby_open(self.world, rng, cx, cy, 4)
                roll = rng.random()
                if roll < 0.28:
                    add_enemy("raider", px, py)
                elif roll < 0.46:
                    add_enemy("hound", px, py)
                elif roll < 0.67:
                    add_enemy("soldier", px, py)
                elif roll < 0.83:
                    add_enemy("specter", px, py)
                elif roll < 0.94:
                    add_enemy("brute", px, py)
                else:
                    add_enemy("turret", px, py)

        for _ in range(18):
            px, py = rng.choice(open_tiles)
            if (px, py) in used or math.dist((px, py), (self.player.x, self.player.y)) < 6:
                continue
            add_enemy(rng.choice(["raider", "hound", "soldier", "specter"]), px, py)

        bx, by = boss_room
        self.enemies.append(Enemy(bx + 0.5, by + 0.5, ENEMY_TYPES["overmind"], ENEMY_TYPES["overmind"].hp))

        def add_pickup(x: int, y: int, kind: str, amount: int, color: Tuple[int, int, int], weapon_unlock: Optional[int] = None) -> None:
            self.pickups.append(Pickup(x + 0.5, y + 0.5, kind, amount, color, weapon_unlock))
            used.add((x, y))

        unlock_rooms = room_centers[1:4]
        unlocks = [(1, "shells", 18, ORANGE), (2, "cells", 60, CYAN), (3, "rockets", 10, RED)]
        for (cx, cy), (weapon_idx, ammo_kind, amount, color) in zip(unlock_rooms, unlocks):
            px, py = find_nearby_open(self.world, rng, cx, cy, 3)
            add_pickup(px, py, f"unlock_{WEAPONS[weapon_idx].name}", amount, color, weapon_idx)
            # extra ammo nearby
            for _ in range(2):
                ax, ay = find_nearby_open(self.world, rng, cx, cy, 4)
                add_pickup(ax, ay, ammo_kind, amount // 2, color)

        for _ in range(50):
            px, py = rng.choice(open_tiles)
            if (px, py) in used:
                continue
            roll = rng.random()
            if roll < 0.35:
                add_pickup(px, py, "light", rng.randint(10, 26), (230, 230, 170))
            elif roll < 0.52:
                add_pickup(px, py, "shells", rng.randint(4, 10), ORANGE)
            elif roll < 0.68:
                add_pickup(px, py, "cells", rng.randint(15, 30), CYAN)
            elif roll < 0.79:
                add_pickup(px, py, "rockets", rng.randint(1, 4), RED)
            elif roll < 0.93:
                add_pickup(px, py, "medkit", rng.randint(15, 35), GREEN)
            else:
                add_pickup(px, py, "armor", rng.randint(8, 22), (120, 160, 255))

    def run(self) -> None:
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.033)
            self.handle_events()
            if not self.dead and not self.win:
                self.update(dt)
            self.draw()
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
                if event.key == pygame.K_m:
                    self.show_map = not self.show_map
                if event.key == pygame.K_r and (self.dead or self.win):
                    self.reset_world()
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    idx = event.key - pygame.K_1
                    if idx in self.player.owned_weapons:
                        self.player.weapon_index = idx
                if event.key == pygame.K_TAB:
                    grab = not pygame.event.get_grab()
                    pygame.event.set_grab(grab)
                    pygame.mouse.set_visible(not grab)
                if event.key == pygame.K_r and not self.dead and not self.win:
                    self.try_reload()
                if event.key == pygame.K_SPACE and not self.dead and not self.win:
                    self.try_fire()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.dead and not self.win:
                self.try_fire()

    def try_reload(self) -> None:
        weapon = self.player.active_weapon()
        if self.player.reload_timer > 0:
            return
        clip_now = self.player.clip[self.player.weapon_index]
        reserve = self.player.reserve[weapon.ammo_type]
        if clip_now >= weapon.clip_size or reserve <= 0:
            return
        self.player.reload_timer = weapon.reload_time
        self.player.reloading = True

    def try_fire(self) -> None:
        if self.player.fire_timer > 0 or self.player.reload_timer > 0:
            return
        weapon = self.player.active_weapon()
        if self.player.clip[self.player.weapon_index] <= 0:
            self.try_reload()
            return
        self.player.fire_timer = weapon.fire_delay
        self.player.clip[self.player.weapon_index] -= 1
        self.player.weapon_sway += weapon.kick

        if weapon.projectile_speed > 0:
            ang = self.player.angle
            self.projectiles.append(
                Projectile(
                    self.player.x + math.cos(ang) * 0.5,
                    self.player.y + math.sin(ang) * 0.5,
                    math.cos(ang),
                    math.sin(ang),
                    weapon.projectile_speed,
                    int(weapon.damage),
                    "player",
                    0.14,
                    2.2,
                    weapon.color,
                    weapon.splash,
                )
            )
        else:
            for _ in range(weapon.pellet_count):
                ang = self.player.angle + random.uniform(-weapon.spread / 2, weapon.spread / 2)
                self.cast_hitscan(ang, weapon.damage, weapon.max_range)

    def cast_hitscan(self, angle: float, damage: float, max_range: float) -> None:
        step = 0.05
        dist = 0.0
        while dist < max_range:
            dist += step
            px = self.player.x + math.cos(angle) * dist
            py = self.player.y + math.sin(angle) * dist
            if is_solid(self.world, px, py):
                return
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                if (enemy.x - px) ** 2 + (enemy.y - py) ** 2 <= (enemy.kind.radius + 0.03) ** 2:
                    enemy.hp -= damage
                    enemy.phase += 1.7
                    if enemy.hp <= 0:
                        enemy.alive = False
                        self.player.score += enemy.kind.score
                    return

    def update(self, dt: float) -> None:
        self.time_alive += dt
        self.player.fire_timer = max(0.0, self.player.fire_timer - dt)
        if self.player.reload_timer > 0:
            self.player.reload_timer = max(0.0, self.player.reload_timer - dt)
            if self.player.reloading and self.player.reload_timer == 0:
                weapon = self.player.active_weapon()
                clip_idx = self.player.weapon_index
                needed = weapon.clip_size - self.player.clip[clip_idx]
                if needed > 0:
                    reserve = self.player.reserve[weapon.ammo_type]
                    if reserve > 0:
                        load = min(needed, reserve)
                        self.player.clip[clip_idx] += load
                        self.player.reserve[weapon.ammo_type] -= load
                self.player.reloading = False
        self.player.hurt_flash = max(0.0, self.player.hurt_flash - dt * 2.4)
        self.player.weapon_sway = max(0.0, self.player.weapon_sway - dt * 0.16)

        self.update_player(dt)
        self.update_enemies(dt)
        self.update_projectiles(dt)
        self.update_pickups()

        boss_alive = any(enemy.alive and enemy.kind.boss for enemy in self.enemies)
        if not boss_alive:
            self.win = True
        if self.player.health <= 0:
            self.dead = True

    def damage_player(self, amount: int) -> None:
        if self.dead or self.win:
            return
        if self.player.armor > 0:
            absorbed = min(self.player.armor, int(amount * 0.45))
            self.player.armor -= absorbed
            amount -= absorbed
        self.player.health -= amount
        self.player.hurt_flash = min(1.0, self.player.hurt_flash + 0.5)

    def update_player(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        if pygame.event.get_grab():
            mx, _ = pygame.mouse.get_rel()
            self.player.angle = (self.player.angle + mx * MOUSE_SENS) % math.tau
        else:
            pygame.mouse.get_rel()

        turn = 0.0
        if keys[pygame.K_LEFT]:
            turn -= 1
        if keys[pygame.K_RIGHT]:
            turn += 1
        self.player.angle = (self.player.angle + turn * TURN_SPEED * dt) % math.tau

        move_speed = MOVE_SPEED * (SPRINT_MULT if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
        dx = dy = 0.0
        ca, sa = math.cos(self.player.angle), math.sin(self.player.angle)
        if keys[pygame.K_w]:
            dx += ca * move_speed * dt
            dy += sa * move_speed * dt
        if keys[pygame.K_s]:
            dx -= ca * move_speed * dt
            dy -= sa * move_speed * dt
        if keys[pygame.K_a]:
            dx += math.cos(self.player.angle - math.pi / 2) * move_speed * dt
            dy += math.sin(self.player.angle - math.pi / 2) * move_speed * dt
        if keys[pygame.K_d]:
            dx += math.cos(self.player.angle + math.pi / 2) * move_speed * dt
            dy += math.sin(self.player.angle + math.pi / 2) * move_speed * dt

        moving = abs(dx) + abs(dy) > 0.0001
        if moving:
            self.player.bob_phase += dt * 10.0 * (1.25 if move_speed > MOVE_SPEED else 1.0)

        self.move_with_collisions(self.player, dx, dy, PLAYER_RADIUS)

        if pygame.mouse.get_pressed()[0]:
            self.try_fire()

    def move_with_collisions(self, obj, dx: float, dy: float, radius: float) -> None:
        new_x = obj.x + dx
        if not circle_hits_wall(self.world, new_x, obj.y, radius):
            obj.x = new_x
        new_y = obj.y + dy
        if not circle_hits_wall(self.world, obj.x, new_y, radius):
            obj.y = new_y

    def update_enemies(self, dt: float) -> None:
        boss = next((enemy for enemy in self.enemies if enemy.alive and enemy.kind.boss), None)
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            enemy.cooldown = max(0.0, enemy.cooldown - dt)
            enemy.phase += dt * 3.0
            dx = self.player.x - enemy.x
            dy = self.player.y - enemy.y
            dist = math.hypot(dx, dy)
            los = dist < 18 and self.line_of_sight(enemy.x, enemy.y, self.player.x, self.player.y)
            kind = enemy.kind

            move_x = move_y = 0.0
            if kind.stationary:
                pass
            elif kind.boss:
                if dist > 5.5:
                    move_x = dx / max(0.001, dist) * kind.speed * dt
                    move_y = dy / max(0.001, dist) * kind.speed * dt
                elif dist < 3.3:
                    move_x = -dx / max(0.001, dist) * kind.speed * dt
                    move_y = -dy / max(0.001, dist) * kind.speed * dt
                strafe_ang = math.atan2(dy, dx) + math.pi / 2
                move_x += math.cos(strafe_ang) * math.sin(enemy.phase) * kind.speed * 0.5 * dt
                move_y += math.sin(strafe_ang) * math.sin(enemy.phase) * kind.speed * 0.5 * dt
            elif kind.ranged:
                desired = 6.5 if kind.name == "Soldier" else 7.5
                if dist > desired and los:
                    move_x = dx / max(0.001, dist) * kind.speed * dt
                    move_y = dy / max(0.001, dist) * kind.speed * dt
                elif dist < desired - 1.7:
                    move_x = -dx / max(0.001, dist) * kind.speed * dt
                    move_y = -dy / max(0.001, dist) * kind.speed * dt
                strafe_ang = math.atan2(dy, dx) + (math.pi / 2 if math.sin(enemy.phase) > 0 else -math.pi / 2)
                move_x += math.cos(strafe_ang) * kind.speed * 0.35 * dt
                move_y += math.sin(strafe_ang) * kind.speed * 0.35 * dt
            elif kind.name == "Specter":
                angle = math.atan2(dy, dx)
                weave = math.sin(enemy.phase * 3) * 0.9
                move_x = math.cos(angle + weave) * kind.speed * dt
                move_y = math.sin(angle + weave) * kind.speed * dt
            else:
                if dist > 0.1:
                    move_x = dx / dist * kind.speed * dt
                    move_y = dy / dist * kind.speed * dt

            if abs(move_x) > 0 or abs(move_y) > 0:
                self.move_with_collisions(enemy, move_x, move_y, kind.radius)

            if kind.ranged and los and dist <= kind.attack_range and enemy.cooldown <= 0:
                enemy.cooldown = kind.cooldown
                angle = math.atan2(dy, dx)
                if kind.boss:
                    for off in (-0.18, 0.0, 0.18):
                        self.projectiles.append(
                            Projectile(enemy.x, enemy.y, math.cos(angle + off), math.sin(angle + off), kind.projectile_speed,
                                       kind.damage, "enemy", 0.12, 2.4, kind.color)
                        )
                else:
                    self.projectiles.append(
                        Projectile(enemy.x, enemy.y, math.cos(angle), math.sin(angle), kind.projectile_speed,
                                   kind.damage, "enemy", 0.10, 2.2, kind.color)
                    )
            elif dist <= kind.melee_range and enemy.cooldown <= 0:
                enemy.cooldown = kind.cooldown
                self.damage_player(kind.damage)

        # wake the boss up visually by nudging if player gets near the room
        if boss and math.dist((boss.x, boss.y), (self.player.x, self.player.y)) < 14:
            boss.phase += dt * 2

    def update_projectiles(self, dt: float) -> None:
        for proj in self.projectiles[:]:
            proj.ttl -= dt
            if proj.ttl <= 0:
                if proj.splash > 0:
                    self.explode(proj.x, proj.y, proj.damage, proj.splash, proj.owner)
                self.projectiles.remove(proj)
                continue

            nx = proj.x + proj.dx * proj.speed * dt
            ny = proj.y + proj.dy * proj.speed * dt
            if is_solid(self.world, nx, ny):
                if proj.splash > 0:
                    self.explode(nx, ny, proj.damage, proj.splash, proj.owner)
                self.projectiles.remove(proj)
                continue

            proj.x, proj.y = nx, ny
            if proj.owner == "enemy":
                if math.dist((proj.x, proj.y), (self.player.x, self.player.y)) <= PLAYER_RADIUS + proj.radius:
                    self.damage_player(proj.damage)
                    self.projectiles.remove(proj)
            else:
                hit_enemy = None
                for enemy in self.enemies:
                    if enemy.alive and math.dist((proj.x, proj.y), (enemy.x, enemy.y)) <= enemy.kind.radius + proj.radius:
                        hit_enemy = enemy
                        break
                if hit_enemy:
                    if proj.splash > 0:
                        self.explode(proj.x, proj.y, proj.damage, proj.splash, proj.owner)
                    else:
                        hit_enemy.hp -= proj.damage
                        if hit_enemy.hp <= 0:
                            hit_enemy.alive = False
                            self.player.score += hit_enemy.kind.score
                    self.projectiles.remove(proj)

    def explode(self, x: float, y: float, damage: int, radius: float, owner: str) -> None:
        if owner == "player":
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                d = math.dist((x, y), (enemy.x, enemy.y))
                if d <= radius + enemy.kind.radius:
                    scaled = max(10, int(damage * (1 - d / max(0.1, radius))))
                    enemy.hp -= scaled
                    if enemy.hp <= 0:
                        enemy.alive = False
                        self.player.score += enemy.kind.score
        else:
            d = math.dist((x, y), (self.player.x, self.player.y))
            if d <= radius + PLAYER_RADIUS:
                scaled = max(8, int(damage * (1 - d / max(0.1, radius))))
                self.damage_player(scaled)

    def update_pickups(self) -> None:
        for pickup in self.pickups:
            if not pickup.alive:
                continue
            if math.dist((pickup.x, pickup.y), (self.player.x, self.player.y)) > 0.7:
                continue
            pickup.alive = False
            if pickup.kind == "medkit":
                self.player.health = min(100, self.player.health + pickup.amount)
            elif pickup.kind == "armor":
                self.player.armor = min(100, self.player.armor + pickup.amount)
            elif pickup.kind.startswith("unlock_") and pickup.weapon_unlock is not None:
                idx = pickup.weapon_unlock
                self.player.owned_weapons.add(idx)
                self.player.weapon_index = idx
                self.player.reserve[WEAPONS[idx].ammo_type] += pickup.amount
                if self.player.clip[idx] == 0:
                    load = min(WEAPONS[idx].clip_size, self.player.reserve[WEAPONS[idx].ammo_type])
                    self.player.clip[idx] += load
                    self.player.reserve[WEAPONS[idx].ammo_type] -= load
            else:
                self.player.reserve[pickup.kind] += pickup.amount

    def line_of_sight(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        dx = x2 - x1
        dy = y2 - y1
        steps = int(max(abs(dx), abs(dy)) * 18)
        for i in range(1, steps):
            t = i / max(1, steps)
            if is_solid(self.world, x1 + dx * t, y1 + dy * t):
                return False
        return True

    def draw(self) -> None:
        self.screen.fill(BLACK)
        self.cast_walls()
        self.draw_sprites()
        self.draw_weapon()
        self.draw_hud()
        if self.show_map:
            self.draw_minimap()
        if self.player.hurt_flash > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 40, 40, int(80 * self.player.hurt_flash)))
            self.screen.blit(overlay, (0, 0))
        if self.dead:
            self.draw_center_message("YOU DIED", "Press R to restart the level")
        elif self.win:
            self.draw_center_message("BOSS DEFEATED", f"Score {self.player.score}  |  Press R to play again")

    def cast_walls(self) -> None:
        pygame.draw.rect(self.screen, SKY, (0, 0, WIDTH, HALF_H))
        pygame.draw.rect(self.screen, FLOOR, (0, HALF_H, WIDTH, HALF_H))
        start_angle = self.player.angle - HALF_FOV
        ox, oy = self.player.x, self.player.y

        for ray in range(RAY_COUNT):
            ray_angle = start_angle + ray / RAY_COUNT * FOV
            sin_a = math.sin(ray_angle)
            cos_a = math.cos(ray_angle)

            map_x, map_y = int(ox), int(oy)
            delta_dist_x = abs(1 / cos_a) if abs(cos_a) > 1e-6 else 1e6
            delta_dist_y = abs(1 / sin_a) if abs(sin_a) > 1e-6 else 1e6

            if cos_a < 0:
                step_x = -1
                side_dist_x = (ox - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - ox) * delta_dist_x
            if sin_a < 0:
                step_y = -1
                side_dist_y = (oy - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - oy) * delta_dist_y

            side = 0
            wall_val = 1
            while True:
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1
                wall_val = self.world[map_y][map_x]
                if wall_val > 0:
                    break

            if side == 0:
                depth = (map_x - ox + (1 - step_x) / 2) / max(1e-6, cos_a)
            else:
                depth = (map_y - oy + (1 - step_y) / 2) / max(1e-6, sin_a)
            depth *= math.cos(self.player.angle - ray_angle)
            depth = max(0.0001, depth)
            self.depth_buffer[ray] = depth
            proj_height = min(int(SCREEN_DIST / depth), HEIGHT * 2)
            color = WALL_COLORS.get(wall_val, (140, 140, 140))
            shade = max(0.18, 1.0 - depth / 22)
            if side == 1:
                shade *= 0.78
            shaded = tuple(min(255, max(0, int(c * shade))) for c in color)
            x = int(ray * RAY_STEP)
            y = HALF_H - proj_height // 2
            pygame.draw.rect(self.screen, shaded, (x, y, int(RAY_STEP + 1), proj_height))

    def draw_sprites(self) -> None:
        sprites = []
        for enemy in self.enemies:
            if enemy.alive:
                sprites.append((math.dist((self.player.x, self.player.y), (enemy.x, enemy.y)), enemy, enemy.kind.color, enemy.kind.radius, True))
        for pickup in self.pickups:
            if pickup.alive:
                sprites.append((math.dist((self.player.x, self.player.y), (pickup.x, pickup.y)), pickup, pickup.color, 0.14, False))
        for proj in self.projectiles:
            sprites.append((math.dist((self.player.x, self.player.y), (proj.x, proj.y)), proj, proj.color, proj.radius, False))
        sprites.sort(key=lambda item: item[0], reverse=True)

        for dist, obj, color, radius, is_enemy in sprites:
            dx = obj.x - self.player.x
            dy = obj.y - self.player.y
            theta = math.atan2(dy, dx)
            gamma = (theta - self.player.angle + math.pi) % (math.tau) - math.pi
            if abs(gamma) > HALF_FOV + 0.35:
                continue
            corrected = dist * math.cos(gamma)
            if corrected <= 0.12:
                continue
            proj_height = SCREEN_DIST / corrected * (1.6 if is_enemy else 0.8)
            sprite_h = max(8, int(proj_height * (1.5 if is_enemy else 0.65)))
            sprite_w = max(6, int(proj_height * (1.0 if is_enemy else 0.65)))
            screen_x = int(HALF_W + math.tan(gamma) * SCREEN_DIST - sprite_w / 2)
            screen_y = HALF_H - sprite_h // 2
            buffer_x = min(RAY_COUNT - 1, max(0, int((screen_x + sprite_w / 2) / RAY_STEP)))
            if corrected > self.depth_buffer[buffer_x] + radius:
                continue

            rect = pygame.Rect(screen_x, screen_y, sprite_w, sprite_h)
            if is_enemy:
                pygame.draw.rect(self.screen, color, rect, border_radius=max(4, sprite_w // 8))
                eye_y = screen_y + sprite_h * 0.28
                eye_r = max(2, sprite_w // 10)
                pygame.draw.circle(self.screen, WHITE, (screen_x + sprite_w // 3, int(eye_y)), eye_r)
                pygame.draw.circle(self.screen, WHITE, (screen_x + sprite_w * 2 // 3, int(eye_y)), eye_r)
                hp_ratio = max(0.0, obj.hp / obj.kind.hp)
                pygame.draw.rect(self.screen, (35, 20, 20), (screen_x, screen_y - 8, sprite_w, 5))
                pygame.draw.rect(self.screen, (110, 240, 110), (screen_x, screen_y - 8, int(sprite_w * hp_ratio), 5))
            elif isinstance(obj, Pickup):
                pygame.draw.ellipse(self.screen, color, rect)
                pygame.draw.ellipse(self.screen, WHITE, rect, 2)
            else:
                pygame.draw.ellipse(self.screen, color, rect)

    def draw_weapon(self) -> None:
        weapon = self.player.active_weapon()
        sway_x = math.sin(self.player.bob_phase) * 8 + self.player.weapon_sway * -260
        sway_y = abs(math.cos(self.player.bob_phase * 0.7)) * 5 + self.player.weapon_sway * 120
        cx = HALF_W + sway_x
        base_y = HEIGHT - 84 + sway_y

        if weapon.name == "Pistol":
            pygame.draw.rect(self.screen, (150, 150, 150), (cx - 36, base_y - 16, 72, 42), border_radius=8)
            pygame.draw.rect(self.screen, (70, 70, 78), (cx - 10, base_y - 48, 22, 40), border_radius=4)
        elif weapon.name == "Scattergun":
            pygame.draw.rect(self.screen, (105, 65, 30), (cx - 80, base_y - 6, 160, 26), border_radius=8)
            pygame.draw.rect(self.screen, (82, 82, 82), (cx - 18, base_y - 28, 140, 18), border_radius=5)
        elif weapon.name == "Pulse Rifle":
            pygame.draw.rect(self.screen, (70, 120, 135), (cx - 92, base_y - 16, 184, 32), border_radius=10)
            pygame.draw.rect(self.screen, (120, 220, 255), (cx - 18, base_y - 40, 110, 14), border_radius=6)
        elif weapon.name == "Nova Launcher":
            pygame.draw.rect(self.screen, (100, 55, 55), (cx - 88, base_y - 8, 176, 30), border_radius=10)
            pygame.draw.rect(self.screen, (180, 180, 180), (cx - 18, base_y - 34, 150, 20), border_radius=7)
            pygame.draw.circle(self.screen, (255, 110, 110), (int(cx + 102), int(base_y - 24)), 12)

        cross_color = (255, 255, 255)
        pygame.draw.line(self.screen, cross_color, (HALF_W - 8, HALF_H), (HALF_W + 8, HALF_H), 2)
        pygame.draw.line(self.screen, cross_color, (HALF_W, HALF_H - 8), (HALF_W, HALF_H + 8), 2)

    def draw_hud(self) -> None:
        weapon = self.player.active_weapon()
        ammo_clip = self.player.clip[self.player.weapon_index]
        ammo_reserve = self.player.reserve[weapon.ammo_type]

        pygame.draw.rect(self.screen, (18, 18, 20), (16, HEIGHT - 112, 310, 94), border_radius=12)
        pygame.draw.rect(self.screen, (26, 26, 30), (WIDTH - 280, 16, 264, 90), border_radius=12)

        self.draw_bar(28, HEIGHT - 92, 180, 16, self.player.health / 100, (190, 40, 40), "HP")
        self.draw_bar(28, HEIGHT - 62, 180, 16, self.player.armor / 100, (60, 120, 230), "AR")

        weapon_text = self.font.render(f"{weapon.name}", True, WHITE)
        ammo_text = self.big_font.render(f"{ammo_clip:02d}/{ammo_reserve:03d}", True, weapon.color)
        score_text = self.font.render(f"Score {self.player.score}", True, YELLOW)
        enemy_count = sum(1 for e in self.enemies if e.alive and not e.kind.boss)
        boss = next((e for e in self.enemies if e.alive and e.kind.boss), None)
        boss_text = self.font.render("Boss active" if boss else "Boss down", True, PURPLE if boss else GREEN)
        remain_text = self.font.render(f"Targets left {enemy_count + (1 if boss else 0)}", True, WHITE)
        hint_text = self.small_font.render("WASD move  Mouse look  Click/Space fire  R reload  1-4 weapons  M map", True, (190, 190, 200))

        self.screen.blit(weapon_text, (224, HEIGHT - 101))
        self.screen.blit(score_text, (224, HEIGHT - 76))
        self.screen.blit(ammo_text, (WIDTH - 260, 28))
        self.screen.blit(remain_text, (WIDTH - 260, 62))
        self.screen.blit(boss_text, (WIDTH - 260, 82))
        self.screen.blit(hint_text, (16, 16))

        if boss:
            ratio = max(0.0, boss.hp / boss.kind.hp)
            pygame.draw.rect(self.screen, (36, 14, 26), (WIDTH * 0.18, 16, WIDTH * 0.64, 16), border_radius=8)
            pygame.draw.rect(self.screen, (255, 70, 140), (WIDTH * 0.18, 16, WIDTH * 0.64 * ratio, 16), border_radius=8)
            label = self.small_font.render("OVERMIND CORE", True, WHITE)
            self.screen.blit(label, (WIDTH * 0.48 - label.get_width() / 2, 17))

        if self.player.reload_timer > 0:
            pct = self.player.reload_timer / max(0.001, weapon.reload_time)
            pygame.draw.rect(self.screen, (20, 20, 20), (WIDTH - 280, 112, 260, 10), border_radius=5)
            pygame.draw.rect(self.screen, weapon.color, (WIDTH - 280, 112, 260 * (1 - pct), 10), border_radius=5)

    def draw_bar(self, x: int, y: int, w: int, h: int, ratio: float, color: Tuple[int, int, int], label: str) -> None:
        pygame.draw.rect(self.screen, (50, 50, 55), (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, color, (x, y, int(w * max(0, min(1, ratio))), h), border_radius=6)
        txt = self.small_font.render(label, True, WHITE)
        self.screen.blit(txt, (x - 22, y - 1))

    def draw_minimap(self) -> None:
        scale = 4
        surf = pygame.Surface((MAP_SIZE * scale, MAP_SIZE * scale), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 180))
        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                cell = self.world[y][x]
                if cell > 0:
                    color = WALL_COLORS.get(cell, (100, 100, 100))
                    pygame.draw.rect(surf, color, (x * scale, y * scale, scale, scale))
        for pickup in self.pickups:
            if pickup.alive:
                pygame.draw.circle(surf, pickup.color, (int(pickup.x * scale), int(pickup.y * scale)), 2)
        for enemy in self.enemies:
            if enemy.alive:
                pygame.draw.circle(surf, enemy.kind.color, (int(enemy.x * scale), int(enemy.y * scale)), 2 if not enemy.kind.boss else 4)
        pygame.draw.circle(surf, WHITE, (int(self.player.x * scale), int(self.player.y * scale)), 3)
        line_end = (
            int((self.player.x + math.cos(self.player.angle) * 2) * scale),
            int((self.player.y + math.sin(self.player.angle) * 2) * scale),
        )
        pygame.draw.line(surf, WHITE, (int(self.player.x * scale), int(self.player.y * scale)), line_end, 1)
        self.screen.blit(surf, (WIDTH - surf.get_width() - 18, HEIGHT - surf.get_height() - 18))

    def draw_center_message(self, title: str, subtitle: str) -> None:
        panel = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 120))
        self.screen.blit(panel, (0, 0))
        t1 = self.big_font.render(title, True, WHITE)
        t2 = self.font.render(subtitle, True, WHITE)
        self.screen.blit(t1, (HALF_W - t1.get_width() // 2, HALF_H - 38))
        self.screen.blit(t2, (HALF_W - t2.get_width() // 2, HALF_H + 6))


# ---------------------------
# World generation helpers
# ---------------------------
def generate_level(rng: random.Random) -> Tuple[List[List[int]], Tuple[int, int], List[Tuple[int, int]], Tuple[int, int]]:
    world = [[1 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
    rooms: List[Tuple[int, int, int, int]] = []
    centers: List[Tuple[int, int]] = []

    for _ in range(60):
        w = rng.randrange(5, 11, 2)
        h = rng.randrange(5, 11, 2)
        x = rng.randrange(1, MAP_SIZE - w - 1, 2)
        y = rng.randrange(1, MAP_SIZE - h - 1, 2)
        candidate = (x, y, w, h)
        if any(rects_intersect(candidate, room, margin=2) for room in rooms):
            continue
        carve_room(world, x, y, w, h, rng.randint(0, 3))
        rooms.append(candidate)
        centers.append((x + w // 2, y + h // 2))
        if len(rooms) >= 15:
            break

    centers.sort(key=lambda c: c[0] + c[1] * 0.15)
    for i in range(1, len(centers)):
        x1, y1 = centers[i - 1]
        x2, y2 = centers[i]
        carve_corridor(world, x1, y1, x2, y2, rng)
    # extra loops
    for _ in range(6):
        a, b = rng.sample(centers, 2)
        carve_corridor(world, a[0], a[1], b[0], b[1], rng)

    # Boss chamber in far quadrant
    boss_cx, boss_cy = MAP_SIZE - 9, MAP_SIZE - 9
    carve_room(world, boss_cx - 4, boss_cy - 4, 9, 9, 3)
    boss_room = (boss_cx, boss_cy)
    if centers:
        carve_corridor(world, centers[-1][0], centers[-1][1], boss_cx, boss_cy, rng)
    else:
        centers.append((3, 3))
        carve_room(world, 1, 1, 7, 7, 1)
        carve_corridor(world, 4, 4, boss_cx, boss_cy, rng)

    # Additional side pockets / halls to make the map feel larger
    for _ in range(40):
        x = rng.randint(2, MAP_SIZE - 3)
        y = rng.randint(2, MAP_SIZE - 3)
        if world[y][x] == 0:
            for _ in range(rng.randint(2, 6)):
                dir_x, dir_y = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                nx, ny = x + dir_x, y + dir_y
                if 1 <= nx < MAP_SIZE - 1 and 1 <= ny < MAP_SIZE - 1:
                    world[ny][nx] = 0
                    x, y = nx, ny

    start = centers[0]
    return world, start, centers, boss_room


def carve_room(world: List[List[int]], x: int, y: int, w: int, h: int, wall_variant: int) -> None:
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            world[yy][xx] = 0
    # decorate edges around room with alternate wall types
    wall_type = 1 + (wall_variant % 4)
    for xx in range(x - 1, x + w + 1):
        if 0 <= x - 1 < MAP_SIZE and 0 <= xx < MAP_SIZE:
            if 0 <= y - 1 < MAP_SIZE:
                world[y - 1][xx] = wall_type
            if 0 <= y + h < MAP_SIZE:
                world[y + h][xx] = wall_type
    for yy in range(y - 1, y + h + 1):
        if 0 <= yy < MAP_SIZE:
            if 0 <= x - 1 < MAP_SIZE:
                world[yy][x - 1] = wall_type
            if 0 <= x + w < MAP_SIZE:
                world[yy][x + w] = wall_type


def carve_corridor(world: List[List[int]], x1: int, y1: int, x2: int, y2: int, rng: random.Random) -> None:
    if rng.random() < 0.5:
        carve_hall_h(world, x1, x2, y1, 2 + int(rng.random() < 0.4), rng)
        carve_hall_v(world, y1, y2, x2, 2 + int(rng.random() < 0.4), rng)
    else:
        carve_hall_v(world, y1, y2, x1, 2 + int(rng.random() < 0.4), rng)
        carve_hall_h(world, x1, x2, y2, 2 + int(rng.random() < 0.4), rng)


def carve_hall_h(world: List[List[int]], x1: int, x2: int, y: int, thickness: int, rng: random.Random) -> None:
    if x1 > x2:
        x1, x2 = x2, x1
    for x in range(x1, x2 + 1):
        for o in range(-thickness // 2, thickness // 2 + 1):
            yy = y + o
            if 1 <= x < MAP_SIZE - 1 and 1 <= yy < MAP_SIZE - 1:
                world[yy][x] = 0
        if rng.random() < 0.07:
            for ext in range(1, rng.randint(2, 4)):
                yy = y + rng.choice([-1, 1]) * ext
                if 1 <= yy < MAP_SIZE - 1:
                    world[yy][x] = 0


def carve_hall_v(world: List[List[int]], y1: int, y2: int, x: int, thickness: int, rng: random.Random) -> None:
    if y1 > y2:
        y1, y2 = y2, y1
    for y in range(y1, y2 + 1):
        for o in range(-thickness // 2, thickness // 2 + 1):
            xx = x + o
            if 1 <= y < MAP_SIZE - 1 and 1 <= xx < MAP_SIZE - 1:
                world[y][xx] = 0
        if rng.random() < 0.07:
            for ext in range(1, rng.randint(2, 4)):
                xx = x + rng.choice([-1, 1]) * ext
                if 1 <= xx < MAP_SIZE - 1:
                    world[y][xx] = 0


def rects_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], margin: int = 0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + margin < bx or bx + bw + margin < ax or ay + ah + margin < by or by + bh + margin < ay)


def find_nearby_open(world: List[List[int]], rng: random.Random, cx: int, cy: int, radius: int) -> Tuple[int, int]:
    options = []
    for y in range(max(1, cy - radius), min(MAP_SIZE - 1, cy + radius + 1)):
        for x in range(max(1, cx - radius), min(MAP_SIZE - 1, cx + radius + 1)):
            if world[y][x] == 0:
                options.append((x, y))
    return rng.choice(options) if options else (cx, cy)


def is_solid(world: List[List[int]], x: float, y: float) -> bool:
    xi = int(x)
    yi = int(y)
    if not (0 <= xi < MAP_SIZE and 0 <= yi < MAP_SIZE):
        return True
    return world[yi][xi] > 0


def circle_hits_wall(world: List[List[int]], x: float, y: float, radius: float) -> bool:
    for ox in (-radius, radius):
        for oy in (-radius, radius):
            if is_solid(world, x + ox, y + oy):
                return True
    return False


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
