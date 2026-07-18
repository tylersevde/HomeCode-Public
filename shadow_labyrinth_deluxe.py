import math
import random
import sys
from collections import deque

import pygame
from pygame.math import Vector2

# ------------------------------------------------------------
# Shadow Labyrinth Deluxe
# Single-file top-down dungeon crawler made with pygame.
# Explore a massive labyrinth, collect rune keys, unlock gates,
# gather loot, use healing flasks, and defeat the final boss.
# ------------------------------------------------------------

pygame.init()
pygame.font.init()

SCREEN_W = 1280
SCREEN_H = 720
FPS = 60
TILE = 48
MAZE_W = 61
MAZE_H = 61
WORLD_W = MAZE_W * TILE
WORLD_H = MAZE_H * TILE

BG = (10, 10, 14)
FLOOR = (28, 28, 36)
FLOOR_ALT = (33, 33, 42)
WALL = (70, 74, 90)
WALL_EDGE = (104, 108, 130)
PLAYER_BODY = (105, 180, 255)
PLAYER_TRIM = (220, 235, 255)
SHIELD = (90, 200, 240)
SWORD = (240, 240, 255)
ENEMY = (55, 20, 75)
ENEMY_EYE = (230, 80, 120)
CULTIST = (70, 24, 90)
STALKER = (30, 30, 50)
BRUTE = (85, 26, 44)
BOSS = (130, 30, 40)
BOSS_CORE = (255, 130, 150)
PROJECTILE = (170, 80, 230)
LOOT = (255, 215, 90)
FLASK = (80, 220, 140)
KEY = (120, 255, 255)
DOOR = (85, 180, 210)
GLOW = (80, 40, 120)
WHITE = (245, 245, 245)
GREEN = (80, 220, 120)
YELLOW = (250, 210, 80)
RED = (220, 80, 80)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Shadow Labyrinth Deluxe")
clock = pygame.time.Clock()

FONT = pygame.font.SysFont("consolas", 22)
BIG_FONT = pygame.font.SysFont("consolas", 48, bold=True)
SMALL_FONT = pygame.font.SysFont("consolas", 16)
TINY_FONT = pygame.font.SysFont("consolas", 13)


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def angle_to_vec(angle_rad):
    return Vector2(math.cos(angle_rad), math.sin(angle_rad))


def normalize_or_zero(vec):
    if vec.length_squared() == 0:
        return Vector2()
    return vec.normalize()


def circle_rect_collision(cx, cy, radius, rect):
    closest_x = clamp(cx, rect.left, rect.right)
    closest_y = clamp(cy, rect.top, rect.bottom)
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy < radius * radius


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0

    def update(self, target):
        self.x = clamp(target.x - SCREEN_W // 2, 0, WORLD_W - SCREEN_W)
        self.y = clamp(target.y - SCREEN_H // 2, 0, WORLD_H - SCREEN_H)

    def apply_point(self, pos):
        return int(pos[0] - self.x), int(pos[1] - self.y)


class FloatingText:
    def __init__(self, x, y, text, color, life=0.8):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.life = life
        self.max_life = life

    def update(self, dt):
        self.life -= dt
        self.y -= 30 * dt

    def draw(self, surf, camera):
        if self.life <= 0:
            return
        alpha = int(255 * (self.life / self.max_life))
        txt = SMALL_FONT.render(self.text, True, self.color)
        txt.set_alpha(alpha)
        pos = camera.apply_point((self.x, self.y))
        surf.blit(txt, (pos[0] - txt.get_width() // 2, pos[1]))


class Pickup:
    def __init__(self, x, y, kind, amount=1):
        self.pos = Vector2(x, y)
        self.kind = kind
        self.amount = amount
        self.radius = 12 if kind == "loot" else 14
        self.alive = True
        self.bob = random.uniform(0, math.tau)

    def update(self, dt):
        self.bob += dt * 3.2

    def collect(self, player, texts):
        if self.kind == "loot":
            player.loot += self.amount
            texts.append(FloatingText(self.pos.x, self.pos.y - 18, f"+{self.amount} gold", LOOT))
        elif self.kind == "flask":
            player.flasks += self.amount
            texts.append(FloatingText(self.pos.x, self.pos.y - 18, f"+{self.amount} flask", FLASK))
        elif self.kind == "key":
            player.keys += self.amount
            texts.append(FloatingText(self.pos.x, self.pos.y - 18, f"+{self.amount} key", KEY))
        self.alive = False

    def draw(self, surf, camera):
        px, py = camera.apply_point(self.pos)
        py += int(math.sin(self.bob) * 4)

        if self.kind == "loot":
            points = [(px, py - 10), (px + 8, py), (px, py + 10), (px - 8, py)]
            pygame.draw.polygon(surf, LOOT, points)
            pygame.draw.polygon(surf, WHITE, points, 2)
        elif self.kind == "flask":
            pygame.draw.rect(surf, FLASK, (px - 7, py - 9, 14, 18), border_radius=4)
            pygame.draw.rect(surf, WHITE, (px - 7, py - 9, 14, 18), 2, border_radius=4)
            pygame.draw.rect(surf, (170, 255, 220), (px - 4, py - 4, 8, 8), border_radius=2)
            pygame.draw.rect(surf, (190, 160, 90), (px - 4, py - 13, 8, 5), border_radius=2)
        elif self.kind == "key":
            pygame.draw.circle(surf, KEY, (px - 4, py), 5, 2)
            pygame.draw.rect(surf, KEY, (px, py - 2, 10, 4), border_radius=2)
            pygame.draw.rect(surf, KEY, (px + 8, py - 2, 2, 6))
            pygame.draw.rect(surf, KEY, (px + 4, py - 2, 2, 8))


class Projectile:
    def __init__(self, x, y, velocity, damage, owner, radius=8, color=PROJECTILE):
        self.pos = Vector2(x, y)
        self.vel = Vector2(velocity)
        self.damage = damage
        self.owner = owner
        self.radius = radius
        self.color = color
        self.alive = True

    def update(self, dt, level, player, texts):
        self.pos += self.vel * dt

        if not (0 <= self.pos.x < WORLD_W and 0 <= self.pos.y < WORLD_H):
            self.alive = False
            return

        if level.circle_hits_blocker(self.pos.x, self.pos.y, self.radius):
            self.alive = False
            return

        if self.owner == "enemy" and (self.pos - player.pos).length() < self.radius + player.radius:
            blocked = player.take_hit(self.damage, self.pos)
            self.alive = False
            if blocked:
                texts.append(FloatingText(player.pos.x, player.pos.y - 26, "BLOCK", SHIELD))
            else:
                texts.append(FloatingText(player.pos.x, player.pos.y - 26, f"-{self.damage}", RED))

    def draw(self, surf, camera):
        pos = camera.apply_point(self.pos)
        pygame.draw.circle(surf, self.color, pos, self.radius)
        pygame.draw.circle(surf, WHITE, pos, max(2, self.radius // 3), 1)


class Door:
    def __init__(self, cell, label):
        self.cell = cell
        self.label = label
        self.locked = True

    @property
    def rect(self):
        return pygame.Rect(self.cell[0] * TILE, self.cell[1] * TILE, TILE, TILE)

    def draw(self, surf, camera):
        if not self.locked:
            return
        rect = self.rect.move(-camera.x, -camera.y)
        pygame.draw.rect(surf, (30, 50, 58), rect)
        pygame.draw.rect(surf, DOOR, rect, 3)
        inner = rect.inflate(-14, -14)
        pygame.draw.rect(surf, (65, 120, 140), inner, border_radius=4)
        rune = TINY_FONT.render(str(self.label), True, WHITE)
        surf.blit(rune, (rect.centerx - rune.get_width() // 2, rect.centery - rune.get_height() // 2))


class Player:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.radius = 16
        self.speed = 190
        self.hp = 120
        self.max_hp = 120
        self.facing = Vector2(1, 0)
        self.attack_timer = 0
        self.attack_cd = 0
        self.attack_id = 0
        self.shielding = False
        self.invuln = 0
        self.keys = 0
        self.flasks = 1
        self.loot = 0
        self.flask_cd = 0

    def update(self, dt, level):
        keys = pygame.key.get_pressed()
        move = Vector2(
            (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0),
            (1 if keys[pygame.K_s] else 0) - (1 if keys[pygame.K_w] else 0),
        )
        move = normalize_or_zero(move)

        mouse_pos = Vector2(pygame.mouse.get_pos())
        world_mouse = mouse_pos + Vector2(level.camera.x, level.camera.y)
        if (world_mouse - self.pos).length_squared() > 0:
            self.facing = (world_mouse - self.pos).normalize()
        elif move.length_squared() > 0:
            self.facing = move.normalize()

        self.shielding = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or pygame.mouse.get_pressed()[2]
        cur_speed = self.speed * (0.55 if self.shielding else 1.0)
        delta = move * cur_speed * dt
        self.move(delta.x, 0, level)
        self.move(0, delta.y, level)

        self.attack_timer = max(0, self.attack_timer - dt)
        self.attack_cd = max(0, self.attack_cd - dt)
        self.invuln = max(0, self.invuln - dt)
        self.flask_cd = max(0, self.flask_cd - dt)

    def move(self, dx, dy, level):
        if dx == 0 and dy == 0:
            return
        new_x = self.pos.x + dx
        new_y = self.pos.y + dy
        if not level.circle_hits_blocker(new_x, self.pos.y, self.radius):
            self.pos.x = new_x
        if not level.circle_hits_blocker(self.pos.x, new_y, self.radius):
            self.pos.y = new_y

    def try_attack(self):
        if self.attack_cd <= 0:
            self.attack_timer = 0.16
            self.attack_cd = 0.34
            self.attack_id += 1
            return True
        return False

    def try_use_flask(self, texts):
        if self.flasks <= 0 or self.flask_cd > 0 or self.hp >= self.max_hp:
            return False
        heal = min(35, self.max_hp - self.hp)
        self.flasks -= 1
        self.hp += heal
        self.flask_cd = 0.45
        texts.append(FloatingText(self.pos.x, self.pos.y - 26, f"+{heal}", GREEN))
        return True

    def sword_hits(self, target_pos, target_radius):
        if self.attack_timer <= 0:
            return False
        to_target = Vector2(target_pos) - self.pos
        dist = to_target.length()
        if dist > 74 + target_radius:
            return False
        if dist == 0:
            return True
        dir_to_target = to_target.normalize()
        return self.facing.dot(dir_to_target) > 0.35

    def take_hit(self, amount, source_pos):
        if self.invuln > 0:
            return False
        source_dir = normalize_or_zero(self.pos - Vector2(source_pos))
        blocked = self.shielding and self.facing.dot(source_dir) > 0.25
        if blocked:
            self.invuln = 0.12
            return True
        self.hp -= amount
        self.invuln = 0.28
        return False

    def draw(self, surf, camera):
        pos = camera.apply_point(self.pos)

        if self.shielding:
            shield_center = self.pos + self.facing * 18
            shield_pos = camera.apply_point(shield_center)
            pygame.draw.circle(surf, (40, 110, 140), shield_pos, 18)
            pygame.draw.circle(surf, SHIELD, shield_pos, 18, 3)

        body_color = PLAYER_BODY if self.invuln <= 0 else (170, 220, 255)
        pygame.draw.circle(surf, body_color, pos, self.radius)
        eye = self.pos + self.facing * 7
        eye_pos = camera.apply_point(eye)
        pygame.draw.circle(surf, PLAYER_TRIM, eye_pos, 3)

        if self.attack_timer > 0:
            sword_tip = self.pos + self.facing * 42
            left = self.pos + self.facing.rotate(35) * 24
            right = self.pos + self.facing.rotate(-35) * 24
            pygame.draw.polygon(
                surf,
                SWORD,
                [camera.apply_point(left), camera.apply_point(sword_tip), camera.apply_point(right)],
            )
            pygame.draw.arc(
                surf,
                (180, 220, 255),
                pygame.Rect(pos[0] - 56, pos[1] - 56, 112, 112),
                math.atan2(self.facing.y, self.facing.x) - 0.55,
                math.atan2(self.facing.y, self.facing.x) + 0.55,
                3,
            )


class Enemy:
    def __init__(self, x, y, kind):
        self.pos = Vector2(x, y)
        self.kind = kind
        self.alive = True
        self.last_hit_by_attack = -1
        self.wander_target = self.pos.copy()
        self.wander_timer = random.uniform(0.5, 2.0)
        self.attack_cd = 0
        self.ranged_cd = random.uniform(0.6, 1.8)

        if kind == "stalker":
            self.radius = 14
            self.speed = random.uniform(125, 145)
            self.hp = self.max_hp = 26
            self.damage = 8
            self.detect = 380
            self.color = STALKER
            self.eye = (140, 220, 255)
            self.name = "Stalker"
            self.loot_drop = 3
        elif kind == "cultist":
            self.radius = 15
            self.speed = random.uniform(80, 95)
            self.hp = self.max_hp = 30
            self.damage = 8
            self.detect = 430
            self.color = CULTIST
            self.eye = (230, 150, 255)
            self.name = "Cultist"
            self.loot_drop = 4
        elif kind == "brute":
            self.radius = 20
            self.speed = random.uniform(65, 80)
            self.hp = self.max_hp = 70
            self.damage = 16
            self.detect = 290
            self.color = BRUTE
            self.eye = (255, 120, 120)
            self.name = "Brute"
            self.loot_drop = 7
        else:
            self.radius = 16
            self.speed = random.uniform(85, 110)
            self.hp = self.max_hp = 40
            self.damage = 10
            self.detect = 320
            self.color = ENEMY
            self.eye = ENEMY_EYE
            self.name = "Shade"
            self.loot_drop = 5

    def update(self, dt, level, player, projectiles, texts):
        if not self.alive:
            return

        self.attack_cd = max(0, self.attack_cd - dt)
        self.ranged_cd = max(0, self.ranged_cd - dt)
        to_player = player.pos - self.pos
        dist = to_player.length()
        direction = Vector2()

        if self.kind == "cultist":
            if dist < self.detect:
                if dist > 240:
                    direction = normalize_or_zero(to_player)
                elif dist < 145:
                    direction = normalize_or_zero(-to_player)
                else:
                    direction = normalize_or_zero(to_player.rotate(90 if random.random() < 0.5 else -90))
                if dist < 360 and self.ranged_cd <= 0:
                    self.fire(player, projectiles)
                    self.ranged_cd = random.uniform(1.15, 1.75)
            else:
                direction = self._wander(dt, level)
        else:
            if dist < self.detect:
                direction = normalize_or_zero(to_player)
            else:
                direction = self._wander(dt, level)

        move = direction * self.speed * dt
        self.move(move.x, 0, level)
        self.move(0, move.y, level)

        melee_range = self.radius + player.radius + (8 if self.kind == "brute" else 4)
        if dist < melee_range and self.attack_cd <= 0 and self.kind != "cultist":
            blocked = player.take_hit(self.damage, self.pos)
            self.attack_cd = 1.15 if self.kind == "brute" else 0.85 if self.kind == "shade" else 0.6
            if blocked:
                texts.append(FloatingText(player.pos.x, player.pos.y - 26, "BLOCK", SHIELD))
            else:
                texts.append(FloatingText(player.pos.x, player.pos.y - 26, f"-{self.damage}", RED))

    def _wander(self, dt, level):
        self.wander_timer -= dt
        if self.wander_timer <= 0 or (self.wander_target - self.pos).length() < 8:
            self.wander_timer = random.uniform(1.2, 2.8)
            for _ in range(10):
                ang = random.uniform(0, math.tau)
                step = angle_to_vec(ang) * random.uniform(36, 130)
                candidate = self.pos + step
                if level.is_walkable_px(candidate.x, candidate.y):
                    self.wander_target = candidate
                    break
        return normalize_or_zero(self.wander_target - self.pos)

    def move(self, dx, dy, level):
        if dx == 0 and dy == 0:
            return
        new_x = self.pos.x + dx
        new_y = self.pos.y + dy
        if not level.circle_hits_blocker(new_x, self.pos.y, self.radius):
            self.pos.x = new_x
        if not level.circle_hits_blocker(self.pos.x, new_y, self.radius):
            self.pos.y = new_y

    def fire(self, player, projectiles):
        direction = normalize_or_zero(player.pos - self.pos)
        if direction.length_squared() == 0:
            return
        velocity = direction * 225
        projectiles.append(Projectile(self.pos.x, self.pos.y, velocity, self.damage, "enemy", radius=7, color=(195, 120, 255)))

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False

    def draw(self, surf, camera):
        if not self.alive:
            return
        pos = camera.apply_point(self.pos)
        glow_radius = self.radius + 8 if self.kind != "brute" else self.radius + 12
        pygame.draw.circle(surf, GLOW, pos, glow_radius)
        pygame.draw.circle(surf, self.color, pos, self.radius)
        if self.kind == "brute":
            pygame.draw.rect(surf, self.eye, (pos[0] - 8, pos[1] - 4, 16, 4), border_radius=2)
        else:
            pygame.draw.circle(surf, self.eye, (pos[0] - 5, pos[1] - 2), 2)
            pygame.draw.circle(surf, self.eye, (pos[0] + 5, pos[1] - 2), 2)

        hp_ratio = self.hp / self.max_hp
        bar_rect = pygame.Rect(pos[0] - 18, pos[1] - self.radius - 12, 36, 5)
        pygame.draw.rect(surf, (40, 20, 20), bar_rect)
        pygame.draw.rect(surf, RED, (bar_rect.x, bar_rect.y, int(bar_rect.w * hp_ratio), bar_rect.h))


class Boss:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.radius = 34
        self.speed = 85
        self.hp = 340
        self.max_hp = 340
        self.damage = 18
        self.attack_cd = 0
        self.volley_cd = 2.3
        self.charge_cd = 4.8
        self.charge_dir = Vector2()
        self.charge_time = 0
        self.alive = True
        self.last_hit_by_attack = -1

    def update(self, dt, level, player, projectiles, texts):
        if not self.alive:
            return
        self.attack_cd = max(0, self.attack_cd - dt)
        self.volley_cd = max(0, self.volley_cd - dt)
        self.charge_cd = max(0, self.charge_cd - dt)

        to_player = player.pos - self.pos
        dist = to_player.length()
        direction = normalize_or_zero(to_player)

        if self.charge_time > 0:
            self.charge_time -= dt
            move = self.charge_dir * 290 * dt
        else:
            move = direction * self.speed * dt
            if self.volley_cd <= 0:
                self.fire_volley(projectiles)
                self.volley_cd = 1.75 if self.hp < self.max_hp * 0.5 else 2.4
            if self.charge_cd <= 0 and dist > 120:
                self.charge_dir = direction
                self.charge_time = 0.6
                self.charge_cd = 4.0

        self.move(move.x, 0, level)
        self.move(0, move.y, level)

        if dist < self.radius + player.radius + 8 and self.attack_cd <= 0:
            blocked = player.take_hit(self.damage, self.pos)
            self.attack_cd = 1.0
            if blocked:
                texts.append(FloatingText(player.pos.x, player.pos.y - 32, "BLOCK", SHIELD))
            else:
                texts.append(FloatingText(player.pos.x, player.pos.y - 32, f"-{self.damage}", RED))

    def fire_volley(self, projectiles):
        count = 8 if self.hp > self.max_hp * 0.5 else 12
        speed = 200 if self.hp > self.max_hp * 0.5 else 245
        start_angle = random.uniform(0, math.tau)
        for i in range(count):
            ang = start_angle + (math.tau / count) * i
            vel = angle_to_vec(ang) * speed
            projectiles.append(Projectile(self.pos.x, self.pos.y, vel, 12, "enemy", radius=9, color=PROJECTILE))

    def move(self, dx, dy, level):
        if dx == 0 and dy == 0:
            return
        new_x = self.pos.x + dx
        new_y = self.pos.y + dy
        if not level.circle_hits_blocker(new_x, self.pos.y, self.radius):
            self.pos.x = new_x
        if not level.circle_hits_blocker(self.pos.x, new_y, self.radius):
            self.pos.y = new_y

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False

    def draw(self, surf, camera):
        if not self.alive:
            return
        pos = camera.apply_point(self.pos)
        pygame.draw.circle(surf, (70, 20, 30), pos, 52)
        pygame.draw.circle(surf, BOSS, pos, self.radius)
        pygame.draw.circle(surf, BOSS_CORE, pos, 12)
        horn1 = (pos[0] - 18, pos[1] - 38)
        horn2 = (pos[0] + 18, pos[1] - 38)
        pygame.draw.polygon(surf, BOSS, [(horn1[0], horn1[1]), (horn1[0] - 10, horn1[1] - 18), (horn1[0] + 2, horn1[1] - 8)])
        pygame.draw.polygon(surf, BOSS, [(horn2[0], horn2[1]), (horn2[0] + 10, horn2[1] - 18), (horn2[0] - 2, horn2[1] - 8)])


class Level:
    def __init__(self):
        self.grid = [[1 for _ in range(MAZE_W)] for _ in range(MAZE_H)]
        self.camera = Camera()
        self.explored = [[False for _ in range(MAZE_W)] for _ in range(MAZE_H)]
        self.start_cell = (1, 1)
        self.boss_cell = (MAZE_W - 6, MAZE_H - 6)
        self.doors = []
        self.generate_maze()

    def cell_center(self, cell):
        return Vector2(cell[0] * TILE + TILE // 2, cell[1] * TILE + TILE // 2)

    def generate_maze(self):
        stack = [self.start_cell]
        self.grid[self.start_cell[1]][self.start_cell[0]] = 0
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < MAZE_W - 1 and 1 <= ny < MAZE_H - 1 and self.grid[ny][nx] == 1:
                    neighbors.append((nx, ny, dx, dy))
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                self.grid[y + dy // 2][x + dx // 2] = 0
                self.grid[ny][nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        for _ in range(220):
            x = random.randrange(2, MAZE_W - 2)
            y = random.randrange(2, MAZE_H - 2)
            if self.grid[y][x] == 1:
                openings = 0
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if self.grid[y + dy][x + dx] == 0:
                        openings += 1
                if openings >= 2 and random.random() < 0.26:
                    self.grid[y][x] = 0

        distances = self.compute_distances(self.start_cell)
        candidates = [
            cell for cell in distances
            if 5 <= cell[0] < MAZE_W - 5 and 5 <= cell[1] < MAZE_H - 5
        ]
        self.boss_cell = max(candidates, key=lambda c: distances[c]) if candidates else max(distances, key=distances.get)
        self.build_boss_wing()

    def build_boss_wing(self):
        bx, by = self.boss_cell

        for y in range(by - 4, by + 5):
            for x in range(bx - 4, bx + 5):
                if 0 <= x < MAZE_W and 0 <= y < MAZE_H:
                    self.grid[y][x] = 1

        for y in range(by - 3, by + 4):
            for x in range(bx - 3, bx + 4):
                if 0 <= x < MAZE_W and 0 <= y < MAZE_H:
                    self.grid[y][x] = 0

        corridor_start = max(2, bx - 12)
        for x in range(corridor_start, bx - 3):
            self.grid[by][x] = 0
            if by - 1 >= 0:
                self.grid[by - 1][x] = 1
            if by + 1 < MAZE_H:
                self.grid[by + 1][x] = 1

        for x in range(corridor_start, bx - 3):
            if by - 2 >= 0:
                self.grid[by - 2][x] = 1
            if by + 2 < MAZE_H:
                self.grid[by + 2][x] = 1

        connector = self.find_nearest_existing_floor((corridor_start - 1, by), max_x=corridor_start - 1)
        if connector:
            cx, cy = connector
            for x in range(min(cx, corridor_start), max(cx, corridor_start) + 1):
                self.grid[cy][x] = 0
            for y in range(min(cy, by), max(cy, by) + 1):
                self.grid[y][corridor_start] = 0

        door_cells = [(bx - 9, by), (bx - 7, by), (bx - 5, by)]
        self.doors = [Door(cell, idx + 1) for idx, cell in enumerate(door_cells)]

    def find_nearest_existing_floor(self, start, max_x=None):
        q = deque([start])
        seen = {start}
        while q:
            x, y = q.popleft()
            if 0 <= x < MAZE_W and 0 <= y < MAZE_H:
                if self.grid[y][x] == 0 and (max_x is None or x <= max_x):
                    return (x, y)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < MAZE_W and 0 <= ny < MAZE_H and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
        return None

    def compute_distances(self, start):
        q = deque([start])
        dist = {start: 0}
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < MAZE_W and 0 <= ny < MAZE_H and self.grid[ny][nx] == 0 and (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
        return dist

    def is_walkable_cell(self, cx, cy):
        if not (0 <= cx < MAZE_W and 0 <= cy < MAZE_H):
            return False
        if self.grid[cy][cx] == 1:
            return False
        for door in self.doors:
            if door.locked and door.cell == (cx, cy):
                return False
        return True

    def is_walkable_px(self, x, y):
        return self.is_walkable_cell(int(x // TILE), int(y // TILE))

    def circle_hits_blocker(self, x, y, radius):
        tile_x = int(x // TILE)
        tile_y = int(y // TILE)
        for gy in range(tile_y - 1, tile_y + 2):
            for gx in range(tile_x - 1, tile_x + 2):
                if 0 <= gx < MAZE_W and 0 <= gy < MAZE_H:
                    blocked = self.grid[gy][gx] == 1
                    if not blocked:
                        for door in self.doors:
                            if door.locked and door.cell == (gx, gy):
                                blocked = True
                                break
                    if blocked:
                        rect = pygame.Rect(gx * TILE, gy * TILE, TILE, TILE)
                        if circle_rect_collision(x, y, radius, rect):
                            return True
        return False

    def update_exploration(self, player):
        px = int(player.pos.x // TILE)
        py = int(player.pos.y // TILE)
        for y in range(py - 4, py + 5):
            for x in range(px - 5, px + 6):
                if 0 <= x < MAZE_W and 0 <= y < MAZE_H:
                    self.explored[y][x] = True

    def handle_doors(self, player, texts, game):
        for door in self.doors:
            if not door.locked:
                continue
            if circle_rect_collision(player.pos.x, player.pos.y, player.radius + 2, door.rect):
                if player.keys > 0:
                    player.keys -= 1
                    door.locked = False
                    texts.append(FloatingText(door.rect.centerx, door.rect.centery - 18, f"Gate {door.label} opened", KEY, life=1.1))
                elif game.message_cd <= 0:
                    texts.append(FloatingText(door.rect.centerx, door.rect.centery - 18, f"Gate {door.label} needs a key", WHITE, life=1.0))
                    game.message_cd = 0.8

    def draw(self, surf):
        surf.fill(BG)
        start_x = max(0, int(self.camera.x // TILE) - 1)
        start_y = max(0, int(self.camera.y // TILE) - 1)
        end_x = min(MAZE_W, int((self.camera.x + SCREEN_W) // TILE) + 2)
        end_y = min(MAZE_H, int((self.camera.y + SCREEN_H) // TILE) + 2)

        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                rect = pygame.Rect(x * TILE - self.camera.x, y * TILE - self.camera.y, TILE, TILE)
                if self.grid[y][x] == 1:
                    pygame.draw.rect(surf, WALL, rect)
                    pygame.draw.rect(surf, WALL_EDGE, rect, 2)
                else:
                    color = FLOOR if (x + y) % 2 == 0 else FLOOR_ALT
                    pygame.draw.rect(surf, color, rect)

        arena = pygame.Rect((self.boss_cell[0] - 4) * TILE - self.camera.x, (self.boss_cell[1] - 4) * TILE - self.camera.y, TILE * 9, TILE * 9)
        pygame.draw.rect(surf, (55, 16, 24), arena, 3)

        for door in self.doors:
            door.draw(surf, self.camera)

    def draw_minimap(self, surf, player, boss_alive):
        map_scale = 4
        pad = 14
        mini_w = MAZE_W * map_scale
        mini_h = MAZE_H * map_scale
        bg_rect = pygame.Rect(SCREEN_W - mini_w - pad * 2, pad, mini_w + pad, mini_h + pad)
        pygame.draw.rect(surf, (12, 12, 18), bg_rect, border_radius=10)
        pygame.draw.rect(surf, (60, 60, 80), bg_rect, 2, border_radius=10)

        ox = bg_rect.x + 7
        oy = bg_rect.y + 7
        for y in range(MAZE_H):
            for x in range(MAZE_W):
                if not self.explored[y][x]:
                    continue
                if self.grid[y][x] == 1:
                    color = (85, 88, 110)
                else:
                    color = (185, 185, 205)
                pygame.draw.rect(surf, color, (ox + x * map_scale, oy + y * map_scale, map_scale, map_scale))

        for door in self.doors:
            if door.locked and self.explored[door.cell[1]][door.cell[0]]:
                pygame.draw.rect(surf, DOOR, (ox + door.cell[0] * map_scale, oy + door.cell[1] * map_scale, map_scale + 1, map_scale + 1))

        px = int(player.pos.x // TILE)
        py = int(player.pos.y // TILE)
        pygame.draw.rect(surf, PLAYER_BODY, (ox + px * map_scale, oy + py * map_scale, map_scale + 1, map_scale + 1))
        if boss_alive and self.explored[self.boss_cell[1]][self.boss_cell[0]]:
            pygame.draw.rect(surf, BOSS_CORE, (ox + self.boss_cell[0] * map_scale, oy + self.boss_cell[1] * map_scale, map_scale + 1, map_scale + 1))


class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        random.seed()
        self.level = Level()
        spawn = self.level.cell_center(self.level.start_cell)
        self.player = Player(spawn.x, spawn.y)
        self.projectiles = []
        self.texts = []
        self.enemies = []
        self.pickups = []
        self.boss = None
        self.state = "title"
        self.enemy_count = 54
        self.message_cd = 0
        self.spawn_enemies()
        self.spawn_pickups()
        self.spawn_boss()

    def available_open_cells(self):
        cells = []
        for y in range(1, MAZE_H - 1):
            for x in range(1, MAZE_W - 1):
                if self.level.grid[y][x] == 0:
                    if abs(x - self.level.start_cell[0]) < 3 and abs(y - self.level.start_cell[1]) < 3:
                        continue
                    if abs(x - self.level.boss_cell[0]) < 9 and abs(y - self.level.boss_cell[1]) < 9:
                        continue
                    cells.append((x, y))
        return cells

    def spawn_enemies(self):
        taken = {self.level.start_cell, self.level.boss_cell}
        attempts = 0
        kinds = ["shade"] * 26 + ["stalker"] * 14 + ["cultist"] * 10 + ["brute"] * 8
        while len(self.enemies) < self.enemy_count and attempts < 9000:
            attempts += 1
            x = random.randrange(1, MAZE_W - 1)
            y = random.randrange(1, MAZE_H - 1)
            if self.level.grid[y][x] != 0:
                continue
            if (x, y) in taken:
                continue
            if abs(x - self.level.boss_cell[0]) < 8 and abs(y - self.level.boss_cell[1]) < 8:
                continue
            pos = self.level.cell_center((x, y))
            if (pos - self.player.pos).length() < 220:
                continue
            kind = random.choice(kinds)
            taken.add((x, y))
            self.enemies.append(Enemy(pos.x, pos.y, kind))

    def spawn_pickups(self):
        distances = self.level.compute_distances(self.level.start_cell)
        safe_cells = [
            cell for cell in distances
            if abs(cell[0] - self.level.boss_cell[0]) > 9 or abs(cell[1] - self.level.boss_cell[1]) > 9
        ]
        safe_cells = [cell for cell in safe_cells if distances[cell] > 8]
        safe_cells.sort(key=lambda c: distances[c])

        key_indices = [int(len(safe_cells) * 0.32), int(len(safe_cells) * 0.58), int(len(safe_cells) * 0.82)]
        used = set()
        for idx in key_indices:
            if not safe_cells:
                break
            idx = clamp(idx, 0, len(safe_cells) - 1)
            cell = safe_cells[idx]
            jitter = 0
            while cell in used and idx + jitter + 1 < len(safe_cells):
                jitter += 1
                cell = safe_cells[idx + jitter]
            used.add(cell)
            pos = self.level.cell_center(cell)
            self.pickups.append(Pickup(pos.x, pos.y, "key", 1))

        pool = [cell for cell in self.available_open_cells() if cell not in used]
        loot_cells = random.sample(pool, 28)
        for cell in loot_cells:
            used.add(cell)
            pos = self.level.cell_center(cell)
            self.pickups.append(Pickup(pos.x, pos.y, "loot", random.randint(3, 8)))

        flask_candidates = [cell for cell in self.available_open_cells() if cell not in used]
        for cell in random.sample(flask_candidates, 8):
            used.add(cell)
            pos = self.level.cell_center(cell)
            self.pickups.append(Pickup(pos.x, pos.y, "flask", 1))

    def spawn_boss(self):
        boss_pos = self.level.cell_center(self.level.boss_cell)
        self.boss = Boss(boss_pos.x, boss_pos.y)

    def drop_from_enemy(self, enemy):
        if enemy.kind == "brute":
            self.pickups.append(Pickup(enemy.pos.x, enemy.pos.y, "loot", random.randint(6, 12)))
            if random.random() < 0.25:
                self.pickups.append(Pickup(enemy.pos.x + 10, enemy.pos.y, "flask", 1))
        elif enemy.kind == "cultist":
            self.pickups.append(Pickup(enemy.pos.x, enemy.pos.y, "loot", random.randint(4, 8)))
        elif enemy.kind == "stalker":
            if random.random() < 0.45:
                self.pickups.append(Pickup(enemy.pos.x, enemy.pos.y, "loot", random.randint(2, 5)))
        else:
            if random.random() < 0.6:
                self.pickups.append(Pickup(enemy.pos.x, enemy.pos.y, "loot", random.randint(3, 6)))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if self.state == "title":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.state = "playing"
                elif self.state in ("win", "lose"):
                    if event.key == pygame.K_r:
                        self.reset()
                        self.state = "playing"
                elif self.state == "playing":
                    if event.key == pygame.K_ESCAPE:
                        self.state = "title"
                    if event.key == pygame.K_SPACE:
                        self.player.try_attack()
                    if event.key in (pygame.K_f, pygame.K_q):
                        self.player.try_use_flask(self.texts)
            if event.type == pygame.MOUSEBUTTONDOWN and self.state == "playing":
                if event.button == 1:
                    self.player.try_attack()

    def handle_player_hits(self):
        for enemy in self.enemies:
            if enemy.alive and enemy.last_hit_by_attack != self.player.attack_id and self.player.sword_hits(enemy.pos, enemy.radius):
                damage = 28 if enemy.kind == "stalker" else 24 if enemy.kind == "cultist" else 20 if enemy.kind == "shade" else 18
                enemy.take_damage(damage)
                enemy.last_hit_by_attack = self.player.attack_id
                self.texts.append(FloatingText(enemy.pos.x, enemy.pos.y - 20, str(damage), YELLOW))
                if not enemy.alive:
                    self.drop_from_enemy(enemy)

        if self.boss and self.boss.alive and self.boss.last_hit_by_attack != self.player.attack_id and self.player.sword_hits(self.boss.pos, self.boss.radius):
            damage = 15
            self.boss.take_damage(damage)
            self.boss.last_hit_by_attack = self.player.attack_id
            self.texts.append(FloatingText(self.boss.pos.x, self.boss.pos.y - 44, str(damage), YELLOW))

    def handle_pickups(self):
        for pickup in self.pickups:
            pickup.update(dt)
            if pickup.alive and (pickup.pos - self.player.pos).length() < pickup.radius + self.player.radius + 2:
                pickup.collect(self.player, self.texts)
        self.pickups = [p for p in self.pickups if p.alive]

    def update(self, dt):
        if self.state != "playing":
            return

        self.message_cd = max(0, self.message_cd - dt)
        self.player.update(dt, self.level)
        self.level.handle_doors(self.player, self.texts, self)
        self.level.camera.update(self.player.pos)
        self.level.update_exploration(self.player)

        for enemy in self.enemies:
            enemy.update(dt, self.level, self.player, self.projectiles, self.texts)
        self.handle_player_hits()
        self.enemies = [e for e in self.enemies if e.alive]

        if self.boss and self.boss.alive:
            self.boss.update(dt, self.level, self.player, self.projectiles, self.texts)
            if not self.boss.alive:
                self.pickups.append(Pickup(self.boss.pos.x, self.boss.pos.y, "loot", 50))
                self.state = "win"

        for projectile in self.projectiles:
            projectile.update(dt, self.level, self.player, self.texts)
        self.projectiles = [p for p in self.projectiles if p.alive]

        self.handle_pickups()

        for text in self.texts:
            text.update(dt)
        self.texts = [t for t in self.texts if t.life > 0]

        if self.player.hp <= 0:
            self.state = "lose"

    def draw_hud(self):
        hp_ratio = self.player.hp / self.player.max_hp
        pygame.draw.rect(screen, (25, 25, 30), (18, 18, 320, 26), border_radius=8)
        pygame.draw.rect(screen, RED, (21, 21, int(314 * hp_ratio), 20), border_radius=8)
        hp_text = FONT.render(f"HP {max(0, self.player.hp)}/{self.player.max_hp}", True, WHITE)
        screen.blit(hp_text, (28, 18))

        info = FONT.render(
            f"Keys: {self.player.keys}   Flasks: {self.player.flasks}   Gold: {self.player.loot}",
            True,
            WHITE,
        )
        screen.blit(info, (18, 56))

        enemy_text = FONT.render(f"Shades left: {len(self.enemies)}", True, WHITE)
        screen.blit(enemy_text, (18, 88))

        shield_text = SMALL_FONT.render(
            "LMB / SPACE = sword    SHIFT / RMB = shield    F or Q = use flask",
            True,
            (210, 210, 220),
        )
        screen.blit(shield_text, (18, 120))

        if self.boss and self.boss.alive:
            boss_ratio = self.boss.hp / self.boss.max_hp
            pygame.draw.rect(screen, (25, 25, 30), (SCREEN_W // 2 - 220, 18, 440, 20), border_radius=8)
            pygame.draw.rect(screen, BOSS, (SCREEN_W // 2 - 217, 21, int(434 * boss_ratio), 14), border_radius=8)
            txt = SMALL_FONT.render("DREAD LORD", True, WHITE)
            screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, 40))

        locked = sum(1 for door in self.level.doors if door.locked)
        if locked > 0:
            gate_txt = SMALL_FONT.render(f"Rune gates sealed: {locked}", True, DOOR)
            screen.blit(gate_txt, (18, 148))

        self.level.draw_minimap(screen, self.player, self.boss.alive if self.boss else False)

    def draw_title(self):
        screen.fill((6, 6, 10))
        title = BIG_FONT.render("SHADOW LABYRINTH DELUXE", True, WHITE)
        prompt = FONT.render("Press ENTER or SPACE to begin", True, (210, 210, 220))
        lines = [
            "Explore one enormous labyrinth.",
            "Collect rune keys to open sealed gates.",
            "Gather loot and healing flasks to survive.",
            "Fight shades, stalkers, cultists, brutes, and the final boss.",
        ]
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 130))
        for i, line in enumerate(lines):
            txt = FONT.render(line, True, (185, 185, 205))
            screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, 250 + i * 38))
        screen.blit(prompt, (SCREEN_W // 2 - prompt.get_width() // 2, 500))

    def draw_end(self, win):
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 165))
        screen.blit(overlay, (0, 0))
        title = BIG_FONT.render("YOU WON" if win else "YOU DIED", True, WHITE if win else RED)
        loot = FONT.render(f"Gold collected: {self.player.loot}", True, LOOT)
        prompt = FONT.render("Press R to play again", True, WHITE)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, SCREEN_H // 2 - 80))
        screen.blit(loot, (SCREEN_W // 2 - loot.get_width() // 2, SCREEN_H // 2 - 18))
        screen.blit(prompt, (SCREEN_W // 2 - prompt.get_width() // 2, SCREEN_H // 2 + 24))

    def draw(self):
        if self.state == "title":
            self.draw_title()
            pygame.display.flip()
            return

        self.level.draw(screen)

        for pickup in self.pickups:
            pickup.draw(screen, self.level.camera)
        for enemy in self.enemies:
            enemy.draw(screen, self.level.camera)
        if self.boss and self.boss.alive:
            self.boss.draw(screen, self.level.camera)
        for projectile in self.projectiles:
            projectile.draw(screen, self.level.camera)
        self.player.draw(screen, self.level.camera)
        for text in self.texts:
            text.draw(screen, self.level.camera)

        self.draw_hud()

        if self.state == "win":
            self.draw_end(True)
        elif self.state == "lose":
            self.draw_end(False)

        pygame.display.flip()


def main():
    game = Game()
    while True:
        dt = clock.tick(FPS) / 1000.0
        game.handle_events()
        game.update(dt)
        game.draw()


if __name__ == "__main__":
    main()
