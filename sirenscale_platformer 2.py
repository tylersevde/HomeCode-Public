import math
import random
import sys
from dataclasses import dataclass

import pygame

WIDTH, HEIGHT = 1000, 600
FPS = 60
TITLE = "Sirenscale: Marsh of Teeth"

GRAVITY = 0.58
PLAYER_SPEED = 4.4
AIR_SPEED = 3.8
JUMP_POWER = -11.5
DOUBLE_JUMP_POWER = -10.5
ATTACK_TIME = 10
INVULN_TIME = 60

SKY = (24, 27, 45)
MIST = (65, 88, 104)
GROUND = (42, 74, 53)
MUD = (66, 95, 71)
WATER = (29, 78, 103)
RED = (200, 70, 70)
WHITE = (240, 240, 240)
BLACK = (15, 15, 15)
GOLD = (230, 200, 85)
PURPLE = (134, 77, 164)
CYAN = (88, 190, 190)
ORANGE = (220, 140, 70)


LEVELS = [
    {
        "name": "Level 1 - The Sinking Road",
        "world_width": 2600,
        "spawn": (80, 440),
        "exit": pygame.Rect(2475, 415, 70, 95),
        "platforms": [
            (0, 510, 560, 90),
            (640, 470, 170, 130),
            (890, 420, 180, 180),
            (1170, 355, 210, 245),
            (1480, 430, 170, 170),
            (1770, 380, 180, 220),
            (2030, 330, 180, 270),
            (2290, 260, 150, 340),
            (2460, 510, 140, 90),
        ],
        "hazards": [
            (560, 565, 80, 35),
            (810, 565, 80, 35),
            (1070, 565, 100, 35),
            (1650, 565, 110, 35),
        ],
        "enemies": [
            ("crawler", 700, 430),
            ("crawler", 1240, 315),
            ("spitter", 1550, 390),
            ("crawler", 2070, 290),
        ],
        "collectibles": [
            (790, 420),
            (1465, 380),
            (2240, 280),
        ],
    },
    {
        "name": "Level 2 - Iron Bog Causeway",
        "world_width": 3000,
        "spawn": (90, 430),
        "exit": pygame.Rect(2850, 135, 85, 105),
        "platforms": [
            (0, 500, 420, 100),
            (490, 460, 130, 140),
            (690, 415, 130, 185),
            (900, 360, 140, 240),
            (1120, 310, 170, 290),
            (1380, 355, 170, 245),
            (1640, 410, 130, 190),
            (1830, 470, 130, 130),
            (2030, 420, 170, 180),
            (2280, 345, 150, 255),
            (2490, 260, 140, 340),
            (2690, 180, 130, 420),
            (2850, 130, 140, 470),
        ],
        "hazards": [
            (420, 565, 70, 35),
            (620, 565, 70, 35),
            (820, 565, 80, 35),
            (1290, 565, 90, 35),
            (1770, 565, 60, 35),
            (1960, 565, 70, 35),
        ],
        "enemies": [
            ("crawler", 535, 420),
            ("spitter", 940, 320),
            ("crawler", 1405, 315),
            ("spitter", 2050, 380),
            ("crawler", 2500, 220),
        ],
        "collectibles": [
            (880, 320),
            (1590, 370),
            (2410, 310),
            (2820, 100),
        ],
    },
    {
        "name": "Level 3 - The Drowned Teeth",
        "world_width": 3400,
        "spawn": (90, 420),
        "exit": pygame.Rect(3210, 120, 95, 105),
        "platforms": [
            (0, 500, 350, 100),
            (420, 455, 110, 145),
            (590, 395, 110, 205),
            (760, 335, 120, 265),
            (950, 275, 130, 325),
            (1150, 220, 130, 380),
            (1350, 260, 150, 340),
            (1560, 320, 130, 280),
            (1750, 390, 110, 210),
            (1930, 450, 110, 150),
            (2140, 400, 140, 200),
            (2340, 345, 130, 255),
            (2520, 280, 120, 320),
            (2700, 220, 130, 380),
            (2890, 165, 130, 435),
            (3070, 120, 130, 480),
            (3210, 120, 150, 480),
        ],
        "hazards": [
            (350, 565, 70, 35),
            (530, 565, 60, 35),
            (700, 565, 60, 35),
            (1080, 565, 70, 35),
            (1500, 565, 60, 35),
            (1860, 565, 70, 35),
            (2280, 565, 60, 35),
        ],
        "enemies": [
            ("crawler", 430, 415),
            ("spitter", 980, 235),
            ("crawler", 1360, 220),
            ("spitter", 2170, 360),
            ("crawler", 2540, 240),
            ("spitter", 2890, 125),
        ],
        "collectibles": [
            (730, 310),
            (1260, 190),
            (1730, 360),
            (2505, 250),
            (3190, 80),
        ],
    },
]


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    color: tuple
    size: int

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12
        self.life -= 1

    def draw(self, surf, camera_x):
        if self.life > 0:
            pygame.draw.circle(surf, self.color, (int(self.x - camera_x), int(self.y)), max(1, self.size))


class Bullet:
    def __init__(self, x, y, vx, color=CYAN, damage=1, owner="enemy"):
        self.rect = pygame.Rect(x, y, 10, 6)
        self.vx = vx
        self.color = color
        self.damage = damage
        self.owner = owner
        self.alive = True

    def update(self, platforms):
        self.rect.x += int(self.vx)
        for p in platforms:
            if self.rect.colliderect(p):
                self.alive = False
                break

    def draw(self, surf, camera_x):
        pygame.draw.rect(surf, self.color, self.rect.move(-camera_x, 0), border_radius=2)


class Entity:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.hp = 1
        self.max_hp = 1
        self.alive = True

    def physics(self, solids):
        self.rect.x += int(round(self.vx))
        for s in solids:
            if self.rect.colliderect(s):
                if self.vx > 0:
                    self.rect.right = s.left
                elif self.vx < 0:
                    self.rect.left = s.right
                self.vx = 0

        self.vy += GRAVITY
        self.rect.y += int(round(self.vy))
        self.on_ground = False
        for s in solids:
            if self.rect.colliderect(s):
                if self.vy > 0:
                    self.rect.bottom = s.top
                    self.on_ground = True
                elif self.vy < 0:
                    self.rect.top = s.bottom
                self.vy = 0


class Player(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 34, 54)
        self.hp = 6
        self.max_hp = 6
        self.facing = 1
        self.jumps_left = 2
        self.attack_timer = 0
        self.invuln = 0
        self.score = 0
        self.form = 0

    def update(self, keys, solids):
        speed = PLAYER_SPEED if self.on_ground else AIR_SPEED
        move = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move += 1
        self.vx = move * speed
        if move != 0:
            self.facing = 1 if move > 0 else -1
        self.physics(solids)
        if self.on_ground:
            self.jumps_left = 2
        if self.attack_timer > 0:
            self.attack_timer -= 1
        if self.invuln > 0:
            self.invuln -= 1
        self.form = (self.form + 1) % 120

    def jump(self):
        if self.jumps_left > 0:
            self.vy = JUMP_POWER if self.jumps_left == 2 else DOUBLE_JUMP_POWER
            self.jumps_left -= 1

    def attack_rect(self):
        if self.attack_timer <= 0:
            return None
        if self.facing > 0:
            return pygame.Rect(self.rect.right - 2, self.rect.y + 8, 36, self.rect.h - 16)
        return pygame.Rect(self.rect.x - 34, self.rect.y + 8, 36, self.rect.h - 16)

    def take_hit(self, dmg):
        if self.invuln <= 0:
            self.hp -= dmg
            self.invuln = INVULN_TIME
            if self.hp <= 0:
                self.alive = False

    def draw(self, surf, camera_x):
        x = self.rect.x - camera_x
        y = self.rect.y
        flicker = self.invuln > 0 and (self.invuln // 3) % 2 == 0
        if flicker:
            return
        body = pygame.Rect(x, y + 10, self.rect.w, self.rect.h - 10)
        pygame.draw.rect(surf, (66, 149, 124), body, border_radius=10)
        pygame.draw.rect(surf, (96, 195, 168), (x + 6, y + 16, self.rect.w - 12, 14), border_radius=7)
        pygame.draw.rect(surf, (110, 79, 46), (x - 6, y + 5, self.rect.w + 12, 16), border_radius=7)
        eye_offset = 7 if self.facing > 0 else 3
        pygame.draw.circle(surf, WHITE, (x + 10 + eye_offset, y + 16), 4)
        pygame.draw.circle(surf, BLACK, (x + 11 + eye_offset, y + 16), 2)
        pygame.draw.circle(surf, WHITE, (x + 20 + eye_offset, y + 16), 4)
        pygame.draw.circle(surf, BLACK, (x + 21 + eye_offset, y + 16), 2)
        shimmer = 2 if (self.form // 8) % 2 == 0 else -2
        for i in range(3):
            pygame.draw.arc(surf, (90, 220 - i * 30, 180 + i * 10), (x - 4 - i * 2, y + 10 + i * 4, self.rect.w + 8 + i * 4, 28), 0.2, 2.1, 2)
        if self.attack_timer > 0:
            ar = self.attack_rect()
            if ar:
                ar = ar.move(-camera_x, 0)
                pygame.draw.rect(surf, (120, 220, 255), ar, 3, border_radius=8)
                pygame.draw.line(surf, WHITE, ar.midleft, ar.midright, 2)


class Crawler(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 32, 30)
        self.start_x = x
        self.range = 95
        self.speed = 1.5
        self.hp = 2
        self.max_hp = 2
        self.facing = -1

    def update(self, solids, player):
        if abs(player.rect.centerx - self.rect.centerx) < 170 and abs(player.rect.centery - self.rect.centery) < 60:
            self.facing = 1 if player.rect.centerx > self.rect.centerx else -1
            self.vx = self.facing * 2.0
        else:
            if self.rect.x < self.start_x - self.range:
                self.facing = 1
            elif self.rect.x > self.start_x + self.range:
                self.facing = -1
            self.vx = self.facing * self.speed
        self.physics(solids)

    def draw(self, surf, camera_x):
        r = self.rect.move(-camera_x, 0)
        pygame.draw.ellipse(surf, (121, 153, 83), r)
        pygame.draw.ellipse(surf, (82, 110, 58), (r.x + 5, r.y + 4, r.w - 10, r.h - 8))
        pygame.draw.circle(surf, RED, (r.centerx + (5 if self.facing > 0 else -5), r.y + 10), 3)


class Spitter(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 34, 42)
        self.hp = 3
        self.max_hp = 3
        self.cooldown = random.randint(30, 90)
        self.facing = -1

    def update(self, solids, player, bullets):
        dx = player.rect.centerx - self.rect.centerx
        self.facing = 1 if dx > 0 else -1
        if abs(dx) < 340 and abs(player.rect.centery - self.rect.centery) < 120:
            self.vx = 1.0 * self.facing
        else:
            self.vx = 0
        self.physics(solids)
        self.cooldown -= 1
        if self.cooldown <= 0 and abs(dx) < 420 and abs(player.rect.centery - self.rect.centery) < 100:
            bx = self.rect.centerx + self.facing * 16
            by = self.rect.centery - 4
            bullets.append(Bullet(bx, by, 6 * self.facing, color=PURPLE, damage=1, owner="enemy"))
            self.cooldown = random.randint(65, 110)

    def draw(self, surf, camera_x):
        r = self.rect.move(-camera_x, 0)
        pygame.draw.rect(surf, (122, 87, 156), r, border_radius=10)
        pygame.draw.rect(surf, (180, 145, 200), (r.x + 5, r.y + 6, r.w - 10, 12), border_radius=6)
        mouth = pygame.Rect(r.centerx - 9 + (4 if self.facing > 0 else -4), r.y + 20, 18, 8)
        pygame.draw.ellipse(surf, (230, 130, 170), mouth)


class Collectible:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 18, 18)
        self.angle = random.random() * math.tau
        self.collected = False

    def update(self):
        self.angle += 0.08

    def draw(self, surf, camera_x):
        x = self.rect.centerx - camera_x
        y = self.rect.centery + math.sin(self.angle) * 6
        points = [
            (x, y - 10),
            (x + 7, y - 2),
            (x + 11, y + 10),
            (x, y + 4),
            (x - 11, y + 10),
            (x - 7, y - 2),
        ]
        pygame.draw.polygon(surf, GOLD, points)
        pygame.draw.polygon(surf, WHITE, points, 2)


class Boss(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 120, 120)
        self.hp = 28
        self.max_hp = 28
        self.cooldown = 80
        self.phase = 0
        self.facing = -1
        self.timer = 0

    def update(self, solids, player, bullets, enemies):
        dx = player.rect.centerx - self.rect.centerx
        self.facing = 1 if dx > 0 else -1
        dist = abs(dx)
        self.timer += 1
        self.cooldown -= 1

        if dist > 120:
            self.vx = 1.8 * self.facing
        else:
            self.vx = 0

        if self.on_ground and self.cooldown < 0 and dist > 150 and random.random() < 0.02:
            self.vy = -12.5
            self.cooldown = 50

        self.physics(solids)

        if self.cooldown <= 0:
            if self.phase == 0:
                for offset in (-16, 0, 16):
                    bullets.append(Bullet(self.rect.centerx, self.rect.centery + offset, 7 * self.facing, color=ORANGE, damage=1, owner="boss"))
                self.cooldown = 80
            elif self.phase == 1:
                for _ in range(2):
                    spawn_x = self.rect.centerx + random.choice([-120, 120])
                    enemies.append(Crawler(spawn_x, self.rect.bottom - 30))
                self.cooldown = 120
            else:
                for angle in (-0.25, 0, 0.25):
                    b = Bullet(self.rect.centerx, self.rect.centery, 8 * self.facing, color=(110, 220, 255), damage=1, owner="boss")
                    b.vy = angle * 10
                    bullets.append(b)
                self.cooldown = 70

        if self.hp <= self.max_hp * 0.66:
            self.phase = 1
        if self.hp <= self.max_hp * 0.33:
            self.phase = 2

    def draw(self, surf, camera_x):
        r = self.rect.move(-camera_x, 0)
        pygame.draw.rect(surf, (84, 126, 145), r, border_radius=18)
        pygame.draw.rect(surf, (130, 196, 204), (r.x + 12, r.y + 18, r.w - 24, 24), border_radius=8)
        pygame.draw.rect(surf, (75, 61, 45), (r.x - 12, r.y + 8, r.w + 24, 20), border_radius=10)
        pygame.draw.circle(surf, WHITE, (r.x + 36 + (12 if self.facing > 0 else 0), r.y + 40), 10)
        pygame.draw.circle(surf, WHITE, (r.x + 72 + (12 if self.facing > 0 else 0), r.y + 40), 10)
        pygame.draw.circle(surf, BLACK, (r.x + 39 + (12 if self.facing > 0 else 0), r.y + 40), 5)
        pygame.draw.circle(surf, BLACK, (r.x + 75 + (12 if self.facing > 0 else 0), r.y + 40), 5)
        pygame.draw.arc(surf, RED, (r.x + 25, r.y + 50, 70, 28), 0.2, 2.9, 4)
        for i in range(6):
            pygame.draw.line(surf, (180, 250, 250), (r.x + 12 + i * 18, r.y + 88), (r.x + 25 + i * 18, r.y + 106), 3)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.big_font = pygame.font.SysFont("arial", 48, bold=True)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.running = True
        self.state = "play"
        self.level_index = 0
        self.camera_x = 0
        self.reset_level(0)

    def reset_level(self, index):
        self.level_index = index
        level = LEVELS[index]
        self.world_width = level["world_width"]
        self.level_name = level["name"]
        sx, sy = level["spawn"]
        if hasattr(self, "player") and self.player.alive:
            score = self.player.score
            hp = min(self.player.max_hp, self.player.hp + 1)
            self.player = Player(sx, sy)
            self.player.score = score
            self.player.hp = hp
        else:
            self.player = Player(sx, sy)
        self.platforms = [pygame.Rect(*p) for p in level["platforms"]]
        self.hazards = [pygame.Rect(*h) for h in level["hazards"]]
        self.exit = level["exit"].copy()
        self.enemies = []
        for kind, x, y in level["enemies"]:
            self.enemies.append(Crawler(x, y) if kind == "crawler" else Spitter(x, y))
        self.collectibles = [Collectible(x, y) for x, y in level["collectibles"]]
        self.bullets = []
        self.particles = []
        self.boss = None
        self.boss_gate = None
        self.boss_arena = None
        self.stage_clear_timer = 0
        self.camera_x = 0

        if index == 2:
            self.boss = Boss(3330, 380)
            self.boss_arena = pygame.Rect(3300, 0, 820, HEIGHT)
            self.boss_gate = pygame.Rect(3260, 350, 20, 250)
            self.platforms.extend([
                pygame.Rect(3300, 500, 820, 100),
                pygame.Rect(3400, 400, 150, 200),
                pygame.Rect(3630, 320, 150, 280),
                pygame.Rect(3870, 410, 160, 190),
            ])
            self.exit = pygame.Rect(4020, 305, 70, 105)
            self.world_width = 4200

    def spawn_hit_particles(self, x, y, color, amount=8):
        for _ in range(amount):
            self.particles.append(
                Particle(
                    x, y,
                    random.uniform(-3, 3),
                    random.uniform(-4, -1),
                    random.randint(18, 30),
                    color,
                    random.randint(2, 4),
                )
            )

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if self.state in ("win", "gameover") and event.key == pygame.K_r:
                    self.state = "play"
                    self.player = Player(0, 0)
                    self.reset_level(0)
                if self.state == "play":
                    if event.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
                        self.player.jump()
                    if event.key in (pygame.K_j, pygame.K_k, pygame.K_LCTRL, pygame.K_RETURN):
                        self.player.attack_timer = ATTACK_TIME

    def update(self):
        if self.state != "play":
            return
        keys = pygame.key.get_pressed()
        self.player.update(keys, self.platforms + ([self.boss_gate] if self.boss_gate else []))

        if self.boss_gate and self.player.rect.right > self.boss_gate.left and (self.boss is None or self.boss.hp > 0):
            pass
        elif self.boss_gate and self.boss and self.boss.hp <= 0:
            self.platforms = [p for p in self.platforms if p != self.boss_gate]
            self.boss_gate = None

        for c in self.collectibles:
            if not c.collected:
                c.update()
                if self.player.rect.colliderect(c.rect):
                    c.collected = True
                    self.player.score += 1
                    self.player.hp = min(self.player.max_hp, self.player.hp + 1)
                    self.spawn_hit_particles(c.rect.centerx, c.rect.centery, GOLD, 14)

        attack_rect = self.player.attack_rect()
        for enemy in list(self.enemies):
            if isinstance(enemy, Crawler):
                enemy.update(self.platforms + ([self.boss_gate] if self.boss_gate else []), self.player)
            else:
                enemy.update(self.platforms + ([self.boss_gate] if self.boss_gate else []), self.player, self.bullets)
            if self.player.rect.colliderect(enemy.rect):
                self.player.take_hit(1)
            if attack_rect and enemy.rect.colliderect(attack_rect):
                enemy.hp -= 1
                enemy.vx += 4 * self.player.facing
                self.spawn_hit_particles(enemy.rect.centerx, enemy.rect.centery, CYAN, 8)
            if enemy.hp <= 0:
                self.enemies.remove(enemy)
                self.spawn_hit_particles(enemy.rect.centerx, enemy.rect.centery, RED, 12)

        if self.boss and self.boss.hp > 0:
            solids = self.platforms + ([self.boss_gate] if self.boss_gate else [])
            self.boss.update(solids, self.player, self.bullets, self.enemies)
            if self.player.rect.colliderect(self.boss.rect):
                self.player.take_hit(1)
            if attack_rect and self.boss.rect.colliderect(attack_rect):
                self.boss.hp -= 1
                self.spawn_hit_particles(self.boss.rect.centerx, self.boss.rect.centery, CYAN, 10)
        
        for b in list(self.bullets):
            if hasattr(b, 'vy'):
                b.rect.y += int(round(b.vy))
            b.update(self.platforms + ([self.boss_gate] if self.boss_gate else []))
            if b.rect.colliderect(self.player.rect):
                self.player.take_hit(b.damage)
                b.alive = False
            if b.rect.right < 0 or b.rect.left > self.world_width:
                b.alive = False
            if not b.alive:
                self.bullets.remove(b)
                self.spawn_hit_particles(b.rect.centerx, b.rect.centery, b.color, 4)

        for hz in self.hazards:
            if self.player.rect.colliderect(hz):
                self.player.take_hit(1)
                self.player.rect.y -= 40
                self.player.vy = -6
            for enemy in self.enemies:
                if enemy.rect.colliderect(hz):
                    enemy.hp = 0
            if self.boss and self.boss.hp > 0 and self.boss.rect.colliderect(hz):
                self.boss.hp -= 1

        for p in list(self.particles):
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        if not self.player.alive:
            self.state = "gameover"

        if self.level_index < 2:
            if self.player.rect.colliderect(self.exit):
                self.stage_clear_timer += 1
                if self.stage_clear_timer > 18:
                    self.reset_level(self.level_index + 1)
            else:
                self.stage_clear_timer = 0
        else:
            if self.boss and self.boss.hp <= 0 and self.player.rect.colliderect(self.exit):
                self.state = "win"

        self.camera_x = max(0, min(self.player.rect.centerx - WIDTH // 2, self.world_width - WIDTH))

    def draw_background(self):
        self.screen.fill(SKY)
        for i in range(7):
            y = 120 + i * 45
            offset = (self.camera_x * (0.15 + i * 0.02)) % (WIDTH + 200)
            pygame.draw.ellipse(self.screen, (35 + i * 5, 55 + i * 5, 70 + i * 4), (-100 - offset, y, 400, 80))
            pygame.draw.ellipse(self.screen, (35 + i * 5, 55 + i * 5, 70 + i * 4), (250 - offset, y + 10, 470, 90))
            pygame.draw.ellipse(self.screen, (35 + i * 5, 55 + i * 5, 70 + i * 4), (670 - offset, y + 5, 360, 80))
        pygame.draw.rect(self.screen, MIST, (0, HEIGHT - 120, WIDTH, 120))
        for x in range(0, WIDTH, 80):
            pygame.draw.line(self.screen, (55, 90, 105), (x, HEIGHT - 120), (x + 20, HEIGHT), 2)

    def draw_world(self):
        for hz in self.hazards:
            pygame.draw.rect(self.screen, WATER, hz.move(-self.camera_x, 0), border_radius=8)
            for i in range(4):
                pygame.draw.arc(self.screen, (70, 170, 210), (hz.x - self.camera_x + i * 18, hz.y + 6, 18, 12), 0, math.pi, 2)

        for p in self.platforms:
            pr = p.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, GROUND, pr, border_radius=8)
            pygame.draw.rect(self.screen, MUD, (pr.x, pr.y + 12, pr.w, pr.h - 12), border_radius=8)
            for i in range(max(1, pr.w // 40)):
                xx = pr.x + i * 40 + 8
                pygame.draw.line(self.screen, (32, 58, 37), (xx, pr.y + 10), (xx + 8, pr.y + 24), 2)

        if self.boss_gate:
            gate = self.boss_gate.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, (28, 104, 110), gate)
            for yy in range(gate.y, gate.bottom, 16):
                pygame.draw.line(self.screen, (120, 230, 240), (gate.x, yy), (gate.right, yy + 8), 2)

        ex = self.exit.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, (82, 62, 115), ex, border_radius=8)
        pygame.draw.rect(self.screen, (160, 120, 220), (ex.x + 10, ex.y + 10, ex.w - 20, ex.h - 20), border_radius=6)
        pygame.draw.circle(self.screen, WHITE, ex.midtop, 10)

        for c in self.collectibles:
            if not c.collected:
                c.draw(self.screen, self.camera_x)

        for e in self.enemies:
            e.draw(self.screen, self.camera_x)

        if self.boss and self.boss.hp > 0:
            self.boss.draw(self.screen, self.camera_x)

        for b in self.bullets:
            b.draw(self.screen, self.camera_x)

        self.player.draw(self.screen, self.camera_x)

        for p in self.particles:
            p.draw(self.screen, self.camera_x)

    def draw_ui(self):
        title = self.font.render(self.level_name, True, WHITE)
        self.screen.blit(title, (18, 14))
        for i in range(self.player.max_hp):
            color = (210, 70, 70) if i < self.player.hp else (70, 45, 45)
            pygame.draw.rect(self.screen, color, (18 + i * 28, 50, 22, 18), border_radius=5)
        self.screen.blit(self.small_font.render(f"Relics: {self.player.score}", True, GOLD), (18, 78))
        self.screen.blit(self.small_font.render("Move: A/D  Jump: Space/W  Attack: J  Restart: R", True, WHITE), (18, HEIGHT - 28))

        if self.boss and self.boss.hp > 0:
            pygame.draw.rect(self.screen, (40, 40, 55), (WIDTH // 2 - 170, 18, 340, 22), border_radius=8)
            fill = int(336 * (self.boss.hp / self.boss.max_hp))
            pygame.draw.rect(self.screen, (70, 195, 215), (WIDTH // 2 - 168, 20, fill, 18), border_radius=7)
            txt = self.small_font.render("Boss: The Mire King", True, WHITE)
            self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 44))

        if self.state == "gameover":
            msg = self.big_font.render("Sirenscale Fell", True, WHITE)
            self.screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 210))
            sub = self.font.render("Press R to restart", True, WHITE)
            self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 270))
        elif self.state == "win":
            msg = self.big_font.render("The Mire King Is Broken", True, WHITE)
            self.screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 200))
            sub = self.font.render(f"Relics gathered: {self.player.score}   Press R to play again", True, GOLD)
            self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 264))

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw_background()
            self.draw_world()
            self.draw_ui()
            pygame.display.flip()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
