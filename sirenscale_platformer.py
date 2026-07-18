import math
import random
import sys
from dataclasses import dataclass

import pygame


WIDTH, HEIGHT = 800, 480
FPS = 60
TITLE = "Sirenscale: Marsh of Teeth"

GRAVITY = 0.58
PLAYER_SPEED = 4.4
AIR_SPEED = 3.8
JUMP_POWER = -11.5
DOUBLE_JUMP_POWER = -10.5
ATTACK_TIME = 10
INVULN_TIME = 60

SKY_TOP = (10, 18, 34)
SKY_BOTTOM = (30, 58, 74)
MIST = (65, 88, 104)
GROUND = (52, 82, 58)
MUD = (66, 95, 71)
WATER = (29, 78, 103)
RED = (200, 70, 70)
WHITE = (240, 240, 240)
BLACK = (15, 15, 15)
GOLD = (230, 200, 85)
PURPLE = (134, 77, 164)
CYAN = (88, 190, 190)
ORANGE = (220, 140, 70)
TEAL = (72, 210, 180)

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


def clamp(value, low, high):
    return max(low, min(high, value))


def blend(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_vertical_gradient(surface, top_color, bottom_color):
    w, h = surface.get_size()
    for y in range(h):
        c = blend(top_color, bottom_color, y / max(1, h - 1))
        pygame.draw.line(surface, c, (0, y), (w, y))


def circle_glow(surface, color, center, radius, alpha):
    if radius <= 0:
        return
    glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for i in range(radius, 0, -4):
        a = int(alpha * (i / radius) ** 2)
        pygame.draw.circle(glow, (*color, a), (radius, radius), i)
    surface.blit(glow, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_PREMULTIPLIED)


def draw_shadow(surface, rect, lift=0, alpha=60, squish=0.55):
    w = max(8, int(rect.w * 0.95))
    h = max(4, int(rect.h * squish * 0.24))
    sh = pygame.Surface((w + 12, h + 8), pygame.SRCALPHA)
    pygame.draw.ellipse(sh, (0, 0, 0, alpha), (6, 2, w, h))
    surface.blit(sh, (rect.centerx - (w + 12) // 2, rect.bottom - h // 2 + lift))


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    color: tuple
    size: int
    fade: bool = True
    glow: bool = False

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12
        self.vx *= 0.985
        self.life -= 1

    def draw(self, surf, camera_x):
        if self.life <= 0:
            return
        px = int(self.x - camera_x)
        py = int(self.y)
        alpha = clamp(self.life * 10, 0, 255) if self.fade else 255
        size = max(1, self.size)
        stamp = pygame.Surface((size * 6, size * 6), pygame.SRCALPHA)
        if self.glow:
            pygame.draw.circle(stamp, (*self.color, alpha // 3), (size * 3, size * 3), size * 2)
        pygame.draw.circle(stamp, (*self.color, alpha), (size * 3, size * 3), size)
        surf.blit(stamp, (px - size * 3, py - size * 3))


class Bullet:
    def __init__(self, x, y, vx, color=CYAN, damage=1, owner="enemy"):
        self.rect = pygame.Rect(x, y, 10, 6)
        self.vx = vx
        self.color = color
        self.damage = damage
        self.owner = owner
        self.alive = True
        self.trail = []
        self.vy = 0.0

    def update(self, platforms):
        self.trail.append((self.rect.centerx, self.rect.centery))
        if len(self.trail) > 5:
            self.trail.pop(0)
        self.rect.x += int(self.vx)
        for p in platforms:
            if self.rect.colliderect(p):
                self.alive = False
                break

    def draw(self, surf, camera_x):
        for i, pos in enumerate(self.trail):
            t = (i + 1) / max(1, len(self.trail))
            circle_glow(surf, self.color, (int(pos[0] - camera_x), pos[1]), int(3 + 4 * t), int(30 * t))
        r = self.rect.move(-camera_x, 0)
        circle_glow(surf, self.color, r.center, 8, 70)
        pygame.draw.rect(surf, self.color, r, border_radius=3)
        pygame.draw.rect(surf, WHITE, (r.x + 1, r.y + 1, max(2, r.w - 4), max(2, r.h - 2)), border_radius=2)


class Entity:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.hp = 1
        self.max_hp = 1
        self.alive = True
        self.prev_on_ground = False

    def physics(self, solids):
        self.prev_on_ground = self.on_ground
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
        self.step_timer = 0

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
            self.step_timer += 1
        else:
            self.step_timer = 0
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
            return True
        return False

    def draw(self, surf, camera_x):
        x = self.rect.x - camera_x
        y = self.rect.y
        if self.invuln > 0 and (self.invuln // 3) % 2 == 0:
            return
        screen_rect = pygame.Rect(x, y, self.rect.w, self.rect.h)
        draw_shadow(surf, screen_rect, alpha=70)

        bob = math.sin(self.form * 0.25) * (1 if self.on_ground else 0.3)
        cape_sway = math.sin(self.form * 0.21) * 4
        body = pygame.Rect(x + 4, int(y + 12 + bob), self.rect.w - 8, self.rect.h - 14)
        coat = pygame.Rect(x + 1, int(y + 10 + bob), self.rect.w - 2, self.rect.h - 8)
        head = pygame.Rect(x + 4, int(y + 2 + bob), self.rect.w - 8, 20)

        # back fins / silhouette
        fin_color = (60, 160, 150)
        fin_pts = [
            (x + 5, y + 24),
            (x - 6, y + 34 + cape_sway * 0.3),
            (x + 4, y + 44),
            (x + 2, y + 54),
            (x + 14, y + 44),
        ]
        fin_pts2 = [
            (x + self.rect.w - 5, y + 26),
            (x + self.rect.w + 7, y + 35 - cape_sway * 0.3),
            (x + self.rect.w - 2, y + 45),
            (x + self.rect.w + 3, y + 54),
            (x + self.rect.w - 13, y + 44),
        ]
        pygame.draw.polygon(surf, fin_color, fin_pts)
        pygame.draw.polygon(surf, fin_color, fin_pts2)

        pygame.draw.rect(surf, (31, 63, 59), coat.move(0, 2), border_radius=12)
        pygame.draw.rect(surf, (55, 138, 118), coat, border_radius=12)
        pygame.draw.rect(surf, (89, 193, 168), body, border_radius=10)

        # scale shimmer
        for row in range(3):
            yy = body.y + 5 + row * 9
            for col in range(3):
                xx = body.x + 5 + col * 8 + (row % 2) * 3
                pygame.draw.arc(surf, (130, 245 - row * 18, 205), (xx, yy, 8, 6), 0.1, 3.1, 1)

        # rusty iron arm + silhouette arm
        arm_x = x + self.rect.w - 2 if self.facing > 0 else x - 8
        pygame.draw.rect(surf, (86, 64, 48), (arm_x, y + 24, 10, 18), border_radius=5)
        pygame.draw.rect(surf, (128, 100, 70), (arm_x + 1, y + 25, 8, 6), border_radius=3)
        pygame.draw.rect(surf, (42, 28, 22), (arm_x - 1, y + 38, 12, 5), border_radius=3)

        hair = [
            (x - 6, y + 10),
            (x + 6, y + 2),
            (x + 16, y + 1),
            (x + 26, y + 6),
            (x + 40, y + 14),
            (x + 34, y + 24),
            (x + 20, y + 18),
            (x + 7, y + 22),
        ]
        pygame.draw.polygon(surf, (86, 73, 44), hair)
        pygame.draw.rect(surf, (72, 133, 126), head, border_radius=9)
        pygame.draw.rect(surf, (110, 205, 193), (head.x + 2, head.y + 3, head.w - 4, 8), border_radius=6)

        eye_shift = 5 if self.facing > 0 else -5
        eye_y = head.y + 11
        pygame.draw.circle(surf, WHITE, (head.centerx - 7 + eye_shift, eye_y), 4)
        pygame.draw.circle(surf, WHITE, (head.centerx + 6 + eye_shift, eye_y), 4)
        pygame.draw.circle(surf, BLACK, (head.centerx - 6 + eye_shift, eye_y), 2)
        pygame.draw.circle(surf, BLACK, (head.centerx + 7 + eye_shift, eye_y), 2)
        pygame.draw.arc(surf, (155, 235, 230), (head.x + 2, head.y + 2, head.w - 4, head.h + 10), 0.3, 2.7, 2)
        circle_glow(surf, (90, 220, 210), (head.centerx, head.centery), 22, 26)

        # movement highlights
        if not self.on_ground:
            for i in range(2):
                pygame.draw.line(surf, (120, 230, 210), (x + 10 + i * 12, y + self.rect.h), (x + 4 + i * 10, y + self.rect.h + 9), 2)

        if self.attack_timer > 0:
            ar = self.attack_rect()
            if ar:
                ar = ar.move(-camera_x, 0)
                slash = pygame.Surface((ar.w + 28, ar.h + 26), pygame.SRCALPHA)
                pts = [(6, slash.get_height() // 2), (slash.get_width() - 7, 6), (slash.get_width() - 2, slash.get_height() // 2), (slash.get_width() - 12, slash.get_height() - 6)]
                if self.facing < 0:
                    pts = [(slash.get_width() - x, y) for x, y in pts]
                pygame.draw.polygon(slash, (110, 235, 255, 70), pts)
                pygame.draw.polygon(slash, (220, 250, 255, 135), pts, 2)
                surf.blit(slash, (ar.x - 12, ar.y - 10))
                circle_glow(surf, (110, 235, 255), ar.center, 22, 40)


class Crawler(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 32, 30)
        self.start_x = x
        self.range = 95
        self.speed = 1.5
        self.hp = 2
        self.max_hp = 2
        self.facing = -1
        self.anim = random.random() * math.tau

    def update(self, solids, player):
        self.anim += 0.18
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
        draw_shadow(surf, r, alpha=45)
        body = pygame.Rect(r.x, r.y + 4, r.w, r.h - 4)
        pygame.draw.ellipse(surf, (70, 104, 64), body)
        pygame.draw.ellipse(surf, (122, 165, 96), (body.x + 2, body.y + 4, body.w - 4, body.h - 8))
        for i in range(4):
            lx = body.x + 6 + i * 6
            ly = body.bottom - 3
            swing = math.sin(self.anim + i) * 3
            pygame.draw.line(surf, (42, 57, 38), (lx, ly), (lx + swing, ly + 8), 2)
        jaw_x = r.centerx + (7 if self.facing > 0 else -7)
        pygame.draw.circle(surf, (220, 70, 70), (jaw_x, r.y + 13), 4)
        pygame.draw.circle(surf, WHITE, (jaw_x, r.y + 13), 1)


class Spitter(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 34, 42)
        self.hp = 3
        self.max_hp = 3
        self.cooldown = random.randint(30, 90)
        self.facing = -1
        self.anim = random.random() * math.tau

    def update(self, solids, player, bullets):
        self.anim += 0.12
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
        draw_shadow(surf, r, alpha=55)
        circle_glow(surf, (145, 90, 185), (r.centerx, r.centery), 24, 28)
        pygame.draw.rect(surf, (95, 60, 122), r, border_radius=11)
        pygame.draw.rect(surf, (165, 128, 198), (r.x + 4, r.y + 6, r.w - 8, 13), border_radius=6)
        hood = [(r.x + 3, r.y + 18), (r.centerx, r.y - 2), (r.right - 3, r.y + 18), (r.right - 7, r.bottom - 4), (r.x + 7, r.bottom - 4)]
        pygame.draw.polygon(surf, (118, 84, 150), hood)
        mouth = pygame.Rect(r.centerx - 9 + (4 if self.facing > 0 else -4), r.y + 22, 18, 8)
        pygame.draw.ellipse(surf, (230, 130, 170), mouth)
        pygame.draw.line(surf, (255, 205, 235), mouth.midleft, mouth.midright, 2)


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
        circle_glow(surf, (230, 200, 85), (x, int(y)), 18, 60)
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
        pygame.draw.line(surf, WHITE, (x, y - 10), (x, y + 4), 1)


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
        draw_shadow(surf, r, alpha=80)
        pulse = 24 + int(math.sin(self.timer * 0.12) * 4)
        circle_glow(surf, (70, 195, 215), (r.centerx, r.centery), pulse + 22, 34)
        body = pygame.Rect(r.x + 8, r.y + 8, r.w - 16, r.h - 12)
        pygame.draw.rect(surf, (67, 109, 121), body, border_radius=22)
        pygame.draw.rect(surf, (114, 175, 184), (body.x + 10, body.y + 12, body.w - 20, 22), border_radius=10)
        crest = [(r.x + 14, r.y + 26), (r.centerx, r.y - 8), (r.right - 14, r.y + 26), (r.centerx, r.y + 18)]
        pygame.draw.polygon(surf, (76, 60, 48), crest)
        eye_shift = 11 if self.facing > 0 else -11
        for ex in (r.centerx - 18, r.centerx + 18):
            pygame.draw.circle(surf, WHITE, (ex + eye_shift, r.y + 38), 10)
            pygame.draw.circle(surf, BLACK, (ex + eye_shift + 3, r.y + 38), 5)
        pygame.draw.arc(surf, RED, (r.x + 22, r.y + 52, 76, 28), 0.22, 2.92, 4)
        for i in range(6):
            pygame.draw.line(surf, (180, 250, 250), (r.x + 15 + i * 16, r.y + 88), (r.x + 24 + i * 16, r.y + 109), 3)
        for i in range(4):
            fy = r.y + 64 + i * 10
            pygame.draw.line(surf, (74, 145, 148), (r.x + 6, fy), (r.x - 12, fy + 6), 3)
            pygame.draw.line(surf, (74, 145, 148), (r.right - 6, fy), (r.right + 12, fy + 6), 3)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.scene = pygame.Surface((WIDTH, HEIGHT)).convert_alpha()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.big_font = pygame.font.SysFont("arial", 48, bold=True)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.running = True
        self.state = "play"
        self.level_index = 0
        self.camera_x = 0
        self.frame = 0
        self.shake_timer = 0
        self.shake_strength = 0
        self.stars = [(random.randint(0, WIDTH - 1), random.randint(0, HEIGHT // 2), random.randint(1, 2), random.random() * math.tau) for _ in range(75)]
        self.hill_layers = []
        self.reeds = []
        self.reset_level(0)

    def make_decor(self):
        self.hill_layers = []
        for li in range(3):
            rng = random.Random(9000 + li + self.level_index * 123)
            points = [(-140, HEIGHT)]
            x = -140
            base = 210 + li * 42
            while x < self.world_width + 220:
                points.append((x, base + rng.randint(-26, 32)))
                x += rng.randint(70, 130)
            points.extend([(self.world_width + 220, HEIGHT), (-140, HEIGHT)])
            self.hill_layers.append(points)

        rng = random.Random(3000 + self.level_index * 77)
        self.reeds = []
        for _ in range(140):
            wx = rng.randint(0, self.world_width)
            hy = rng.randint(HEIGHT - 110, HEIGHT - 12)
            h = rng.randint(10, 34)
            lean = rng.randint(-5, 5)
            self.reeds.append((wx, hy, h, lean))

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
        self.stage_clear_timer = 0
        self.camera_x = 0
        self.make_decor()

        if index == 2:
            self.boss = Boss(3330, 380)
            self.boss_gate = pygame.Rect(3260, 350, 20, 250)
            self.platforms.extend([
                pygame.Rect(3300, 500, 820, 100),
                pygame.Rect(3400, 400, 150, 200),
                pygame.Rect(3630, 320, 150, 280),
                pygame.Rect(3870, 410, 160, 190),
            ])
            self.exit = pygame.Rect(4020, 305, 70, 105)
            self.world_width = 4200
            self.make_decor()

    def rumble(self, strength=4, frames=8):
        self.shake_strength = max(self.shake_strength, strength)
        self.shake_timer = max(self.shake_timer, frames)

    def spawn_hit_particles(self, x, y, color, amount=8, glow=False):
        for _ in range(amount):
            self.particles.append(
                Particle(
                    x,
                    y,
                    random.uniform(-3, 3),
                    random.uniform(-4, -1),
                    random.randint(18, 30),
                    color,
                    random.randint(2, 4),
                    True,
                    glow,
                )
            )

    def spawn_foot_dust(self, x, y, color=(150, 175, 145)):
        for _ in range(3):
            self.particles.append(Particle(x, y, random.uniform(-1.5, 1.5), random.uniform(-2.5, -0.5), 18, color, random.randint(2, 3)))

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
                        self.spawn_hit_particles(self.player.rect.centerx, self.player.rect.bottom, CYAN, 6, glow=True)
                    if event.key in (pygame.K_j, pygame.K_k, pygame.K_LCTRL, pygame.K_RETURN):
                        self.player.attack_timer = ATTACK_TIME
                        ax = self.player.rect.centerx + self.player.facing * 18
                        ay = self.player.rect.centery
                        self.spawn_hit_particles(ax, ay, CYAN, 5, glow=True)

    def update(self):
        self.frame += 1
        if self.state != "play":
            return
        keys = pygame.key.get_pressed()
        was_grounded = self.player.on_ground
        prev_hp = self.player.hp
        self.player.update(keys, self.platforms + ([self.boss_gate] if self.boss_gate else []))

        if self.player.on_ground and not was_grounded:
            self.spawn_foot_dust(self.player.rect.centerx, self.player.rect.bottom)
            self.rumble(2, 5)
        elif self.player.on_ground and abs(self.player.vx) > 0.3 and self.frame % 9 == 0:
            self.spawn_foot_dust(self.player.rect.centerx - self.player.facing * 6, self.player.rect.bottom)

        if self.boss_gate and self.boss and self.boss.hp <= 0:
            self.platforms = [p for p in self.platforms if p != self.boss_gate]
            self.boss_gate = None

        for c in self.collectibles:
            if not c.collected:
                c.update()
                if self.player.rect.colliderect(c.rect):
                    c.collected = True
                    self.player.score += 1
                    self.player.hp = min(self.player.max_hp, self.player.hp + 1)
                    self.spawn_hit_particles(c.rect.centerx, c.rect.centery, GOLD, 14, glow=True)
                    self.rumble(2, 4)

        attack_rect = self.player.attack_rect()
        for enemy in list(self.enemies):
            solids = self.platforms + ([self.boss_gate] if self.boss_gate else [])
            if isinstance(enemy, Crawler):
                enemy.update(solids, self.player)
            else:
                enemy.update(solids, self.player, self.bullets)
            if self.player.rect.colliderect(enemy.rect):
                if self.player.take_hit(1):
                    self.rumble(5, 10)
                    self.spawn_hit_particles(self.player.rect.centerx, self.player.rect.centery, RED, 10, glow=True)
            if attack_rect and enemy.rect.colliderect(attack_rect):
                enemy.hp -= 1
                enemy.vx += 4 * self.player.facing
                self.spawn_hit_particles(enemy.rect.centerx, enemy.rect.centery, CYAN, 8, glow=True)
                self.rumble(2, 4)
            if enemy.hp <= 0:
                self.enemies.remove(enemy)
                self.spawn_hit_particles(enemy.rect.centerx, enemy.rect.centery, RED, 12, glow=True)
                self.rumble(4, 7)

        if self.boss and self.boss.hp > 0:
            solids = self.platforms + ([self.boss_gate] if self.boss_gate else [])
            old_phase = self.boss.phase
            self.boss.update(solids, self.player, self.bullets, self.enemies)
            if self.player.rect.colliderect(self.boss.rect):
                if self.player.take_hit(1):
                    self.rumble(6, 12)
                    self.spawn_hit_particles(self.player.rect.centerx, self.player.rect.centery, RED, 10, glow=True)
            if attack_rect and self.boss.rect.colliderect(attack_rect):
                self.boss.hp -= 1
                self.spawn_hit_particles(self.boss.rect.centerx, self.boss.rect.centery, CYAN, 10, glow=True)
                self.rumble(3, 5)
            if self.boss.phase != old_phase:
                self.rumble(7, 12)

        for b in list(self.bullets):
            b.rect.y += int(round(b.vy))
            b.update(self.platforms + ([self.boss_gate] if self.boss_gate else []))
            if b.rect.colliderect(self.player.rect):
                if self.player.take_hit(b.damage):
                    self.rumble(5, 10)
                    self.spawn_hit_particles(self.player.rect.centerx, self.player.rect.centery, RED, 10, glow=True)
                b.alive = False
            if b.rect.right < 0 or b.rect.left > self.world_width:
                b.alive = False
            if not b.alive:
                self.bullets.remove(b)
                self.spawn_hit_particles(b.rect.centerx, b.rect.centery, b.color, 4, glow=True)

        for hz in self.hazards:
            if self.player.rect.colliderect(hz):
                if self.player.take_hit(1):
                    self.rumble(4, 8)
                    self.spawn_hit_particles(self.player.rect.centerx, self.player.rect.bottom, (70, 180, 210), 12, glow=True)
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

        if self.player.hp < prev_hp:
            self.rumble(5, 10)

        if not self.player.alive:
            self.state = "gameover"

        if self.level_index < 2:
            if self.player.rect.colliderect(self.exit):
                self.stage_clear_timer += 1
                self.spawn_hit_particles(self.exit.centerx, self.exit.centery, PURPLE, 2, glow=True)
                if self.stage_clear_timer > 18:
                    self.reset_level(self.level_index + 1)
            else:
                self.stage_clear_timer = 0
        else:
            if self.boss and self.boss.hp <= 0 and self.player.rect.colliderect(self.exit):
                self.state = "win"

        self.camera_x = max(0, min(self.player.rect.centerx - WIDTH // 2, self.world_width - WIDTH))
        if self.shake_timer > 0:
            self.shake_timer -= 1
        else:
            self.shake_strength = 0

    def draw_background(self):
        surf = self.scene
        draw_vertical_gradient(surf, SKY_TOP, SKY_BOTTOM)

        moon_x = WIDTH - 95
        moon_y = 70
        circle_glow(surf, (140, 210, 230), (moon_x, moon_y), 38, 40)
        pygame.draw.circle(surf, (190, 220, 230), (moon_x, moon_y), 22)
        pygame.draw.circle(surf, (132, 162, 175), (moon_x + 7, moon_y - 4), 18)

        for sx, sy, rad, phase in self.stars:
            twinkle = 180 + int(75 * math.sin(self.frame * 0.03 + phase))
            pygame.draw.circle(surf, (twinkle, twinkle, min(255, twinkle + 10)), (sx, sy), rad)

        colors = [(19, 31, 44), (24, 45, 53), (34, 62, 62)]
        speeds = [0.14, 0.24, 0.36]
        for layer_idx, points in enumerate(self.hill_layers):
            pts = []
            for px, py in points:
                pts.append((px - self.camera_x * speeds[layer_idx], py))
            pygame.draw.polygon(surf, colors[layer_idx], pts)

        for i in range(3):
            fog = pygame.Surface((WIDTH, 90), pygame.SRCALPHA)
            for y in range(90):
                alpha = int((30 - i * 6) * (1 - y / 90))
                pygame.draw.line(fog, (155, 190, 195, alpha), (0, y), (WIDTH, y))
            surf.blit(fog, (0, 120 + i * 70 + math.sin(self.frame * 0.01 + i) * 8))

        # foreground marsh haze
        haze = pygame.Surface((WIDTH, 150), pygame.SRCALPHA)
        for y in range(150):
            a = int(68 * (y / 150))
            pygame.draw.line(haze, (85, 110, 110, a), (0, y), (WIDTH, y))
        surf.blit(haze, (0, HEIGHT - 150))

        for wx, hy, h, lean in self.reeds:
            sx = int(wx - self.camera_x * 0.85)
            if -12 <= sx <= WIDTH + 12:
                sway = math.sin(self.frame * 0.04 + wx * 0.02) * 2
                pygame.draw.line(surf, (42, 88, 61), (sx, hy), (sx + lean + sway, hy - h), 2)
                if h > 18:
                    pygame.draw.line(surf, (72, 122, 78), (sx + 1, hy - h // 2), (sx + lean + sway + 4, hy - h + 6), 1)

    def draw_platform(self, pr, idx):
        surf = self.scene
        seed = idx * 971 + self.level_index * 137
        rng = random.Random(seed)
        shadow = pygame.Rect(pr.x + 6, pr.y + 8, pr.w, pr.h)
        pygame.draw.rect(surf, (14, 24, 22), shadow, border_radius=10)
        pygame.draw.rect(surf, GROUND, pr, border_radius=10)
        inner = pygame.Rect(pr.x, pr.y + 12, pr.w, pr.h - 12)
        pygame.draw.rect(surf, MUD, inner, border_radius=10)
        top_h = min(16, max(10, pr.h // 6))
        pygame.draw.rect(surf, (78, 120, 72), (pr.x + 2, pr.y + 2, pr.w - 4, top_h), border_radius=9)
        pygame.draw.rect(surf, (112, 163, 92), (pr.x + 3, pr.y + 2, pr.w - 6, max(5, top_h // 2)), border_radius=8)

        for x in range(pr.x + 8, pr.right - 8, 9):
            grass_h = 4 + ((x + idx) % 5)
            pygame.draw.line(surf, (92, 155, 85), (x, pr.y + top_h - 1), (x - 2, pr.y + top_h - grass_h), 1)
            pygame.draw.line(surf, (145, 195, 120), (x + 1, pr.y + top_h - 1), (x + 1, pr.y + top_h - grass_h + 1), 1)

        crack_count = max(1, pr.w // 60)
        for _ in range(crack_count):
            cx = rng.randint(pr.x + 12, pr.right - 12)
            cy = rng.randint(pr.y + 18, min(pr.bottom - 8, pr.y + 40))
            pygame.draw.line(surf, (40, 58, 36), (cx, cy), (cx + rng.randint(-8, 8), cy + rng.randint(6, 18)), 2)
            pygame.draw.line(surf, (58, 76, 50), (cx, cy + 3), (cx + rng.randint(-6, 6), cy + rng.randint(10, 20)), 1)

        for x in range(pr.x + 14, pr.right - 14, 40):
            pygame.draw.line(surf, (32, 58, 37), (x, pr.y + 14), (x + 8, pr.y + 28), 2)

    def draw_hazard(self, hz):
        surf = self.scene
        r = hz.move(-self.camera_x, 0)
        pygame.draw.rect(surf, (12, 38, 50), r, border_radius=10)
        water = pygame.Rect(r.x, r.y + 4, r.w, r.h - 4)
        pygame.draw.rect(surf, WATER, water, border_radius=10)
        pygame.draw.rect(surf, (44, 118, 145), (water.x + 2, water.y + 2, water.w - 4, max(3, water.h // 2)), border_radius=8)
        for i in range(0, r.w + 20, 18):
            wave_y = r.y + 8 + math.sin(self.frame * 0.15 + i * 0.3 + hz.x * 0.01) * 3
            pygame.draw.arc(surf, (110, 205, 225), (r.x + i - 8, wave_y, 18, 10), 0, math.pi, 2)
        for i in range(max(1, r.w // 24)):
            bx = r.x + 8 + (i * 24 + self.frame * 2 + hz.x // 9) % max(12, r.w - 16)
            by = r.y + 12 + math.sin(self.frame * 0.08 + i + hz.x * 0.01) * 6
            pygame.draw.circle(surf, (170, 230, 240), (int(bx), int(by)), 1)

    def draw_exit(self):
        surf = self.scene
        ex = self.exit.move(-self.camera_x, 0)
        circle_glow(surf, (160, 120, 220), ex.center, 30, 28)
        pygame.draw.rect(surf, (68, 52, 96), ex, border_radius=10)
        pygame.draw.rect(surf, (150, 120, 210), (ex.x + 9, ex.y + 10, ex.w - 18, ex.h - 16), border_radius=8)
        for i in range(4):
            yy = ex.y + 14 + i * 16 + math.sin(self.frame * 0.1 + i) * 2
            pygame.draw.line(surf, (220, 215, 255), (ex.x + 12, yy), (ex.right - 12, yy), 1)
        pygame.draw.circle(surf, WHITE, ex.midtop, 10)

    def draw_gate(self):
        if not self.boss_gate:
            return
        surf = self.scene
        gate = self.boss_gate.move(-self.camera_x, 0)
        circle_glow(surf, (100, 220, 230), gate.center, 26, 30)
        pygame.draw.rect(surf, (18, 84, 90), gate, border_radius=4)
        pygame.draw.rect(surf, (90, 215, 220), (gate.x + 5, gate.y, gate.w - 10, gate.h), border_radius=3)
        for yy in range(gate.y, gate.bottom, 16):
            pygame.draw.line(surf, (170, 240, 245), (gate.x, yy), (gate.right, yy + 8), 2)

    def draw_world(self):
        for hz in self.hazards:
            self.draw_hazard(hz)

        for i, p in enumerate(self.platforms):
            pr = p.move(-self.camera_x, 0)
            self.draw_platform(pr, i)

        self.draw_gate()
        self.draw_exit()

        for c in self.collectibles:
            if not c.collected:
                c.draw(self.scene, self.camera_x)

        for e in self.enemies:
            e.draw(self.scene, self.camera_x)

        if self.boss and self.boss.hp > 0:
            self.boss.draw(self.scene, self.camera_x)

        for b in self.bullets:
            b.draw(self.scene, self.camera_x)

        self.player.draw(self.scene, self.camera_x)

        for p in self.particles:
            p.draw(self.scene, self.camera_x)

        # ambient cyan drizzle sparks
        for i in range(8):
            sx = (i * 121 + self.frame * 3) % (WIDTH + 40) - 20
            sy = (i * 43 + self.frame * 2) % HEIGHT
            pygame.draw.line(self.scene, (120, 180, 190, 40), (sx, sy), (sx - 6, sy + 10), 1)

    def draw_post_fx(self):
        vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(vignette, (0, 0, 0, 0), (0, 0, WIDTH, HEIGHT))
        for i in range(7):
            pad = i * 10
            alpha = 18
            pygame.draw.rect(vignette, (0, 0, 0, alpha), (pad, pad, WIDTH - pad * 2, HEIGHT - pad * 2), width=18)
        self.screen.blit(vignette, (0, 0))

        if self.player.invuln > 0:
            hurt = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(hurt, (120, 20, 20, 18), (0, 0, WIDTH, HEIGHT))
            self.screen.blit(hurt, (0, 0))

    def draw_ui(self):
        title = self.font.render(self.level_name, True, WHITE)
        self.screen.blit(title, (18, 14))
        for i in range(self.player.max_hp):
            x = 18 + i * 28
            box = pygame.Rect(x, 50, 22, 18)
            pygame.draw.rect(self.screen, (36, 30, 32), box, border_radius=6)
            if i < self.player.hp:
                pygame.draw.rect(self.screen, (210, 78, 78), box.inflate(-2, -2), border_radius=5)
                pygame.draw.rect(self.screen, (255, 145, 145), (x + 4, 53, 12, 5), border_radius=3)
            else:
                pygame.draw.rect(self.screen, (70, 45, 45), box.inflate(-2, -2), border_radius=5)
        relic = self.small_font.render(f"Relics: {self.player.score}", True, GOLD)
        self.screen.blit(relic, (18, 78))
        controls = self.small_font.render("Move: A/D  Jump: Space/W  Attack: J  Restart: R", True, WHITE)
        self.screen.blit(controls, (18, HEIGHT - 28))

        if self.boss and self.boss.hp > 0:
            outer = pygame.Rect(WIDTH // 2 - 170, 18, 340, 22)
            pygame.draw.rect(self.screen, (24, 28, 36), outer, border_radius=9)
            fill = int(336 * (self.boss.hp / self.boss.max_hp))
            pygame.draw.rect(self.screen, (70, 195, 215), (WIDTH // 2 - 168, 20, fill, 18), border_radius=8)
            pygame.draw.rect(self.screen, (170, 245, 255), (WIDTH // 2 - 168, 20, fill, 6), border_radius=6)
            txt = self.small_font.render("Boss: The Mire King", True, WHITE)
            self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 44))

        if self.state == "gameover":
            msg = self.big_font.render("Sirenscale Fell", True, WHITE)
            self.screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 206))
            sub = self.font.render("Press R to restart", True, WHITE)
            self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 266))
        elif self.state == "win":
            msg = self.big_font.render("The Mire King Is Broken", True, WHITE)
            self.screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 196))
            sub = self.font.render(f"Relics gathered: {self.player.score}   Press R to play again", True, GOLD)
            self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 260))

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.scene.fill((0, 0, 0, 0))
            self.draw_background()
            self.draw_world()
            sx = random.randint(-self.shake_strength, self.shake_strength) if self.shake_timer > 0 else 0
            sy = random.randint(-self.shake_strength, self.shake_strength) if self.shake_timer > 0 else 0
            self.screen.fill(BLACK)
            self.screen.blit(self.scene, (sx, sy))
            self.draw_post_fx()
            self.draw_ui()
            pygame.display.flip()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
