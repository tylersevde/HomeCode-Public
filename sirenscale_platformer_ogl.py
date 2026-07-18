import math
import random
import sys
from dataclasses import dataclass

import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_LINE_LOOP,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POLYGON,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLE_FAN,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glViewport,
)


WIDTH, HEIGHT = 800, 480
FPS = 60
TITLE = "Sirenscale: Marsh of Teeth (PyOpenGL)"

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
SKY = (12, 20, 36)

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


class GLRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.font_cache = {}
        self.text_cache = {}
        self.configure_viewport(width, height)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_TEXTURE_2D)

    def configure_viewport(self, width: int, height: int):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def begin(self):
        glClearColor(*(c / 255.0 for c in SKY), 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

    @staticmethod
    def color(color, alpha=255):
        glColor4f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha / 255.0)

    def rect(self, x, y, w, h, color, alpha=255):
        self.color(color, alpha)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def rect_gradient(self, x, y, w, h, top_color, bottom_color, alpha=255):
        glBegin(GL_QUADS)
        self.color(top_color, alpha)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        self.color(bottom_color, alpha)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def line(self, x1, y1, x2, y2, color, width=1, alpha=255):
        glLineWidth(width)
        self.color(color, alpha)
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()
        glLineWidth(1)

    def poly(self, points, color, alpha=255):
        self.color(color, alpha)
        glBegin(GL_POLYGON)
        for x, y in points:
            glVertex2f(x, y)
        glEnd()

    def circle(self, cx, cy, radius, color, alpha=255, segments=28):
        self.color(color, alpha)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(segments + 1):
            angle = math.tau * i / segments
            glVertex2f(cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
        glEnd()

    def circle_outline(self, cx, cy, radius, color, width=2, alpha=255, segments=32):
        glLineWidth(width)
        self.color(color, alpha)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = math.tau * i / segments
            glVertex2f(cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
        glEnd()
        glLineWidth(1)

    def ellipse(self, x, y, w, h, color, alpha=255, segments=32):
        cx = x + w / 2
        cy = y + h / 2
        rx = w / 2
        ry = h / 2
        self.color(color, alpha)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(segments + 1):
            angle = math.tau * i / segments
            glVertex2f(cx + math.cos(angle) * rx, cy + math.sin(angle) * ry)
        glEnd()

    def ellipse_outline(self, x, y, w, h, color, width=2, alpha=255, segments=32):
        cx = x + w / 2
        cy = y + h / 2
        rx = w / 2
        ry = h / 2
        glLineWidth(width)
        self.color(color, alpha)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = math.tau * i / segments
            glVertex2f(cx + math.cos(angle) * rx, cy + math.sin(angle) * ry)
        glEnd()
        glLineWidth(1)

    def arc(self, x, y, w, h, start, end, color, width=2, alpha=255, segments=28):
        cx = x + w / 2
        cy = y + h / 2
        rx = w / 2
        ry = h / 2
        glLineWidth(width)
        self.color(color, alpha)
        glBegin(GL_LINE_STRIP)
        for i in range(segments + 1):
            t = i / segments
            angle = start + (end - start) * t
            glVertex2f(cx + math.cos(angle) * rx, cy + math.sin(angle) * ry)
        glEnd()
        glLineWidth(1)

    def glow_circle(self, cx, cy, radius, color, strength=120, layers=6):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for i in range(layers, 0, -1):
            alpha = int(strength * (i / layers) ** 2)
            self.circle(cx, cy, radius * (0.35 + i / layers), color, alpha=alpha)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def shadow(self, x, y, w, h, alpha=70):
        self.ellipse(x, y, w, h, BLACK, alpha=alpha, segments=28)

    def get_font(self, size=24, bold=False):
        key = (size, bold)
        if key not in self.font_cache:
            self.font_cache[key] = pygame.font.SysFont("arial", size, bold=bold)
        return self.font_cache[key]

    def get_text_texture(self, text, size, color, bold=False):
        key = (text, size, color, bold)
        cached = self.text_cache.get(key)
        if cached is not None:
            return cached
        font = self.get_font(size=size, bold=bold)
        surf = font.render(text, True, color)
        w, h = surf.get_size()
        data = pygame.image.tostring(surf, "RGBA", True)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 9729)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 9729)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        payload = (tex, w, h)
        self.text_cache[key] = payload
        return payload

    def draw_text(self, x, y, text, size=24, color=WHITE, bold=False, centered=False):
        tex, w, h = self.get_text_texture(text, size, color, bold)
        if centered:
            x -= w / 2
            y -= h / 2
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        self.color((255, 255, 255), 255)
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
        return w, h

    def shutdown(self):
        for tex, _, _ in self.text_cache.values():
            glDeleteTextures(int(tex))
        self.text_cache.clear()


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

    def draw(self, renderer: GLRenderer, camera_x: float, offset_x=0.0, offset_y=0.0):
        if self.life <= 0:
            return
        px = self.x - camera_x + offset_x
        py = self.y + offset_y
        alpha = clamp(self.life * 10, 0, 255) if self.fade else 255
        if self.glow:
            renderer.glow_circle(px, py, self.size * 2.4, self.color, strength=max(30, alpha // 3), layers=4)
        renderer.circle(px, py, max(1, self.size), self.color, alpha=alpha)


class Bullet:
    def __init__(self, x, y, vx, color=CYAN, damage=1, owner="enemy"):
        self.rect = pygame.Rect(x, y, 10, 6)
        self.vx = vx
        self.vy = 0.0
        self.color = color
        self.damage = damage
        self.owner = owner
        self.alive = True
        self.trail = []

    def update(self, solids):
        self.trail.append((self.rect.centerx, self.rect.centery))
        if len(self.trail) > 5:
            self.trail.pop(0)
        self.rect.x += int(self.vx)
        for solid in solids:
            if self.rect.colliderect(solid):
                self.alive = False
                break

    def draw(self, renderer: GLRenderer, camera_x: float, offset_x=0.0, offset_y=0.0):
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(90 * (i + 1) / max(1, len(self.trail)))
            renderer.rect(tx - camera_x - 4 + offset_x, ty - 2 + offset_y, 8, 4, self.color, alpha)
        renderer.glow_circle(self.rect.centerx - camera_x + offset_x, self.rect.centery + offset_y, 6, self.color, strength=55, layers=3)
        renderer.rect(self.rect.x - camera_x + offset_x, self.rect.y + offset_y, self.rect.w, self.rect.h, self.color)


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
        self.vy += GRAVITY
        self.rect.x += int(self.vx)
        for solid in solids:
            if self.rect.colliderect(solid):
                if self.vx > 0:
                    self.rect.right = solid.left
                elif self.vx < 0:
                    self.rect.left = solid.right
                self.vx = 0
        self.rect.y += int(self.vy)
        self.on_ground = False
        for solid in solids:
            if self.rect.colliderect(solid):
                if self.vy > 0:
                    self.rect.bottom = solid.top
                    self.on_ground = True
                elif self.vy < 0:
                    self.rect.top = solid.bottom
                self.vy = 0


class Player(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, 28, 52)
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

    def draw(self, renderer: GLRenderer, camera_x: float, offset_x=0.0, offset_y=0.0):
        x = self.rect.x - camera_x + offset_x
        y = self.rect.y + offset_y
        flicker = self.invuln > 0 and (self.invuln // 3) % 2 == 0
        if flicker:
            return
        renderer.shadow(x + 1, y + self.rect.h - 8, self.rect.w + 8, 10, alpha=70)
        renderer.glow_circle(x + self.rect.w / 2, y + self.rect.h / 2, 24, TEAL, strength=22, layers=3)
        renderer.rect(x, y + 10, self.rect.w, self.rect.h - 10, (66, 149, 124))
        renderer.rect(x + 4, y + 15, self.rect.w - 8, 14, (96, 195, 168))
        renderer.rect(x - 4, y + 4, self.rect.w + 8, 16, (110, 79, 46))
        eye_offset = 6 if self.facing > 0 else 2
        renderer.circle(x + 10 + eye_offset, y + 17, 4, WHITE)
        renderer.circle(x + 10.8 + eye_offset, y + 17, 1.8, BLACK)
        renderer.circle(x + 20 + eye_offset, y + 17, 4, WHITE)
        renderer.circle(x + 20.8 + eye_offset, y + 17, 1.8, BLACK)
        for i in range(3):
            renderer.arc(x - 4 - i * 2, y + 10 + i * 4, self.rect.w + 8 + i * 4, 28,
                         0.2, 2.1, (90, 220 - i * 30, 180 + i * 10), width=2, alpha=200)
        if self.attack_timer > 0:
            ar = self.attack_rect()
            if ar:
                ar = ar.move(-camera_x + int(offset_x), int(offset_y))
                renderer.glow_circle(ar.centerx, ar.centery, max(ar.w, ar.h) * 0.6, CYAN, strength=80, layers=4)
                renderer.rect(ar.x, ar.y, ar.w, ar.h, (120, 220, 255), alpha=50)
                renderer.line(ar.left, ar.centery, ar.right, ar.centery, WHITE, width=2)


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

    def draw(self, renderer: GLRenderer, camera_x: float, offset_x=0.0, offset_y=0.0):
        r = self.rect.move(-int(camera_x) + int(offset_x), int(offset_y))
        renderer.shadow(r.x + 2, r.bottom - 4, r.w, 8, alpha=60)
        renderer.ellipse(r.x, r.y, r.w, r.h, (121, 153, 83))
        renderer.ellipse(r.x + 5, r.y + 4, r.w - 10, r.h - 8, (82, 110, 58))
        renderer.circle(r.centerx + (5 if self.facing > 0 else -5), r.y + 10, 3, RED)


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

    def draw(self, renderer: GLRenderer, camera_x: float, offset_x=0.0, offset_y=0.0):
        r = self.rect.move(-int(camera_x) + int(offset_x), int(offset_y))
        renderer.shadow(r.x + 2, r.bottom - 4, r.w, 8, alpha=55)
        renderer.glow_circle(r.centerx, r.centery, 18, PURPLE, strength=18, layers=3)
        renderer.rect(r.x, r.y, r.w, r.h, (122, 87, 156))
        renderer.rect(r.x + 5, r.y + 6, r.w - 10, 12, (180, 145, 200))
        mouth_x = r.centerx + (-9 if self.facing < 0 else -1)
        renderer.ellipse(mouth_x, r.y + 20, 18, 8, (230, 130, 170))


class Collectible:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 18, 18)
        self.angle = random.random() * math.tau
        self.collected = False

    def update(self):
        self.angle += 0.08

    def draw(self, renderer: GLRenderer, camera_x: float, offset_x=0.0, offset_y=0.0):
        x = self.rect.centerx - camera_x + offset_x
        y = self.rect.centery + math.sin(self.angle) * 6 + offset_y
        renderer.glow_circle(x, y, 18, GOLD, strength=95, layers=5)
        points = [
            (x, y - 10),
            (x + 7, y - 2),
            (x + 11, y + 10),
            (x, y + 4),
            (x - 11, y + 10),
            (x - 7, y - 2),
        ]
        renderer.poly(points, GOLD)
        renderer.line(points[0][0], points[0][1], points[2][0], points[2][1], WHITE, width=2, alpha=180)
        renderer.line(points[0][0], points[0][1], points[4][0], points[4][1], WHITE, width=2, alpha=180)


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

        self.vx = 1.8 * self.facing if dist > 120 else 0

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

    def draw(self, renderer: GLRenderer, camera_x: float, offset_x=0.0, offset_y=0.0):
        r = self.rect.move(-int(camera_x) + int(offset_x), int(offset_y))
        renderer.shadow(r.x + 10, r.bottom - 10, r.w - 12, 16, alpha=80)
        renderer.glow_circle(r.centerx, r.centery, 55, (80, 200, 210), strength=35, layers=4)
        renderer.rect(r.x, r.y, r.w, r.h, (84, 126, 145))
        renderer.rect(r.x + 12, r.y + 18, r.w - 24, 24, (130, 196, 204))
        renderer.rect(r.x - 12, r.y + 8, r.w + 24, 20, (75, 61, 45))
        eye_shift = 12 if self.facing > 0 else 0
        renderer.circle(r.x + 36 + eye_shift, r.y + 40, 10, WHITE)
        renderer.circle(r.x + 72 + eye_shift, r.y + 40, 10, WHITE)
        renderer.circle(r.x + 39 + eye_shift, r.y + 40, 4.5, BLACK)
        renderer.circle(r.x + 75 + eye_shift, r.y + 40, 4.5, BLACK)
        renderer.arc(r.x + 25, r.y + 50, 70, 28, 0.2, 2.9, RED, width=4)
        for i in range(6):
            renderer.line(r.x + 12 + i * 18, r.y + 88, r.x + 25 + i * 18, r.y + 106, (180, 250, 250), width=3)


class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption(TITLE)
        pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
        self.clock = pygame.time.Clock()
        self.renderer = GLRenderer(WIDTH, HEIGHT)
        self.running = True
        self.state = "play"
        self.level_index = 0
        self.camera_x = 0
        self.shake_timer = 0
        self.shake_strength = 0
        self.water_phase = 0.0
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
        self.enemies = [Crawler(x, y) if kind == "crawler" else Spitter(x, y) for kind, x, y in level["enemies"]]
        self.collectibles = [Collectible(x, y) for x, y in level["collectibles"]]
        self.bullets = []
        self.particles = []
        self.boss = None
        self.boss_gate = None
        self.stage_clear_timer = 0
        self.camera_x = 0

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
                    glow=glow,
                )
            )

    def add_shake(self, strength=5, duration=10):
        self.shake_strength = max(self.shake_strength, strength)
        self.shake_timer = max(self.shake_timer, duration)

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
                        self.spawn_hit_particles(self.player.rect.centerx, self.player.rect.bottom, TEAL, 8, glow=True)
                    if event.key in (pygame.K_j, pygame.K_k, pygame.K_LCTRL, pygame.K_RETURN):
                        self.player.attack_timer = ATTACK_TIME
                        self.add_shake(2, 5)

    def update(self):
        if self.state != "play":
            return
        keys = pygame.key.get_pressed()
        solids = self.platforms + ([self.boss_gate] if self.boss_gate else [])
        self.player.update(keys, solids)
        self.water_phase += 0.09

        if self.boss_gate and self.boss and self.boss.hp <= 0:
            self.platforms = [p for p in self.platforms if p != self.boss_gate]
            self.boss_gate = None

        if self.player.on_ground and abs(self.player.vx) > 0.1 and random.random() < 0.25:
            self.particles.append(Particle(self.player.rect.centerx, self.player.rect.bottom - 2,
                                           random.uniform(-0.8, 0.8), random.uniform(-1.5, -0.4),
                                           12, MUD, 2))

        for collectible in self.collectibles:
            if not collectible.collected:
                collectible.update()
                if self.player.rect.colliderect(collectible.rect):
                    collectible.collected = True
                    self.player.score += 1
                    self.player.hp = min(self.player.max_hp, self.player.hp + 1)
                    self.spawn_hit_particles(collectible.rect.centerx, collectible.rect.centery, GOLD, 14, glow=True)

        attack_rect = self.player.attack_rect()
        for enemy in list(self.enemies):
            if isinstance(enemy, Crawler):
                enemy.update(solids, self.player)
            else:
                enemy.update(solids, self.player, self.bullets)
            if self.player.rect.colliderect(enemy.rect):
                self.player.take_hit(1)
                self.add_shake(4, 8)
            if attack_rect and enemy.rect.colliderect(attack_rect):
                enemy.hp -= 1
                enemy.vx += 4 * self.player.facing
                self.spawn_hit_particles(enemy.rect.centerx, enemy.rect.centery, CYAN, 8, glow=True)
                self.add_shake(3, 6)
            if enemy.hp <= 0:
                self.enemies.remove(enemy)
                self.spawn_hit_particles(enemy.rect.centerx, enemy.rect.centery, RED, 12, glow=True)

        if self.boss and self.boss.hp > 0:
            self.boss.update(solids, self.player, self.bullets, self.enemies)
            if self.player.rect.colliderect(self.boss.rect):
                self.player.take_hit(1)
                self.add_shake(6, 10)
            if attack_rect and self.boss.rect.colliderect(attack_rect):
                self.boss.hp -= 1
                self.spawn_hit_particles(self.boss.rect.centerx, self.boss.rect.centery, CYAN, 10, glow=True)
                self.add_shake(5, 8)

        for bullet in list(self.bullets):
            bullet.rect.y += int(round(bullet.vy))
            bullet.update(solids)
            if bullet.rect.colliderect(self.player.rect):
                self.player.take_hit(bullet.damage)
                bullet.alive = False
                self.add_shake(4, 8)
            if bullet.rect.right < 0 or bullet.rect.left > self.world_width:
                bullet.alive = False
            if not bullet.alive:
                self.bullets.remove(bullet)
                self.spawn_hit_particles(bullet.rect.centerx, bullet.rect.centery, bullet.color, 4, glow=True)

        for hazard in self.hazards:
            if self.player.rect.colliderect(hazard):
                self.player.take_hit(1)
                self.player.rect.y -= 40
                self.player.vy = -6
                self.add_shake(5, 9)
            for enemy in self.enemies:
                if enemy.rect.colliderect(hazard):
                    enemy.hp = 0
            if self.boss and self.boss.hp > 0 and self.boss.rect.colliderect(hazard):
                self.boss.hp -= 1

        for particle in list(self.particles):
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)

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
        if self.shake_timer > 0:
            self.shake_timer -= 1
            self.shake_strength *= 0.88
        else:
            self.shake_strength = 0

    def draw_background(self, ox=0.0, oy=0.0):
        r = self.renderer
        r.rect_gradient(0 + ox, 0 + oy, WIDTH, HEIGHT, SKY_TOP, SKY_BOTTOM)

        for i in range(6):
            y = 80 + i * 42 + oy
            speed = 0.12 + i * 0.03
            offset = (self.camera_x * speed) % (WIDTH + 260)
            color = (26 + i * 6, 46 + i * 7, 64 + i * 7)
            r.ellipse(-140 - offset + ox, y, 420, 90, color, alpha=175)
            r.ellipse(170 - offset + ox, y + 10, 500, 110, color, alpha=175)
            r.ellipse(620 - offset + ox, y + 5, 380, 88, color, alpha=175)

        for i in range(18):
            px = ((i * 173) - self.camera_x * 0.22) % (WIDTH + 80) - 40
            py = 40 + (i * 37) % 180
            r.glow_circle(px + ox, py + oy, 2 + (i % 3), (110, 180, 210), strength=40, layers=2)

        r.rect_gradient(0 + ox, HEIGHT - 140 + oy, WIDTH, 140, MIST, (25, 40, 54), alpha=220)
        for x in range(0, WIDTH + 100, 55):
            r.line(x + ox, HEIGHT - 120 + oy, x + 20 + ox, HEIGHT + oy, (55, 90, 105), width=2, alpha=120)

    def draw_hazard(self, rect, ox=0.0, oy=0.0):
        r = self.renderer
        x = rect.x - self.camera_x + ox
        y = rect.y + oy
        r.rect_gradient(x, y, rect.w, rect.h, (38, 104, 138), WATER)
        for i in range(max(2, rect.w // 18)):
            wave_x = x + i * 18
            wave_y = y + 8 + math.sin(self.water_phase + i * 0.7) * 3
            r.arc(wave_x, wave_y, 18, 10, 0.0, math.pi, (92, 196, 225), width=2, alpha=210)
        r.glow_circle(x + rect.w / 2, y + rect.h / 2, max(rect.w * 0.25, 10), (50, 180, 220), strength=16, layers=2)

    def draw_platform(self, rect, ox=0.0, oy=0.0):
        r = self.renderer
        x = rect.x - self.camera_x + ox
        y = rect.y + oy
        r.rect_gradient(x, y, rect.w, rect.h, GROUND, MUD)
        r.rect(x, y, rect.w, 12, (88, 136, 92), alpha=220)
        for i in range(max(1, rect.w // 32)):
            xx = x + i * 32 + 8
            r.line(xx, y + 10, xx + 7, y + 24, (32, 58, 37), width=2, alpha=180)
            if i % 2 == 0:
                r.line(xx + 4, y + 10, xx + 1, y + 4, (96, 160, 102), width=2, alpha=170)
        for i in range(max(1, rect.h // 28)):
            yy = y + 18 + i * 28
            r.line(x + 8, yy, x + rect.w - 10, yy, (74, 88, 64), width=1, alpha=70)

    def draw_gate(self, rect, ox=0.0, oy=0.0):
        r = self.renderer
        x = rect.x - self.camera_x + ox
        y = rect.y + oy
        r.rect(x, y, rect.w, rect.h, (28, 104, 110), alpha=180)
        for yy in range(int(y), int(y + rect.h), 16):
            r.line(x, yy, x + rect.w, yy + 8, (120, 230, 240), width=2, alpha=170)
        r.glow_circle(x + rect.w / 2, y + rect.h / 2, 18, CYAN, strength=25, layers=3)

    def draw_exit(self, rect, ox=0.0, oy=0.0):
        r = self.renderer
        x = rect.x - self.camera_x + ox
        y = rect.y + oy
        r.glow_circle(x + rect.w / 2, y + rect.h / 2, 40, (136, 96, 220), strength=36, layers=4)
        r.rect(x, y, rect.w, rect.h, (82, 62, 115), alpha=220)
        r.rect(x + 10, y + 10, rect.w - 20, rect.h - 20, (160, 120, 220), alpha=210)
        r.circle(x + rect.w / 2, y, 10, WHITE)

    def draw_world(self, ox=0.0, oy=0.0):
        for hazard in self.hazards:
            self.draw_hazard(hazard, ox, oy)
        for platform in self.platforms:
            self.draw_platform(platform, ox, oy)
        if self.boss_gate:
            self.draw_gate(self.boss_gate, ox, oy)

        self.draw_exit(self.exit, ox, oy)

        for collectible in self.collectibles:
            if not collectible.collected:
                collectible.draw(self.renderer, self.camera_x, ox, oy)
        for enemy in self.enemies:
            enemy.draw(self.renderer, self.camera_x, ox, oy)
        if self.boss and self.boss.hp > 0:
            self.boss.draw(self.renderer, self.camera_x, ox, oy)
        for bullet in self.bullets:
            bullet.draw(self.renderer, self.camera_x, ox, oy)
        self.player.draw(self.renderer, self.camera_x, ox, oy)
        for particle in self.particles:
            particle.draw(self.renderer, self.camera_x, ox, oy)

    def draw_ui(self):
        r = self.renderer
        r.draw_text(18, 14, self.level_name, size=24, color=WHITE)
        for i in range(self.player.max_hp):
            color = (210, 70, 70) if i < self.player.hp else (70, 45, 45)
            r.rect(18 + i * 28, 50, 22, 18, color)
        r.draw_text(18, 78, f"Relics: {self.player.score}", size=18, color=GOLD)
        r.draw_text(18, HEIGHT - 28, "Move: A/D  Jump: Space/W  Attack: J  Restart: R", size=18, color=WHITE)

        if self.boss and self.boss.hp > 0:
            r.rect(WIDTH // 2 - 170, 18, 340, 22, (40, 40, 55), alpha=220)
            fill = int(336 * (self.boss.hp / self.boss.max_hp))
            r.rect_gradient(WIDTH // 2 - 168, 20, fill, 18, (70, 195, 215), (110, 235, 245))
            r.draw_text(WIDTH // 2, 44, "Boss: The Mire King", size=18, color=WHITE, centered=True)

        if self.state == "gameover":
            r.glow_circle(WIDTH / 2, 228, 120, RED, strength=30, layers=4)
            r.draw_text(WIDTH // 2, 210, "Sirenscale Fell", size=48, color=WHITE, bold=True, centered=True)
            r.draw_text(WIDTH // 2, 270, "Press R to restart", size=24, color=WHITE, centered=True)
        elif self.state == "win":
            r.glow_circle(WIDTH / 2, 220, 140, CYAN, strength=34, layers=4)
            r.draw_text(WIDTH // 2, 200, "The Mire King Is Broken", size=48, color=WHITE, bold=True, centered=True)
            r.draw_text(WIDTH // 2, 264, f"Relics gathered: {self.player.score}   Press R to play again", size=24, color=GOLD, centered=True)

    def draw(self):
        self.renderer.begin()
        ox = random.uniform(-self.shake_strength, self.shake_strength) if self.shake_timer > 0 else 0.0
        oy = random.uniform(-self.shake_strength * 0.5, self.shake_strength * 0.5) if self.shake_timer > 0 else 0.0
        self.draw_background(ox, oy)
        self.draw_world(ox, oy)
        self.draw_ui()
        pygame.display.flip()

    def run(self):
        try:
            while self.running:
                self.clock.tick(FPS)
                self.handle_events()
                self.update()
                self.draw()
        finally:
            self.renderer.shutdown()
            pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
