import math
import random
import sys
from collections import deque

try:
    import pygame
    from pygame.math import Vector2
    from pygame.locals import DOUBLEBUF, OPENGL
except Exception as exc:
    raise SystemExit(
        "This game requires pygame. Install it with: pip install pygame"
    ) from exc

try:
    from OpenGL.GL import (
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_LINE_LOOP,
        GL_LINE_STRIP,
        GL_LINEAR,
        GL_MODELVIEW,
        GL_ONE,
        GL_ONE_MINUS_SRC_ALPHA,
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
except Exception as exc:
    raise SystemExit(
        "This game requires PyOpenGL. Install it with: pip install PyOpenGL PyOpenGL_accelerate"
    ) from exc

# ------------------------------------------------------------
# Shadow Labyrinth OpenGL
# A PyOpenGL rewrite of the top-down labyrinth crawler.
# ------------------------------------------------------------

SCREEN_W = 800
SCREEN_H = 480
FPS = 60
TILE = 48
MAZE_W = 61
MAZE_H = 61
WORLD_W = MAZE_W * TILE
WORLD_H = MAZE_H * TILE

BG = (0.035, 0.035, 0.055, 1.0)
WHITE = (0.96, 0.96, 0.98, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
PLAYER_BODY = (0.41, 0.71, 1.00, 1.0)
PLAYER_TRIM = (0.86, 0.92, 1.0, 1.0)
PLAYER_CLOAK = (0.13, 0.18, 0.24, 1.0)
SHIELD_C = (0.34, 0.82, 0.95, 1.0)
SWORD_C = (0.93, 0.95, 1.0, 1.0)
FLOOR_A = (0.11, 0.115, 0.15, 1.0)
FLOOR_B = (0.135, 0.14, 0.18, 1.0)
WALL_A = (0.24, 0.25, 0.31, 1.0)
WALL_B = (0.30, 0.31, 0.39, 1.0)
WALL_EDGE = (0.46, 0.48, 0.60, 1.0)
SHADE_C = (0.22, 0.08, 0.30, 1.0)
STALKER_C = (0.13, 0.13, 0.20, 1.0)
CULTIST_C = (0.28, 0.10, 0.36, 1.0)
BRUTE_C = (0.36, 0.11, 0.18, 1.0)
BOSS_C = (0.50, 0.12, 0.16, 1.0)
BOSS_CORE = (1.00, 0.48, 0.58, 1.0)
DOOR_C = (0.35, 0.74, 0.88, 1.0)
KEY_C = (0.55, 0.96, 1.00, 1.0)
FLASK_C = (0.30, 0.88, 0.55, 1.0)
LOOT_C = (1.00, 0.85, 0.36, 1.0)
RED_C = (0.90, 0.26, 0.26, 1.0)
GREEN_C = (0.35, 0.90, 0.50, 1.0)
PURPLE_C = (0.72, 0.44, 0.98, 1.0)


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize(vec):
    if vec.length_squared() == 0:
        return Vector2()
    return vec.normalize()


def angle_vec(angle):
    return Vector2(math.cos(angle), math.sin(angle))


def hash_xy(x, y):
    value = (x * 92837111) ^ (y * 689287499)
    value ^= value >> 13
    return value & 0x7FFFFFFF


def circle_rect_collision(cx, cy, radius, rx, ry, rw, rh):
    closest_x = clamp(cx, rx, rx + rw)
    closest_y = clamp(cy, ry, ry + rh)
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy < radius * radius


class GLText:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 34, bold=True)
        self.cache = {}

    def _get_tex(self, text, color, font):
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
        return self.cache[key]

    def draw(self, text, x, y, color=(245, 245, 245), size="normal", center=False):
        font = self.font if size == "normal" else self.small if size == "small" else self.big
        tex, w, h = self._get_tex(text, color, font)
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


TEXT = None


def setup_gl():
    glViewport(0, 0, SCREEN_W, SCREEN_H)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, SCREEN_W, SCREEN_H, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(*BG)


# -------------------------------
# OpenGL drawing helpers
# -------------------------------

def color4(c):
    glColor4f(*c)


def draw_quad(x, y, w, h, color):
    color4(color)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_quad_gradient(x, y, w, h, c1, c2):
    glBegin(GL_QUADS)
    glColor4f(*c1)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glColor4f(*c2)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_outline(x, y, w, h, color, width=1.0):
    glLineWidth(width)
    color4(color)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_circle(x, y, r, color, segments=24):
    color4(color)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(segments + 1):
        ang = math.tau * i / segments
        glVertex2f(x + math.cos(ang) * r, y + math.sin(ang) * r)
    glEnd()


def draw_ring(x, y, r, color, width=1.5, segments=28):
    glLineWidth(width)
    color4(color)
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        ang = math.tau * i / segments
        glVertex2f(x + math.cos(ang) * r, y + math.sin(ang) * r)
    glEnd()


def draw_arc(x, y, r, start_ang, end_ang, color, width=2.0, segments=18):
    glLineWidth(width)
    color4(color)
    glBegin(GL_LINE_STRIP)
    for i in range(segments + 1):
        t = i / max(1, segments)
        ang = start_ang + (end_ang - start_ang) * t
        glVertex2f(x + math.cos(ang) * r, y + math.sin(ang) * r)
    glEnd()


def draw_line(x1, y1, x2, y2, color, width=1.0):
    glLineWidth(width)
    color4(color)
    glBegin(GL_LINE_STRIP)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def draw_glow(x, y, r, color, alpha=0.2):
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    steps = 5
    for i in range(steps, 0, -1):
        rr = r * i / steps
        a = alpha * (i / steps) * 0.7
        draw_circle(x, y, rr, (color[0], color[1], color[2], a), segments=28)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def draw_vignette():
    draw_quad_gradient(0, 0, SCREEN_W, SCREEN_H, (0, 0, 0, 0.0), (0, 0, 0, 0.22))
    border = 72
    draw_quad(0, 0, SCREEN_W, border, (0, 0, 0, 0.20))
    draw_quad(0, SCREEN_H - border, SCREEN_W, border, (0, 0, 0, 0.20))
    draw_quad(0, 0, border, SCREEN_H, (0, 0, 0, 0.18))
    draw_quad(SCREEN_W - border, 0, border, SCREEN_H, (0, 0, 0, 0.18))


class Camera:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

    def update(self, target, shake):
        self.x = clamp(target.x - SCREEN_W / 2 + shake.x, 0, WORLD_W - SCREEN_W)
        self.y = clamp(target.y - SCREEN_H / 2 + shake.y, 0, WORLD_H - SCREEN_H)

    def world_to_screen(self, pos):
        return pos[0] - self.x, pos[1] - self.y


class FloatingText:
    def __init__(self, x, y, text, color=(245, 245, 245), life=0.9):
        self.pos = Vector2(x, y)
        self.text = text
        self.color = color
        self.life = life
        self.max_life = life

    def update(self, dt):
        self.life -= dt
        self.pos.y -= 32 * dt

    def draw(self, camera):
        if self.life <= 0:
            return
        px, py = camera.world_to_screen(self.pos)
        alpha = int(255 * max(0.0, self.life / self.max_life))
        col = tuple(int(c * 255) for c in self.color[:3])
        TEXT.draw(self.text, px, py, color=(*col,), size="small", center=True)
        draw_quad(px - 4, py - 2, 8, 2, (0, 0, 0, alpha / 255 * 0.18))


class Particle:
    def __init__(self, pos, vel, color, radius=4, life=0.5, drag=0.9, glow=0):
        self.pos = Vector2(pos)
        self.vel = Vector2(vel)
        self.color = color
        self.radius = radius
        self.life = life
        self.max_life = life
        self.drag = drag
        self.glow = glow

    def update(self, dt):
        self.life -= dt
        self.pos += self.vel * dt
        self.vel *= self.drag ** (dt * 60)

    def draw(self, camera):
        if self.life <= 0:
            return
        px, py = camera.world_to_screen(self.pos)
        a = max(0.0, self.life / self.max_life)
        if self.glow > 0:
            draw_glow(px, py, self.glow, self.color, 0.16 * a)
        draw_circle(px, py, max(1.0, self.radius * a), (self.color[0], self.color[1], self.color[2], a), 18)


class SlashEffect:
    def __init__(self, pos, facing):
        self.pos = Vector2(pos)
        self.angle = math.atan2(facing.y, facing.x)
        self.life = 0.15
        self.max_life = self.life

    def update(self, dt):
        self.life -= dt

    def draw(self, camera):
        if self.life <= 0:
            return
        px, py = camera.world_to_screen(self.pos)
        t = 1.0 - self.life / self.max_life
        sweep = 0.55 + t * 0.12
        alpha = 0.85 * (self.life / self.max_life)
        draw_arc(px, py, 54, self.angle - sweep, self.angle + sweep, (0.72, 0.86, 1.0, alpha), width=4.0)
        draw_arc(px, py, 43, self.angle - sweep * 0.8, self.angle + sweep * 0.8, (1.0, 1.0, 1.0, alpha * 0.55), width=2.0)


class Pickup:
    def __init__(self, x, y, kind, amount=1):
        self.pos = Vector2(x, y)
        self.kind = kind
        self.amount = amount
        self.radius = 12
        self.alive = True
        self.bob = random.random() * math.tau

    def update(self, dt):
        self.bob += dt * 3.2

    def collect(self, player, texts):
        if self.kind == "key":
            player.keys += self.amount
            texts.append(FloatingText(self.pos.x, self.pos.y - 18, f"+{self.amount} key", KEY_C))
        elif self.kind == "flask":
            player.flasks += self.amount
            texts.append(FloatingText(self.pos.x, self.pos.y - 18, f"+{self.amount} flask", FLASK_C))
        else:
            player.loot += self.amount
            texts.append(FloatingText(self.pos.x, self.pos.y - 18, f"+{self.amount} gold", LOOT_C))
        self.alive = False

    def draw(self, camera):
        px, py = camera.world_to_screen(self.pos)
        py += math.sin(self.bob) * 4
        if self.kind == "key":
            draw_glow(px, py, 18, KEY_C, 0.12)
            draw_ring(px - 4, py, 5, KEY_C, 2.0)
            draw_quad(px, py - 2, 12, 4, KEY_C)
        elif self.kind == "flask":
            draw_glow(px, py, 18, FLASK_C, 0.12)
            draw_quad(px - 6, py - 9, 12, 16, FLASK_C)
            draw_outline(px - 6, py - 9, 12, 16, WHITE, 1.5)
            draw_quad(px - 3, py - 13, 6, 5, (0.75, 0.62, 0.38, 1.0))
        else:
            draw_glow(px, py, 18, LOOT_C, 0.12)
            draw_circle(px, py, 7, LOOT_C, 4)
            draw_ring(px, py, 7, WHITE, 1.5, 4)


class Projectile:
    def __init__(self, x, y, velocity, damage, owner, radius=6, color=PURPLE_C):
        self.pos = Vector2(x, y)
        self.vel = Vector2(velocity)
        self.damage = damage
        self.owner = owner
        self.radius = radius
        self.color = color
        self.alive = True
        self.trail_time = 0.0

    def update(self, dt, level, player, game):
        self.pos += self.vel * dt
        self.trail_time -= dt
        if self.trail_time <= 0:
            self.trail_time = 0.03
            game.particles.append(Particle(self.pos, -self.vel * 0.03, self.color, radius=2.5, life=0.22, drag=0.83, glow=10))

        if not (0 <= self.pos.x < WORLD_W and 0 <= self.pos.y < WORLD_H):
            self.alive = False
            return
        if level.circle_hits_blocker(self.pos.x, self.pos.y, self.radius):
            self.alive = False
            game.spawn_impact(self.pos, self.color)
            return

        if self.owner == "enemy" and (self.pos - player.pos).length() < self.radius + player.radius:
            result = player.take_hit(self.damage, self.pos)
            self.alive = False
            if result == "blocked":
                game.texts.append(FloatingText(player.pos.x, player.pos.y - 28, "BLOCK", SHIELD_C))
                game.spawn_block(self.pos)
            elif result == "hit":
                game.texts.append(FloatingText(player.pos.x, player.pos.y - 28, f"-{self.damage}", RED_C))
                game.spawn_blood(player.pos)
                game.add_shake(5, 0.16)
                game.flash = 0.12

    def draw(self, camera):
        px, py = camera.world_to_screen(self.pos)
        draw_glow(px, py, self.radius * 2.2, self.color, 0.20)
        draw_circle(px, py, self.radius, self.color, 14)
        draw_ring(px, py, max(2, self.radius - 2), WHITE, 1.0, 14)


class Door:
    def __init__(self, cell, label):
        self.cell = cell
        self.label = label
        self.locked = True

    @property
    def rect(self):
        return (self.cell[0] * TILE, self.cell[1] * TILE, TILE, TILE)

    def draw(self, camera, time_now):
        if not self.locked:
            return
        x, y, w, h = self.rect
        sx, sy = camera.world_to_screen((x, y))
        pulse = 0.12 + 0.07 * (math.sin(time_now * 2.4 + self.label) * 0.5 + 0.5)
        draw_glow(sx + w / 2, sy + h / 2, 26, DOOR_C, pulse)
        draw_quad(sx + 4, sy + 4, w - 8, h - 8, (0.10, 0.18, 0.22, 1.0))
        draw_outline(sx + 4, sy + 4, w - 8, h - 8, DOOR_C, 2.0)
        TEXT.draw(str(self.label), sx + w / 2, sy + h / 2 - 8, color=(245, 245, 255), size="small", center=True)


class Player:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.radius = 16
        self.speed = 190
        self.hp = 120
        self.max_hp = 120
        self.keys = 0
        self.flasks = 1
        self.loot = 0
        self.facing = Vector2(1, 0)
        self.attack_timer = 0.0
        self.attack_cd = 0.0
        self.invuln = 0.0
        self.flask_cd = 0.0
        self.shielding = False

    def update(self, dt, level, mouse_world, key_state):
        move = Vector2(
            (1 if key_state[pygame.K_d] else 0) - (1 if key_state[pygame.K_a] else 0),
            (1 if key_state[pygame.K_s] else 0) - (1 if key_state[pygame.K_w] else 0),
        )
        move = normalize(move)
        if (mouse_world - self.pos).length_squared() > 0:
            self.facing = normalize(mouse_world - self.pos)
        elif move.length_squared() > 0:
            self.facing = move
        self.shielding = key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT] or pygame.mouse.get_pressed()[2]
        speed = self.speed * (0.56 if self.shielding else 1.0)
        delta = move * speed * dt
        self.move(delta.x, 0, level)
        self.move(0, delta.y, level)
        self.attack_timer = max(0.0, self.attack_timer - dt)
        self.attack_cd = max(0.0, self.attack_cd - dt)
        self.invuln = max(0.0, self.invuln - dt)
        self.flask_cd = max(0.0, self.flask_cd - dt)

    def move(self, dx, dy, level):
        if dx == 0 and dy == 0:
            return
        nx = self.pos.x + dx
        ny = self.pos.y + dy
        if not level.circle_hits_blocker(nx, self.pos.y, self.radius):
            self.pos.x = nx
        if not level.circle_hits_blocker(self.pos.x, ny, self.radius):
            self.pos.y = ny

    def try_attack(self):
        if self.attack_cd <= 0:
            self.attack_timer = 0.16
            self.attack_cd = 0.32
            return True
        return False

    def sword_hits(self, target_pos, target_radius):
        if self.attack_timer <= 0:
            return False
        to_target = Vector2(target_pos) - self.pos
        dist = to_target.length()
        if dist > 74 + target_radius:
            return False
        if dist == 0:
            return True
        return self.facing.dot(normalize(to_target)) > 0.35

    def try_use_flask(self, texts):
        if self.flasks <= 0 or self.flask_cd > 0 or self.hp >= self.max_hp:
            return False
        amount = min(35, self.max_hp - self.hp)
        self.hp += amount
        self.flasks -= 1
        self.flask_cd = 0.42
        texts.append(FloatingText(self.pos.x, self.pos.y - 28, f"+{amount}", GREEN_C))
        return True

    def take_hit(self, amount, source_pos):
        if self.invuln > 0:
            return "ignored"
        source_dir = normalize(self.pos - Vector2(source_pos))
        blocked = self.shielding and self.facing.dot(source_dir) > 0.25
        if blocked:
            self.invuln = 0.12
            return "blocked"
        self.hp -= amount
        self.invuln = 0.30
        return "hit"

    def draw(self, camera):
        px, py = camera.world_to_screen(self.pos)
        draw_circle(px, py + 12, 17, (0, 0, 0, 0.28), 18)
        draw_glow(px, py, 28, PLAYER_BODY, 0.11)
        if self.shielding:
            sx = px + self.facing.x * 18
            sy = py + self.facing.y * 18
            draw_glow(sx, sy, 26, SHIELD_C, 0.18)
            draw_circle(sx, sy, 18, (0.14, 0.34, 0.44, 0.95), 20)
            draw_ring(sx, sy, 18, SHIELD_C, 2.4, 20)
        draw_circle(px - self.facing.x * 5, py - self.facing.y * 5, self.radius + 2, PLAYER_CLOAK, 20)
        body = PLAYER_BODY if self.invuln <= 0 else (0.7, 0.85, 1.0, 1.0)
        draw_circle(px, py, self.radius, body, 20)
        draw_ring(px, py, self.radius, (0.28, 0.49, 0.73, 1.0), 1.5, 20)
        eye = self.pos + self.facing * 7
        ex, ey = camera.world_to_screen(eye)
        draw_circle(ex, ey, 3, PLAYER_TRIM, 8)
        if self.attack_timer > 0:
            tip = self.pos + self.facing * 42
            left = self.pos + self.facing.rotate(28) * 22
            right = self.pos + self.facing.rotate(-28) * 22
            tx, ty = camera.world_to_screen(tip)
            lx, ly = camera.world_to_screen(left)
            rx, ry = camera.world_to_screen(right)
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(*SWORD_C)
            glVertex2f(lx, ly)
            glVertex2f(tx, ty)
            glVertex2f(rx, ry)
            glEnd()
            ang = math.atan2(self.facing.y, self.facing.x)
            draw_arc(px, py, 56, ang - 0.6, ang + 0.6, (0.72, 0.86, 1.0, 0.75), width=3.0)


class Enemy:
    def __init__(self, x, y, kind):
        self.pos = Vector2(x, y)
        self.kind = kind
        self.alive = True
        self.last_hit_id = -1
        self.wander_target = self.pos.copy()
        self.wander_timer = random.uniform(0.5, 2.0)
        self.attack_cd = 0.0
        self.ranged_cd = random.uniform(0.7, 1.6)
        self.anim = random.random() * math.tau

        if kind == "stalker":
            self.radius = 14
            self.speed = random.uniform(125, 145)
            self.hp = self.max_hp = 26
            self.damage = 8
            self.detect = 380
            self.color = STALKER_C
            self.eye = (0.62, 0.88, 1.0, 1.0)
            self.loot_drop = 3
            self.glow = (0.40, 0.62, 1.0, 1.0)
        elif kind == "cultist":
            self.radius = 15
            self.speed = random.uniform(82, 96)
            self.hp = self.max_hp = 30
            self.damage = 8
            self.detect = 430
            self.color = CULTIST_C
            self.eye = (0.90, 0.60, 1.0, 1.0)
            self.loot_drop = 4
            self.glow = PURPLE_C
        elif kind == "brute":
            self.radius = 20
            self.speed = random.uniform(64, 80)
            self.hp = self.max_hp = 70
            self.damage = 16
            self.detect = 290
            self.color = BRUTE_C
            self.eye = (1.0, 0.52, 0.52, 1.0)
            self.loot_drop = 7
            self.glow = (0.90, 0.28, 0.28, 1.0)
        else:
            self.radius = 16
            self.speed = random.uniform(88, 110)
            self.hp = self.max_hp = 40
            self.damage = 10
            self.detect = 320
            self.color = SHADE_C
            self.eye = (0.90, 0.36, 0.48, 1.0)
            self.loot_drop = 5
            self.glow = (0.66, 0.30, 0.76, 1.0)

    def move(self, dx, dy, level):
        nx = self.pos.x + dx
        ny = self.pos.y + dy
        if not level.circle_hits_blocker(nx, self.pos.y, self.radius):
            self.pos.x = nx
        if not level.circle_hits_blocker(self.pos.x, ny, self.radius):
            self.pos.y = ny

    def _wander(self, dt, level):
        self.wander_timer -= dt
        if self.wander_timer <= 0 or (self.wander_target - self.pos).length() < 8:
            self.wander_timer = random.uniform(1.0, 2.8)
            for _ in range(10):
                ang = random.random() * math.tau
                candidate = self.pos + angle_vec(ang) * random.uniform(36, 130)
                if level.is_walkable_px(candidate.x, candidate.y):
                    self.wander_target = candidate
                    break
        return normalize(self.wander_target - self.pos)

    def fire(self, player, projectiles):
        direction = normalize(player.pos - self.pos)
        if direction.length_squared() == 0:
            return
        projectiles.append(Projectile(self.pos.x, self.pos.y, direction * 225, self.damage, "enemy", radius=7, color=PURPLE_C))

    def update(self, dt, level, player, projectiles, game):
        if not self.alive:
            return
        self.anim += dt * (5.0 if self.kind == "stalker" else 3.0)
        self.attack_cd = max(0.0, self.attack_cd - dt)
        self.ranged_cd = max(0.0, self.ranged_cd - dt)
        to_player = player.pos - self.pos
        dist = to_player.length()

        if self.kind == "cultist":
            if dist < self.detect:
                if dist > 240:
                    direction = normalize(to_player)
                elif dist < 145:
                    direction = normalize(-to_player)
                else:
                    direction = normalize(to_player.rotate(90 if random.random() < 0.5 else -90))
                if dist < 360 and self.ranged_cd <= 0:
                    self.fire(player, projectiles)
                    self.ranged_cd = random.uniform(1.15, 1.75)
                    game.spawn_cast(self.pos, self.glow)
            else:
                direction = self._wander(dt, level)
        else:
            direction = normalize(to_player) if dist < self.detect else self._wander(dt, level)

        move = direction * self.speed * dt
        self.move(move.x, 0, level)
        self.move(0, move.y, level)

        melee_range = self.radius + player.radius + (8 if self.kind == "brute" else 4)
        if dist < melee_range and self.attack_cd <= 0 and self.kind != "cultist":
            result = player.take_hit(self.damage, self.pos)
            self.attack_cd = 1.10 if self.kind == "brute" else 0.82 if self.kind == "shade" else 0.6
            if result == "blocked":
                game.texts.append(FloatingText(player.pos.x, player.pos.y - 28, "BLOCK", SHIELD_C))
                game.spawn_block(player.pos)
            elif result == "hit":
                game.texts.append(FloatingText(player.pos.x, player.pos.y - 28, f"-{self.damage}", RED_C))
                game.spawn_blood(player.pos)
                game.add_shake(5 if self.kind != "brute" else 7, 0.16)
                game.flash = 0.10

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False

    def draw(self, camera):
        if not self.alive:
            return
        px, py = camera.world_to_screen(self.pos)
        py += math.sin(self.anim) * 1.4
        draw_circle(px, py + self.radius * 0.7, self.radius * 1.15, (0, 0, 0, 0.22), 18)
        draw_glow(px, py, self.radius + 12, self.glow, 0.10)
        draw_circle(px, py, self.radius, self.color, 20)
        draw_ring(px, py, self.radius, (0.06, 0.04, 0.08, 1.0), 1.2, 20)
        if self.kind == "brute":
            draw_quad(px - 9, py - 4, 18, 4, self.eye)
            draw_quad(px - 10, py + 7, 20, 4, (0.55, 0.22, 0.26, 1.0))
        else:
            draw_circle(px - 5, py - 2, 2.4, self.eye, 8)
            draw_circle(px + 5, py - 2, 2.4, self.eye, 8)
            draw_arc(px, py + 3, 8, 0, math.pi, (0.55, 0.24, 0.32, 1.0), width=1.2, segments=10)
        hp_ratio = max(0.0, self.hp / self.max_hp)
        draw_quad(px - 18, py - self.radius - 12, 36, 5, (0.15, 0.08, 0.08, 0.9))
        draw_quad(px - 18, py - self.radius - 12, 36 * hp_ratio, 5, RED_C)


class Boss:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.radius = 34
        self.speed = 85
        self.hp = 340
        self.max_hp = 340
        self.damage = 18
        self.attack_cd = 0.0
        self.volley_cd = 2.3
        self.charge_cd = 4.8
        self.charge_time = 0.0
        self.charge_dir = Vector2()
        self.anim = 0.0
        self.alive = True
        self.last_hit_id = -1

    def fire_volley(self, projectiles):
        count = 8 if self.hp > self.max_hp * 0.5 else 12
        offset = random.random() * math.tau
        for i in range(count):
            ang = offset + math.tau * i / count
            projectiles.append(Projectile(self.pos.x, self.pos.y, angle_vec(ang) * 190, 12, "enemy", radius=7, color=BOSS_CORE))

    def update(self, dt, level, player, projectiles, game):
        if not self.alive:
            return
        self.anim += dt * 2.8
        self.attack_cd = max(0.0, self.attack_cd - dt)
        self.volley_cd = max(0.0, self.volley_cd - dt)
        self.charge_cd = max(0.0, self.charge_cd - dt)
        to_player = player.pos - self.pos
        dist = to_player.length()
        direction = normalize(to_player)

        if self.charge_time > 0:
            self.charge_time -= dt
            move = self.charge_dir * 290 * dt
            if random.random() < 0.50:
                game.particles.append(Particle(self.pos, Vector2(random.uniform(-20, 20), random.uniform(-20, 20)), BOSS_CORE, radius=4, life=0.28, drag=0.85, glow=14))
        else:
            move = direction * self.speed * dt
            if self.volley_cd <= 0:
                self.fire_volley(projectiles)
                self.volley_cd = 1.7 if self.hp < self.max_hp * 0.5 else 2.35
                game.spawn_cast(self.pos, BOSS_CORE, 30)
                game.flash = 0.08
            if self.charge_cd <= 0 and dist > 120:
                self.charge_dir = direction
                self.charge_time = 0.55
                self.charge_cd = 4.4 if self.hp < self.max_hp * 0.5 else 5.0
                game.add_shake(8, 0.22)
                game.spawn_cast(self.pos, BOSS_CORE, 40)

        self.move(move.x, 0, level)
        self.move(0, move.y, level)

        if dist < self.radius + player.radius + 12 and self.attack_cd <= 0:
            result = player.take_hit(self.damage, self.pos)
            self.attack_cd = 0.95
            if result == "blocked":
                game.texts.append(FloatingText(player.pos.x, player.pos.y - 28, "BLOCK", SHIELD_C))
                game.spawn_block(player.pos)
            elif result == "hit":
                game.texts.append(FloatingText(player.pos.x, player.pos.y - 28, f"-{self.damage}", RED_C))
                game.spawn_blood(player.pos)
                game.add_shake(8, 0.18)
                game.flash = 0.14

    def move(self, dx, dy, level):
        nx = self.pos.x + dx
        ny = self.pos.y + dy
        if not level.circle_hits_blocker(nx, self.pos.y, self.radius):
            self.pos.x = nx
        if not level.circle_hits_blocker(self.pos.x, ny, self.radius):
            self.pos.y = ny

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False

    def draw(self, camera):
        if not self.alive:
            return
        px, py = camera.world_to_screen(self.pos)
        py += math.sin(self.anim) * 2.0
        draw_circle(px, py + 24, 34, (0, 0, 0, 0.30), 28)
        draw_glow(px, py, 52, BOSS_CORE, 0.12)
        draw_glow(px, py, 34, BOSS_C, 0.16)
        draw_circle(px, py, self.radius, BOSS_C, 28)
        draw_ring(px, py, self.radius, (0.12, 0.02, 0.02, 1.0), 2.0, 28)
        draw_circle(px, py, 12 + math.sin(self.anim * 1.4) * 1.5, BOSS_CORE, 18)
        draw_circle(px - 11, py - 5, 4, (1.0, 0.85, 0.88, 1.0), 10)
        draw_circle(px + 11, py - 5, 4, (1.0, 0.85, 0.88, 1.0), 10)
        draw_arc(px, py + 6, 16, 0, math.pi, (0.82, 0.46, 0.50, 1.0), width=2.0, segments=12)


class Level:
    def __init__(self):
        self.grid = [[1 for _ in range(MAZE_W)] for _ in range(MAZE_H)]
        self.start_cell = (1, 1)
        self.boss_cell = (MAZE_W - 6, MAZE_H - 6)
        self.doors = []
        self.explored = [[False for _ in range(MAZE_W)] for _ in range(MAZE_H)]
        self.generate_maze()

    def cell_center(self, cell):
        return Vector2(cell[0] * TILE + TILE / 2, cell[1] * TILE + TILE / 2)

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

    def find_nearest_existing_floor(self, start, max_x=None):
        q = deque([start])
        seen = {start}
        while q:
            x, y = q.popleft()
            if 0 <= x < MAZE_W and 0 <= y < MAZE_H:
                if self.grid[y][x] == 0 and (max_x is None or x <= max_x):
                    return x, y
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (x + dx, y + dy)
                    if 0 <= nxt[0] < MAZE_W and 0 <= nxt[1] < MAZE_H and nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
        return None

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
        connector = self.find_nearest_existing_floor((corridor_start - 1, by), max_x=corridor_start - 1)
        if connector:
            cx, cy = connector
            for x in range(min(cx, corridor_start), max(cx, corridor_start) + 1):
                self.grid[cy][x] = 0
            for y in range(min(cy, by), max(cy, by) + 1):
                self.grid[y][corridor_start] = 0
        door_cells = [(bx - 9, by), (bx - 7, by), (bx - 5, by)]
        self.doors = [Door(cell, i + 1) for i, cell in enumerate(door_cells)]

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
        candidates = [c for c in distances if 5 <= c[0] < MAZE_W - 5 and 5 <= c[1] < MAZE_H - 5]
        self.boss_cell = max(candidates, key=lambda c: distances[c]) if candidates else max(distances, key=distances.get)
        self.build_boss_wing()

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
        tx = int(x // TILE)
        ty = int(y // TILE)
        for gy in range(ty - 1, ty + 2):
            for gx in range(tx - 1, tx + 2):
                if 0 <= gx < MAZE_W and 0 <= gy < MAZE_H:
                    blocked = self.grid[gy][gx] == 1
                    if not blocked:
                        for door in self.doors:
                            if door.locked and door.cell == (gx, gy):
                                blocked = True
                                break
                    if blocked and circle_rect_collision(x, y, radius, gx * TILE, gy * TILE, TILE, TILE):
                        return True
        return False

    def update_exploration(self, player):
        px = int(player.pos.x // TILE)
        py = int(player.pos.y // TILE)
        for y in range(py - 4, py + 5):
            for x in range(px - 5, px + 6):
                if 0 <= x < MAZE_W and 0 <= y < MAZE_H:
                    self.explored[y][x] = True

    def handle_doors(self, player, game):
        for door in self.doors:
            if not door.locked:
                continue
            rx, ry, rw, rh = door.rect
            if circle_rect_collision(player.pos.x, player.pos.y, player.radius + 2, rx, ry, rw, rh):
                if player.keys > 0:
                    player.keys -= 1
                    door.locked = False
                    game.texts.append(FloatingText(rx + rw / 2, ry - 10, f"Gate {door.label} opened", KEY_C, life=1.1))
                    game.spawn_gate(Vector2(rx + rw / 2, ry + rh / 2))
                    game.flash = 0.06
                elif game.message_cd <= 0:
                    game.texts.append(FloatingText(rx + rw / 2, ry - 10, f"Gate {door.label} needs a key", WHITE, life=1.0))
                    game.message_cd = 0.8

    def tile_variant(self, x, y):
        return hash_xy(x, y) % 4

    def draw(self, camera, time_now):
        start_x = max(0, int(camera.x // TILE) - 1)
        start_y = max(0, int(camera.y // TILE) - 1)
        end_x = min(MAZE_W, int((camera.x + SCREEN_W) // TILE) + 2)
        end_y = min(MAZE_H, int((camera.y + SCREEN_H) // TILE) + 2)
        bx, by = self.boss_cell
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                sx = x * TILE - camera.x
                sy = y * TILE - camera.y
                in_boss = abs(x - bx) <= 3 and abs(y - by) <= 3
                if self.grid[y][x] == 1:
                    shade = 0.0 if self.tile_variant(x, y) % 2 == 0 else 0.03
                    draw_quad_gradient(sx, sy, TILE, TILE, (WALL_A[0] + shade, WALL_A[1] + shade, WALL_A[2] + shade, 1.0), (WALL_B[0], WALL_B[1], WALL_B[2], 1.0))
                    draw_outline(sx + 1, sy + 1, TILE - 2, TILE - 2, WALL_EDGE, 1.0)
                    if (x + y) % 2 == 0:
                        draw_line(sx + 6, sy + 12, sx + TILE - 6, sy + 12, (0.22, 0.23, 0.30, 1.0), 1.0)
                        draw_line(sx + 6, sy + TILE - 12, sx + TILE - 6, sy + TILE - 12, (0.18, 0.19, 0.26, 1.0), 1.0)
                else:
                    if in_boss:
                        c1 = (0.17, 0.07, 0.09, 1.0)
                        c2 = (0.24, 0.09, 0.11, 1.0)
                    else:
                        c1 = FLOOR_A if self.tile_variant(x, y) % 2 == 0 else FLOOR_B
                        c2 = FLOOR_B if self.tile_variant(x, y) % 2 == 0 else FLOOR_A
                    draw_quad_gradient(sx, sy, TILE, TILE, c1, c2)
                    draw_outline(sx, sy, TILE, TILE, (0.14, 0.14, 0.18, 0.35), 1.0)
                    if (x + y) % 9 == 0:
                        alpha = 0.02 + 0.03 * (math.sin(time_now * 1.4 + x * 0.2 + y * 0.3) * 0.5 + 0.5)
                        draw_glow(sx + TILE / 2, sy + TILE / 2, 10, (0.42, 0.30, 0.58, 1.0), alpha)

        arena_x = (self.boss_cell[0] - 4) * TILE - camera.x
        arena_y = (self.boss_cell[1] - 4) * TILE - camera.y
        draw_outline(arena_x, arena_y, TILE * 9, TILE * 9, (0.40, 0.15, 0.18, 0.85), 2.0)
        draw_glow(arena_x + TILE * 4.5, arena_y + TILE * 4.5, 48, (0.55, 0.18, 0.22, 1.0), 0.06)
        for door in self.doors:
            door.draw(camera, time_now)

    def draw_minimap(self, player, boss_alive):
        scale = 4
        pad = 14
        mini_w = MAZE_W * scale
        mini_h = MAZE_H * scale
        bx = SCREEN_W - mini_w - pad * 2
        by = pad
        draw_quad(bx, by, mini_w + pad, mini_h + pad, (0.05, 0.05, 0.08, 0.92))
        draw_outline(bx, by, mini_w + pad, mini_h + pad, (0.26, 0.26, 0.36, 1.0), 2.0)
        ox = bx + 7
        oy = by + 7
        for y in range(MAZE_H):
            for x in range(MAZE_W):
                if not self.explored[y][x]:
                    continue
                color = (0.34, 0.35, 0.45, 1.0) if self.grid[y][x] == 1 else (0.74, 0.74, 0.82, 1.0)
                draw_quad(ox + x * scale, oy + y * scale, scale, scale, color)
        for door in self.doors:
            if door.locked and self.explored[door.cell[1]][door.cell[0]]:
                draw_quad(ox + door.cell[0] * scale, oy + door.cell[1] * scale, scale + 1, scale + 1, DOOR_C)
        px = int(player.pos.x // TILE)
        py = int(player.pos.y // TILE)
        draw_quad(ox + px * scale - 1, oy + py * scale - 1, scale + 2, scale + 2, PLAYER_BODY)
        if boss_alive and self.explored[self.boss_cell[1]][self.boss_cell[0]]:
            draw_quad(ox + self.boss_cell[0] * scale, oy + self.boss_cell[1] * scale, scale + 1, scale + 1, BOSS_CORE)


class Game:
    def __init__(self):
        self.level = Level()
        self.camera = Camera()
        self.player = Player(*self.level.cell_center(self.level.start_cell))
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.effects = []
        self.pickups = []
        self.texts = []
        self.message_cd = 0.0
        self.time_now = 0.0
        self.shake_power = 0.0
        self.shake_time = 0.0
        self.flash = 0.0
        self.state = "title"
        self.victory_time = 0.0
        self.spawn_world()

    def add_shake(self, amount, duration):
        self.shake_power = max(self.shake_power, amount)
        self.shake_time = max(self.shake_time, duration)

    def spawn_impact(self, pos, color):
        for _ in range(10):
            ang = random.random() * math.tau
            spd = random.uniform(50, 140)
            self.particles.append(Particle(pos, angle_vec(ang) * spd, color, radius=3, life=0.34, drag=0.87, glow=8))

    def spawn_blood(self, pos):
        for _ in range(12):
            ang = random.random() * math.tau
            spd = random.uniform(50, 180)
            self.particles.append(Particle(pos, angle_vec(ang) * spd, RED_C, radius=3.5, life=0.42, drag=0.88, glow=6))

    def spawn_block(self, pos):
        for _ in range(10):
            ang = random.random() * math.tau
            spd = random.uniform(40, 140)
            self.particles.append(Particle(pos, angle_vec(ang) * spd, SHIELD_C, radius=2.7, life=0.30, drag=0.86, glow=8))

    def spawn_cast(self, pos, color, count=18):
        for _ in range(count):
            ang = random.random() * math.tau
            spd = random.uniform(40, 120)
            self.particles.append(Particle(pos, angle_vec(ang) * spd, color, radius=3.0, life=0.42, drag=0.90, glow=12))

    def spawn_gate(self, pos):
        for _ in range(18):
            ang = random.random() * math.tau
            spd = random.uniform(60, 180)
            self.particles.append(Particle(pos, angle_vec(ang) * spd, DOOR_C, radius=3.0, life=0.48, drag=0.90, glow=10))

    def spawn_world(self):
        distances = self.level.compute_distances(self.level.start_cell)
        floor_cells = list(distances.keys())
        floor_cells.sort(key=lambda c: distances[c])
        boss_forbidden = set()
        bx, by = self.level.boss_cell
        for y in range(by - 4, by + 5):
            for x in range(bx - 4, bx + 5):
                boss_forbidden.add((x, y))

        enemy_pool = [c for c in floor_cells if distances[c] > 10 and c not in boss_forbidden]
        random.shuffle(enemy_pool)
        for i, cell in enumerate(enemy_pool[:95]):
            pos = self.level.cell_center(cell)
            roll = random.random()
            if roll < 0.18:
                kind = "brute"
            elif roll < 0.42:
                kind = "cultist"
            elif roll < 0.68:
                kind = "stalker"
            else:
                kind = "shade"
            if i < 10:
                kind = "shade"
            self.enemies.append(Enemy(pos.x, pos.y, kind))

        boss_pos = self.level.cell_center(self.level.boss_cell)
        self.boss = Boss(boss_pos.x, boss_pos.y)

        good_path = [c for c in floor_cells if 14 < distances[c] < distances[self.level.boss_cell] - 10 and c[0] < self.level.boss_cell[0] - 10]
        random.shuffle(good_path)
        for idx, cell in enumerate(good_path[:3]):
            pos = self.level.cell_center(cell)
            self.pickups.append(Pickup(pos.x, pos.y, "key", 1))
        for cell in good_path[3:15]:
            pos = self.level.cell_center(cell)
            self.pickups.append(Pickup(pos.x, pos.y, "flask", 1 if random.random() < 0.3 else 0))
            self.pickups.append(Pickup(pos.x + random.uniform(-8, 8), pos.y + random.uniform(-8, 8), "loot", random.randint(2, 7)))
        for cell in good_path[15:45]:
            pos = self.level.cell_center(cell)
            self.pickups.append(Pickup(pos.x + random.uniform(-10, 10), pos.y + random.uniform(-10, 10), "loot", random.randint(1, 5)))
        self.pickups = [p for p in self.pickups if p.amount > 0]

    def restart(self):
        self.__init__()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if self.state in ("gameover", "victory") and event.key == pygame.K_r:
                self.restart()
            elif self.state == "title" and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.state = "playing"
            elif self.state == "playing":
                if event.key == pygame.K_f:
                    if self.player.try_use_flask(self.texts):
                        self.spawn_cast(self.player.pos, FLASK_C, 14)
                        self.flash = 0.05
        if self.state != "playing":
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.player.try_attack():
                self.effects.append(SlashEffect(self.player.pos, self.player.facing))
                self.add_shake(2, 0.06)

    def update(self, dt):
        self.time_now += dt
        self.message_cd = max(0.0, self.message_cd - dt)
        self.flash = max(0.0, self.flash - dt)
        if self.shake_time > 0:
            self.shake_time -= dt
            if self.shake_time <= 0:
                self.shake_power = 0.0
        if self.state == "title":
            return
        if self.state == "victory":
            self.victory_time += dt
            for obj in self.particles:
                obj.update(dt)
            self.particles = [p for p in self.particles if p.life > 0]
            return
        if self.state == "gameover":
            return

        mx, my = pygame.mouse.get_pos()
        mouse_world = Vector2(mx + self.camera.x, my + self.camera.y)
        self.player.update(dt, self.level, mouse_world, pygame.key.get_pressed())
        self.level.handle_doors(self.player, self)
        self.level.update_exploration(self.player)

        shake = Vector2()
        if self.shake_power > 0 and self.shake_time > 0:
            shake = Vector2(random.uniform(-self.shake_power, self.shake_power), random.uniform(-self.shake_power, self.shake_power))
        self.camera.update(self.player.pos, shake)

        for pickup in self.pickups:
            pickup.update(dt)
            if pickup.alive and (pickup.pos - self.player.pos).length() < pickup.radius + self.player.radius:
                pickup.collect(self.player, self.texts)
                if pickup.kind == "key":
                    self.spawn_cast(pickup.pos, KEY_C, 12)
                elif pickup.kind == "flask":
                    self.spawn_cast(pickup.pos, FLASK_C, 12)
                else:
                    self.spawn_cast(pickup.pos, LOOT_C, 8)
        self.pickups = [p for p in self.pickups if p.alive]

        for enemy in self.enemies:
            enemy.update(dt, self.level, self.player, self.projectiles, self)
            if enemy.alive and self.player.sword_hits(enemy.pos, enemy.radius):
                attack_id = id(self.effects[-1]) if self.effects else 0
                if enemy.last_hit_id != attack_id:
                    enemy.last_hit_id = attack_id
                    damage = 18 if enemy.kind != "brute" else 14
                    enemy.take_damage(damage)
                    self.texts.append(FloatingText(enemy.pos.x, enemy.pos.y - 18, str(damage), WHITE))
                    self.spawn_impact(enemy.pos, PLAYER_BODY)
                    self.add_shake(3, 0.08)
                    if not enemy.alive:
                        self.spawn_cast(enemy.pos, enemy.glow, 16)
                        if random.random() < 0.22:
                            self.pickups.append(Pickup(enemy.pos.x, enemy.pos.y, "flask", 1))
                        if random.random() < 0.35:
                            self.pickups.append(Pickup(enemy.pos.x + random.uniform(-6, 6), enemy.pos.y + random.uniform(-6, 6), "loot", enemy.loot_drop))
        self.enemies = [e for e in self.enemies if e.alive]

        if self.boss.alive:
            self.boss.update(dt, self.level, self.player, self.projectiles, self)
            if self.player.sword_hits(self.boss.pos, self.boss.radius):
                attack_id = id(self.effects[-1]) if self.effects else 0
                if self.boss.last_hit_id != attack_id:
                    self.boss.last_hit_id = attack_id
                    damage = 12
                    self.boss.take_damage(damage)
                    self.texts.append(FloatingText(self.boss.pos.x, self.boss.pos.y - 24, str(damage), WHITE))
                    self.spawn_impact(self.boss.pos, PLAYER_BODY)
                    self.add_shake(4, 0.10)
                    if not self.boss.alive:
                        self.spawn_cast(self.boss.pos, BOSS_CORE, 40)
                        self.add_shake(12, 0.45)
                        self.flash = 0.28
                        self.state = "victory"
        
        for proj in self.projectiles:
            proj.update(dt, self.level, self.player, self)
        self.projectiles = [p for p in self.projectiles if p.alive]

        for fx in self.effects:
            fx.update(dt)
        self.effects = [fx for fx in self.effects if fx.life > 0]

        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.life > 0]

        for t in self.texts:
            t.update(dt)
        self.texts = [t for t in self.texts if t.life > 0]

        if self.player.hp <= 0:
            self.state = "gameover"

    def draw_hud(self):
        draw_quad(14, 14, 340, 84, (0.06, 0.06, 0.09, 0.85))
        draw_outline(14, 14, 340, 84, (0.25, 0.27, 0.36, 1.0), 2.0)
        draw_quad(28, 34, 180, 16, (0.14, 0.08, 0.08, 0.95))
        draw_quad(28, 34, 180 * max(0.0, self.player.hp / self.player.max_hp), 16, RED_C)
        draw_outline(28, 34, 180, 16, (0.8, 0.8, 0.84, 1.0), 1.5)
        TEXT.draw(f"HP {self.player.hp}/{self.player.max_hp}", 28, 16, color=(245, 245, 255))
        TEXT.draw(f"Keys: {self.player.keys}", 230, 28, color=(160, 250, 255))
        TEXT.draw(f"Flasks: {self.player.flasks}", 230, 48, color=(110, 240, 150))
        TEXT.draw(f"Gold: {self.player.loot}", 230, 68, color=(255, 220, 100))
        TEXT.draw("LMB Attack  RMB/Shift Shield  F Flask", 18, SCREEN_H - 24, color=(210, 210, 220), size="small")

        if self.boss.alive:
            ratio = max(0.0, self.boss.hp / self.boss.max_hp)
            w = 340
            x = SCREEN_W / 2 - w / 2
            y = 18
            draw_quad(x, y, w, 18, (0.12, 0.05, 0.05, 0.92))
            draw_quad(x, y, w * ratio, 18, BOSS_CORE)
            draw_outline(x, y, w, 18, (0.95, 0.78, 0.82, 1.0), 1.8)
            TEXT.draw("The Heart of Night", SCREEN_W / 2, y - 16, color=(255, 210, 220), size="small", center=True)

        self.level.draw_minimap(self.player, self.boss.alive)

    def draw_world(self):
        self.level.draw(self.camera, self.time_now)
        for pickup in self.pickups:
            pickup.draw(self.camera)
        for proj in self.projectiles:
            proj.draw(self.camera)
        for enemy in self.enemies:
            enemy.draw(self.camera)
        if self.boss.alive:
            self.boss.draw(self.camera)
        self.player.draw(self.camera)
        for fx in self.effects:
            fx.draw(self.camera)
        for p in self.particles:
            p.draw(self.camera)
        for t in self.texts:
            t.draw(self.camera)

    def draw_overlay(self):
        draw_vignette()
        if self.flash > 0:
            draw_quad(0, 0, SCREEN_W, SCREEN_H, (1.0, 1.0, 1.0, self.flash * 0.55))

    def draw_title(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for i in range(20):
            x = (i * 47 + (self.time_now * 22)) % (SCREEN_W + 140) - 70
            y = 40 + (i * 23) % (SCREEN_H - 80)
            draw_glow(x, y, 34 + (i % 5) * 8, (0.30, 0.18, 0.42, 1.0), 0.03)
        draw_quad(90, 82, 620, 250, (0.04, 0.04, 0.06, 0.82))
        draw_outline(90, 82, 620, 250, (0.28, 0.30, 0.40, 1.0), 2.0)
        TEXT.draw("SHADOW LABYRINTH OPENGL", SCREEN_W / 2, 122, color=(245, 245, 255), size="big", center=True)
        TEXT.draw("One massive dungeon. Sword. Shield. Dark enemies. Final boss.", SCREEN_W / 2, 186, color=(210, 210, 225), center=True)
        TEXT.draw("This version renders with PyOpenGL while keeping pygame for input/window management.", SCREEN_W / 2, 214, color=(165, 200, 255), size="small", center=True)
        TEXT.draw("WASD move   Mouse aim   LMB attack   RMB/Shift shield   F flask", SCREEN_W / 2, 258, color=(180, 255, 220), center=True)
        TEXT.draw("Press Enter or Space to descend", SCREEN_W / 2, 304, color=(255, 220, 120), center=True)
        draw_vignette()

    def draw_end_state(self):
        draw_quad(0, 0, SCREEN_W, SCREEN_H, (0.0, 0.0, 0.0, 0.45))
        if self.state == "gameover":
            TEXT.draw("You fell in the labyrinth", SCREEN_W / 2, SCREEN_H / 2 - 24, color=(255, 200, 200), size="big", center=True)
            TEXT.draw("Press R to restart", SCREEN_W / 2, SCREEN_H / 2 + 24, color=(240, 240, 255), center=True)
        elif self.state == "victory":
            TEXT.draw("The Heart of Night is broken", SCREEN_W / 2, SCREEN_H / 2 - 24, color=(255, 220, 230), size="big", center=True)
            TEXT.draw("Press R to descend again", SCREEN_W / 2, SCREEN_H / 2 + 24, color=(240, 240, 255), center=True)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.state == "title":
            self.draw_title()
        else:
            self.draw_world()
            self.draw_hud()
            self.draw_overlay()
            if self.state in ("gameover", "victory"):
                self.draw_end_state()


def main():
    pygame.init()
    pygame.display.set_caption("Shadow Labyrinth OpenGL")
    pygame.display.set_mode((SCREEN_W, SCREEN_H), DOUBLEBUF | OPENGL)
    setup_gl()
    global TEXT
    TEXT = GLText()
    clock = pygame.time.Clock()
    game = Game()

    try:
        running = True
        while running:
            dt = min(0.033, clock.tick(FPS) / 1000.0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    game.handle_event(event)
            game.update(dt)
            game.draw()
            pygame.display.flip()
    finally:
        if TEXT is not None:
            TEXT.cleanup()
        pygame.quit()


if __name__ == "__main__":
    main()
