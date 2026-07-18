import math
import random
import sys
from dataclasses import dataclass

import pygame

# ------------------------------------------------------------
# Tactical FPS Prototype
# 800x480 pseudo-3D first-person shooter built in Pygame.
# Inspired by modern arcade military shooters, but uses only
# original placeholder visuals and gameplay systems.
# ------------------------------------------------------------

WIDTH, HEIGHT = 800, 480
HALF_W, HALF_H = WIDTH // 2, HEIGHT // 2
FPS = 60
FOV = math.pi / 3
HALF_FOV = FOV / 2
NUM_RAYS = 400
SCALE = WIDTH // NUM_RAYS
MAX_DEPTH = 22
DELTA_ANGLE = FOV / NUM_RAYS
SCREEN_DIST = HALF_W / math.tan(HALF_FOV)
TEXTURE_SHADE = 0.92
MOUSE_SENS = 0.0025

PLAYER_SPEED = 3.7
SPRINT_SPEED = 5.8
ROT_SPEED = 2.3
PLAYER_RADIUS = 0.18
SHOOT_COOLDOWN = 0.12
RELOAD_TIME = 1.2
MAX_AMMO = 24
DAMAGE = 34
MAX_HP = 100

ENEMY_BASE_SPEED = 1.3
ENEMY_ATTACK_RANGE = 0.9
ENEMY_ATTACK_DAMAGE = 9
ENEMY_ATTACK_COOLDOWN = 0.8

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
for j, row in enumerate(MAP_STR):
    for i, ch in enumerate(row):
        if ch != ".":
            WORLD_MAP[(i, j)] = int(ch)
        else:
            FREE_CELLS.append((i + 0.5, j + 0.5))
MAP_W, MAP_H = len(MAP_STR[0]), len(MAP_STR)

WALL_COLORS = {
    1: (75, 91, 112),
    2: (103, 97, 84),
    3: (83, 108, 86),
}

SKY_TOP = (17, 22, 31)
SKY_BOTTOM = (55, 70, 93)
FLOOR_TOP = (44, 44, 46)
FLOOR_BOTTOM = (16, 16, 18)
RETICLE = (214, 236, 255)
HUD = (190, 215, 235)
ACCENT = (255, 195, 82)
WARNING = (255, 106, 106)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)


@dataclass
class Enemy:
    x: float
    y: float
    hp: int = 100
    speed: float = ENEMY_BASE_SPEED
    attack_timer: float = 0.0
    anim: float = 0.0

    @property
    def alive(self):
        return self.hp > 0


class Player:
    def __init__(self):
        self.x = 2.5
        self.y = 2.5
        self.angle = 0.0
        self.hp = MAX_HP
        self.ammo = MAX_AMMO
        self.shoot_timer = 0.0
        self.reload_timer = 0.0
        self.weapon_kick = 0.0
        self.damage_flash = 0.0
        self.steps = 0.0
        self.score = 0
        self.kills = 0

    def pos(self):
        return self.x, self.y


def cell_blocked(x, y):
    return (int(x), int(y)) in WORLD_MAP


def clamp(value, low, high):
    return max(low, min(high, value))


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tactical FPS Prototype")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 14, bold=True)
        self.big_font = pygame.font.SysFont("consolas", 48, bold=True)
        self.player = Player()
        self.wave = 1
        self.enemies = []
        self.spawn_wave(self.wave)
        self.crosshair_hit = 0.0
        self.game_over = False
        self.victory_flash = 0.0
        self.zbuffer = [MAX_DEPTH for _ in range(NUM_RAYS)]
        self.enemy_surface_cache = {}
        self.grab_mouse(True)
        random.seed()

    def grab_mouse(self, enabled=True):
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        if enabled:
            pygame.mouse.get_rel()

    def reset(self):
        self.player = Player()
        self.wave = 1
        self.enemies.clear()
        self.spawn_wave(self.wave)
        self.crosshair_hit = 0.0
        self.game_over = False
        self.victory_flash = 0.0
        self.grab_mouse(True)

    def spawn_wave(self, wave_num):
        self.enemies.clear()
        needed = 4 + wave_num * 2
        random.shuffle(FREE_CELLS)
        picks = FREE_CELLS[: min(needed * 3, len(FREE_CELLS))]
        for px, py in picks:
            if math.hypot(px - self.player.x, py - self.player.y) < 3.5:
                continue
            speed = ENEMY_BASE_SPEED + min(0.9, wave_num * 0.08)
            hp = 70 + wave_num * 8
            self.enemies.append(Enemy(px, py, hp=hp, speed=speed + random.uniform(-0.12, 0.18)))
            if len(self.enemies) >= needed:
                break

    def run(self):
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.03)
            self.handle_events()
            if not self.game_over:
                self.update(dt)
            self.draw()
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
                elif event.key == pygame.K_r:
                    if self.game_over:
                        self.reset()
                    elif self.player.ammo < MAX_AMMO and self.player.reload_timer <= 0:
                        self.player.reload_timer = RELOAD_TIME
                elif event.key == pygame.K_TAB:
                    grabbed = pygame.event.get_grab()
                    self.grab_mouse(not grabbed)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.game_over:
                self.try_shoot()

    def update(self, dt):
        self.player.shoot_timer = max(0.0, self.player.shoot_timer - dt)
        self.player.reload_timer = max(0.0, self.player.reload_timer - dt)
        self.player.weapon_kick = max(0.0, self.player.weapon_kick - dt * 5.0)
        self.player.damage_flash = max(0.0, self.player.damage_flash - dt * 2.2)
        self.crosshair_hit = max(0.0, self.crosshair_hit - dt * 3.5)
        self.victory_flash = max(0.0, self.victory_flash - dt * 2.0)

        if self.player.reload_timer == 0 and self.player.ammo < MAX_AMMO:
            self.player.ammo = MAX_AMMO

        if pygame.event.get_grab():
            dx, _ = pygame.mouse.get_rel()
            self.player.angle += dx * MOUSE_SENS

        keys = pygame.key.get_pressed()
        rot = 0.0
        if keys[pygame.K_LEFT]:
            rot -= ROT_SPEED * dt
        if keys[pygame.K_RIGHT]:
            rot += ROT_SPEED * dt
        self.player.angle += rot
        self.player.angle %= math.tau

        move_speed = SPRINT_SPEED if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else PLAYER_SPEED
        sin_a = math.sin(self.player.angle)
        cos_a = math.cos(self.player.angle)
        dx = dy = 0.0
        if keys[pygame.K_w]:
            dx += cos_a * move_speed * dt
            dy += sin_a * move_speed * dt
        if keys[pygame.K_s]:
            dx -= cos_a * move_speed * dt
            dy -= sin_a * move_speed * dt
        if keys[pygame.K_a]:
            dx += math.sin(self.player.angle) * move_speed * dt
            dy -= math.cos(self.player.angle) * move_speed * dt
        if keys[pygame.K_d]:
            dx -= math.sin(self.player.angle) * move_speed * dt
            dy += math.cos(self.player.angle) * move_speed * dt

        moving = abs(dx) + abs(dy) > 0.0001
        if moving:
            self.player.steps += dt * (11.0 if move_speed > PLAYER_SPEED else 8.0)
        self.move_player(dx, dy)

        if pygame.mouse.get_pressed()[0] and not self.game_over:
            self.try_shoot()

        self.update_enemies(dt)

        alive_count = sum(1 for e in self.enemies if e.alive)
        if alive_count == 0:
            self.wave += 1
            self.victory_flash = 0.35
            self.spawn_wave(self.wave)

        if self.player.hp <= 0:
            self.game_over = True
            self.grab_mouse(False)

    def move_player(self, dx, dy):
        nx = self.player.x + dx
        ny = self.player.y + dy

        if not self.circle_collides(nx, self.player.y, PLAYER_RADIUS):
            self.player.x = nx
        if not self.circle_collides(self.player.x, ny, PLAYER_RADIUS):
            self.player.y = ny

    def circle_collides(self, x, y, radius):
        left = int(x - radius)
        right = int(x + radius)
        top = int(y - radius)
        bottom = int(y + radius)
        for gy in range(top, bottom + 1):
            for gx in range(left, right + 1):
                if (gx, gy) not in WORLD_MAP:
                    continue
                nearest_x = clamp(x, gx, gx + 1)
                nearest_y = clamp(y, gy, gy + 1)
                if (x - nearest_x) ** 2 + (y - nearest_y) ** 2 < radius ** 2:
                    return True
        return False

    def try_shoot(self):
        if self.player.shoot_timer > 0 or self.player.reload_timer > 0:
            return
        if self.player.ammo <= 0:
            self.player.reload_timer = RELOAD_TIME
            return
        self.player.shoot_timer = SHOOT_COOLDOWN
        self.player.ammo -= 1
        self.player.weapon_kick = 1.0
        shot_hit = self.hitscan_shot()
        if shot_hit:
            self.crosshair_hit = 0.6
        if self.player.ammo <= 0:
            self.player.reload_timer = RELOAD_TIME

    def hitscan_shot(self):
        best_enemy = None
        best_angle = 999
        best_dist = 999
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            delta = (ang - self.player.angle + math.pi) % math.tau - math.pi
            if abs(delta) > math.radians(5.2):
                continue
            if not self.has_line_of_sight(self.player.x, self.player.y, enemy.x, enemy.y):
                continue
            if abs(delta) < best_angle or (abs(delta) <= best_angle + 1e-5 and dist < best_dist):
                best_enemy = enemy
                best_angle = abs(delta)
                best_dist = dist
        if best_enemy:
            best_enemy.hp -= DAMAGE
            if best_enemy.hp <= 0:
                self.player.score += 100
                self.player.kills += 1
            else:
                self.player.score += 10
            return True
        return False

    def has_line_of_sight(self, x1, y1, x2, y2):
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(4, int(dist * 14))
        for i in range(1, steps):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            if cell_blocked(x, y):
                return False
        return True

    def update_enemies(self, dt):
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            enemy.anim += dt * 6.0
            dx = self.player.x - enemy.x
            dy = self.player.y - enemy.y
            dist = math.hypot(dx, dy)
            enemy.attack_timer = max(0.0, enemy.attack_timer - dt)

            if dist <= ENEMY_ATTACK_RANGE:
                if enemy.attack_timer <= 0.0:
                    enemy.attack_timer = ENEMY_ATTACK_COOLDOWN + random.uniform(0.0, 0.2)
                    self.player.hp = max(0, self.player.hp - ENEMY_ATTACK_DAMAGE)
                    self.player.damage_flash = 0.8
                continue

            if dist > 0.001:
                move = enemy.speed * dt
                nx = enemy.x + dx / dist * move
                ny = enemy.y + dy / dist * move

                if not self.circle_collides(nx, enemy.y, 0.22):
                    enemy.x = nx
                if not self.circle_collides(enemy.x, ny, 0.22):
                    enemy.y = ny

    # -------------------------- Rendering --------------------------
    def draw(self):
        self.draw_background()
        self.zbuffer = self.cast_walls()
        self.draw_enemies()
        self.draw_weapon()
        self.draw_hud()
        if self.game_over:
            self.draw_game_over()

    def draw_background(self):
        for y in range(HALF_H):
            t = y / max(1, HALF_H - 1)
            color = (
                int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * t),
                int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * t),
                int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * t),
            )
            pygame.draw.line(self.screen, color, (0, y), (WIDTH, y))
        for y in range(HALF_H, HEIGHT):
            t = (y - HALF_H) / max(1, HALF_H - 1)
            color = (
                int(FLOOR_TOP[0] + (FLOOR_BOTTOM[0] - FLOOR_TOP[0]) * t),
                int(FLOOR_TOP[1] + (FLOOR_BOTTOM[1] - FLOOR_TOP[1]) * t),
                int(FLOOR_TOP[2] + (FLOOR_BOTTOM[2] - FLOOR_TOP[2]) * t),
            )
            pygame.draw.line(self.screen, color, (0, y), (WIDTH, y))

        pygame.draw.rect(self.screen, (25, 27, 31), (0, HALF_H + 40, WIDTH, HEIGHT - HALF_H - 40), 1)
        if self.victory_flash > 0:
            flash = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            flash.fill((255, 225, 130, int(65 * self.victory_flash)))
            self.screen.blit(flash, (0, 0))

    def cast_walls(self):
        zbuffer = [MAX_DEPTH for _ in range(NUM_RAYS)]
        ox, oy = self.player.x, self.player.y
        xm, ym = int(ox), int(oy)
        cur_angle = self.player.angle - HALF_FOV

        for ray in range(NUM_RAYS):
            sin_a = math.sin(cur_angle)
            cos_a = math.cos(cur_angle)
            sin_a = sin_a if abs(sin_a) > 1e-6 else 1e-6
            cos_a = cos_a if abs(cos_a) > 1e-6 else 1e-6

            x_map, y_map = xm, ym

            # verticals
            if cos_a > 0:
                x_vert = x_map + 1
                dx = 1
            else:
                x_vert = x_map - 1e-6
                dx = -1
            depth_vert = (x_vert - ox) / cos_a
            y_vert = oy + depth_vert * sin_a
            delta_depth = dx / cos_a
            dy = delta_depth * sin_a
            wall_type_v = 1
            for _ in range(MAX_DEPTH * 2):
                tile_v = (int(x_vert + (dx - 1) / 2), int(y_vert))
                if tile_v in WORLD_MAP:
                    wall_type_v = WORLD_MAP[tile_v]
                    break
                x_vert += dx
                y_vert += dy
                depth_vert += delta_depth

            # horizontals
            if sin_a > 0:
                y_hor = y_map + 1
                dy_h = 1
            else:
                y_hor = y_map - 1e-6
                dy_h = -1
            depth_hor = (y_hor - oy) / sin_a
            x_hor = ox + depth_hor * cos_a
            delta_depth_h = dy_h / sin_a
            dx_h = delta_depth_h * cos_a
            wall_type_h = 1
            for _ in range(MAX_DEPTH * 2):
                tile_h = (int(x_hor), int(y_hor + (dy_h - 1) / 2))
                if tile_h in WORLD_MAP:
                    wall_type_h = WORLD_MAP[tile_h]
                    break
                x_hor += dx_h
                y_hor += dy_h
                depth_hor += delta_depth_h

            if depth_vert < depth_hor:
                depth = depth_vert
                wall_type = wall_type_v
                shade_mod = 0.86
            else:
                depth = depth_hor
                wall_type = wall_type_h
                shade_mod = 1.0

            depth = max(depth, 0.0001)
            depth *= math.cos(self.player.angle - cur_angle)
            zbuffer[ray] = depth
            proj_h = min(int(SCREEN_DIST / depth), HEIGHT * 2)

            base = WALL_COLORS.get(wall_type, (110, 110, 110))
            fog = max(0.18, 1 / (1 + depth * depth * 0.06))
            shade = shade_mod * fog
            color = (
                int(base[0] * shade),
                int(base[1] * shade),
                int(base[2] * shade),
            )
            x = ray * SCALE
            wall_y = HALF_H - proj_h // 2
            pygame.draw.rect(self.screen, color, (x, wall_y, SCALE + 1, proj_h))
            if proj_h < HEIGHT:
                pygame.draw.rect(self.screen, (0, 0, 0), (x, wall_y, SCALE + 1, proj_h), 1)
            cur_angle += DELTA_ANGLE
        return zbuffer

    def get_enemy_surface(self, size):
        size = max(24, min(240, int(size)))
        key = size
        if key in self.enemy_surface_cache:
            return self.enemy_surface_cache[key]
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx = size // 2
        body_w = int(size * 0.42)
        body_h = int(size * 0.5)
        head_r = int(size * 0.12)
        shoulder_y = int(size * 0.34)
        pygame.draw.ellipse(surf, (22, 24, 28, 245), (cx - body_w // 2, shoulder_y, body_w, body_h))
        pygame.draw.circle(surf, (36, 42, 49, 250), (cx, int(size * 0.21)), head_r)
        pygame.draw.rect(surf, (54, 60, 70, 230), (cx - int(body_w * 0.18), int(size * 0.34), int(body_w * 0.36), int(body_h * 0.48)))
        pygame.draw.line(surf, (170, 38, 38, 220), (cx - int(size * 0.08), int(size * 0.22)), (cx - int(size * 0.03), int(size * 0.22)), 2)
        pygame.draw.line(surf, (170, 38, 38, 220), (cx + int(size * 0.03), int(size * 0.22)), (cx + int(size * 0.08), int(size * 0.22)), 2)
        self.enemy_surface_cache[key] = surf
        return surf

    def draw_enemies(self):
        render_list = []
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            dist = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            delta = (theta - self.player.angle + math.pi) % math.tau - math.pi
            if abs(delta) > HALF_FOV + 0.35:
                continue
            dist *= math.cos(delta)
            if dist <= 0.15:
                continue
            proj = SCREEN_DIST / max(dist, 0.2)
            sprite_size = min(int(proj * 0.95), HEIGHT * 2)
            screen_x = int(HALF_W + math.tan(delta) * SCREEN_DIST - sprite_size / 2)
            bob = int(math.sin(enemy.anim) * max(2, sprite_size * 0.02))
            screen_y = HALF_H - sprite_size // 2 + bob + int(sprite_size * 0.05)
            render_list.append((dist, enemy, screen_x, screen_y, sprite_size))

        render_list.sort(key=lambda item: item[0], reverse=True)
        for dist, enemy, sx, sy, size in render_list:
            surf = pygame.transform.smoothscale(self.get_enemy_surface(size), (size, size))
            start_ray = max(0, sx // SCALE)
            end_ray = min(NUM_RAYS - 1, (sx + size) // SCALE)
            visible = False
            for ray in range(start_ray, end_ray + 1):
                if dist < self.zbuffer[ray] + 0.15:
                    visible = True
                    break
            if not visible:
                continue
            tint = max(80, min(255, int(255 / (1 + dist * 0.2))))
            surf = surf.copy()
            surf.fill((tint, tint, tint, 0), special_flags=pygame.BLEND_RGB_MULT)
            self.screen.blit(surf, (sx, sy))
            # health bar if centered enough
            if abs((sx + size // 2) - HALF_W) < 160 and dist < 6.5:
                bar_w = int(size * 0.5)
                bar_x = sx + size // 2 - bar_w // 2
                bar_y = sy - 10
                hp_ratio = max(0.0, enemy.hp / (70 + self.wave * 8))
                pygame.draw.rect(self.screen, (20, 20, 22), (bar_x, bar_y, bar_w, 5))
                pygame.draw.rect(self.screen, WARNING, (bar_x, bar_y, int(bar_w * hp_ratio), 5))

    def draw_weapon(self):
        bob = math.sin(self.player.steps) * 4
        kick = self.player.weapon_kick * 16
        gun_w, gun_h = 260, 140
        x = HALF_W - gun_w // 2
        y = HEIGHT - gun_h + 26 + int(bob) + int(kick * 0.15)

        shadow = pygame.Surface((gun_w + 20, gun_h + 20), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 85), (22, gun_h - 10, gun_w - 40, 30))
        self.screen.blit(shadow, (x - 10, y - 10))

        # rifle body
        pygame.draw.polygon(self.screen, (25, 27, 32), [
            (x + 25, y + 108), (x + 85, y + 65), (x + 182, y + 60),
            (x + 236, y + 79), (x + 215, y + 113), (x + 100, y + 118)
        ])
        pygame.draw.rect(self.screen, (45, 51, 58), (x + 92, y + 56, 120, 18), border_radius=4)
        pygame.draw.rect(self.screen, (72, 80, 90), (x + 143, y + 43, 50, 16), border_radius=4)
        pygame.draw.rect(self.screen, (86, 76, 59), (x + 65, y + 82, 28, 36), border_radius=5)
        pygame.draw.rect(self.screen, (98, 104, 114), (x + 212, y + 67, 48, 8), border_radius=3)
        pygame.draw.rect(self.screen, (18, 20, 22), (x + 64, y + 108, 44, 14), border_radius=5)

        if self.player.shoot_timer > SHOOT_COOLDOWN * 0.55:
            flash_x = x + 260
            flash_y = y + 71
            pygame.draw.polygon(self.screen, ACCENT, [
                (flash_x, flash_y),
                (flash_x + 46, flash_y - 8),
                (flash_x + 32, flash_y),
                (flash_x + 46, flash_y + 8)
            ])
            pygame.draw.circle(self.screen, (255, 235, 180), (flash_x + 18, flash_y), 10)

    def draw_hud(self):
        # vignette / damage
        if self.player.damage_flash > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((180, 22, 22, int(90 * self.player.damage_flash)))
            self.screen.blit(overlay, (0, 0))

        # top info
        wave_text = self.font.render(f"WAVE {self.wave:02d}", True, HUD)
        score_text = self.small_font.render(f"SCORE {self.player.score:05d}", True, HUD)
        kills_text = self.small_font.render(f"KILLS {self.player.kills:03d}", True, HUD)
        self.screen.blit(wave_text, (16, 16))
        self.screen.blit(score_text, (16, 40))
        self.screen.blit(kills_text, (16, 58))

        # ammo
        ammo_color = WARNING if self.player.ammo <= 5 else WHITE
        ammo_text = self.big_font.render(f"{self.player.ammo:02d}", True, ammo_color)
        self.screen.blit(ammo_text, (WIDTH - ammo_text.get_width() - 18, HEIGHT - 70))
        label = self.small_font.render("AUTO", True, HUD)
        self.screen.blit(label, (WIDTH - 64, HEIGHT - 82))
        if self.player.reload_timer > 0:
            reload_t = self.small_font.render("RELOADING", True, ACCENT)
            self.screen.blit(reload_t, (WIDTH - 118, HEIGHT - 100))

        # health bar
        bar_x, bar_y = 18, HEIGHT - 34
        bar_w, bar_h = 240, 14
        pygame.draw.rect(self.screen, (18, 20, 24), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        hp_ratio = self.player.hp / MAX_HP
        hp_col = WARNING if hp_ratio < 0.35 else (78, 196, 122)
        pygame.draw.rect(self.screen, hp_col, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h), border_radius=4)
        hp_text = self.small_font.render(f"HP {self.player.hp:03d}", True, BLACK if hp_ratio > 0.5 else WHITE)
        self.screen.blit(hp_text, (bar_x + 8, bar_y - 1))

        # crosshair
        c = WARNING if self.crosshair_hit > 0 else RETICLE
        cx, cy = HALF_W, HALF_H
        gap = 6 + int(self.player.weapon_kick * 4)
        ln = 10
        pygame.draw.line(self.screen, c, (cx - gap - ln, cy), (cx - gap, cy), 2)
        pygame.draw.line(self.screen, c, (cx + gap, cy), (cx + gap + ln, cy), 2)
        pygame.draw.line(self.screen, c, (cx, cy - gap - ln), (cx, cy - gap), 2)
        pygame.draw.line(self.screen, c, (cx, cy + gap), (cx, cy + gap + ln), 2)
        pygame.draw.circle(self.screen, c, (cx, cy), 2)

        # controls hint
        hint = self.small_font.render("WASD move   Shift sprint   Mouse aim   LMB shoot   R reload   Tab free mouse", True, HUD)
        self.screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 20))

    def draw_game_over(self):
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 155))
        self.screen.blit(shade, (0, 0))
        title = self.big_font.render("MISSION FAILED", True, WARNING)
        subtitle = self.font.render("Press R to restart", True, WHITE)
        detail = self.small_font.render(f"Final score {self.player.score}   Waves cleared {self.wave - 1}   Kills {self.player.kills}", True, HUD)
        self.screen.blit(title, (HALF_W - title.get_width() // 2, HALF_H - 54))
        self.screen.blit(subtitle, (HALF_W - subtitle.get_width() // 2, HALF_H + 4))
        self.screen.blit(detail, (HALF_W - detail.get_width() // 2, HALF_H + 30))


if __name__ == "__main__":
    Game().run()
