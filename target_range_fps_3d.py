import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    import pygame
except ImportError:
    print("This game requires pygame. Install it with: pip install pygame")
    raise


WIDTH, HEIGHT = 720, 360
HALF_W, HALF_H = WIDTH // 2, HEIGHT // 2
FPS = 60
FOV = math.radians(70)
HALF_FOV = FOV / 2
RAY_COUNT = 220
SCREEN_DIST = HALF_W / math.tan(HALF_FOV)
MAX_DEPTH = 32.0
TEX_SIZE = 64
MOUSE_SENS = 0.003
MOVE_SPEED = 3.4
SPRINT_MULT = 1.45
TURN_SPEED = 2.4
PLAYER_RADIUS = 0.20
MAP_W, MAP_H = 40, 32
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)


@dataclass
class Weapon:
    name: str
    damage: int
    pellet_count: int
    spread: float
    fire_delay: float
    clip_size: int
    reload_time: float
    max_range: float
    kick: float
    color: Tuple[int, int, int]


WEAPONS = [
    Weapon("Revolver", 44, 1, math.radians(1.2), 0.34, 6, 1.8, 24.0, 0.020, (215, 195, 160)),
    Weapon("9mm Pistol", 24, 1, math.radians(0.8), 0.17, 15, 1.25, 20.0, 0.010, (185, 185, 195)),
    Weapon("Bolt Rifle", 92, 1, math.radians(0.25), 1.05, 5, 1.9, 30.0, 0.028, (145, 110, 85)),
    Weapon("Pump Shotgun", 12, 8, math.radians(7.2), 0.92, 8, 2.15, 14.0, 0.050, (120, 85, 50)),
]


@dataclass
class Player:
    x: float
    y: float
    angle: float
    health: int = 100
    score: int = 0
    weapon_index: int = 0
    clip: Dict[int, int] = field(default_factory=lambda: {i: w.clip_size for i, w in enumerate(WEAPONS)})
    reserve: Dict[int, int] = field(default_factory=lambda: {0: 90, 1: 180, 2: 40, 3: 64})
    fire_timer: float = 0.0
    reload_timer: float = 0.0
    reloading: bool = False
    bob_phase: float = 0.0
    kick_back: float = 0.0
    hit_marker: float = 0.0

    def weapon(self) -> Weapon:
        return WEAPONS[self.weapon_index]


@dataclass
class Target:
    x: float
    y: float
    kind: str
    radius: float
    base_score: int
    hp: int = 1
    cooldown: float = 0.0
    phase: float = 0.0
    vx: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.0
    flash: float = 0.0
    visible: bool = True
    spin_dir: float = 1.0


class RangeGame:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Procedural Target Range 3D")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.big_font = pygame.font.SysFont("consolas", 32, bold=True)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.range_seed = random.randint(1000, 999999)
        self.reset_world(new_seed=False)

    def reset_world(self, new_seed: bool = False) -> None:
        if new_seed:
            self.range_seed = random.randint(1000, 999999)
        self.rng = random.Random(self.range_seed)
        self.world = build_world()
        self.textures, self.texture_columns, self.weapon_finish = build_texture_pack(self.rng)
        self.player = Player(19.5, 27.2, -math.pi / 2)
        self.targets = spawn_targets(self.rng)
        self.depth_buffer = [MAX_DEPTH for _ in range(RAY_COUNT)]
        self.message = ""
        self.message_timer = 0.0

    def run(self) -> None:
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.033)
            self.handle_events()
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
                if event.key == pygame.K_TAB:
                    grab = not pygame.event.get_grab()
                    pygame.event.set_grab(grab)
                    pygame.mouse.set_visible(not grab)
                if event.key == pygame.K_r:
                    self.try_reload()
                if event.key == pygame.K_n:
                    self.reset_world(new_seed=True)
                if event.key == pygame.K_F5:
                    self.reset_world(new_seed=True)
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    self.switch_weapon(event.key - pygame.K_1)
                if event.key == pygame.K_SPACE:
                    self.try_fire()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.try_fire()

    def switch_weapon(self, index: int) -> None:
        if index == self.player.weapon_index:
            return
        self.player.weapon_index = index
        if self.player.reloading:
            self.player.reload_timer = 0.0
            self.player.reloading = False

    def try_reload(self) -> None:
        if self.player.reload_timer > 0:
            return
        weapon = self.player.weapon()
        clip_now = self.player.clip[self.player.weapon_index]
        reserve = self.player.reserve[self.player.weapon_index]
        if clip_now >= weapon.clip_size or reserve <= 0:
            return
        self.player.reload_timer = weapon.reload_time
        self.player.reloading = True
        self.message = f"Reloading {weapon.name}"
        self.message_timer = 0.5

    def try_fire(self) -> None:
        if self.player.fire_timer > 0 or self.player.reload_timer > 0:
            return
        weapon = self.player.weapon()
        clip = self.player.clip[self.player.weapon_index]
        if clip <= 0:
            self.try_reload()
            return
        self.player.clip[self.player.weapon_index] -= 1
        self.player.fire_timer = weapon.fire_delay
        self.player.kick_back += weapon.kick
        did_hit = False
        for _ in range(weapon.pellet_count):
            ang = self.player.angle + self.rng.uniform(-weapon.spread / 2, weapon.spread / 2)
            did_hit = self.cast_hitscan(ang, weapon.damage, weapon.max_range) or did_hit
        if did_hit:
            self.player.hit_marker = 0.16

    def cast_hitscan(self, angle: float, damage: int, max_range: float) -> bool:
        step = 0.045
        dist = 0.0
        while dist < max_range:
            dist += step
            px = self.player.x + math.cos(angle) * dist
            py = self.player.y + math.sin(angle) * dist
            if is_solid(self.world, px, py):
                return False
            for target in self.targets:
                if target.cooldown > 0 or not target.visible:
                    continue
                if (target.x - px) ** 2 + (target.y - py) ** 2 <= (target.radius + 0.02) ** 2:
                    target.hp -= damage
                    target.flash = 0.18
                    if target.hp <= 0:
                        self.register_target_hit(target, dist)
                    return True
        return False

    def register_target_hit(self, target: Target, dist: float) -> None:
        bonus = int(dist * 5)
        points = target.base_score + bonus
        self.player.score += points
        target.hp = 1
        target.cooldown = 0.22 if target.kind != "popup" else 1.2
        if target.kind == "bullseye":
            target.flash = 0.22
        elif target.kind == "plate":
            target.flash = 0.22
        elif target.kind == "popup":
            target.visible = False
            target.phase = 0.0
        elif target.kind == "spinner":
            target.flash = 0.25
            target.spin_dir *= -1.0
        self.message = f"Hit {target.kind} +{points}"
        self.message_timer = 0.45

    def update(self, dt: float) -> None:
        self.player.fire_timer = max(0.0, self.player.fire_timer - dt)
        self.player.hit_marker = max(0.0, self.player.hit_marker - dt)
        self.player.kick_back = max(0.0, self.player.kick_back - dt * 0.12)
        if self.player.reload_timer > 0:
            self.player.reload_timer = max(0.0, self.player.reload_timer - dt)
            if self.player.reloading and self.player.reload_timer <= 0:
                weapon = self.player.weapon()
                idx = self.player.weapon_index
                need = weapon.clip_size - self.player.clip[idx]
                reserve = self.player.reserve[idx]
                load = min(need, reserve)
                self.player.clip[idx] += load
                self.player.reserve[idx] -= load
                self.player.reloading = False
        self.message_timer = max(0.0, self.message_timer - dt)
        self.update_player(dt)
        self.update_targets(dt)

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
        dx = 0.0
        dy = 0.0
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

        step_len = math.hypot(dx, dy)
        max_step = move_speed * dt
        if step_len > max_step > 0.0:
            scale = max_step / step_len
            dx *= scale
            dy *= scale

        if abs(dx) + abs(dy) > 0.0001:
            self.player.bob_phase += dt * 9.5
        self.move_with_collisions(dx, dy)

        if pygame.mouse.get_pressed()[0]:
            self.try_fire()

    def move_with_collisions(self, dx: float, dy: float) -> None:
        new_x = self.player.x + dx
        if not circle_hits_wall(self.world, new_x, self.player.y, PLAYER_RADIUS):
            self.player.x = new_x
        new_y = self.player.y + dy
        if not circle_hits_wall(self.world, self.player.x, new_y, PLAYER_RADIUS):
            self.player.y = new_y

    def update_targets(self, dt: float) -> None:
        for target in self.targets:
            target.cooldown = max(0.0, target.cooldown - dt)
            target.flash = max(0.0, target.flash - dt)
            target.phase += dt
            if target.kind == "plate":
                target.x += target.vx * dt
                if target.x <= target.x_min:
                    target.x = target.x_min
                    target.vx = abs(target.vx)
                elif target.x >= target.x_max:
                    target.x = target.x_max
                    target.vx = -abs(target.vx)
                target.visible = True
            elif target.kind == "popup":
                if target.cooldown <= 0:
                    cycle = math.sin(target.phase * 2.0)
                    target.visible = cycle > -0.25
                else:
                    target.visible = False
            elif target.kind == "spinner":
                target.visible = True
            else:
                target.visible = True

    def draw(self) -> None:
        self.draw_background()
        self.cast_walls()
        self.draw_sprites()
        self.draw_weapon()
        self.draw_hud()

    def draw_background(self) -> None:
        self.screen.fill((0, 0, 0))
        ceiling_top = self.textures["ceiling_top"]
        ceiling_bottom = self.textures["ceiling_bottom"]
        floor_top = self.textures["floor_top"]
        floor_bottom = self.textures["floor_bottom"]
        pygame.draw.rect(self.screen, ceiling_top, (0, 0, WIDTH, HALF_H // 2))
        pygame.draw.rect(self.screen, ceiling_bottom, (0, HALF_H // 2, WIDTH, HALF_H // 2))
        pygame.draw.rect(self.screen, floor_top, (0, HALF_H, WIDTH, HALF_H // 2))
        pygame.draw.rect(self.screen, floor_bottom, (0, HALF_H + HALF_H // 2, WIDTH, HALF_H // 2))
        for y in range(HALF_H + 8, HEIGHT, 24):
            shade = 30 + (y - HALF_H) // 8
            pygame.draw.line(self.screen, (shade, shade, shade), (0, y), (WIDTH, y), 1)
        for x in range(0, WIDTH, 48):
            pygame.draw.line(self.screen, (35, 35, 35), (x, HALF_H), (x, HEIGHT), 1)

    def cast_walls(self) -> None:
        ox, oy = self.player.x, self.player.y
        start_angle = self.player.angle - HALF_FOV
        for ray in range(RAY_COUNT):
            ray_angle = start_angle + ray / RAY_COUNT * FOV
            sin_a = math.sin(ray_angle)
            cos_a = math.cos(ray_angle)

            map_x, map_y = int(ox), int(oy)
            delta_x = abs(1 / cos_a) if abs(cos_a) > 1e-6 else 1e6
            delta_y = abs(1 / sin_a) if abs(sin_a) > 1e-6 else 1e6

            if cos_a < 0:
                step_x = -1
                side_dist_x = (ox - map_x) * delta_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - ox) * delta_x
            if sin_a < 0:
                step_y = -1
                side_dist_y = (oy - map_y) * delta_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - oy) * delta_y

            side = 0
            wall_val = 1
            while True:
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_y
                    map_y += step_y
                    side = 1
                wall_val = self.world[map_y][map_x]
                if wall_val > 0:
                    break

            if side == 0:
                depth = (map_x - ox + (1 - step_x) / 2) / safe_ray_divisor(cos_a)
                hit = oy + depth * sin_a
            else:
                depth = (map_y - oy + (1 - step_y) / 2) / safe_ray_divisor(sin_a)
                hit = ox + depth * cos_a
            tex_x = int((hit - math.floor(hit)) * TEX_SIZE) % TEX_SIZE

            corrected = max(0.0001, depth * math.cos(self.player.angle - ray_angle))
            self.depth_buffer[ray] = corrected
            proj_height = min(int(SCREEN_DIST / corrected), HEIGHT * 2)
            y = HALF_H - proj_height // 2
            x0, x1 = ray_column_bounds(ray)

            tex_key = f"wall_{wall_val}"
            tex_col = self.texture_columns[tex_key][tex_x]
            scaled = pygame.transform.scale(tex_col, (x1 - x0, proj_height))
            if side == 1:
                shade = 0.78
                shade_surf = scaled.copy()
                shade_surf.fill((int(255 * shade), int(255 * shade), int(255 * shade)), special_flags=pygame.BLEND_RGB_MULT)
                scaled = shade_surf
            fog = max(0.28, 1.0 - corrected / 24.0)
            fogged = scaled.copy()
            fogged.fill((int(255 * fog), int(255 * fog), int(255 * fog)), special_flags=pygame.BLEND_RGB_MULT)
            self.screen.blit(fogged, (x0, y))

    def draw_sprites(self) -> None:
        sprites = []
        for target in self.targets:
            if target.visible:
                dist = math.dist((self.player.x, self.player.y), (target.x, target.y))
                sprites.append((dist, target))
        sprites.sort(key=lambda item: item[0], reverse=True)

        for dist, target in sprites:
            dx = target.x - self.player.x
            dy = target.y - self.player.y
            theta = math.atan2(dy, dx)
            gamma = (theta - self.player.angle + math.pi) % (math.tau) - math.pi
            if abs(gamma) > HALF_FOV + 0.35:
                continue
            corrected = dist * math.cos(gamma)
            if corrected <= 0.12:
                continue
            proj = SCREEN_DIST / corrected
            height = max(12, int(proj * 1.45))
            width = max(8, int(proj * 0.95))
            if target.kind == "spinner":
                width = max(6, int(width * max(0.18, abs(math.sin(target.phase * 4.0 * target.spin_dir)))))
            elif target.kind == "plate":
                width = int(width * 0.75)
                height = int(height * 0.75)
            elif target.kind == "popup":
                height = int(height * 1.15)
            screen_x = int(HALF_W + math.tan(gamma) * SCREEN_DIST - width / 2)
            screen_y = HALF_H - height // 2
            buffer_x = min(RAY_COUNT - 1, max(0, int((screen_x + width / 2) * RAY_COUNT / WIDTH)))
            if corrected > self.depth_buffer[buffer_x] + target.radius:
                continue

            key = f"target_{target.kind}"
            surf = self.textures[key]
            if target.flash > 0:
                surf = surf.copy()
                surf.fill((255, 255, 255), special_flags=pygame.BLEND_RGB_ADD)
            scaled = pygame.transform.scale(surf, (width, height))
            self.screen.blit(scaled, (screen_x, screen_y))

    def draw_weapon(self) -> None:
        weapon = self.player.weapon()
        finish = self.weapon_finish[self.player.weapon_index]
        sway_x = math.sin(self.player.bob_phase) * 7 + self.player.kick_back * -280
        sway_y = abs(math.cos(self.player.bob_phase * 0.7)) * 5 + self.player.kick_back * 140
        cx = HALF_W + sway_x
        base_y = HEIGHT - 82 + sway_y

        if weapon.name == "Revolver":
            pygame.draw.rect(self.screen, finish[0], (cx - 44, base_y - 12, 88, 28), border_radius=8)
            pygame.draw.circle(self.screen, finish[1], (int(cx - 4), int(base_y + 2)), 18)
            pygame.draw.rect(self.screen, finish[2], (cx - 8, base_y - 46, 22, 42), border_radius=4)
        elif weapon.name == "9mm Pistol":
            pygame.draw.rect(self.screen, finish[0], (cx - 38, base_y - 14, 76, 34), border_radius=8)
            pygame.draw.rect(self.screen, finish[1], (cx - 10, base_y - 50, 22, 42), border_radius=4)
            pygame.draw.rect(self.screen, finish[2], (cx - 26, base_y - 42, 48, 10), border_radius=3)
        elif weapon.name == "Bolt Rifle":
            pygame.draw.rect(self.screen, finish[0], (cx - 104, base_y - 5, 200, 18), border_radius=7)
            pygame.draw.rect(self.screen, finish[1], (cx - 6, base_y - 26, 160, 14), border_radius=5)
            pygame.draw.rect(self.screen, finish[2], (cx + 22, base_y - 34, 54, 10), border_radius=4)
            pygame.draw.circle(self.screen, (90, 70, 50), (int(cx - 10), int(base_y + 9)), 10)
        elif weapon.name == "Pump Shotgun":
            pygame.draw.rect(self.screen, finish[0], (cx - 96, base_y - 8, 176, 22), border_radius=7)
            pygame.draw.rect(self.screen, finish[1], (cx - 18, base_y - 28, 152, 16), border_radius=5)
            pygame.draw.rect(self.screen, finish[2], (cx - 92, base_y + 6, 86, 10), border_radius=4)

        cross = (255, 255, 255)
        if self.player.hit_marker > 0:
            cross = (255, 215, 80)
        pygame.draw.line(self.screen, cross, (HALF_W - 8, HALF_H), (HALF_W + 8, HALF_H), 2)
        pygame.draw.line(self.screen, cross, (HALF_W, HALF_H - 8), (HALF_W, HALF_H + 8), 2)
        if self.player.hit_marker > 0:
            size = 14
            pygame.draw.line(self.screen, cross, (HALF_W - size, HALF_H - size), (HALF_W - size // 2, HALF_H - size // 2), 2)
            pygame.draw.line(self.screen, cross, (HALF_W + size, HALF_H - size), (HALF_W + size // 2, HALF_H - size // 2), 2)
            pygame.draw.line(self.screen, cross, (HALF_W - size, HALF_H + size), (HALF_W - size // 2, HALF_H + size // 2), 2)
            pygame.draw.line(self.screen, cross, (HALF_W + size, HALF_H + size), (HALF_W + size // 2, HALF_H + size // 2), 2)

    def draw_hud(self) -> None:
        weapon = self.player.weapon()
        clip = self.player.clip[self.player.weapon_index]
        reserve = self.player.reserve[self.player.weapon_index]
        lane = int((self.player.x - 4) / 5)
        lane_text = max(1, min(6, lane + 1))
        panel = pygame.Surface((WIDTH, 88), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 110))
        self.screen.blit(panel, (0, HEIGHT - 88))

        left = [
            f"Weapon: {weapon.name}",
            f"Ammo: {clip} / {reserve}",
            f"Score: {self.player.score}",
            f"Seed: {self.range_seed}",
        ]
        for i, text in enumerate(left):
            self.screen.blit(self.font.render(text, True, WHITE), (16, HEIGHT - 80 + i * 18))

        right = [
            f"Lane {lane_text}",
            "1-4 switch  |  R reload",
            "WASD move   |  mouse look",
            "N randomize textures/range",
        ]
        for i, text in enumerate(right):
            surf = self.small_font.render(text, True, (225, 225, 225))
            self.screen.blit(surf, (WIDTH - surf.get_width() - 16, HEIGHT - 80 + i * 18))

        if self.player.reload_timer > 0:
            ratio = 1.0 - (self.player.reload_timer / max(0.001, weapon.reload_time))
            pygame.draw.rect(self.screen, (50, 50, 50), (HALF_W - 100, HEIGHT - 30, 200, 10), border_radius=4)
            pygame.draw.rect(self.screen, (90, 220, 120), (HALF_W - 100, HEIGHT - 30, int(200 * ratio), 10), border_radius=4)
        if self.message_timer > 0 and self.message:
            msg = self.big_font.render(self.message, True, (250, 225, 120))
            self.screen.blit(msg, (HALF_W - msg.get_width() // 2, 28))


def safe_ray_divisor(value: float) -> float:
    if abs(value) >= 1e-6:
        return value
    return 1e-6 if value >= 0.0 else -1e-6


def ray_column_bounds(ray: int) -> Tuple[int, int]:
    x0 = ray * WIDTH // RAY_COUNT
    x1 = (ray + 1) * WIDTH // RAY_COUNT
    return x0, max(x0 + 1, x1)


def build_world() -> List[List[int]]:
    world = [[1 for _ in range(MAP_W)] for _ in range(MAP_H)]
    for y in range(1, MAP_H - 1):
        for x in range(1, MAP_W - 1):
            world[y][x] = 0

    for y in range(3, MAP_H - 2):
        world[y][2] = 2
        world[y][MAP_W - 3] = 2

    for x in range(3, MAP_W - 3):
        world[2][x] = 4
        world[MAP_H - 3][x] = 3

    for x in range(6, MAP_W - 5, 5):
        for y in range(23, MAP_H - 3):
            world[y][x] = 2

    for x in range(4, MAP_W - 4):
        world[22][x] = 3 if x % 2 == 0 else 0

    for x in (8, 14, 20, 26, 32):
        for y in range(10, 14):
            world[y][x] = 2

    for x in range(4, MAP_W - 4):
        if x % 4 in (0, 1):
            world[6][x] = 3

    return world


def spawn_targets(rng: random.Random) -> List[Target]:
    targets: List[Target] = []
    lane_centers = [4.5, 9.5, 14.5, 19.5, 24.5, 29.5, 34.5]
    distances = [18.5, 15.0, 11.5, 8.0]
    for i, x in enumerate(lane_centers[:-1]):
        targets.append(Target(x=x, y=distances[i % len(distances)], kind="bullseye", radius=0.28, base_score=40))
    for x in lane_centers[1:6:2]:
        speed = rng.choice([1.2, 1.6, 2.0]) * (1 if rng.random() < 0.5 else -1)
        targets.append(Target(x=x, y=13.5, kind="plate", radius=0.22, base_score=65, vx=speed, x_min=x - 1.2, x_max=x + 1.2))
    for x in lane_centers[::2]:
        targets.append(Target(x=x, y=9.0, kind="popup", radius=0.24, base_score=85, phase=rng.random() * math.tau))
    for x in (11.5, 19.5, 27.5):
        targets.append(Target(x=x, y=5.0, kind="spinner", radius=0.24, base_score=100, phase=rng.random() * math.tau))
    return targets


def build_texture_pack(rng: random.Random):
    textures: Dict[str, pygame.Surface] = {}
    textures["wall_1"] = make_concrete_texture(rng)
    textures["wall_2"] = make_wood_texture(rng)
    textures["wall_3"] = make_hazard_texture(rng)
    textures["wall_4"] = make_backstop_texture(rng)
    textures["target_bullseye"] = make_bullseye_texture(rng)
    textures["target_plate"] = make_plate_texture(rng)
    textures["target_popup"] = make_popup_texture(rng)
    textures["target_spinner"] = make_spinner_texture(rng)
    textures["ceiling_top"] = rand_tint((35, 40, 46), rng, 14)
    textures["ceiling_bottom"] = rand_tint((48, 52, 58), rng, 12)
    textures["floor_top"] = rand_tint((50, 46, 43), rng, 16)
    textures["floor_bottom"] = rand_tint((36, 32, 30), rng, 12)

    columns = {}
    for key, surf in textures.items():
        if key.startswith("wall_"):
            columns[key] = [surf.subsurface((x, 0, 1, TEX_SIZE)).copy() for x in range(TEX_SIZE)]

    weapon_finish = []
    for base in ((110, 110, 120), (85, 85, 95), (95, 70, 55), (90, 65, 45)):
        weapon_finish.append(
            (
                rand_tint(base, rng, 35),
                rand_tint(tuple(min(255, c + 45) for c in base), rng, 20),
                rand_tint(tuple(max(20, c - 20) for c in base), rng, 20),
            )
        )
    return textures, columns, weapon_finish


def rand_tint(base: Tuple[int, int, int], rng: random.Random, spread: int) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, c + rng.randint(-spread, spread))) for c in base)


def make_concrete_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE))
    base = rand_tint((105, 108, 114), rng, 18)
    surf.fill(base)
    for _ in range(1200):
        x = rng.randrange(TEX_SIZE)
        y = rng.randrange(TEX_SIZE)
        shade = rng.randint(-24, 24)
        c = tuple(max(0, min(255, ch + shade)) for ch in base)
        surf.set_at((x, y), c)
    for _ in range(18):
        y = rng.randrange(TEX_SIZE)
        pygame.draw.line(surf, rand_tint((90, 90, 92), rng, 18), (0, y), (TEX_SIZE, y), 1)
    return surf.convert()


def make_wood_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE))
    base = rand_tint((124, 89, 58), rng, 24)
    surf.fill(base)
    for x in range(TEX_SIZE):
        grain = int(14 * math.sin(x / 4.2 + rng.random() * 0.4)) + rng.randint(-10, 10)
        color = tuple(max(0, min(255, c + grain)) for c in base)
        pygame.draw.line(surf, color, (x, 0), (x, TEX_SIZE))
    for y in range(10, TEX_SIZE, 18):
        pygame.draw.line(surf, rand_tint((80, 50, 34), rng, 15), (0, y), (TEX_SIZE, y), 2)
    return surf.convert()


def make_hazard_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE))
    surf.fill(rand_tint((65, 65, 68), rng, 10))
    yellow = rand_tint((230, 195, 40), rng, 20)
    for offset in range(-TEX_SIZE, TEX_SIZE * 2, 16):
        pygame.draw.polygon(surf, yellow, [(offset, 0), (offset + 8, 0), (offset + TEX_SIZE, TEX_SIZE), (offset + TEX_SIZE - 8, TEX_SIZE)])
    return surf.convert()


def make_backstop_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE))
    base = rand_tint((60, 48, 38), rng, 12)
    surf.fill(base)
    for y in range(0, TEX_SIZE, 6):
        pygame.draw.line(surf, rand_tint((90, 72, 55), rng, 10), (0, y), (TEX_SIZE, y), 3)
    for _ in range(200):
        x = rng.randrange(TEX_SIZE)
        y = rng.randrange(TEX_SIZE)
        pygame.draw.circle(surf, rand_tint((32, 24, 18), rng, 8), (x, y), rng.randint(1, 2))
    return surf.convert()


def make_bullseye_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE), pygame.SRCALPHA)
    paper = rand_tint((230, 225, 205), rng, 10)
    surf.fill((0, 0, 0, 0))
    pygame.draw.rect(surf, paper, (8, 4, 48, 56), border_radius=4)
    cx, cy = TEX_SIZE // 2, TEX_SIZE // 2
    ring_colors = [
        rand_tint((245, 245, 235), rng, 8),
        rand_tint((225, 40, 40), rng, 12),
        rand_tint((30, 30, 30), rng, 10),
        rand_tint((225, 40, 40), rng, 12),
        rand_tint((250, 225, 60), rng, 10),
    ]
    radii = [22, 18, 14, 9, 4]
    for color, radius in zip(ring_colors, radii):
        pygame.draw.circle(surf, color, (cx, cy), radius)
    return surf.convert_alpha()


def make_plate_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    outer = rand_tint((210, 210, 215), rng, 20)
    inner = rand_tint((140, 145, 155), rng, 20)
    pygame.draw.circle(surf, outer, (32, 28), 16)
    pygame.draw.circle(surf, inner, (32, 28), 10)
    pygame.draw.rect(surf, rand_tint((60, 60, 65), rng, 12), (29, 42, 6, 16))
    return surf.convert_alpha()


def make_popup_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    body = rand_tint((90, 70, 55), rng, 12)
    head = rand_tint((115, 90, 70), rng, 12)
    pygame.draw.ellipse(surf, body, (12, 18, 40, 34))
    pygame.draw.circle(surf, head, (32, 14), 10)
    pygame.draw.rect(surf, rand_tint((55, 45, 38), rng, 10), (30, 46, 4, 18))
    return surf.convert_alpha()


def make_spinner_texture(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((TEX_SIZE, TEX_SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    left = rand_tint((220, 70, 70), rng, 18)
    right = rand_tint((60, 180, 230), rng, 18)
    pygame.draw.circle(surf, left, (20, 30), 12)
    pygame.draw.circle(surf, right, (44, 30), 12)
    pygame.draw.rect(surf, rand_tint((80, 80, 85), rng, 12), (30, 10, 4, 42))
    return surf.convert_alpha()


def is_solid(world: List[List[int]], x: float, y: float) -> bool:
    ix, iy = int(x), int(y)
    if iy < 0 or iy >= len(world) or ix < 0 or ix >= len(world[0]):
        return True
    return world[iy][ix] > 0


def circle_hits_wall(world: List[List[int]], x: float, y: float, radius: float) -> bool:
    for ox in (-radius, radius):
        for oy in (-radius, radius):
            if is_solid(world, x + ox, y + oy):
                return True
    return False


if __name__ == "__main__":
    RangeGame().run()
