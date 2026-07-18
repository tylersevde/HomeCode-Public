import math
import random
import sys
from collections import deque

try:
    import pygame
    from pygame.math import Vector2
    from pygame.locals import DOUBLEBUF, OPENGL
except Exception as exc:
    raise SystemExit("This game requires pygame. Install it with: pip install pygame") from exc

try:
    from OpenGL.GL import (
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
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


SCREEN_W = 800
SCREEN_H = 480
FPS = 60
TITLE = "Endless Maze"

CELL_SIZE = 56
WALL_THICKNESS = 8
CHUNK_CELLS = 12
CHUNK_SIZE = CHUNK_CELLS * CELL_SIZE
PLAYER_RADIUS = 12
MOVE_SPEED = 180.0
SPRINT_SPEED = 290.0
TRAIL_LEN = 42
CHUNK_KEEP_RADIUS = 4
MINIMAP_RADIUS = 10

BG = (0.03, 0.035, 0.055, 1.0)
WHITE = (0.94, 0.96, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
FLOOR_A = (0.10, 0.11, 0.15, 1.0)
FLOOR_B = (0.12, 0.13, 0.18, 1.0)
FLOOR_C = (0.08, 0.09, 0.13, 1.0)
WALL_A = (0.22, 0.25, 0.34, 1.0)
WALL_B = (0.30, 0.34, 0.46, 1.0)
WALL_EDGE = (0.44, 0.49, 0.64, 1.0)
PLAYER_CORE = (0.52, 0.82, 1.0, 1.0)
PLAYER_RING = (0.88, 0.95, 1.0, 1.0)
PLAYER_DIR = (0.60, 1.00, 0.82, 1.0)
BEACON_C = (0.58, 0.42, 1.0, 1.0)
BEACON_CORE = (0.94, 0.88, 1.0, 1.0)
TRAIL_C = (0.45, 0.80, 1.0, 0.65)
MAP_BG = (0.02, 0.03, 0.05, 0.70)
MAP_DISCOVERED = (0.30, 0.38, 0.52, 0.92)
MAP_CURRENT = (0.78, 0.95, 1.0, 1.0)
MAP_UNSEEN = (0.08, 0.10, 0.15, 0.80)
FOG_C = (0.12, 0.13, 0.18, 0.09)

DIRS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

TEXT = None


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def mix_hash(*values):
    h = 0x345678
    for value in values:
        v = int(value) & 0xFFFFFFFF
        h = ((h ^ (v + 0x9E3779B9 + ((h << 6) & 0xFFFFFFFF) + (h >> 2))) * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 16
    return h & 0x7FFFFFFF


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


class Camera:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

    def update(self, target):
        self.x += (target.x - SCREEN_W / 2 - self.x) * 0.16
        self.y += (target.y - SCREEN_H / 2 - self.y) * 0.16

    def world_to_screen(self, pos):
        return pos[0] - self.x, pos[1] - self.y


class MazeChunk:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.seed = mix_hash(cx, cy, 0xE11D)
        self.opens = {(x, y): set() for y in range(CHUNK_CELLS) for x in range(CHUNK_CELLS)}
        self.beacons = []
        self.generate()

    @staticmethod
    def vertical_boundary_doors(boundary_x, chunk_y):
        rng = random.Random(mix_hash(boundary_x, chunk_y, 0xA11CE))
        count = 1 + (1 if rng.random() < 0.38 else 0)
        slots = list(range(CHUNK_CELLS))
        rng.shuffle(slots)
        return sorted(slots[:count])

    @staticmethod
    def horizontal_boundary_doors(chunk_x, boundary_y):
        rng = random.Random(mix_hash(chunk_x, boundary_y, 0xB11CE))
        count = 1 + (1 if rng.random() < 0.38 else 0)
        slots = list(range(CHUNK_CELLS))
        rng.shuffle(slots)
        return sorted(slots[:count])

    def carve(self, x, y, direction):
        self.opens[(x, y)].add(direction)
        dx, dy = DIRS[direction]
        nx, ny = x + dx, y + dy
        if 0 <= nx < CHUNK_CELLS and 0 <= ny < CHUNK_CELLS:
            self.opens[(nx, ny)].add(OPPOSITE[direction])

    def generate(self):
        rng = random.Random(self.seed)
        start = (rng.randrange(CHUNK_CELLS), rng.randrange(CHUNK_CELLS))
        stack = [start]
        visited = {start}

        while stack:
            x, y = stack[-1]
            dirs = ["N", "E", "S", "W"]
            rng.shuffle(dirs)
            advanced = False
            for direction in dirs:
                dx, dy = DIRS[direction]
                nx, ny = x + dx, y + dy
                if 0 <= nx < CHUNK_CELLS and 0 <= ny < CHUNK_CELLS and (nx, ny) not in visited:
                    self.carve(x, y, direction)
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                    advanced = True
                    break
            if not advanced:
                stack.pop()

        for _ in range(CHUNK_CELLS * 2):
            x = rng.randrange(CHUNK_CELLS)
            y = rng.randrange(CHUNK_CELLS)
            direction = rng.choice(["N", "E", "S", "W"])
            dx, dy = DIRS[direction]
            nx, ny = x + dx, y + dy
            if 0 <= nx < CHUNK_CELLS and 0 <= ny < CHUNK_CELLS:
                self.carve(x, y, direction)

        for row in self.vertical_boundary_doors(self.cx, self.cy):
            self.opens[(0, row)].add("W")
        for row in self.vertical_boundary_doors(self.cx + 1, self.cy):
            self.opens[(CHUNK_CELLS - 1, row)].add("E")
        for col in self.horizontal_boundary_doors(self.cx, self.cy):
            self.opens[(col, 0)].add("N")
        for col in self.horizontal_boundary_doors(self.cx, self.cy + 1):
            self.opens[(col, CHUNK_CELLS - 1)].add("S")

        if mix_hash(self.cx, self.cy, 777) % 3 == 0:
            for _ in range(1 + mix_hash(self.cx, self.cy, 991) % 2):
                bx = rng.randrange(1, CHUNK_CELLS - 1)
                by = rng.randrange(1, CHUNK_CELLS - 1)
                self.beacons.append((bx, by))


class MazeWorld:
    def __init__(self):
        self.chunks = {}

    def get_chunk(self, cx, cy):
        key = (cx, cy)
        if key not in self.chunks:
            self.chunks[key] = MazeChunk(cx, cy)
        return self.chunks[key]

    def get_openings(self, cell_x, cell_y):
        cx = math.floor(cell_x / CHUNK_CELLS)
        cy = math.floor(cell_y / CHUNK_CELLS)
        lx = cell_x - cx * CHUNK_CELLS
        ly = cell_y - cy * CHUNK_CELLS
        return self.get_chunk(cx, cy).opens[(lx, ly)]

    def prune_far_chunks(self, center_chunk):
        cx, cy = center_chunk
        keep = {}
        for key, chunk in self.chunks.items():
            if abs(key[0] - cx) <= CHUNK_KEEP_RADIUS and abs(key[1] - cy) <= CHUNK_KEEP_RADIUS:
                keep[key] = chunk
        self.chunks = keep

    def wall_rects_near(self, x, y, radius):
        min_cell_x = math.floor((x - radius) / CELL_SIZE) - 1
        max_cell_x = math.floor((x + radius) / CELL_SIZE) + 1
        min_cell_y = math.floor((y - radius) / CELL_SIZE) - 1
        max_cell_y = math.floor((y + radius) / CELL_SIZE) + 1
        rects = []
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                opens = self.get_openings(cell_x, cell_y)
                wx = cell_x * CELL_SIZE
                wy = cell_y * CELL_SIZE
                if "N" not in opens:
                    rects.append((wx, wy, CELL_SIZE, WALL_THICKNESS))
                if "W" not in opens:
                    rects.append((wx, wy, WALL_THICKNESS, CELL_SIZE))
        return rects


class Player:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.dir = Vector2(1, 0)
        self.trail = deque(maxlen=TRAIL_LEN)
        self.distance_record = 0.0
        self.discovered = set()

    def current_cell(self):
        return (math.floor(self.pos.x / CELL_SIZE), math.floor(self.pos.y / CELL_SIZE))

    def current_chunk(self):
        cell_x, cell_y = self.current_cell()
        return (math.floor(cell_x / CHUNK_CELLS), math.floor(cell_y / CHUNK_CELLS))


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        pygame.display.set_mode((SCREEN_W, SCREEN_H), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.running = True
        self.started = False
        self.show_minimap = True
        self.camera = Camera()
        self.world = MazeWorld()
        self.player = Player(CELL_SIZE * 1.5, CELL_SIZE * 1.5)
        self.start_pos = Vector2(self.player.pos)
        self.time_alive = 0.0
        self.explored_chunks = set()
        self.dust_phase = 0.0
        setup_gl()
        global TEXT
        TEXT = GLText()

    def mark_discovery(self):
        cell = self.player.current_cell()
        self.player.discovered.add(cell)
        self.explored_chunks.add(self.player.current_chunk())

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self.started = True
                elif event.key == pygame.K_TAB:
                    self.show_minimap = not self.show_minimap

    def move_player(self, dt):
        keys = pygame.key.get_pressed()
        axis = Vector2(
            (1 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0),
            (1 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0) - (1 if keys[pygame.K_w] or keys[pygame.K_UP] else 0),
        )
        if axis.length_squared() > 0:
            axis = axis.normalize()
            self.player.dir = axis
        speed = SPRINT_SPEED if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else MOVE_SPEED
        vx = axis.x * speed
        vy = axis.y * speed

        new_x = self.player.pos.x + vx * dt
        new_rect = pygame.Rect(new_x - PLAYER_RADIUS, self.player.pos.y - PLAYER_RADIUS, PLAYER_RADIUS * 2, PLAYER_RADIUS * 2)
        for rect in self.world.wall_rects_near(new_x, self.player.pos.y, PLAYER_RADIUS + WALL_THICKNESS + 2):
            wall = pygame.Rect(rect)
            if new_rect.colliderect(wall):
                if vx > 0:
                    new_x = wall.left - PLAYER_RADIUS
                elif vx < 0:
                    new_x = wall.right + PLAYER_RADIUS
                new_rect.x = int(new_x - PLAYER_RADIUS)
        self.player.pos.x = new_x

        new_y = self.player.pos.y + vy * dt
        new_rect = pygame.Rect(self.player.pos.x - PLAYER_RADIUS, new_y - PLAYER_RADIUS, PLAYER_RADIUS * 2, PLAYER_RADIUS * 2)
        for rect in self.world.wall_rects_near(self.player.pos.x, new_y, PLAYER_RADIUS + WALL_THICKNESS + 2):
            wall = pygame.Rect(rect)
            if new_rect.colliderect(wall):
                if vy > 0:
                    new_y = wall.top - PLAYER_RADIUS
                elif vy < 0:
                    new_y = wall.bottom + PLAYER_RADIUS
                new_rect.y = int(new_y - PLAYER_RADIUS)
        self.player.pos.y = new_y

        self.player.trail.append(Vector2(self.player.pos))
        self.player.distance_record = max(self.player.distance_record, self.player.pos.distance_to(self.start_pos))

    def update(self, dt):
        self.handle_events()
        if not self.running:
            return
        if not self.started:
            return
        self.time_alive += dt
        self.dust_phase += dt
        self.move_player(dt)
        self.mark_discovery()
        self.camera.update(self.player.pos)
        self.world.prune_far_chunks(self.player.current_chunk())

    def draw_floor(self):
        min_cell_x = math.floor(self.camera.x / CELL_SIZE) - 2
        max_cell_x = math.floor((self.camera.x + SCREEN_W) / CELL_SIZE) + 2
        min_cell_y = math.floor(self.camera.y / CELL_SIZE) - 2
        max_cell_y = math.floor((self.camera.y + SCREEN_H) / CELL_SIZE) + 2
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                sx = cell_x * CELL_SIZE - self.camera.x
                sy = cell_y * CELL_SIZE - self.camera.y
                h = mix_hash(cell_x, cell_y, 313)
                color = FLOOR_A if h % 3 == 0 else FLOOR_B if h % 3 == 1 else FLOOR_C
                draw_quad(sx, sy, CELL_SIZE, CELL_SIZE, color)
                inset = 6 + (h % 4)
                alpha = 0.04 + (h % 5) * 0.01
                draw_quad(sx + inset, sy + inset, CELL_SIZE - inset * 2, CELL_SIZE - inset * 2, (1.0, 1.0, 1.0, alpha))

    def draw_walls(self):
        min_cell_x = math.floor(self.camera.x / CELL_SIZE) - 2
        max_cell_x = math.floor((self.camera.x + SCREEN_W) / CELL_SIZE) + 2
        min_cell_y = math.floor(self.camera.y / CELL_SIZE) - 2
        max_cell_y = math.floor((self.camera.y + SCREEN_H) / CELL_SIZE) + 2
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                sx = cell_x * CELL_SIZE - self.camera.x
                sy = cell_y * CELL_SIZE - self.camera.y
                opens = self.world.get_openings(cell_x, cell_y)
                tone = WALL_A if mix_hash(cell_x, cell_y, 919) % 2 == 0 else WALL_B
                if "N" not in opens:
                    draw_quad(sx, sy, CELL_SIZE, WALL_THICKNESS, tone)
                    draw_quad(sx, sy, CELL_SIZE, 2, WALL_EDGE)
                if "W" not in opens:
                    draw_quad(sx, sy, WALL_THICKNESS, CELL_SIZE, tone)
                    draw_quad(sx, sy, 2, CELL_SIZE, WALL_EDGE)

    def draw_beacons(self):
        min_chunk_x = math.floor(self.camera.x / CHUNK_SIZE) - 1
        max_chunk_x = math.floor((self.camera.x + SCREEN_W) / CHUNK_SIZE) + 1
        min_chunk_y = math.floor(self.camera.y / CHUNK_SIZE) - 1
        max_chunk_y = math.floor((self.camera.y + SCREEN_H) / CHUNK_SIZE) + 1
        pulse = 0.5 + 0.5 * math.sin(self.time_alive * 2.2)
        for cy in range(min_chunk_y, max_chunk_y + 1):
            for cx in range(min_chunk_x, max_chunk_x + 1):
                chunk = self.world.get_chunk(cx, cy)
                for bx, by in chunk.beacons:
                    wx = cx * CHUNK_SIZE + bx * CELL_SIZE + CELL_SIZE / 2
                    wy = cy * CHUNK_SIZE + by * CELL_SIZE + CELL_SIZE / 2
                    sx, sy = self.camera.world_to_screen((wx, wy))
                    draw_glow(sx, sy, 18 + pulse * 8, BEACON_C, 0.15)
                    draw_circle(sx, sy, 6 + pulse * 2, BEACON_CORE, 18)
                    draw_ring(sx, sy, 13 + pulse * 2, BEACON_C, width=2.0)

    def draw_trail(self):
        points = list(self.player.trail)
        for i, point in enumerate(points):
            t = (i + 1) / max(1, len(points))
            sx, sy = self.camera.world_to_screen(point)
            draw_circle(sx, sy, 2.5 + t * 2.5, (TRAIL_C[0], TRAIL_C[1], TRAIL_C[2], 0.08 + t * 0.25), 14)

    def draw_player(self):
        px, py = self.camera.world_to_screen(self.player.pos)
        draw_glow(px, py, 32, PLAYER_CORE, 0.16)
        draw_circle(px, py, PLAYER_RADIUS + 3, PLAYER_RING, 24)
        draw_circle(px, py, PLAYER_RADIUS, PLAYER_CORE, 24)
        tip = self.player.pos + self.player.dir * 20
        tx, ty = self.camera.world_to_screen(tip)
        draw_line(px, py, tx, ty, PLAYER_DIR, width=3.0)

    def draw_ambient_fog(self):
        for i in range(7):
            x = (i * 137 + self.dust_phase * 24) % (SCREEN_W + 180) - 90
            y = 70 + i * 52 + math.sin(self.dust_phase * 0.7 + i * 1.3) * 12
            draw_glow(x, y, 70, FOG_C, 0.09)

    def draw_minimap(self):
        if not self.show_minimap:
            return
        map_cell = 6
        width = (MINIMAP_RADIUS * 2 + 1) * map_cell
        height = width
        ox = SCREEN_W - width - 18
        oy = 18
        draw_quad(ox - 8, oy - 8, width + 16, height + 16, MAP_BG)
        current = self.player.current_cell()
        for dy in range(-MINIMAP_RADIUS, MINIMAP_RADIUS + 1):
            for dx in range(-MINIMAP_RADIUS, MINIMAP_RADIUS + 1):
                cell = (current[0] + dx, current[1] + dy)
                px = ox + (dx + MINIMAP_RADIUS) * map_cell
                py = oy + (dy + MINIMAP_RADIUS) * map_cell
                if cell == current:
                    draw_quad(px, py, map_cell - 1, map_cell - 1, MAP_CURRENT)
                elif cell in self.player.discovered:
                    draw_quad(px, py, map_cell - 1, map_cell - 1, MAP_DISCOVERED)
                else:
                    draw_quad(px, py, map_cell - 1, map_cell - 1, MAP_UNSEEN)
        TEXT.draw("TAB minimap", ox, oy + height + 8, color=(215, 225, 240), size="small")

    def draw_hud(self):
        dist = int(self.player.pos.distance_to(self.start_pos) / CELL_SIZE)
        farthest = int(self.player.distance_record / CELL_SIZE)
        TEXT.draw(f"Cells from start: {dist}", 16, 14, color=(232, 240, 255))
        TEXT.draw(f"Farthest reached: {farthest}", 16, 36, color=(198, 220, 255), size="small")
        TEXT.draw(f"Cells discovered: {len(self.player.discovered)}", 16, 56, color=(198, 220, 255), size="small")
        TEXT.draw(f"Chunks explored: {len(self.explored_chunks)}", 16, 74, color=(198, 220, 255), size="small")
        TEXT.draw("Move: WASD / Arrows   Sprint: Shift   Quit: Esc", 16, SCREEN_H - 26, color=(220, 228, 245), size="small")

    def draw_title(self):
        draw_quad(0, 0, SCREEN_W, SCREEN_H, (0.01, 0.015, 0.03, 1.0))
        for i in range(12):
            draw_glow(
                80 + i * 62,
                100 + math.sin(i * 1.7 + self.dust_phase) * 10,
                50,
                (0.20 + i * 0.015, 0.26, 0.42, 0.14),
                0.08,
            )
        TEXT.draw("ENDLESS MAZE", SCREEN_W / 2, 120, color=(240, 246, 255), size="big", center=True)
        TEXT.draw("A never-ending procedurally generated labyrinth in PyOpenGL", SCREEN_W / 2, 180, color=(188, 214, 255), center=True)
        TEXT.draw("Explore as far as you can. The maze keeps going.", SCREEN_W / 2, 212, color=(168, 190, 225), center=True)
        TEXT.draw("Press SPACE or ENTER to begin", SCREEN_W / 2, 270, color=(232, 240, 255), center=True)
        TEXT.draw("WASD / Arrows to move, Shift to sprint, Tab to toggle minimap", SCREEN_W / 2, 306, color=(192, 205, 228), size="small", center=True)
        TEXT.draw("Esc quits", SCREEN_W / 2, 330, color=(192, 205, 228), size="small", center=True)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        if not self.started:
            self.draw_title()
            pygame.display.flip()
            return
        self.draw_floor()
        self.draw_beacons()
        self.draw_walls()
        self.draw_trail()
        self.draw_player()
        self.draw_ambient_fog()
        self.draw_minimap()
        self.draw_hud()
        draw_vignette()
        pygame.display.flip()

    def run(self):
        self.mark_discovery()
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.update(dt)
            self.draw()
        self.cleanup()

    def cleanup(self):
        global TEXT
        if TEXT is not None:
            TEXT.cleanup()
        pygame.quit()


# ---------- drawing helpers ----------

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



def draw_circle(x, y, r, color, segments=28):
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
    glBegin(0x0002)  # GL_LINE_LOOP
    for i in range(segments):
        ang = math.tau * i / segments
        glVertex2f(x + math.cos(ang) * r, y + math.sin(ang) * r)
    glEnd()



def draw_line(x1, y1, x2, y2, color, width=1.0):
    glLineWidth(width)
    color4(color)
    glBegin(0x0003)  # GL_LINE_STRIP
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()



def draw_glow(x, y, r, color, alpha=0.16):
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    for i in range(5, 0, -1):
        rr = r * i / 5
        aa = alpha * (i / 5) * 0.7
        draw_circle(x, y, rr, (color[0], color[1], color[2], aa), 24)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)



def draw_vignette():
    border = 80
    draw_quad(0, 0, SCREEN_W, border, (0.0, 0.0, 0.0, 0.18))
    draw_quad(0, SCREEN_H - border, SCREEN_W, border, (0.0, 0.0, 0.0, 0.20))
    draw_quad(0, 0, border, SCREEN_H, (0.0, 0.0, 0.0, 0.14))
    draw_quad(SCREEN_W - border, 0, border, SCREEN_H, (0.0, 0.0, 0.0, 0.14))


if __name__ == "__main__":
    try:
        Game().run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
