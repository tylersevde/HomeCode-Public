import sys  
import math  
import random  
from array import array  
  
import pygame  
  
pygame.mixer.pre_init(44100, -16, 1, 512)  
  
# --------------------------------------------------  
# Settings  
# --------------------------------------------------  
BOARD_SIZE = 8  
SHIPS = [  
    {"name": "Battleship", "length": 4},  
    {"name": "Cruiser", "length": 3},  
    {"name": "Submarine", "length": 3},  
    {"name": "Destroyer", "length": 2},  
    {"name": "Patrol Boat", "length": 2},  
]  
  
CELL_SIZE = 44  
GRID_GAP = 80  
MARGIN = 40  
TOP_MARGIN = 110  
BOTTOM_MARGIN = 155  
  
GRID_PIXELS = BOARD_SIZE * CELL_SIZE  
WINDOW_WIDTH = MARGIN * 2 + GRID_PIXELS * 2 + GRID_GAP  
WINDOW_HEIGHT = TOP_MARGIN + GRID_PIXELS + BOTTOM_MARGIN  
  
FPS = 60  
  
WATER = 0  
SHIP = 1  
HIT = 2  
MISS = 3  
  
PLAYER_ORIGIN = (MARGIN, TOP_MARGIN)  
ENEMY_ORIGIN = (MARGIN + GRID_PIXELS + GRID_GAP, TOP_MARGIN)  
  
# Colors  
BG_TOP = (10, 22, 38)  
BG_BOTTOM = (18, 40, 62)  
  
GRID_BG = (24, 45, 70)  
GRID_LINE = (90, 120, 150)  
  
TEXT = (235, 240, 245)  
SUBTEXT = (185, 195, 208)  
TITLE_COLOR = (140, 220, 255)  
  
PLAYER_LABEL = (120, 255, 180)  
ENEMY_LABEL = (255, 190, 130)  
  
SHIP_HULL = (115, 130, 150)  
SHIP_DECK = (150, 165, 185)  
SHIP_TRIM = (80, 95, 110)  
SHIP_SUNK = (145, 80, 80)  
  
HIT_COLOR = (235, 95, 80)  
MISS_COLOR = (220, 230, 235)  
SUNK_COLOR = (255, 220, 110)  
  
HOVER_GOOD = (70, 165, 110)  
HOVER_BAD = (175, 80, 80)  
  
BUTTON_BG = (35, 52, 76)  
BUTTON_HOVER = (58, 86, 120)  
BUTTON_BORDER = (110, 150, 195)  
BUTTON_TEXT = (240, 244, 248)  
BUTTON_DANGER = (118, 58, 58)  
BUTTON_DANGER_HOVER = (160, 78, 78)  
  
BANNER_BG = (22, 34, 50)  
BANNER_PLAYER = (90, 220, 150)  
BANNER_ENEMY = (255, 170, 110)  
BANNER_NEUTRAL = (140, 210, 255)  
  
FLASH_ORANGE = (255, 180, 70)  
FLASH_YELLOW = (255, 245, 130)  
FLASH_RED = (255, 100, 75)  
SPLASH_BLUE = (120, 210, 255)  
  
  
# --------------------------------------------------  
# Utility  
# --------------------------------------------------  
def clamp(value, low, high):  
    return max(low, min(value, high))  
  
  
def create_board():  
    return [[WATER for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]  
  
  
def in_bounds(row, col):  
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE  
  
  
def can_place_ship(board, row, col, length, horizontal):  
    for i in range(length):  
        r = row if horizontal else row + i  
        c = col + i if horizontal else col  
        if not in_bounds(r, c):  
            return False  
        if board[r][c] != WATER:  
            return False  
    return True  
  
  
def place_ship_cells(board, row, col, length, horizontal):  
    cells = []  
    for i in range(length):  
        r = row if horizontal else row + i  
        c = col + i if horizontal else col  
        board[r][c] = SHIP  
        cells.append((r, c))  
    return cells  
  
  
def remove_ship_cells(board, cells):  
    for r, c in cells:  
        board[r][c] = WATER  
  
  
def create_ship_record(name, cells, horizontal):  
    return {  
        "name": name,  
        "cells": list(cells),  
        "horizontal": horizontal,  
        "hits": 0,  
        "sunk": False,  
    }  
  
  
def place_named_ship(board, ships, row, col, ship_info, horizontal):  
    cells = place_ship_cells(board, row, col, ship_info["length"], horizontal)  
    ship = create_ship_record(ship_info["name"], cells, horizontal)  
    ships.append(ship)  
    return ship  
  
  
def random_place_all_named_ships(board, ship_defs):  
    ships = []  
    for ship_info in ship_defs:  
        while True:  
            row = random.randint(0, BOARD_SIZE - 1)  
            col = random.randint(0, BOARD_SIZE - 1)  
            horizontal = random.choice([True, False])  
            if can_place_ship(board, row, col, ship_info["length"], horizontal):  
                place_named_ship(board, ships, row, col, ship_info, horizontal)  
                break  
    return ships  
  
  
def find_ship_at(ships, row, col):  
    for ship in ships:  
        if (row, col) in ship["cells"]:  
            return ship  
    return None  
  
  
def ships_remaining(ships):  
    return sum(1 for ship in ships if not ship["sunk"])  
  
  
def ship_preview_cells(row, col, length, horizontal):  
    cells = []  
    for i in range(length):  
        r = row if horizontal else row + i  
        c = col + i if horizontal else col  
        cells.append((r, c))  
    return cells  
  
  
def fire_at(board, ships, row, col):  
    if board[row][col] in (HIT, MISS):  
        return "already", None, False  
  
    if board[row][col] == SHIP:  
        board[row][col] = HIT  
        ship = find_ship_at(ships, row, col)  
        sunk_now = False  
        if ship is not None:  
            ship["hits"] += 1  
            if ship["hits"] >= len(ship["cells"]) and not ship["sunk"]:  
                ship["sunk"] = True  
                sunk_now = True  
        return "hit", ship, sunk_now  
  
    board[row][col] = MISS  
    return "miss", None, False  
  
  
def point_to_cell(pos, origin):  
    mx, my = pos  
    x0, y0 = origin  
    if not (x0 <= mx < x0 + GRID_PIXELS and y0 <= my < y0 + GRID_PIXELS):  
        return None  
    col = (mx - x0) // CELL_SIZE  
    row = (my - y0) // CELL_SIZE  
    return int(row), int(col)  
  
  
def cell_center(origin, row, col):  
    x0, y0 = origin  
    return (  
        x0 + col * CELL_SIZE + CELL_SIZE // 2,  
        y0 + row * CELL_SIZE + CELL_SIZE // 2,  
    )  
  
  
def draw_text(surface, text, font, color, x, y, center=False):  
    img = font.render(text, True, color)  
    rect = img.get_rect()  
    if center:  
        rect.center = (x, y)  
    else:  
        rect.topleft = (x, y)  
    surface.blit(img, rect)  
  
  
def draw_alpha_circle(surface, color_rgba, center, radius, width=0):  
    size = radius * 2 + 6  
    temp = pygame.Surface((size, size), pygame.SRCALPHA)  
    pygame.draw.circle(temp, color_rgba, (size // 2, size // 2), radius, width)  
    surface.blit(temp, (center[0] - size // 2, center[1] - size // 2))  
  
  
def draw_alpha_rect(surface, color_rgba, rect, border_radius=0):  
    temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)  
    pygame.draw.rect(temp, color_rgba, temp.get_rect(), border_radius=border_radius)  
    surface.blit(temp, rect.topleft)  
  
  
# --------------------------------------------------  
# Sound  
# --------------------------------------------------  
def generate_tone(freq_hz, duration_ms, volume=0.25, wave="sine"):  
    sample_rate = 44100  
    length = int(sample_rate * duration_ms / 1000)  
    buf = array("h")  
  
    for i in range(length):  
        t = i / sample_rate  
        if wave == "square":  
            sample = 1.0 if math.sin(2 * math.pi * freq_hz * t) >= 0 else -1.0  
        else:  
            sample = math.sin(2 * math.pi * freq_hz * t)  
  
        env = 1.0  
        if i < 120:  
            env = i / 120  
        elif length - i < 240:  
            env = max(0.0, (length - i) / 240)  
  
        value = int(32767 * volume * sample * env)  
        buf.append(value)  
  
    return pygame.mixer.Sound(buffer=buf.tobytes())  
  
  
class SoundBank:  
    def __init__(self):  
        self.available = False  
        self.muted = False  
        self.sounds = {}  
  
        try:  
            if not pygame.mixer.get_init():  
                pygame.mixer.init()  
            self.available = True  
            self._build()  
        except pygame.error:  
            self.available = False  
  
    def _build(self):  
        self.sounds["click"] = generate_tone(560, 70, 0.16, "square")  
        self.sounds["place"] = generate_tone(440, 90, 0.18)  
        self.sounds["rotate"] = generate_tone(670, 75, 0.14, "square")  
        self.sounds["undo"] = generate_tone(320, 110, 0.16)  
        self.sounds["error"] = generate_tone(190, 120, 0.18, "square")  
        self.sounds["miss"] = generate_tone(230, 120, 0.16)  
        self.sounds["hit"] = generate_tone(790, 120, 0.22, "square")  
        self.sounds["sink"] = generate_tone(530, 240, 0.24)  
        self.sounds["win"] = generate_tone(900, 280, 0.24)  
        self.sounds["lose"] = generate_tone(160, 320, 0.24)  
  
    def play(self, name):  
        if self.available and not self.muted and name in self.sounds:  
            self.sounds[name].play()  
  
    def toggle_mute(self):  
        self.muted = not self.muted  
  
  
# --------------------------------------------------  
# AI  
# --------------------------------------------------  
class BaseAI:  
    def __init__(self):  
        self.shots_taken = set()  
  
    def random_open_shot(self):  
        while True:  
            shot = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))  
            if shot not in self.shots_taken:  
                self.shots_taken.add(shot)  
                return shot  
  
    def choose_shot(self):  
        raise NotImplementedError  
  
    def process_result(self, row, col, result, sunk_now):  
        pass  
  
  
class EasyAI(BaseAI):  
    def choose_shot(self):  
        return self.random_open_shot()  
  
  
class NormalAI(BaseAI):  
    def __init__(self):  
        super().__init__()  
        self.target_queue = []  
        self.hunt_cells = self._build_pattern()  
  
    def _build_pattern(self):  
        a = []  
        b = []  
        for r in range(BOARD_SIZE):  
            for c in range(BOARD_SIZE):  
                if (r + c) % 2 == 0:  
                    a.append((r, c))  
                else:  
                    b.append((r, c))  
        random.shuffle(a)  
        random.shuffle(b)  
        return a + b  
  
    def _add_neighbors(self, row, col):  
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]  
        random.shuffle(neighbors)  
        for nr, nc in neighbors:  
            if in_bounds(nr, nc) and (nr, nc) not in self.shots_taken and (nr, nc) not in self.target_queue:  
                self.target_queue.append((nr, nc))  
  
    def choose_shot(self):  
        while self.target_queue:  
            shot = self.target_queue.pop(0)  
            if shot not in self.shots_taken:  
                self.shots_taken.add(shot)  
                return shot  
  
        while self.hunt_cells:  
            shot = self.hunt_cells.pop(0)  
            if shot not in self.shots_taken:  
                self.shots_taken.add(shot)  
                return shot  
  
        return self.random_open_shot()  
  
    def process_result(self, row, col, result, sunk_now):  
        if result == "hit" and not sunk_now:  
            self._add_neighbors(row, col)  
  
  
class MeanAI(BaseAI):  
    def __init__(self):  
        super().__init__()  
        self.target_queue = []  
        self.hit_cluster = []  
        self.hunt_cells = self._build_pattern()  
  
    def _build_pattern(self):  
        a = []  
        b = []  
        for r in range(BOARD_SIZE):  
            for c in range(BOARD_SIZE):  
                if (r + c) % 2 == 0:  
                    a.append((r, c))  
                else:  
                    b.append((r, c))  
        random.shuffle(a)  
        random.shuffle(b)  
        return a + b  
  
    def _enqueue_front(self, cells):  
        for cell in reversed(cells):  
            if in_bounds(*cell) and cell not in self.shots_taken and cell not in self.target_queue:  
                self.target_queue.insert(0, cell)  
  
    def _refresh_targets(self):  
        if not self.hit_cluster:  
            return  
  
        if len(self.hit_cluster) == 1:  
            r, c = self.hit_cluster[0]  
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]  
            random.shuffle(neighbors)  
            self._enqueue_front(neighbors)  
            return  
  
        rows = {r for r, _ in self.hit_cluster}  
        cols = {c for _, c in self.hit_cluster}  
  
        if len(rows) == 1:  
            row = next(iter(rows))  
            cols_sorted = sorted(c for _, c in self.hit_cluster)  
            self._enqueue_front([(row, cols_sorted[0] - 1), (row, cols_sorted[-1] + 1)])  
        elif len(cols) == 1:  
            col = next(iter(cols))  
            rows_sorted = sorted(r for r, _ in self.hit_cluster)  
            self._enqueue_front([(rows_sorted[0] - 1, col), (rows_sorted[-1] + 1, col)])  
  
    def choose_shot(self):  
        while self.target_queue:  
            shot = self.target_queue.pop(0)  
            if shot not in self.shots_taken and in_bounds(*shot):  
                self.shots_taken.add(shot)  
                return shot  
  
        while self.hunt_cells:  
            shot = self.hunt_cells.pop(0)  
            if shot not in self.shots_taken:  
                self.shots_taken.add(shot)  
                return shot  
  
        return self.random_open_shot()  
  
    def process_result(self, row, col, result, sunk_now):  
        if result == "hit":  
            self.hit_cluster.append((row, col))  
            if sunk_now:  
                self.hit_cluster.clear()  
                self.target_queue.clear()  
            else:  
                self._refresh_targets()  
  
  
def create_ai(difficulty):  
    if difficulty == "easy":  
        return EasyAI()  
    if difficulty == "normal":  
        return NormalAI()  
    return MeanAI()  
  
  
# --------------------------------------------------  
# Effects  
# --------------------------------------------------  
def effect_board_origin(board_name):  
    return PLAYER_ORIGIN if board_name == "player" else ENEMY_ORIGIN  
  
  
def draw_effects(surface, effects, now_ms):  
    for effect in effects:  
        origin = effect_board_origin(effect["board"])  
        elapsed = now_ms - effect["start"]  
        duration = effect["duration"]  
        if duration <= 0:  
            continue  
        t = clamp(elapsed / duration, 0.0, 1.0)  
  
        if effect["kind"] == "splash":  
            cx, cy = cell_center(origin, effect["row"], effect["col"])  
            alpha = int(190 * (1 - t))  
            ring_r = int(6 + 16 * t)  
            draw_alpha_circle(surface, (*SPLASH_BLUE, alpha), (cx, cy), ring_r, width=3)  
  
            for i in range(5):  
                angle = (math.pi * 2 / 5) * i + t * 3.2  
                px = cx + math.cos(angle) * (6 + 10 * t)  
                py = cy + math.sin(angle) * (6 + 10 * t)  
                draw_alpha_circle(  
                    surface,  
                    (220, 245, 255, max(0, alpha - 30)),  
                    (int(px), int(py)),  
                    max(1, int(3 - t * 2)),  
                )  
  
        elif effect["kind"] == "explosion":  
            cx, cy = cell_center(origin, effect["row"], effect["col"])  
            outer = int(8 + 15 * t)  
            inner = int(4 + 7 * (1 - t))  
            alpha_outer = int(210 * (1 - t))  
            alpha_inner = int(240 * (1 - t * 0.7))  
  
            draw_alpha_circle(surface, (*FLASH_ORANGE, alpha_outer), (cx, cy), outer)  
            draw_alpha_circle(surface, (*FLASH_YELLOW, alpha_inner), (cx, cy), max(2, inner))  
  
            for i in range(6):  
                angle = (math.pi * 2 / 6) * i + t  
                px = cx + math.cos(angle) * (8 + 12 * t)  
                py = cy + math.sin(angle) * (8 + 12 * t)  
                draw_alpha_circle(  
                    surface,  
                    (*FLASH_RED, max(0, alpha_outer - 25)),  
                    (int(px), int(py)),  
                    max(1, int(3 - t * 2)),  
                )  
  
        elif effect["kind"] == "sink_flash":  
            pulse = 0.5 + 0.5 * math.sin(t * math.pi * 8)  
            alpha = int((1 - t) * (80 + 100 * pulse))  
            for row, col in effect["cells"]:  
                x = origin[0] + col * CELL_SIZE + 2  
                y = origin[1] + row * CELL_SIZE + 2  
                rect = pygame.Rect(x, y, CELL_SIZE - 4, CELL_SIZE - 4)  
                draw_alpha_rect(surface, (255, 230, 90, alpha), rect, border_radius=8)  
  
  
# --------------------------------------------------  
# Drawing  
# --------------------------------------------------  
def draw_ocean_background(surface, now_ms):  
    # gradient  
    for y in range(WINDOW_HEIGHT):  
        t = y / max(1, WINDOW_HEIGHT - 1)  
        r = int(BG_TOP[0] * (1 - t) + BG_BOTTOM[0] * t)  
        g = int(BG_TOP[1] * (1 - t) + BG_BOTTOM[1] * t)  
        b = int(BG_TOP[2] * (1 - t) + BG_BOTTOM[2] * t)  
        pygame.draw.line(surface, (r, g, b), (0, y), (WINDOW_WIDTH, y))  
  
    wave_layer = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)  
  
    # moving wave lines  
    for band in range(7):  
        y_base = 90 + band * 78  
        pts = []  
        for x in range(-40, WINDOW_WIDTH + 40, 18):  
            y = y_base + math.sin((x * 0.014) + now_ms * 0.0012 + band) * (6 + band % 3)  
            pts.append((x, y))  
        if len(pts) >= 2:  
            pygame.draw.lines(wave_layer, (160, 210, 255, 22), False, pts, 2)  
  
    # drifting highlight dots  
    for i in range(26):  
        x = (i * 41 + now_ms * (0.02 + i * 0.0007)) % (WINDOW_WIDTH + 60) - 30  
        y = 70 + (i * 29) % (WINDOW_HEIGHT - 140)  
        alpha = 20 + (i * 7) % 35  
        radius = 1 + (i % 2)  
        pygame.draw.circle(wave_layer, (220, 245, 255, alpha), (int(x), int(y)), radius)  
  
    surface.blit(wave_layer, (0, 0))  
  
  
def draw_button(surface, rect, label, fonts, hovered=False, danger=False):  
    fill = BUTTON_DANGER_HOVER if danger and hovered else BUTTON_DANGER if danger else BUTTON_HOVER if hovered else BUTTON_BG  
    pygame.draw.rect(surface, fill, rect, border_radius=12)  
    pygame.draw.rect(surface, BUTTON_BORDER, rect, 2, border_radius=12)  
    draw_text(surface, label, fonts["small"], BUTTON_TEXT, rect.centerx, rect.centery, center=True)  
  
  
def draw_ship_sprite(surface, origin, ship, show_sunk=False, alpha_dim=False):  
    if not ship["cells"]:  
        return  
  
    rows = [r for r, _ in ship["cells"]]  
    cols = [c for _, c in ship["cells"]]  
    min_r, max_r = min(rows), max(rows)  
    min_c, max_c = min(cols), max(cols)  
  
    x = origin[0] + min_c * CELL_SIZE + 4  
    y = origin[1] + min_r * CELL_SIZE + 6  
    w = (max_c - min_c + 1) * CELL_SIZE - 8  
    h = (max_r - min_r + 1) * CELL_SIZE - 12  
  
    temp = pygame.Surface((w, h), pygame.SRCALPHA)  
  
    hull_color = SHIP_SUNK if ship["sunk"] and show_sunk else SHIP_HULL  
    deck_color = SHIP_DECK if not ship["sunk"] else (185, 120, 110)  
    trim_color = SHIP_TRIM  
  
    if alpha_dim:  
        hull = (*hull_color, 175)  
        deck = (*deck_color, 175)  
        trim = (*trim_color, 175)  
    else:  
        hull = (*hull_color, 255)  
        deck = (*deck_color, 255)  
        trim = (*trim_color, 255)  
  
    if ship["horizontal"]:  
        hull_rect = pygame.Rect(2, h // 2 - 10, w - 4, 20)  
        pygame.draw.rect(temp, hull, hull_rect, border_radius=10)  
        nose = [(w - 10, h // 2 - 10), (w - 2, h // 2), (w - 10, h // 2 + 10)]  
        pygame.draw.polygon(temp, hull, nose)  
        deck_rect = pygame.Rect(10, h // 2 - 6, max(10, w - 28), 12)  
        pygame.draw.rect(temp, deck, deck_rect, border_radius=6)  
  
        for i in range(max(1, len(ship["cells"]) - 1)):  
            tx = 12 + i * max(12, (w - 28) // max(1, len(ship["cells"]) - 1))  
            pygame.draw.rect(temp, trim, pygame.Rect(tx, h // 2 - 11, 4, 22), border_radius=2)  
    else:  
        hull_rect = pygame.Rect(w // 2 - 10, 2, 20, h - 4)  
        pygame.draw.rect(temp, hull, hull_rect, border_radius=10)  
        nose = [(w // 2 - 10, h - 10), (w // 2, h - 2), (w // 2 + 10, h - 10)]  
        pygame.draw.polygon(temp, hull, nose)  
        deck_rect = pygame.Rect(w // 2 - 6, 10, 12, max(10, h - 28))  
        pygame.draw.rect(temp, deck, deck_rect, border_radius=6)  
  
        for i in range(max(1, len(ship["cells"]) - 1)):  
            ty = 12 + i * max(12, (h - 28) // max(1, len(ship["cells"]) - 1))  
            pygame.draw.rect(temp, trim, pygame.Rect(w // 2 - 11, ty, 22, 4), border_radius=2)  
  
    surface.blit(temp, (x, y))  
  
  
def draw_board(surface, board, ships, origin, title, label_color, fonts, hide_ships=False, show_sunk_enemy=False,  
               hover_cells=None, hover_ok=True):  
    x0, y0 = origin  
    hover_cells = hover_cells or []  
    hover_color = HOVER_GOOD if hover_ok else HOVER_BAD  
  
    draw_text(surface, title, fonts["medium"], label_color, x0, y0 - 38)  
  
    for col in range(BOARD_SIZE):  
        label = chr(ord("A") + col)  
        x = x0 + col * CELL_SIZE + CELL_SIZE // 2  
        draw_text(surface, label, fonts["small"], SUBTEXT, x, y0 - 22, center=True)  
  
    for row in range(BOARD_SIZE):  
        label = str(row + 1)  
        y = y0 + row * CELL_SIZE + CELL_SIZE // 2  
        draw_text(surface, label, fonts["small"], SUBTEXT, x0 - 18, y, center=True)  
  
    # cell backgrounds + hover  
    for row in range(BOARD_SIZE):  
        for col in range(BOARD_SIZE):  
            x = x0 + col * CELL_SIZE  
            y = y0 + row * CELL_SIZE  
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)  
            pygame.draw.rect(surface, GRID_BG, rect)  
  
            if (row, col) in hover_cells:  
                inner = rect.inflate(-4, -4)  
                pygame.draw.rect(surface, hover_color, inner, border_radius=7)  
  
    # ship sprites  
    if not hide_ships:  
        for ship in ships:  
            draw_ship_sprite(surface, origin, ship, show_sunk=True, alpha_dim=False)  
    elif show_sunk_enemy:  
        for ship in ships:  
            if ship["sunk"]:  
                draw_ship_sprite(surface, origin, ship, show_sunk=True, alpha_dim=True)  
  
    # hit/miss marks  
    for row in range(BOARD_SIZE):  
        for col in range(BOARD_SIZE):  
            x = x0 + col * CELL_SIZE  
            y = y0 + row * CELL_SIZE  
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)  
            cell = board[row][col]  
  
            if cell == HIT:  
                pygame.draw.circle(surface, HIT_COLOR, rect.center, CELL_SIZE // 3)  
            elif cell == MISS:  
                pygame.draw.circle(surface, MISS_COLOR, rect.center, CELL_SIZE // 6)  
  
    # grid lines  
    for row in range(BOARD_SIZE):  
        for col in range(BOARD_SIZE):  
            x = x0 + col * CELL_SIZE  
            y = y0 + row * CELL_SIZE  
            pygame.draw.rect(surface, GRID_LINE, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), 1)  
  
  
def draw_ship_status(surface, ships, x, y, title, fonts, is_enemy=False):  
    draw_text(surface, title, fonts["small"], SUBTEXT, x, y)  
    y += 24  
    for ship in ships:  
        if ship["sunk"]:  
            label = f"{ship['name']}: SUNK"  
            color = SUNK_COLOR  
        else:  
            if is_enemy:  
                label = f"{ship['name']}: ?"  
            else:  
                remaining = len(ship["cells"]) - ship["hits"]  
                label = f"{ship['name']}: {remaining} left"  
            color = TEXT  
        draw_text(surface, label, fonts["tiny"], color, x, y)  
        y += 18  
  
  
def draw_turn_banner(surface, fonts, game, now_ms):  
    if game.phase == "placement":  
        text = "PLACE YOUR SHIPS"  
        color = BANNER_NEUTRAL  
    elif game.phase == "battle":  
        if game.ai_pending:  
            text = "ENEMY TURN"  
            color = BANNER_ENEMY  
        else:  
            text = "YOUR TURN"  
            color = BANNER_PLAYER  
    elif game.phase == "game_over":  
        text = "YOU WIN" if game.winner == "player" else "YOU LOSE"  
        color = BANNER_PLAYER if game.winner == "player" else BANNER_ENEMY  
    else:  
        return  
  
    pulse = 0.5 + 0.5 * math.sin(now_ms * 0.006)  
    alpha = int(120 + 60 * pulse)  
  
    banner_rect = pygame.Rect(WINDOW_WIDTH // 2 - 150, 58, 300, 38)  
    draw_alpha_rect(surface, (*BANNER_BG, alpha), banner_rect, border_radius=14)  
    pygame.draw.rect(surface, color, banner_rect, 2, border_radius=14)  
    draw_text(surface, text, fonts["small"], color, banner_rect.centerx, banner_rect.centery, center=True)  
  
  
# --------------------------------------------------  
# Game State  
# --------------------------------------------------  
class Game:  
    def __init__(self, sounds):  
        self.sounds = sounds  
        self.full_reset()  
  
    def full_reset(self):  
        self.phase = "menu"  
        self.difficulty = None  
        self.ai = None  
        self.message = "Choose an AI difficulty."  
        self.submessage = ""  
        self.winner = None  
  
        self.player_board = create_board()  
        self.enemy_board = create_board()  
        self.player_ships = []  
        self.enemy_ships = []  
  
        self.horizontal = True  
        self.placement_index = 0  
        self.placed_ships = []  
  
        self.player_turn = True  
        self.ai_pending = False  
        self.ai_move_time = 0  
  
        self.effects = []  
  
    def start_new_match(self, difficulty):  
        self.difficulty = difficulty  
        self.ai = create_ai(difficulty)  
  
        self.player_board = create_board()  
        self.enemy_board = create_board()  
        self.player_ships = []  
        self.enemy_ships = random_place_all_named_ships(self.enemy_board, SHIPS)  
  
        self.horizontal = True  
        self.placement_index = 0  
        self.placed_ships = []  
  
        self.player_turn = True  
        self.ai_pending = False  
        self.ai_move_time = 0  
  
        self.phase = "placement"  
        self.winner = None  
        self.effects = []  
  
        self.message = f"{difficulty.title()} difficulty selected."  
        self.submessage = "Place your ships on the left board."  
        self.sounds.play("click")  
  
    def restart_same_difficulty(self):  
        if self.difficulty is None:  
            self.full_reset()  
        else:  
            self.start_new_match(self.difficulty)  
  
    def current_ship_info(self):  
        if self.placement_index < len(SHIPS):  
            return SHIPS[self.placement_index]  
        return None  
  
    def add_effect(self, kind, board, row=None, col=None, cells=None, duration=600):  
        self.effects.append({  
            "kind": kind,  
            "board": board,  
            "row": row,  
            "col": col,  
            "cells": list(cells) if cells else [],  
            "start": pygame.time.get_ticks(),  
            "duration": duration,  
        })  
  
    def rotate_ship(self):  
        if self.phase == "placement":  
            self.horizontal = not self.horizontal  
            self.message = "Horizontal" if self.horizontal else "Vertical"  
            self.submessage = "Rotate with right click or Space."  
            self.sounds.play("rotate")  
  
    def undo_last_ship(self):  
        if self.phase != "placement":  
            return  
  
        if not self.placed_ships:  
            self.message = "Nothing to undo."  
            self.submessage = ""  
            self.sounds.play("error")  
            return  
  
        last = self.placed_ships.pop()  
        remove_ship_cells(self.player_board, last["cells"])  
        self.player_ships.pop()  
        self.placement_index -= 1  
  
        self.message = f"Removed {last['name']}."  
        self.submessage = "Place it again."  
        self.sounds.play("undo")  
  
    def try_place_current_ship(self, row, col):  
        if self.phase != "placement":  
            return  
  
        ship_info = self.current_ship_info()  
        if ship_info is None:  
            return  
  
        if not can_place_ship(self.player_board, row, col, ship_info["length"], self.horizontal):  
            self.message = "That ship does not fit there."  
            self.submessage = ""  
            self.sounds.play("error")  
            return  
  
        placed = place_named_ship(  
            self.player_board,  
            self.player_ships,  
            row,  
            col,  
            ship_info,  
            self.horizontal,  
        )  
        self.placed_ships.append({"name": placed["name"], "cells": list(placed["cells"])})  
        self.placement_index += 1  
        self.sounds.play("place")  
  
        if self.placement_index >= len(SHIPS):  
            self.phase = "battle"  
            self.message = f"Battle start - {self.difficulty.title()} AI."  
            self.submessage = "Click the enemy board to fire."  
        else:  
            next_ship = self.current_ship_info()  
            self.message = f"Placed {placed['name']}."  
            self.submessage = f"Next ship: {next_ship['name']} ({next_ship['length']})"  
  
    def player_fire(self, row, col):  
        if self.phase != "battle" or not self.player_turn or self.ai_pending:  
            return  
  
        result, hit_ship, sunk_now = fire_at(self.enemy_board, self.enemy_ships, row, col)  
  
        if result == "already":  
            self.message = "You already fired there."  
            self.submessage = ""  
            self.sounds.play("error")  
            return  
  
        cell_text = f"{chr(65 + col)}{row + 1}"  
  
        if result == "hit":  
            self.add_effect("explosion", "enemy", row=row, col=col, duration=650)  
            if sunk_now and hit_ship is not None:  
                self.add_effect("sink_flash", "enemy", cells=hit_ship["cells"], duration=1100)  
                self.message = f"You sank the enemy {hit_ship['name']}!"  
                self.submessage = f"Final hit at {cell_text}"  
                self.sounds.play("sink")  
            else:  
                self.message = f"You hit an enemy ship at {cell_text}!"  
                self.submessage = ""  
                self.sounds.play("hit")  
        else:  
            self.add_effect("splash", "enemy", row=row, col=col, duration=550)  
            self.message = f"You missed at {cell_text}."  
            self.submessage = ""  
            self.sounds.play("miss")  
  
        if ships_remaining(self.enemy_ships) == 0:  
            self.phase = "game_over"  
            self.winner = "player"  
            self.message = "You sank all enemy ships. You win!"  
            self.submessage = "Use Restart or Menu."  
            self.sounds.play("win")  
            return  
  
        self.player_turn = False  
        self.ai_pending = True  
        delay = 850 if self.difficulty == "easy" else 650 if self.difficulty == "normal" else 500  
        self.ai_move_time = pygame.time.get_ticks() + delay  
  
    def ai_fire(self):  
        if self.phase != "battle":  
            return  
  
        row, col = self.ai.choose_shot()  
        result, hit_ship, sunk_now = fire_at(self.player_board, self.player_ships, row, col)  
        self.ai.process_result(row, col, result, sunk_now)  
  
        cell_text = f"{chr(65 + col)}{row + 1}"  
  
        if result == "hit":  
            self.add_effect("explosion", "player", row=row, col=col, duration=650)  
            if sunk_now and hit_ship is not None:  
                self.add_effect("sink_flash", "player", cells=hit_ship["cells"], duration=1100)  
                self.message = f"The AI sank your {hit_ship['name']}!"  
                self.submessage = f"It finished it at {cell_text}"  
                self.sounds.play("sink")  
            else:  
                self.message = f"AI hit your ship at {cell_text}!"  
                self.submessage = ""  
                self.sounds.play("hit")  
        else:  
            self.add_effect("splash", "player", row=row, col=col, duration=550)  
            self.message = f"AI missed at {cell_text}."  
            self.submessage = ""  
            self.sounds.play("miss")  
  
        if ships_remaining(self.player_ships) == 0:  
            self.phase = "game_over"  
            self.winner = "ai"  
            self.message = f"The {self.difficulty.title()} AI sank all your ships."  
            self.submessage = "Use Restart or Menu."  
            self.sounds.play("lose")  
            return  
  
        self.player_turn = True  
  
    def update(self):  
        now = pygame.time.get_ticks()  
        self.effects = [e for e in self.effects if now - e["start"] < e["duration"]]  
  
        if self.ai_pending and now >= self.ai_move_time:  
            self.ai_pending = False  
            self.ai_fire()  
  
  
# --------------------------------------------------  
# Buttons  
# --------------------------------------------------  
def get_menu_buttons():  
    width = 300  
    height = 62  
    x = WINDOW_WIDTH // 2 - width // 2  
    y0 = 190  
    gap = 18  
  
    return [  
        {"label": "Easy", "action": "easy", "rect": pygame.Rect(x, y0 + 0 * (height + gap), width, height)},  
        {"label": "Normal", "action": "normal", "rect": pygame.Rect(x, y0 + 1 * (height + gap), width, height)},  
        {"label": "Mean", "action": "mean", "rect": pygame.Rect(x, y0 + 2 * (height + gap), width, height)},  
        {"label": "Mute: Toggle", "action": "mute", "rect": pygame.Rect(x, y0 + 3 * (height + gap) + 8, width, 56)},  
        {"label": "Quit", "action": "quit", "rect": pygame.Rect(x, y0 + 4 * (height + gap) + 18, width, 56)},  
    ]  
  
  
def get_overlay_buttons():  
    top_y = 16  
    width = 110  
    height = 38  
    gap = 12  
    x = WINDOW_WIDTH - (width * 3 + gap * 2) - 20  
    return [  
        {"label": "Menu", "action": "menu", "rect": pygame.Rect(x, top_y, width, height)},  
        {"label": "Restart", "action": "restart", "rect": pygame.Rect(x + width + gap, top_y, width, height)},  
        {"label": "Mute", "action": "mute", "rect": pygame.Rect(x + (width + gap) * 2, top_y, width, height)},  
    ]  
  
  
# --------------------------------------------------  
# Main  
# --------------------------------------------------  
def main():  
    pygame.init()  
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  
    pygame.display.set_caption("Battleship - Finalish")  
    clock = pygame.time.Clock()  
  
    fonts = {  
        "title": pygame.font.SysFont("arial", 36, bold=True),  
        "medium": pygame.font.SysFont("arial", 24, bold=True),  
        "small": pygame.font.SysFont("arial", 18),  
        "tiny": pygame.font.SysFont("arial", 15),  
        "status": pygame.font.SysFont("arial", 22),  
        "hint": pygame.font.SysFont("arial", 18),  
    }  
  
    sounds = SoundBank()  
    game = Game(sounds)  
  
    running = True  
    while running:  
        clock.tick(FPS)  
        game.update()  
  
        now = pygame.time.get_ticks()  
        mouse_pos = pygame.mouse.get_pos()  
  
        hover_player_cells = []  
        hover_player_ok = True  
        hover_enemy_cells = []  
  
        if game.phase == "placement":  
            cell = point_to_cell(mouse_pos, PLAYER_ORIGIN)  
            if cell is not None:  
                row, col = cell  
                ship_info = game.current_ship_info()  
                if ship_info is not None:  
                    hover_player_cells = ship_preview_cells(row, col, ship_info["length"], game.horizontal)  
                    hover_player_ok = can_place_ship(game.player_board, row, col, ship_info["length"], game.horizontal)  
  
        elif game.phase == "battle" and game.player_turn and not game.ai_pending:  
            cell = point_to_cell(mouse_pos, ENEMY_ORIGIN)  
            if cell is not None:  
                hover_enemy_cells = [cell]  
  
        for event in pygame.event.get():  
            if event.type == pygame.QUIT:  
                running = False  
  
            elif event.type == pygame.KEYDOWN:  
                if game.phase == "menu":  
                    if event.key == pygame.K_1:  
                        game.start_new_match("easy")  
                    elif event.key == pygame.K_2:  
                        game.start_new_match("normal")  
                    elif event.key == pygame.K_3:  
                        game.start_new_match("mean")  
                    elif event.key == pygame.K_m:  
                        sounds.toggle_mute()  
                else:  
                    if event.key == pygame.K_SPACE:  
                        game.rotate_ship()  
                    elif event.key == pygame.K_BACKSPACE:  
                        game.undo_last_ship()  
                    elif event.key == pygame.K_r:  
                        game.restart_same_difficulty()  
                    elif event.key == pygame.K_m:  
                        game.full_reset()  
  
            elif event.type == pygame.MOUSEBUTTONDOWN:  
                if game.phase == "menu" and event.button == 1:  
                    for button in get_menu_buttons():  
                        if button["rect"].collidepoint(event.pos):  
                            action = button["action"]  
                            if action in ("easy", "normal", "mean"):  
                                game.start_new_match(action)  
                            elif action == "mute":  
                                sounds.toggle_mute()  
                                sounds.play("click")  
                            elif action == "quit":  
                                running = False  
  
                elif game.phase in ("placement", "battle", "game_over"):  
                    clicked_overlay = False  
  
                    if event.button == 1:  
                        for button in get_overlay_buttons():  
                            if button["rect"].collidepoint(event.pos):  
                                clicked_overlay = True  
                                sounds.play("click")  
                                if button["action"] == "menu":  
                                    game.full_reset()  
                                elif button["action"] == "restart":  
                                    game.restart_same_difficulty()  
                                elif button["action"] == "mute":  
                                    sounds.toggle_mute()  
                                break  
  
                    if clicked_overlay:  
                        continue  
  
                    if game.phase == "placement":  
                        if event.button == 1:  
                            cell = point_to_cell(event.pos, PLAYER_ORIGIN)  
                            if cell is not None:  
                                game.try_place_current_ship(cell[0], cell[1])  
                        elif event.button == 3:  
                            game.rotate_ship()  
  
                    elif game.phase == "battle":  
                        if event.button == 1:  
                            cell = point_to_cell(event.pos, ENEMY_ORIGIN)  
                            if cell is not None:  
                                game.player_fire(cell[0], cell[1])  
  
        # draw  
        draw_ocean_background(screen, now)  
  
        if game.phase == "menu":  
            draw_text(screen, "BATTLESHIP", fonts["title"], TITLE_COLOR, WINDOW_WIDTH // 2, 72, center=True)  
            draw_text(screen, "Choose AI Difficulty", fonts["medium"], TEXT, WINDOW_WIDTH // 2, 122, center=True)  
            draw_text(screen, "Scrolling ocean, ship sprites, turn banner, and effects", fonts["small"], SUBTEXT,  
                      WINDOW_WIDTH // 2, 150, center=True)  
            draw_text(  
                screen,  
                f"Sound: {'Muted' if sounds.muted else 'On'}{' (audio unavailable)' if not sounds.available else ''}",  
                fonts["small"],  
                SUBTEXT,  
                WINDOW_WIDTH // 2,  
                WINDOW_HEIGHT - 34,  
                center=True,  
            )  
  
            for button in get_menu_buttons():  
                hovered = button["rect"].collidepoint(mouse_pos)  
                danger = button["action"] == "quit"  
                draw_button(screen, button["rect"], button["label"], fonts, hovered=hovered, danger=danger)  
  
            pygame.display.flip()  
            continue  
  
        draw_text(screen, "BATTLESHIP", fonts["title"], TITLE_COLOR, WINDOW_WIDTH // 2, 40, center=True)  
        draw_turn_banner(screen, fonts, game, now)  
  
        for button in get_overlay_buttons():  
            hovered = button["rect"].collidepoint(mouse_pos)  
            label = "Muted" if button["action"] == "mute" and sounds.muted else button["label"]  
            draw_button(screen, button["rect"], label, fonts, hovered=hovered)  
  
        draw_board(  
            screen,  
            game.player_board,  
            game.player_ships,  
            PLAYER_ORIGIN,  
            "Your Fleet",  
            PLAYER_LABEL,  
            fonts,  
            hide_ships=False,  
            show_sunk_enemy=False,  
            hover_cells=hover_player_cells,  
            hover_ok=hover_player_ok,  
        )  
  
        draw_board(  
            screen,  
            game.enemy_board,  
            game.enemy_ships,  
            ENEMY_ORIGIN,  
            "Enemy Waters",  
            ENEMY_LABEL,  
            fonts,  
            hide_ships=True,  
            show_sunk_enemy=True,  
            hover_cells=hover_enemy_cells,  
            hover_ok=True,  
        )  
  
        draw_effects(screen, game.effects, now)  
  
        draw_ship_status(screen, game.player_ships, PLAYER_ORIGIN[0], TOP_MARGIN + GRID_PIXELS + 10, "Your ships", fonts, is_enemy=False)  
        draw_ship_status(screen, game.enemy_ships, ENEMY_ORIGIN[0], TOP_MARGIN + GRID_PIXELS + 10, "Enemy ships", fonts, is_enemy=True)  
  
        info_bar = pygame.Rect(20, WINDOW_HEIGHT - 92, WINDOW_WIDTH - 40, 66)  
        draw_alpha_rect(screen, (14, 24, 36, 225), info_bar, border_radius=12)  
  
        draw_text(screen, game.message, fonts["status"], TEXT, 36, WINDOW_HEIGHT - 82)  
        draw_text(screen, game.submessage, fonts["hint"], SUBTEXT, 36, WINDOW_HEIGHT - 50)  
  
        if game.phase == "placement":  
            ship_info = game.current_ship_info()  
            direction = "Horizontal" if game.horizontal else "Vertical"  
            hint = (  
                f"Difficulty: {game.difficulty.title()} | "  
                f"Left click: place | Right click / Space: rotate | Backspace: undo | "  
                f"Next: {ship_info['name']} ({ship_info['length']}) | {direction}"  
            )  
        elif game.ai_pending:  
            hint = f"{game.difficulty.title()} AI is taking its turn..."  
        elif game.phase == "battle":  
            hint = f"Difficulty: {game.difficulty.title()} | Click enemy board to fire"  
        else:  
            hint = f"Game over | Winner: {'You' if game.winner == 'player' else 'AI'}"  
  
        draw_text(screen, hint, fonts["hint"], SUBTEXT, 36, WINDOW_HEIGHT - 116)  
  
        pygame.display.flip()  
  
    pygame.quit()  
    sys.exit()  
  
  
if __name__ == "__main__":  
    main()  
