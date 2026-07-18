import math
import random
import sys
from dataclasses import dataclass

import pygame


WIDTH, HEIGHT = 800, 480
FPS = 60
TITLE = "STRATEGIC RESPONSE TERMINAL // 800x480"

BG = (4, 8, 10)
PANEL = (8, 16, 18)
PANEL_2 = (10, 22, 26)
GRID = (18, 54, 58)
NEON = (88, 255, 210)
PLAYER = (92, 255, 180)
ENEMY = (255, 110, 120)
TEXT = (170, 230, 215)
WHITE = (235, 255, 245)
YELLOW = (255, 220, 120)
AMBER = (255, 188, 92)
GRAY = (82, 115, 110)
DARK = (18, 36, 38)
RED_DARK = (120, 42, 42)

TOP_H = 36
BOTTOM_H = 36
MAP_RECT = pygame.Rect(8, TOP_H + 8, 464, HEIGHT - TOP_H - BOTTOM_H - 16)
LOG_RECT = pygame.Rect(480, TOP_H + 8, WIDTH - 488, HEIGHT - TOP_H - BOTTOM_H - 16)
TOP_RECT = pygame.Rect(0, 0, WIDTH, TOP_H)
BOTTOM_RECT = pygame.Rect(0, HEIGHT - BOTTOM_H, WIDTH, BOTTOM_H)

COUNTRY_NAMES = {
    "player": "ATLANTIC UNION",
    "enemy": "CONTINENTAL BLOC",
}

DEFCON_INFO = {
    5: "RECON ONLY",
    4: "WARNING NETWORKS ACTIVE",
    3: "ABM NETS HOT",
    2: "LAUNCH AUTHORITY ENABLED",
    1: "GENERAL EXCHANGE",
}

INTRO_LINES = [
    "STRATEGIC RESPONSE TERMINAL V4.8",
    "NODE: CHEYENNE MOUNTAIN SIMULATION CORE",
    "SCENARIO PACKAGE: GLOBAL THERMONET / FICTIONAL BLOCS",
    "STATUS: SAFE TRAINING ENVIRONMENT / NO REAL TARGET DATA",
    "TYPE HELP FOR COMMANDS. TYPE START TO BEGIN.",
]


@dataclass
class City:
    name: str
    x: float
    y: float
    owner: str
    population: int
    alive: bool = True
    hp: float = 100.0


@dataclass
class Silo:
    name: str
    x: float
    y: float
    owner: str
    missiles: int = 8
    alive: bool = True
    cooldown: float = 0.0
    hp: float = 100.0


@dataclass
class ABMSite:
    name: str
    x: float
    y: float
    owner: str
    ammo: int = 4
    alive: bool = True
    cooldown: float = 0.0
    hp: float = 100.0


class Missile:
    def __init__(self, start, target, owner, speed=160.0, label=""):
        self.start = pygame.Vector2(start)
        self.target = pygame.Vector2(target)
        self.pos = pygame.Vector2(start)
        self.owner = owner
        self.speed = speed
        self.label = label
        self.progress = 0.0
        self.distance = max(1.0, self.start.distance_to(self.target))
        self.flight_time = self.distance / self.speed
        self.trail = []

    def update(self, dt):
        self.progress += dt / self.flight_time
        self.progress = min(1.0, self.progress)
        t = self.progress
        peak = 26 + self.distance * 0.07
        x = self.start.x + (self.target.x - self.start.x) * t
        base_y = self.start.y + (self.target.y - self.start.y) * t
        y = base_y - math.sin(math.pi * t) * peak
        self.pos.update(x, y)
        self.trail.append((x, y))
        if len(self.trail) > 24:
            self.trail.pop(0)
        return self.progress >= 1.0


class Interceptor:
    def __init__(self, start, target_missile):
        self.pos = pygame.Vector2(start)
        self.target_missile = target_missile
        self.speed = 320.0
        self.alive = True
        self.trail = []

    def update(self, dt):
        if self.target_missile is None:
            self.alive = False
            return False
        target_pos = pygame.Vector2(self.target_missile.pos)
        delta = target_pos - self.pos
        dist = delta.length()
        if dist < 1:
            self.alive = False
            return True
        if dist > 0:
            delta.scale_to_length(min(self.speed * dt, dist))
            self.pos += delta
        self.trail.append((self.pos.x, self.pos.y))
        if len(self.trail) > 16:
            self.trail.pop(0)
        if self.pos.distance_to(target_pos) < 10:
            self.alive = False
            return True
        return False


class Explosion:
    def __init__(self, pos, airburst=False, radius=54, duration=1.0):
        self.pos = pygame.Vector2(pos)
        self.airburst = airburst
        self.max_radius = radius
        self.duration = duration
        self.timer = 0.0
        self.damage_applied = False

    def update(self, dt):
        self.timer += dt
        return self.timer >= self.duration

    @property
    def radius(self):
        return max(2, self.max_radius * math.sin(min(1.0, self.timer / self.duration) * math.pi))


class Particle:
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        ang = random.random() * math.tau
        speed = random.uniform(20, 85)
        self.vel = pygame.Vector2(math.cos(ang) * speed, math.sin(ang) * speed)
        self.life = random.uniform(0.25, 0.7)
        self.max_life = self.life
        self.color = color

    def update(self, dt):
        self.life -= dt
        self.pos += self.vel * dt
        self.vel *= 0.95
        return self.life <= 0


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_tiny = pygame.font.SysFont("consolas", 11)
        self.font_small = pygame.font.SysFont("consolas", 12)
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_huge = pygame.font.SysFont("consolas", 24, bold=True)
        self.reset()

    def reset(self):
        self.running = True
        self.simulation_started = False
        self.paused = False
        self.state = "terminal"
        self.winner = None
        self.selected_silo_name = None
        self.defcon = 5
        self.defcon_timer = 0.0
        self.defcon_step_time = 13.0
        self.ai_timer = 0.0
        self.ai_interval = 2.1
        self.flash = 0.0
        self.scan_phase = 0.0
        self.cursor_timer = 0.0
        self.input_text = ""
        self.command_history = []
        self.history_index = None
        self.log_lines = []
        self.stars = [(random.randrange(WIDTH), random.randrange(HEIGHT), random.randint(1, 2)) for _ in range(50)]
        self.missiles = []
        self.interceptors = []
        self.explosions = []
        self.particles = []
        self.continents = self.build_continents()
        self.cities, self.silos, self.abm_sites = self.build_world()
        self.add_boot_log()

    def add_boot_log(self):
        self.log_lines.clear()
        for line in INTRO_LINES:
            self.log(line, NEON)
        self.log("AWAITING OPERATOR INPUT.", YELLOW)

    def build_continents(self):
        return [
            [(48, 115), (98, 90), (148, 98), (182, 146), (162, 188), (104, 205), (56, 172)],
            [(150, 250), (206, 232), (244, 246), (262, 286), (246, 335), (196, 356), (158, 318)],
            [(286, 95), (332, 82), (384, 92), (440, 118), (456, 182), (422, 214), (352, 206), (292, 158)],
            [(392, 232), (444, 226), (498, 248), (528, 302), (510, 360), (452, 372), (410, 330), (382, 282)],
            [(410, 96), (450, 92), (466, 116), (464, 152), (438, 164), (404, 144)],
        ]

    def build_world(self):
        cities = [
            City("NORA", 92, 136, "player", 12),
            City("AURIC", 174, 158, "player", 10),
            City("HAVEN", 222, 292, "player", 9),
            City("DELTA", 146, 320, "player", 8),
            City("VESTA", 364, 132, "enemy", 12),
            City("KHAN", 432, 166, "enemy", 10),
            City("ONYX", 512, 304, "enemy", 9),
            City("EMBER", 444, 338, "enemy", 8),
        ]
        silos = [
            Silo("ALPHA", 84, 232, "player", missiles=8),
            Silo("BRAVO", 240, 216, "player", missiles=8),
            Silo("SIGMA", 392, 278, "enemy", missiles=8),
            Silo("OMEGA", 522, 214, "enemy", missiles=8),
        ]
        abm_sites = [
            ABMSite("AEGIS-W", 182, 236, "player", ammo=4),
            ABMSite("AEGIS-E", 456, 228, "enemy", ammo=4),
        ]
        return cities, silos, abm_sites

    def all_assets(self):
        return [obj for obj in self.cities + self.silos + self.abm_sites if obj.alive]

    def get_asset(self, name):
        n = name.strip().upper()
        for obj in self.cities + self.silos + self.abm_sites:
            if obj.name.upper() == n:
                return obj
        return None

    def get_selected_silo(self):
        if not self.selected_silo_name:
            return None
        asset = self.get_asset(self.selected_silo_name)
        return asset if isinstance(asset, Silo) else None

    def log(self, text, color=TEXT):
        wrapped = self.wrap_text(text, 38)
        for line in wrapped:
            self.log_lines.append((line, color))
        self.log_lines = self.log_lines[-200:]

    def wrap_text(self, text, width_chars):
        words = text.split(" ")
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= width_chars:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def handle_command(self, raw):
        cmd = raw.strip()
        if not cmd:
            return
        upper = cmd.upper()
        self.command_history.append(cmd)
        self.history_index = None
        self.log("> " + cmd, WHITE)

        parts = upper.replace(",", " ").split()
        if not parts:
            return

        action = parts[0]

        if action in {"HELP", "?"}:
            for line in [
                "COMMANDS: START, HELP, STATUS, TARGETS, SILOS, CITIES, ENEMY, PLAYER",
                "SELECT <SILO>  |  LAUNCH <TARGET>  |  LAUNCH <SILO> <TARGET>",
                "PAUSE, RESUME, RESTART, CLEAR, QUIT",
                "EXAMPLE: SELECT ALPHA    EXAMPLE: LAUNCH KHAN",
            ]:
                self.log(line, NEON)
            return

        if action == "START":
            if self.simulation_started:
                self.log("SIMULATION ALREADY RUNNING.", YELLOW)
            else:
                self.simulation_started = True
                self.state = "play"
                self.log(f"SIMULATION STARTED. DEFCON {self.defcon}: {DEFCON_INFO[self.defcon]}", YELLOW)
            return

        if action in {"STATUS", "SITREP"}:
            self.print_status()
            return

        if action == "TARGETS":
            enemy_targets = [o.name for o in self.all_assets() if o.owner == "enemy"]
            self.log("ENEMY TARGETS: " + ", ".join(enemy_targets), ENEMY)
            return

        if action == "PLAYER":
            player_assets = [o.name for o in self.all_assets() if o.owner == "player"]
            self.log("PLAYER ASSETS: " + ", ".join(player_assets), PLAYER)
            return

        if action == "ENEMY":
            enemy_assets = [o.name for o in self.all_assets() if o.owner == "enemy"]
            self.log("ENEMY ASSETS: " + ", ".join(enemy_assets), ENEMY)
            return

        if action == "SILOS":
            player_silos = [f"{s.name}[{s.missiles}]" for s in self.silos if s.owner == "player" and s.alive]
            self.log("PLAYER SILOS: " + ", ".join(player_silos), PLAYER)
            return

        if action == "CITIES":
            player_cities = [c.name for c in self.cities if c.owner == "player" and c.alive]
            enemy_cities = [c.name for c in self.cities if c.owner == "enemy" and c.alive]
            self.log("PLAYER CITIES: " + ", ".join(player_cities), PLAYER)
            self.log("ENEMY CITIES: " + ", ".join(enemy_cities), ENEMY)
            return

        if action == "SELECT":
            if len(parts) < 2:
                self.log("USAGE: SELECT <SILO>", YELLOW)
                return
            self.select_silo(parts[1])
            return

        if action in {"LAUNCH", "FIRE"}:
            if len(parts) == 2:
                selected = self.get_selected_silo()
                if not selected:
                    self.log("NO SILO SELECTED. USE SELECT ALPHA OR SELECT BRAVO.", YELLOW)
                    return
                self.launch_command(selected.name, parts[1])
                return
            if len(parts) >= 3:
                self.launch_command(parts[1], parts[2])
                return
            self.log("USAGE: LAUNCH <TARGET> OR LAUNCH <SILO> <TARGET>", YELLOW)
            return

        if action == "PAUSE":
            if not self.simulation_started:
                self.log("SIMULATION NOT RUNNING.", YELLOW)
            else:
                self.paused = True
                self.log("SIMULATION PAUSED.", YELLOW)
            return

        if action == "RESUME":
            if not self.simulation_started:
                self.log("SIMULATION NOT RUNNING.", YELLOW)
            else:
                self.paused = False
                self.log("SIMULATION RESUMED.", YELLOW)
            return

        if action in {"RESTART", "RESET"}:
            self.reset()
            return

        if action in {"CLEAR", "CLS"}:
            self.add_boot_log()
            return

        if action in {"QUIT", "EXIT"}:
            self.running = False
            return

        self.log("UNKNOWN COMMAND. TYPE HELP.", ENEMY)

    def print_status(self):
        player_pop = sum(c.population for c in self.cities if c.owner == "player" and c.alive)
        enemy_pop = sum(c.population for c in self.cities if c.owner == "enemy" and c.alive)
        player_missiles = sum(s.missiles for s in self.silos if s.owner == "player" and s.alive)
        enemy_missiles = sum(s.missiles for s in self.silos if s.owner == "enemy" and s.alive)
        selected = self.get_selected_silo()
        sel_text = selected.name if selected and selected.alive else "NONE"
        self.log(f"DEFCON {self.defcon} // {DEFCON_INFO[self.defcon]}", YELLOW if self.defcon <= 2 else NEON)
        self.log(f"SELECTED SILO: {sel_text}", PLAYER)
        self.log(f"{COUNTRY_NAMES['player']}: POP {player_pop}M / MISSILES {player_missiles}", PLAYER)
        self.log(f"{COUNTRY_NAMES['enemy']}: POP {enemy_pop}M / MISSILES {enemy_missiles}", ENEMY)

    def select_silo(self, silo_name):
        asset = self.get_asset(silo_name)
        if not asset or not isinstance(asset, Silo):
            self.log("SILO NOT FOUND.", ENEMY)
            return
        if asset.owner != "player":
            self.log("ACCESS DENIED. ENEMY SILO CANNOT BE SELECTED.", ENEMY)
            return
        if not asset.alive:
            self.log("SILO OFFLINE.", ENEMY)
            return
        self.selected_silo_name = asset.name
        self.log(f"SILO {asset.name} SELECTED. MISSILES: {asset.missiles}", PLAYER)

    def launch_command(self, silo_name, target_name):
        silo = self.get_asset(silo_name)
        target = self.get_asset(target_name)
        if not silo or not isinstance(silo, Silo):
            self.log("LAUNCH SOURCE INVALID.", ENEMY)
            return
        if silo.owner != "player":
            self.log("ONLY PLAYER SILOS ACCEPT COMMANDS.", ENEMY)
            return
        if not target:
            self.log("TARGET NOT FOUND. USE TARGETS TO LIST VALID ENEMY ASSETS.", ENEMY)
            return
        if target.owner == "player":
            self.log("FRIENDLY FIRE LOCKOUT ENGAGED.", ENEMY)
            return
        if not target.alive:
            self.log("TARGET ALREADY OFFLINE.", YELLOW)
            return
        if self.launch_from_silo(silo, (target.x, target.y), target.name):
            self.log(f"LAUNCH CONFIRMED: {silo.name} -> {target.name}", PLAYER)

    def launch_from_silo(self, silo, target_pos, label=""):
        if not self.simulation_started:
            self.log("SIMULATION NOT STARTED. TYPE START.", YELLOW)
            return False
        if not silo.alive or silo.missiles <= 0:
            self.log(f"{silo.name} UNABLE TO FIRE.", ENEMY)
            return False
        if silo.cooldown > 0:
            self.log(f"{silo.name} IN COOLDOWN.", YELLOW)
            return False
        if self.defcon > 2:
            self.log("LAUNCH DENIED. AUTHORITY NOT AVAILABLE UNTIL DEFCON 2.", YELLOW)
            return False
        self.missiles.append(Missile((silo.x, silo.y), target_pos, silo.owner, speed=random.uniform(150, 174), label=label))
        silo.missiles -= 1
        silo.cooldown = 1.15
        self.flash = 0.24
        return True

    def spawn_explosion(self, pos, radius=54, airburst=False):
        self.explosions.append(Explosion(pos, airburst=airburst, radius=radius, duration=0.95 if airburst else 1.10))
        for _ in range(10 if airburst else 18):
            self.particles.append(Particle(pos, NEON if airburst else AMBER))

    def damage_world(self, explosion):
        if explosion.damage_applied:
            return
        explosion.damage_applied = True
        radius = explosion.max_radius
        multiplier = 0.55 if explosion.airburst else 1.0
        for group in (self.cities, self.silos, self.abm_sites):
            for obj in group:
                if not obj.alive:
                    continue
                dist = pygame.Vector2(obj.x, obj.y).distance_to(explosion.pos)
                if dist <= radius:
                    severity = max(0.2, 1.0 - dist / radius)
                    obj.hp -= 100 * severity * multiplier
                    if obj.hp <= 0:
                        obj.alive = False
                        if isinstance(obj, City):
                            obj.population = 0
                        self.log(f"ASSET LOST: {obj.name}", ENEMY if obj.owner == "player" else PLAYER)

    def update_ai(self, dt):
        self.ai_timer += dt
        if self.ai_timer < self.ai_interval or self.defcon > 2:
            return
        self.ai_timer = 0.0

        enemy_silos = [s for s in self.silos if s.owner == "enemy" and s.alive and s.missiles > 0 and s.cooldown <= 0]
        player_targets = [o for o in self.cities + self.silos + self.abm_sites if o.owner == "player" and o.alive]
        if not enemy_silos or not player_targets:
            return

        silo = random.choice(enemy_silos)
        weighted_targets = []
        for t in player_targets:
            weight = 4 if isinstance(t, City) else 2
            if isinstance(t, ABMSite):
                weight = 3
            weighted_targets.extend([t] * weight)
        target = random.choice(weighted_targets)
        if self.launch_from_enemy_silo(silo, (target.x, target.y), target.name):
            self.log(f"INBOUND WARNING: {silo.name} -> {target.name}", ENEMY)

    def launch_from_enemy_silo(self, silo, target_pos, label=""):
        if not silo.alive or silo.missiles <= 0 or silo.cooldown > 0:
            return False
        self.missiles.append(Missile((silo.x, silo.y), target_pos, silo.owner, speed=random.uniform(148, 172), label=label))
        silo.missiles -= 1
        silo.cooldown = 1.15
        self.flash = 0.22
        return True

    def update_abm(self, dt):
        if self.defcon > 3:
            return
        for site in self.abm_sites:
            if not site.alive:
                continue
            site.cooldown = max(0.0, site.cooldown - dt)
            if site.ammo <= 0 or site.cooldown > 0:
                continue
            hostile = None
            for missile in self.missiles:
                if missile.owner == site.owner:
                    continue
                if pygame.Vector2(site.x, site.y).distance_to(missile.pos) < 110:
                    hostile = missile
                    break
            if hostile:
                self.interceptors.append(Interceptor((site.x, site.y), hostile))
                site.ammo -= 1
                site.cooldown = 2.3
                self.log(f"ABM ENGAGE: {site.name}", YELLOW)

    def update_simulation(self, dt):
        self.scan_phase += dt * 60
        self.flash = max(0.0, self.flash - dt)
        self.cursor_timer += dt
        if not self.simulation_started or self.paused or self.state == "gameover":
            return

        self.defcon_timer += dt
        if self.defcon > 1 and self.defcon_timer >= self.defcon_step_time:
            self.defcon_timer = 0.0
            self.defcon -= 1
            self.log(f"*** DEFCON {self.defcon}: {DEFCON_INFO[self.defcon]} ***", YELLOW)
            self.flash = 0.30

        for silo in self.silos:
            silo.cooldown = max(0.0, silo.cooldown - dt)

        self.update_ai(dt)
        self.update_abm(dt)

        live_missiles = []
        for missile in self.missiles:
            if missile.update(dt):
                self.spawn_explosion(missile.target, radius=56, airburst=False)
            else:
                live_missiles.append(missile)
        self.missiles = live_missiles

        live_interceptors = []
        for interceptor in self.interceptors:
            if interceptor.update(dt):
                target = interceptor.target_missile
                if target in self.missiles:
                    self.missiles.remove(target)
                    self.spawn_explosion(target.pos, radius=26, airburst=True)
                    self.log("INTERCEPT CONFIRMED.", YELLOW)
            elif interceptor.alive:
                live_interceptors.append(interceptor)
        self.interceptors = live_interceptors

        live_explosions = []
        for explosion in self.explosions:
            if explosion.timer >= explosion.duration * 0.24:
                self.damage_world(explosion)
            if not explosion.update(dt):
                live_explosions.append(explosion)
        self.explosions = live_explosions

        live_particles = []
        for particle in self.particles:
            if not particle.update(dt):
                live_particles.append(particle)
        self.particles = live_particles

        self.check_win_state()

    def check_win_state(self):
        player_pop = sum(c.population for c in self.cities if c.owner == "player" and c.alive)
        enemy_pop = sum(c.population for c in self.cities if c.owner == "enemy" and c.alive)
        player_strike = any(s.alive and s.missiles > 0 for s in self.silos if s.owner == "player")
        enemy_strike = any(s.alive and s.missiles > 0 for s in self.silos if s.owner == "enemy")

        if enemy_pop <= 0 < player_pop:
            self.state = "gameover"
            self.winner = "player"
            self.log("OUTCOME: VICTORY", PLAYER)
        elif player_pop <= 0 < enemy_pop:
            self.state = "gameover"
            self.winner = "enemy"
            self.log("OUTCOME: DEFEAT", ENEMY)
        elif not player_strike and not enemy_strike and not self.missiles and not self.explosions:
            self.state = "gameover"
            if player_pop > enemy_pop:
                self.winner = "player"
                self.log("OUTCOME: VICTORY", PLAYER)
            elif enemy_pop > player_pop:
                self.winner = "enemy"
                self.log("OUTCOME: DEFEAT", ENEMY)
            else:
                self.winner = "draw"
                self.log("OUTCOME: STALEMATE", YELLOW)

    def draw_top_bar(self):
        pygame.draw.rect(self.screen, PANEL, TOP_RECT)
        pygame.draw.line(self.screen, NEON, (0, TOP_H - 1), (WIDTH, TOP_H - 1), 1)
        title = self.font_big.render("STRATEGIC RESPONSE TERMINAL", True, WHITE)
        self.screen.blit(title, (8, 8))
        mode = "PAUSED" if self.paused else ("RUNNING" if self.simulation_started and self.state != "gameover" else "STANDBY")
        defcon_color = YELLOW if self.defcon <= 2 else NEON
        right = self.font.render(f"{mode}   DEFCON {self.defcon}   {DEFCON_INFO[self.defcon]}", True, defcon_color)
        self.screen.blit(right, (WIDTH - right.get_width() - 8, 11))

    def draw_grid(self):
        for x in range(MAP_RECT.left, MAP_RECT.right + 1, 28):
            pygame.draw.line(self.screen, GRID, (x, MAP_RECT.top), (x, MAP_RECT.bottom), 1)
        for y in range(MAP_RECT.top, MAP_RECT.bottom + 1, 28):
            pygame.draw.line(self.screen, GRID, (MAP_RECT.left, y), (MAP_RECT.right, y), 1)

    def draw_map(self):
        pygame.draw.rect(self.screen, PANEL_2, MAP_RECT)
        pygame.draw.rect(self.screen, NEON, MAP_RECT, 1)
        self.draw_grid()

        for poly in self.continents:
            pygame.draw.polygon(self.screen, (8, 28, 30), poly)
            pygame.draw.lines(self.screen, NEON, True, poly, 1)

        for i in range(4):
            y = MAP_RECT.top + 48 + i * 66
            pts = []
            for x in range(MAP_RECT.left, MAP_RECT.right, 14):
                pts.append((x, y + math.sin((x * 0.02) + i) * 5))
            pygame.draw.lines(self.screen, (20, 72, 70), False, pts, 1)

        scan_y = MAP_RECT.top + int(self.scan_phase % MAP_RECT.height)
        pygame.draw.line(self.screen, (48, 180, 160), (MAP_RECT.left + 1, scan_y), (MAP_RECT.right - 1, scan_y), 1)

        selected = self.get_selected_silo()

        for city in self.cities:
            color = PLAYER if city.owner == "player" else ENEMY
            if not city.alive:
                color = DARK
            pygame.draw.circle(self.screen, color, (city.x, city.y), 4)
            pygame.draw.circle(self.screen, color, (city.x, city.y), 8, 1)
            label = f"{city.name}" if city.alive else f"{city.name} X"
            self.screen.blit(self.font_tiny.render(label, True, color if city.alive else GRAY), (city.x + 7, city.y - 6))

        for silo in self.silos:
            color = PLAYER if silo.owner == "player" else ENEMY
            if not silo.alive:
                color = DARK
            rect = pygame.Rect(silo.x - 5, silo.y - 5, 10, 10)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, WHITE if selected == silo else color, rect.inflate(6, 6), 1)
            label = f"{silo.name}[{silo.missiles}]"
            self.screen.blit(self.font_tiny.render(label, True, color if silo.alive else GRAY), (silo.x + 8, silo.y - 6))

        for site in self.abm_sites:
            color = PLAYER if site.owner == "player" else ENEMY
            if not site.alive:
                color = DARK
            pygame.draw.circle(self.screen, color, (site.x, site.y), 6, 1)
            pygame.draw.circle(self.screen, color, (site.x, site.y), 28, 1)
            self.screen.blit(self.font_tiny.render(f"{site.name} {site.ammo}", True, color if site.alive else GRAY), (site.x + 8, site.y - 6))

        for missile in self.missiles:
            trail_color = PLAYER if missile.owner == "player" else ENEMY
            if len(missile.trail) >= 2:
                pygame.draw.lines(self.screen, trail_color, False, missile.trail, 2)
            pygame.draw.circle(self.screen, WHITE, (int(missile.pos.x), int(missile.pos.y)), 2)

        for interceptor in self.interceptors:
            if len(interceptor.trail) >= 2:
                pygame.draw.lines(self.screen, YELLOW, False, interceptor.trail, 1)
            pygame.draw.circle(self.screen, YELLOW, (int(interceptor.pos.x), int(interceptor.pos.y)), 2)

        for explosion in self.explosions:
            radius = int(explosion.radius)
            color = NEON if explosion.airburst else AMBER
            pygame.draw.circle(self.screen, color, (int(explosion.pos.x), int(explosion.pos.y)), radius, 1)
            pygame.draw.circle(self.screen, color, (int(explosion.pos.x), int(explosion.pos.y)), max(2, radius // 2), 1)

        for particle in self.particles:
            alpha = max(0, min(255, int(255 * (particle.life / particle.max_life))))
            surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*particle.color, alpha), (2, 2), 2)
            self.screen.blit(surf, (particle.pos.x - 2, particle.pos.y - 2))

        self.screen.blit(self.font_tiny.render("WORLD STATUS MAP", True, WHITE), (MAP_RECT.left + 8, MAP_RECT.top + 6))

    def draw_log_panel(self):
        pygame.draw.rect(self.screen, PANEL_2, LOG_RECT)
        pygame.draw.rect(self.screen, NEON, LOG_RECT, 1)
        self.screen.blit(self.font_tiny.render("COMMAND / EVENT LOG", True, WHITE), (LOG_RECT.left + 8, LOG_RECT.top + 6))

        max_lines = 27
        visible = self.log_lines[-max_lines:]
        y = LOG_RECT.top + 24
        for line, color in visible:
            rendered = self.font_tiny.render(line, True, color)
            self.screen.blit(rendered, (LOG_RECT.left + 8, y))
            y += 12

    def draw_bottom_input(self):
        pygame.draw.rect(self.screen, PANEL, BOTTOM_RECT)
        pygame.draw.line(self.screen, NEON, (0, HEIGHT - BOTTOM_H), (WIDTH, HEIGHT - BOTTOM_H), 1)
        selected = self.get_selected_silo()
        selected_name = selected.name if selected and selected.alive else "NONE"
        prefix = f"SELECTED:{selected_name}   > "
        blink = "_" if int(self.cursor_timer * 2) % 2 == 0 else " "
        rendered = self.font.render(prefix + self.input_text + blink, True, WHITE)
        self.screen.blit(rendered, (8, HEIGHT - 24))

    def draw_status_box(self):
        box = pygame.Rect(12, 46, 188, 58)
        pygame.draw.rect(self.screen, (6, 18, 20), box)
        pygame.draw.rect(self.screen, NEON, box, 1)
        player_pop = sum(c.population for c in self.cities if c.owner == "player" and c.alive)
        enemy_pop = sum(c.population for c in self.cities if c.owner == "enemy" and c.alive)
        player_missiles = sum(s.missiles for s in self.silos if s.owner == "player" and s.alive)
        enemy_missiles = sum(s.missiles for s in self.silos if s.owner == "enemy" and s.alive)
        self.screen.blit(self.font_tiny.render(f"AU POP {player_pop}M  MIS {player_missiles}", True, PLAYER), (18, 58))
        self.screen.blit(self.font_tiny.render(f"CB POP {enemy_pop}M  MIS {enemy_missiles}", True, ENEMY), (18, 74))
        self.screen.blit(self.font_tiny.render("TYPE HELP FOR COMMANDS", True, TEXT), (18, 90))

    def draw_gameover_overlay(self):
        box = pygame.Rect(178, 158, 444, 122)
        color = PLAYER if self.winner == "player" else ENEMY if self.winner == "enemy" else YELLOW
        label = "VICTORY" if self.winner == "player" else "DEFEAT" if self.winner == "enemy" else "STALEMATE"
        pygame.draw.rect(self.screen, (3, 10, 12), box)
        pygame.draw.rect(self.screen, color, box, 2)
        title = self.font_huge.render(label, True, color)
        self.screen.blit(title, (box.centerx - title.get_width() // 2, box.y + 22))
        note = self.font.render("TYPE RESTART TO RUN ANOTHER SIMULATION.", True, WHITE)
        self.screen.blit(note, (box.centerx - note.get_width() // 2, box.y + 68))

    def draw_overlay_fx(self):
        for x, y, size in self.stars:
            self.screen.fill((34, 52, 50), ((x, y), (size, size)))
        if self.flash > 0:
            surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            surf.fill((255, 255, 255, int(82 * self.flash)))
            self.screen.blit(surf, (0, 0))
        scan = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 3):
            pygame.draw.line(scan, (0, 0, 0, 16), (0, y), (WIDTH, y))
        self.screen.blit(scan, (0, 0))

    def draw(self):
        self.screen.fill(BG)
        self.draw_top_bar()
        self.draw_map()
        self.draw_log_panel()
        self.draw_status_box()
        self.draw_bottom_input()
        if self.state == "gameover":
            self.draw_gameover_overlay()
        self.draw_overlay_fx()
        pygame.display.flip()

    def handle_keydown(self, event):
        if event.key == pygame.K_ESCAPE:
            self.running = False
            return
        if event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
            return
        if event.key == pygame.K_RETURN:
            cmd = self.input_text
            self.input_text = ""
            self.handle_command(cmd)
            return
        if event.key == pygame.K_UP:
            if not self.command_history:
                return
            if self.history_index is None:
                self.history_index = len(self.command_history) - 1
            else:
                self.history_index = max(0, self.history_index - 1)
            self.input_text = self.command_history[self.history_index]
            return
        if event.key == pygame.K_DOWN:
            if self.history_index is None:
                return
            self.history_index += 1
            if self.history_index >= len(self.command_history):
                self.history_index = None
                self.input_text = ""
            else:
                self.input_text = self.command_history[self.history_index]
            return
        if event.unicode and event.unicode.isprintable():
            if len(self.input_text) < 60:
                self.input_text += event.unicode

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)

            self.update_simulation(dt)
            self.draw()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
