import math
import random
import sys
from dataclasses import dataclass

import pygame


WIDTH, HEIGHT = 800, 480
FPS = 60
TITLE = "GLOBAL THERMONET // 800x480"

BG = (5, 9, 14)
PANEL = (9, 20, 28)
GRID = (15, 48, 62)
NEON = (76, 228, 255)
PLAYER = (90, 255, 170)
ENEMY = (255, 110, 120)
TEXT = (190, 235, 245)
YELLOW = (255, 228, 120)
WHITE = (245, 250, 255)
GRAY = (90, 120, 130)
DARK = (16, 36, 40)
EXPLOSION = (255, 170, 90)

HUD_H = 56
BOTTOM_H = 42
MAP_RECT = pygame.Rect(10, HUD_H + 8, WIDTH - 20, HEIGHT - HUD_H - BOTTOM_H - 16)
HUD_RECT = pygame.Rect(0, 0, WIDTH, HUD_H)
BOTTOM_RECT = pygame.Rect(0, HEIGHT - BOTTOM_H, WIDTH, BOTTOM_H)

DEFCON_INFO = {
    5: "Recon only.",
    4: "Warning nets online.",
    3: "ABM systems hot.",
    2: "Launch authority open.",
    1: "Full exchange.",
}

COUNTRY_NAMES = {
    "player": "Atlantic Union",
    "enemy": "Continental Bloc",
}


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
    def __init__(self, start, target, owner, speed=160.0):
        self.start = pygame.Vector2(start)
        self.target = pygame.Vector2(target)
        self.pos = pygame.Vector2(start)
        self.owner = owner
        self.speed = speed
        self.progress = 0.0
        self.distance = max(1.0, self.start.distance_to(self.target))
        self.flight_time = self.distance / self.speed
        self.trail = []

    def update(self, dt):
        self.progress += dt / self.flight_time
        self.progress = min(1.0, self.progress)
        t = self.progress
        peak = 35 + self.distance * 0.08
        x = self.start.x + (self.target.x - self.start.x) * t
        base_y = self.start.y + (self.target.y - self.start.y) * t
        y = base_y - math.sin(math.pi * t) * peak
        self.pos.update(x, y)
        self.trail.append((x, y))
        if len(self.trail) > 28:
            self.trail.pop(0)
        return self.progress >= 1.0


class Interceptor:
    def __init__(self, start, target_missile):
        self.pos = pygame.Vector2(start)
        self.target_missile = target_missile
        self.speed = 300.0
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
        speed = random.uniform(20, 90)
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
        self.font_small = pygame.font.SysFont("consolas", 12)
        self.font = pygame.font.SysFont("consolas", 15)
        self.font_big = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_huge = pygame.font.SysFont("consolas", 30, bold=True)
        self.reset()

    def reset(self):
        self.running = True
        self.state = "menu"
        self.paused = False
        self.defcon = 5
        self.defcon_timer = 0.0
        self.defcon_step_time = 14.0
        self.selected_silo = None
        self.message = "Select a green silo, then right-click a target once DEFCON 2 begins."
        self.flash = 0.0
        self.scan_phase = 0.0
        self.ai_timer = 0.0
        self.ai_interval = 2.1
        self.winner = None
        self.missiles = []
        self.interceptors = []
        self.explosions = []
        self.particles = []
        self.stars = [(random.randrange(WIDTH), random.randrange(HEIGHT), random.randint(1, 2)) for _ in range(55)]
        self.continents = self.build_continents()
        self.cities, self.silos, self.abm_sites = self.build_world()

    def build_continents(self):
        # Abstract display shapes for a command-screen aesthetic.
        return [
            [(70, 110), (130, 88), (180, 95), (220, 140), (200, 200), (140, 220), (80, 175)],
            [(180, 250), (245, 230), (290, 245), (315, 290), (290, 350), (235, 375), (190, 330)],
            [(390, 100), (450, 88), (520, 110), (590, 145), (605, 210), (560, 240), (465, 225), (405, 170)],
            [(565, 250), (635, 245), (695, 270), (755, 325), (730, 380), (655, 395), (590, 360), (550, 305)],
            [(665, 115), (720, 102), (755, 130), (768, 173), (745, 205), (695, 198), (650, 160)],
        ]

    def build_world(self):
        cities = [
            City("NORA", 110, 140, "player", 12),
            City("AURIC", 200, 165, "player", 10),
            City("HAVEN", 250, 305, "player", 9),
            City("DELTA", 165, 330, "player", 8),
            City("VESTA", 475, 145, "enemy", 12),
            City("KHAN", 560, 185, "enemy", 10),
            City("ONYX", 665, 315, "enemy", 9),
            City("EMBER", 735, 340, "enemy", 8),
        ]
        silos = [
            Silo("ALPHA", 90, 245, "player", missiles=8),
            Silo("BRAVO", 285, 235, "player", missiles=8),
            Silo("SIGMA", 520, 290, "enemy", missiles=8),
            Silo("OMEGA", 720, 235, "enemy", missiles=8),
        ]
        abm_sites = [
            ABMSite("AEGIS-W", 220, 250, "player", ammo=4),
            ABMSite("AEGIS-E", 615, 245, "enemy", ammo=4),
        ]
        return cities, silos, abm_sites

    def all_assets(self):
        return [obj for obj in self.cities + self.silos + self.abm_sites if obj.alive]

    def pick_target_under_mouse(self, mouse_pos):
        m = pygame.Vector2(mouse_pos)
        for obj in self.all_assets():
            if pygame.Vector2(obj.x, obj.y).distance_to(m) < 15:
                return obj
        return None

    def launch_from_silo(self, silo, target_pos):
        if not silo or not silo.alive or silo.missiles <= 0 or silo.cooldown > 0:
            return False
        if self.defcon > 2:
            self.message = "Launch denied. Wait for DEFCON 2."
            return False
        self.missiles.append(Missile((silo.x, silo.y), target_pos, silo.owner, speed=random.uniform(150, 175)))
        silo.missiles -= 1
        silo.cooldown = 1.1
        self.flash = 0.22
        self.message = f"{silo.name} launch confirmed."
        return True

    def spawn_explosion(self, pos, radius=54, airburst=False):
        self.explosions.append(Explosion(pos, airburst=airburst, radius=radius, duration=0.95 if airburst else 1.12))
        for _ in range(10 if airburst else 18):
            self.particles.append(Particle(pos, NEON if airburst else EXPLOSION))

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
                        self.message = f"{obj.name} destroyed."

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
        self.launch_from_silo(silo, (target.x, target.y))
        self.message = f"WARNING: {COUNTRY_NAMES['enemy']} launch detected."

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
                if pygame.Vector2(site.x, site.y).distance_to(missile.pos) < 120:
                    hostile = missile
                    break
            if hostile:
                self.interceptors.append(Interceptor((site.x, site.y), hostile))
                site.ammo -= 1
                site.cooldown = 2.4

    def update_simulation(self, dt):
        self.scan_phase += dt * 60
        self.flash = max(0.0, self.flash - dt)
        if self.state != "play" or self.paused:
            return

        self.defcon_timer += dt
        if self.defcon > 1 and self.defcon_timer >= self.defcon_step_time:
            self.defcon_timer = 0.0
            self.defcon -= 1
            self.message = f"DEFCON {self.defcon}: {DEFCON_INFO[self.defcon]}"
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
        elif player_pop <= 0 < enemy_pop:
            self.state = "gameover"
            self.winner = "enemy"
        elif not player_strike and not enemy_strike and not self.missiles and not self.explosions:
            self.state = "gameover"
            if player_pop > enemy_pop:
                self.winner = "player"
            elif enemy_pop > player_pop:
                self.winner = "enemy"
            else:
                self.winner = "draw"

    def handle_click(self, pos, button):
        if self.state == "menu":
            self.state = "play"
            self.message = f"Simulation started. DEFCON {self.defcon}. {DEFCON_INFO[self.defcon]}"
            return
        if self.state == "gameover":
            self.reset()
            return
        if self.state != "play":
            return

        target = self.pick_target_under_mouse(pos)
        if button == 1:
            if target and isinstance(target, Silo) and target.owner == "player" and target.alive:
                self.selected_silo = target
                self.message = f"Selected {target.name}. Missiles: {target.missiles}"
            else:
                self.selected_silo = None
        elif button == 3 and self.selected_silo and self.selected_silo.alive:
            if target and target.owner != "player":
                self.launch_from_silo(self.selected_silo, (target.x, target.y))
            else:
                self.launch_from_silo(self.selected_silo, pos)

    def draw_grid(self):
        for x in range(MAP_RECT.left, MAP_RECT.right + 1, 32):
            pygame.draw.line(self.screen, GRID, (x, MAP_RECT.top), (x, MAP_RECT.bottom), 1)
        for y in range(MAP_RECT.top, MAP_RECT.bottom + 1, 32):
            pygame.draw.line(self.screen, GRID, (MAP_RECT.left, y), (MAP_RECT.right, y), 1)

    def draw_map(self):
        pygame.draw.rect(self.screen, PANEL, MAP_RECT, border_radius=4)
        self.draw_grid()

        for poly in self.continents:
            pygame.draw.polygon(self.screen, (10, 32, 42), poly)
            pygame.draw.lines(self.screen, NEON, True, poly, 2)

        for i in range(4):
            y = MAP_RECT.top + 55 + i * 72
            points = []
            for x in range(MAP_RECT.left, MAP_RECT.right, 18):
                points.append((x, y + math.sin((x * 0.02) + i) * 6))
            pygame.draw.lines(self.screen, (18, 70, 92), False, points, 1)

        scan_y = MAP_RECT.top + int(self.scan_phase % MAP_RECT.height)
        pygame.draw.line(self.screen, (30, 120, 145), (MAP_RECT.left + 2, scan_y), (MAP_RECT.right - 2, scan_y), 2)

        hovered = self.pick_target_under_mouse(pygame.mouse.get_pos())

        for city in self.cities:
            color = PLAYER if city.owner == "player" else ENEMY
            if not city.alive:
                color = DARK
            pygame.draw.circle(self.screen, color, (city.x, city.y), 5)
            pygame.draw.circle(self.screen, color, (city.x, city.y), 10, 1)
            label = f"{city.name} {city.population}M" if city.alive else f"{city.name} X"
            self.screen.blit(self.font_small.render(label, True, color if city.alive else GRAY), (city.x + 8, city.y - 7))

        for silo in self.silos:
            color = PLAYER if silo.owner == "player" else ENEMY
            if not silo.alive:
                color = DARK
            rect = pygame.Rect(silo.x - 6, silo.y - 6, 12, 12)
            pygame.draw.rect(self.screen, color, rect, border_radius=2)
            outline = WHITE if silo == self.selected_silo else color
            pygame.draw.rect(self.screen, outline, rect.inflate(8, 8), 1, border_radius=3)
            label = f"{silo.name}[{silo.missiles}]"
            self.screen.blit(self.font_small.render(label, True, color if silo.alive else GRAY), (silo.x + 8, silo.y - 7))

        for site in self.abm_sites:
            color = PLAYER if site.owner == "player" else ENEMY
            if not site.alive:
                color = DARK
            pygame.draw.circle(self.screen, color, (site.x, site.y), 7, 2)
            pygame.draw.circle(self.screen, color, (site.x, site.y), 35, 1)
            self.screen.blit(self.font_small.render(f"{site.name} {site.ammo}", True, color if site.alive else GRAY), (site.x + 8, site.y - 7))

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
            color = NEON if explosion.airburst else EXPLOSION
            pygame.draw.circle(self.screen, color, (int(explosion.pos.x), int(explosion.pos.y)), radius, 2)
            pygame.draw.circle(self.screen, color, (int(explosion.pos.x), int(explosion.pos.y)), max(2, radius // 2), 1)

        for particle in self.particles:
            alpha = max(0, min(255, int(255 * (particle.life / particle.max_life))))
            surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*particle.color, alpha), (2, 2), 2)
            self.screen.blit(surf, (particle.pos.x - 2, particle.pos.y - 2))

        if hovered:
            pygame.draw.circle(self.screen, WHITE, (int(hovered.x), int(hovered.y)), 15, 1)

        mx, my = pygame.mouse.get_pos()
        pygame.draw.line(self.screen, (80, 180, 200), (mx - 8, my), (mx + 8, my), 1)
        pygame.draw.line(self.screen, (80, 180, 200), (mx, my - 8), (mx, my + 8), 1)

    def draw_hud(self):
        pygame.draw.rect(self.screen, (7, 16, 24), HUD_RECT)
        pygame.draw.rect(self.screen, NEON, HUD_RECT, 2)
        pygame.draw.rect(self.screen, (7, 16, 24), BOTTOM_RECT)
        pygame.draw.rect(self.screen, NEON, BOTTOM_RECT, 2)

        self.screen.blit(self.font_big.render("GLOBAL THERMONET", True, WHITE), (10, 10))
        self.screen.blit(self.font_small.render("SIMULATION // STRATEGIC COMMAND", True, TEXT), (12, 32))

        defcon_col = YELLOW if self.defcon <= 2 else NEON
        self.screen.blit(self.font_big.render(f"DEFCON {self.defcon}", True, defcon_col), (610, 8))
        self.screen.blit(self.font_small.render(DEFCON_INFO[self.defcon], True, TEXT), (610, 31))

        player_pop = sum(c.population for c in self.cities if c.owner == "player" and c.alive)
        enemy_pop = sum(c.population for c in self.cities if c.owner == "enemy" and c.alive)
        player_missiles = sum(s.missiles for s in self.silos if s.owner == "player" and s.alive)
        enemy_missiles = sum(s.missiles for s in self.silos if s.owner == "enemy" and s.alive)

        bottom_left = f"{COUNTRY_NAMES['player']}: POP {player_pop}M  MIS {player_missiles}"
        bottom_right = f"{COUNTRY_NAMES['enemy']}: POP {enemy_pop}M  MIS {enemy_missiles}"
        self.screen.blit(self.font_small.render(bottom_left, True, PLAYER), (8, HEIGHT - 30))
        self.screen.blit(self.font_small.render(bottom_right, True, ENEMY), (420, HEIGHT - 30))
        self.screen.blit(self.font_small.render(self.message[:94], True, TEXT), (8, HEIGHT - 15))

    def draw_overlay_fx(self):
        for x, y, size in self.stars:
            self.screen.fill((40, 60, 75), ((x, y), (size, size)))
        if self.flash > 0:
            surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            surf.fill((255, 255, 255, int(90 * self.flash)))
            self.screen.blit(surf, (0, 0))
        scan = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 3):
            pygame.draw.line(scan, (0, 0, 0, 18), (0, y), (WIDTH, y))
        self.screen.blit(scan, (0, 0))

    def draw_menu(self):
        self.screen.fill(BG)
        self.draw_overlay_fx()
        title = self.font_huge.render("SHALL WE PLAY A GAME?", True, WHITE)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 92))
        subtitle = self.font.render("A compact pygame strategy game inspired by movie-style nuclear command sims.", True, NEON)
        self.screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 135))

        lines = [
            "Two fictional blocs. One abstract command map. No real-world target set.",
            "Left click a green silo to arm it.",
            "Right click an enemy city, silo, or open map position to launch.",
            "ABM rings automatically engage nearby hostile missiles at DEFCON 3.",
            "Win by preserving more population than the enemy.",
            "Click to begin.",
        ]
        for i, line in enumerate(lines):
            color = YELLOW if i == len(lines) - 1 else TEXT
            rendered = self.font.render(line, True, color)
            self.screen.blit(rendered, (WIDTH // 2 - rendered.get_width() // 2, 205 + i * 28))

    def draw_gameover(self):
        outcome = {"player": "VICTORY", "enemy": "DEFEAT", "draw": "STALEMATE"}.get(self.winner, "COMPLETE")
        color = PLAYER if self.winner == "player" else ENEMY if self.winner == "enemy" else YELLOW
        box = pygame.Rect(200, 145, 400, 150)
        pygame.draw.rect(self.screen, (6, 16, 24), box, border_radius=6)
        pygame.draw.rect(self.screen, color, box, 2, border_radius=6)
        txt = self.font_huge.render(outcome, True, color)
        self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 172))
        player_pop = sum(c.population for c in self.cities if c.owner == "player" and c.alive)
        enemy_pop = sum(c.population for c in self.cities if c.owner == "enemy" and c.alive)
        detail = self.font.render(f"Survivors // {player_pop}M vs {enemy_pop}M", True, TEXT)
        self.screen.blit(detail, (WIDTH // 2 - detail.get_width() // 2, 224))
        prompt = self.font.render("Click to restart.", True, WHITE)
        self.screen.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2, 252))

    def draw(self):
        self.screen.fill(BG)
        if self.state == "menu":
            self.draw_menu()
        else:
            self.draw_map()
            self.draw_hud()
            if self.state == "gameover":
                self.draw_gameover()
        self.draw_overlay_fx()
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_SPACE and self.state == "play":
                        self.paused = not self.paused
                        self.message = "Simulation paused." if self.paused else "Simulation resumed."
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos, event.button)

            self.update_simulation(dt)
            self.draw()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
