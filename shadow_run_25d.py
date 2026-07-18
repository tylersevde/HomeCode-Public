
import math
import random
import sys
import pygame

# ------------------------------------------------------------
# Shadow Run 2.5D
# A simple faux-2.5D platformer with 3 levels and a final boss.
#
# Controls:
#   A / Left Arrow   = move left
#   D / Right Arrow  = move right
#   W / Up Arrow     = move "deeper" into screen lane
#   S / Down Arrow   = move "closer" lane
#   Space            = jump
#   J                = attack
#   R                = restart after game over / win
#   Esc              = quit
#
# Install:
#   pip install pygame
# Run:
#   python shadow_run_25d.py
# ------------------------------------------------------------

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 800, 480
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shadow Run 2.5D")
CLOCK = pygame.time.Clock()

FPS = 60
GRAVITY = 0.75
PLAYER_SPEED = 4.0
JUMP_VELOCITY = -12.5
GROUND_Y = 430

WHITE = (245, 245, 245)
BLACK = (15, 15, 20)
SKY = (110, 160, 220)
FAR_HILL = (80, 110, 160)
MID_HILL = (65, 90, 135)
GROUND = (70, 65, 55)
PLATFORM = (120, 95, 70)
PLATFORM_TOP = (175, 145, 110)
PLAYER_COLOR = (245, 210, 80)
ENEMY_COLOR = (220, 90, 90)
BOSS_COLOR = (160, 60, 200)
BULLET_COLOR = (255, 240, 110)
BOSS_BULLET = (255, 90, 180)
UI_PANEL = (20, 20, 30)

FONT = pygame.font.SysFont("arial", 24)
BIG_FONT = pygame.font.SysFont("arial", 52, bold=True)

LANES = [0, 18, 36]  # visual offset for faux depth


def clamp(value, low, high):
    return max(low, min(high, value))


class Platform:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surf, cam_x, lane_shift=0):
        body = pygame.Rect(self.rect.x - cam_x, self.rect.y + lane_shift, self.rect.w, self.rect.h)
        top = pygame.Rect(body.x, body.y, body.w, 8)
        pygame.draw.rect(surf, PLATFORM, body, border_radius=4)
        pygame.draw.rect(surf, PLATFORM_TOP, top, border_radius=4)
        pygame.draw.line(surf, BLACK, (body.x, body.bottom), (body.right, body.bottom), 2)


class Bullet:
    def __init__(self, x, y, vx, vy, color, damage=1, from_player=True, radius=6, lane=1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.damage = damage
        self.from_player = from_player
        self.radius = radius
        self.alive = True
        self.lane = lane

    @property
    def rect(self):
        return pygame.Rect(int(self.x - self.radius), int(self.y - self.radius), self.radius * 2, self.radius * 2)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < -200 or self.x > 4000 or self.y < -200 or self.y > HEIGHT + 200:
            self.alive = False

    def draw(self, surf, cam_x, lane_shift=0):
        pygame.draw.circle(surf, self.color, (int(self.x - cam_x), int(self.y + lane_shift)), self.radius)


class Fighter:
    def __init__(self, x, y, w, h, color):
        self.rect = pygame.Rect(x, y, w, h)
        self.vx = 0
        self.vy = 0
        self.color = color
        self.on_ground = False
        self.facing = 1
        self.lane = 1
        self.max_hp = 5
        self.hp = self.max_hp
        self.invuln = 0
        self.attack_timer = 0

    def lane_shift(self):
        return LANES[self.lane]

    def hurt(self, amount):
        if self.invuln <= 0:
            self.hp -= amount
            self.invuln = 30

    def draw_body(self, surf, cam_x):
        shift = self.lane_shift()
        draw_rect = pygame.Rect(self.rect.x - cam_x, self.rect.y + shift, self.rect.w, self.rect.h)
        shadow = pygame.Rect(draw_rect.x + 6, GROUND_Y + shift + 5, draw_rect.w - 8, 10)
        pygame.draw.ellipse(surf, (0, 0, 0, 80), shadow)
        if self.invuln > 0 and (self.invuln // 4) % 2 == 0:
            return
        pygame.draw.rect(surf, self.color, draw_rect, border_radius=10)
        eye_x = draw_rect.centerx + 8 * self.facing
        pygame.draw.circle(surf, BLACK, (eye_x, draw_rect.y + 14), 3)

    def physics(self, platforms):
        self.vy += GRAVITY
        self.rect.x += int(self.vx)

        for p in platforms:
            if self.rect.colliderect(p.rect):
                if self.vx > 0:
                    self.rect.right = p.rect.left
                elif self.vx < 0:
                    self.rect.left = p.rect.right

        self.rect.y += int(self.vy)
        self.on_ground = False
        for p in platforms:
            if self.rect.colliderect(p.rect):
                if self.vy > 0:
                    self.rect.bottom = p.rect.top
                    self.vy = 0
                    self.on_ground = True
                elif self.vy < 0:
                    self.rect.top = p.rect.bottom
                    self.vy = 0

        if self.rect.bottom > GROUND_Y:
            self.rect.bottom = GROUND_Y
            self.vy = 0
            self.on_ground = True

        if self.invuln > 0:
            self.invuln -= 1
        if self.attack_timer > 0:
            self.attack_timer -= 1


class Player(Fighter):
    def __init__(self, x, y):
        super().__init__(x, y, 34, 54, PLAYER_COLOR)
        self.max_hp = 8
        self.hp = self.max_hp
        self.score = 0
        self.spawn = (x, y)
        self.attack_cooldown = 0

    def reset_to_spawn(self):
        self.rect.topleft = self.spawn
        self.vx = self.vy = 0
        self.lane = 1
        self.hp = self.max_hp
        self.invuln = 60

    def handle_input(self, keys):
        self.vx = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.vx = -PLAYER_SPEED
            self.facing = -1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.vx = PLAYER_SPEED
            self.facing = 1

    def try_jump(self):
        if self.on_ground:
            self.vy = JUMP_VELOCITY

    def try_lane_up(self):
        self.lane = clamp(self.lane - 1, 0, 2)

    def try_lane_down(self):
        self.lane = clamp(self.lane + 1, 0, 2)

    def try_attack(self, bullets):
        if self.attack_cooldown <= 0:
            bx = self.rect.centerx + self.facing * 24
            by = self.rect.centery - 8
            bullets.append(Bullet(
                bx, by, self.facing * 8, 0, BULLET_COLOR,
                damage=1, from_player=True, radius=5, lane=self.lane
            ))
            self.attack_cooldown = 16
            self.attack_timer = 8

    def update(self, platforms):
        self.physics(platforms)
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1


class WalkerEnemy(Fighter):
    def __init__(self, x, y, patrol_left, patrol_right, lane=1):
        super().__init__(x, y, 30, 44, ENEMY_COLOR)
        self.patrol_left = patrol_left
        self.patrol_right = patrol_right
        self.vx = 1.5
        self.lane = lane
        self.max_hp = 2
        self.hp = self.max_hp

    def update(self, platforms, player):
        if self.rect.x <= self.patrol_left:
            self.vx = abs(self.vx)
            self.facing = 1
        if self.rect.x >= self.patrol_right:
            self.vx = -abs(self.vx)
            self.facing = -1

        # aggro if same lane and nearby
        if abs(player.rect.centerx - self.rect.centerx) < 180 and player.lane == self.lane:
            self.vx = 2.4 if player.rect.centerx > self.rect.centerx else -2.4
            self.facing = 1 if self.vx > 0 else -1

        self.physics(platforms)


class HopperEnemy(Fighter):
    def __init__(self, x, y, lane=1):
        super().__init__(x, y, 30, 38, (95, 230, 180))
        self.lane = lane
        self.jump_timer = random.randint(40, 90)
        self.max_hp = 2
        self.hp = self.max_hp

    def update(self, platforms, player):
        self.vx = 1.2 if player.rect.centerx > self.rect.centerx else -1.2
        self.facing = 1 if self.vx > 0 else -1
        self.jump_timer -= 1
        if self.jump_timer <= 0 and self.on_ground:
            self.vy = -10.5
            self.jump_timer = random.randint(50, 100)
        self.physics(platforms)


class Boss(Fighter):
    def __init__(self, x, y):
        super().__init__(x, y, 90, 110, BOSS_COLOR)
        self.max_hp = 24
        self.hp = self.max_hp
        self.phase = 1
        self.shot_timer = 50
        self.jump_timer = 120
        self.lane_timer = 90

    def update(self, platforms, player, bullets):
        if self.hp <= self.max_hp * 0.55:
            self.phase = 2

        speed = 2.0 if self.phase == 1 else 3.0
        self.vx = speed if player.rect.centerx > self.rect.centerx else -speed
        self.facing = 1 if self.vx > 0 else -1

        self.shot_timer -= 1
        self.jump_timer -= 1
        self.lane_timer -= 1

        if self.shot_timer <= 0:
            dx = player.rect.centerx - self.rect.centerx
            dy = (player.rect.centery + player.lane_shift()) - (self.rect.centery + self.lane_shift())
            dist = max(1, math.hypot(dx, dy))
            speed_shot = 5 if self.phase == 1 else 6.5
            bullets.append(Bullet(
                self.rect.centerx + self.facing * 30,
                self.rect.centery - 15,
                dx / dist * speed_shot,
                dy / dist * speed_shot,
                BOSS_BULLET,
                damage=1,
                from_player=False,
                radius=8 if self.phase == 1 else 10,
                lane=self.lane
            ))
            self.shot_timer = 45 if self.phase == 1 else 25

        if self.jump_timer <= 0 and self.on_ground:
            self.vy = -12 if self.phase == 1 else -14
            self.jump_timer = 110 if self.phase == 1 else 70

        if self.lane_timer <= 0:
            self.lane = random.randint(0, 2)
            self.lane_timer = 85 if self.phase == 1 else 50

        self.physics(platforms)

    def draw_body(self, surf, cam_x):
        shift = self.lane_shift()
        draw_rect = pygame.Rect(self.rect.x - cam_x, self.rect.y + shift, self.rect.w, self.rect.h)
        shadow = pygame.Rect(draw_rect.x + 10, GROUND_Y + shift + 8, draw_rect.w - 16, 16)
        pygame.draw.ellipse(surf, (0, 0, 0, 90), shadow)
        if self.invuln > 0 and (self.invuln // 3) % 2 == 0:
            return
        pygame.draw.rect(surf, self.color, draw_rect, border_radius=16)
        pygame.draw.rect(surf, (220, 210, 245), (draw_rect.x + 12, draw_rect.y + 20, 18, 18), border_radius=8)
        pygame.draw.rect(surf, (220, 210, 245), (draw_rect.right - 30, draw_rect.y + 20, 18, 18), border_radius=8)


class Goal:
    def __init__(self, x, y, w=40, h=70):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surf, cam_x, lane_shift=0):
        r = pygame.Rect(self.rect.x - cam_x, self.rect.y + lane_shift, self.rect.w, self.rect.h)
        pygame.draw.rect(surf, (90, 220, 255), r, border_radius=10)
        pygame.draw.rect(surf, WHITE, r, 3, border_radius=10)


class Level:
    def __init__(self, name, width, platforms, enemies, goal=None, boss=None):
        self.name = name
        self.width = width
        self.platforms = platforms
        self.enemies = enemies
        self.goal = goal
        self.boss = boss


def make_levels():
    levels = []

    # Level 1
    plats1 = [
        Platform(0, GROUND_Y, 1500, 200),
        Platform(220, 360, 120, 20),
        Platform(430, 310, 140, 20),
        Platform(690, 350, 140, 20),
        Platform(930, 290, 140, 20),
        Platform(1180, 340, 120, 20),
    ]
    enemies1 = [
        WalkerEnemy(330, 316, 260, 520, lane=1),
        HopperEnemy(760, 312, lane=2),
        WalkerEnemy(1030, 246, 960, 1100, lane=0),
    ]
    goal1 = Goal(1360, 360)
    levels.append(Level("Level 1 - Rust Flats", 1500, plats1, enemies1, goal=goal1))

    # Level 2
    plats2 = [
        Platform(0, GROUND_Y, 1800, 200),
        Platform(180, 370, 120, 20),
        Platform(390, 330, 100, 20),
        Platform(560, 285, 120, 20),
        Platform(760, 245, 130, 20),
        Platform(980, 290, 130, 20),
        Platform(1220, 335, 120, 20),
        Platform(1450, 290, 140, 20),
    ]
    enemies2 = [
        WalkerEnemy(235, 326, 180, 450, lane=2),
        HopperEnemy(615, 241, lane=1),
        WalkerEnemy(825, 201, 770, 900, lane=0),
        HopperEnemy(1260, 291, lane=2),
        WalkerEnemy(1500, 246, 1460, 1590, lane=1),
    ]
    goal2 = Goal(1690, 360)
    levels.append(Level("Level 2 - Neon Steps", 1800, plats2, enemies2, goal=goal2))

    # Level 3
    plats3 = [
        Platform(0, GROUND_Y, 2000, 200),
        Platform(240, 360, 120, 20),
        Platform(470, 300, 130, 20),
        Platform(750, 340, 130, 20),
        Platform(980, 260, 140, 20),
        Platform(1240, 220, 140, 20),
        Platform(1520, 280, 140, 20),
        Platform(1750, 340, 140, 20),
    ]
    enemies3 = [
        HopperEnemy(280, 316, lane=1),
        WalkerEnemy(520, 256, 480, 620, lane=0),
        WalkerEnemy(830, 296, 760, 920, lane=2),
        HopperEnemy(1030, 216, lane=1),
        WalkerEnemy(1300, 176, 1250, 1380, lane=0),
        HopperEnemy(1570, 236, lane=2),
    ]
    goal3 = Goal(1890, 360)
    levels.append(Level("Level 3 - Sky Foundry", 2000, plats3, enemies3, goal=goal3))

    # Boss level
    plats4 = [
        Platform(0, GROUND_Y, 1700, 200),
        Platform(420, 330, 140, 20),
        Platform(1050, 330, 140, 20),
    ]
    boss = Boss(1320, 320)
    levels.append(Level("Final Boss - Void Engine", 1700, plats4, [], boss=boss))

    return levels


def draw_background(surf, cam_x):
    surf.fill(SKY)

    # far parallax
    for i in range(-1, 6):
        x = i * 260 - (cam_x * 0.2) % 260
        pygame.draw.circle(surf, FAR_HILL, (int(x + 130), 310), 170)

    # mid parallax
    for i in range(-1, 7):
        x = i * 200 - (cam_x * 0.45) % 200
        pygame.draw.circle(surf, MID_HILL, (int(x + 100), 360), 130)

    # ground strips to fake 2.5D depth lanes
    pygame.draw.rect(surf, (84, 79, 68), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    pygame.draw.line(surf, (180, 180, 180), (0, GROUND_Y + LANES[0]), (WIDTH, GROUND_Y + LANES[0]), 2)
    pygame.draw.line(surf, (140, 140, 140), (0, GROUND_Y + LANES[1]), (WIDTH, GROUND_Y + LANES[1]), 2)
    pygame.draw.line(surf, (100, 100, 100), (0, GROUND_Y + LANES[2]), (WIDTH, GROUND_Y + LANES[2]), 2)


def draw_ui(surf, player, level_name, level_index, total_levels, boss=None):
    pygame.draw.rect(surf, UI_PANEL, (10, 10, 320, 92), border_radius=12)
    hp_text = FONT.render(f"HP: {player.hp}/{player.max_hp}", True, WHITE)
    lvl_text = FONT.render(f"{level_name} ({level_index + 1}/{total_levels})", True, WHITE)
    lane_text = FONT.render(f"Lane: {player.lane + 1}/3", True, WHITE)

    surf.blit(hp_text, (20, 18))
    surf.blit(lvl_text, (20, 46))
    surf.blit(lane_text, (20, 74))

    if boss:
        pygame.draw.rect(surf, UI_PANEL, (WIDTH - 320, 10, 300, 66), border_radius=12)
        text = FONT.render("BOSS", True, WHITE)
        surf.blit(text, (WIDTH - 300, 18))
        bar_x, bar_y, bar_w, bar_h = WIDTH - 300, 46, 260, 18
        pygame.draw.rect(surf, (60, 60, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=8)
        fill = int((boss.hp / boss.max_hp) * bar_w)
        pygame.draw.rect(surf, (220, 70, 190), (bar_x, bar_y, fill, bar_h), border_radius=8)


def draw_center_message(surf, title, subtitle):
    title_s = BIG_FONT.render(title, True, WHITE)
    sub_s = FONT.render(subtitle, True, WHITE)
    surf.blit(title_s, (WIDTH // 2 - title_s.get_width() // 2, HEIGHT // 2 - 60))
    surf.blit(sub_s, (WIDTH // 2 - sub_s.get_width() // 2, HEIGHT // 2 + 8))


def respawn_enemies(level):
    # Recreate level to reset enemies cleanly
    fresh = make_levels()
    idx = 0
    for i, lv in enumerate(fresh):
        if lv.name == level.name:
            idx = i
            break
    return fresh[idx]


def main():
    levels = make_levels()
    level_index = 0
    player = Player(60, 360)
    game_over = False
    victory = False
    transition_timer = 80
    bullets = []

    while True:
        current_level = levels[level_index]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if not game_over and not victory:
                    if event.key == pygame.K_SPACE:
                        player.try_jump()
                    elif event.key in (pygame.K_w, pygame.K_UP):
                        player.try_lane_up()
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        player.try_lane_down()
                    elif event.key == pygame.K_j:
                        player.try_attack(bullets)
                if event.key == pygame.K_r and (game_over or victory):
                    levels = make_levels()
                    level_index = 0
                    player = Player(60, 360)
                    bullets = []
                    game_over = False
                    victory = False
                    transition_timer = 80

        keys = pygame.key.get_pressed()

        if not game_over and not victory:
            player.handle_input(keys)

            if transition_timer > 0:
                transition_timer -= 1
            else:
                player.update(current_level.platforms)

                # update enemies
                for enemy in current_level.enemies[:]:
                    enemy.update(current_level.platforms, player)
                    if enemy.hp <= 0:
                        current_level.enemies.remove(enemy)
                        player.score += 100

                # boss
                if current_level.boss and current_level.boss.hp > 0:
                    current_level.boss.update(current_level.platforms, player, bullets)
                elif current_level.boss and current_level.boss.hp <= 0:
                    victory = True

                # direct contact damage
                for enemy in current_level.enemies:
                    if player.rect.colliderect(enemy.rect) and player.lane == enemy.lane:
                        player.hurt(1)

                if current_level.boss and current_level.boss.hp > 0:
                    if player.rect.colliderect(current_level.boss.rect) and player.lane == current_level.boss.lane:
                        player.hurt(1)

                # bullets
                for bullet in bullets[:]:
                    bullet.update()

                    # platform collision
                    for p in current_level.platforms:
                        if bullet.rect.colliderect(p.rect):
                            bullet.alive = False
                            break

                    if bullet.from_player:
                        hit = False
                        for enemy in current_level.enemies:
                            if bullet.rect.colliderect(enemy.rect) and bullet.alive and bullet.lane == enemy.lane:
                                enemy.hurt(bullet.damage)
                                bullet.alive = False
                                hit = True
                                break
                        if (not hit) and current_level.boss and current_level.boss.hp > 0:
                            boss = current_level.boss
                            if bullet.rect.colliderect(boss.rect) and bullet.alive and bullet.lane == boss.lane:
                                boss.hurt(bullet.damage)
                                bullet.alive = False
                    else:
                        if bullet.rect.colliderect(player.rect) and bullet.alive and bullet.lane == player.lane:
                            player.hurt(bullet.damage)
                            bullet.alive = False

                    if not bullet.alive:
                        bullets.remove(bullet)

                # level progression
                if current_level.goal and player.rect.colliderect(current_level.goal.rect):
                    level_index += 1
                    if level_index >= len(levels):
                        victory = True
                    else:
                        player.rect.topleft = (60, 360)
                        player.spawn = (60, 360)
                        player.vx = player.vy = 0
                        player.lane = 1
                        bullets.clear()
                        transition_timer = 80

                # player death
                if player.hp <= 0:
                    game_over = True

        cam_x = clamp(player.rect.centerx - WIDTH // 2, 0, current_level.width - WIDTH)

        draw_background(SCREEN, cam_x)

        for lane_i in range(3):
            lane_shift = LANES[lane_i]
            for p in current_level.platforms:
                p.draw(SCREEN, cam_x, lane_shift=0 if lane_i == 1 else 0)

        # Goal
        if current_level.goal:
            current_level.goal.draw(SCREEN, cam_x, 0)

        # draw characters sorted by lane for fake depth
        drawables = []
        for e in current_level.enemies:
            drawables.append((e.lane, e.rect.bottom, e))
        if current_level.boss and current_level.boss.hp > 0:
            drawables.append((current_level.boss.lane, current_level.boss.rect.bottom, current_level.boss))
        drawables.append((player.lane, player.rect.bottom, player))
        drawables.sort(key=lambda item: (item[0], item[1]))

        for _, _, obj in drawables:
            obj.draw_body(SCREEN, cam_x)

        for bullet in bullets:
            bullet.draw(SCREEN, cam_x, LANES[bullet.lane])

        draw_ui(SCREEN, player, current_level.name, level_index, len(levels), boss=current_level.boss if current_level.boss and current_level.boss.hp > 0 else None)

        if transition_timer > 0 and not game_over and not victory:
            draw_center_message(SCREEN, current_level.name, "Get ready...")

        if game_over:
            draw_center_message(SCREEN, "GAME OVER", "Press R to restart")
        elif victory:
            draw_center_message(SCREEN, "YOU WIN", "Press R to play again")

        pygame.display.flip()
        CLOCK.tick(FPS)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("The game crashed:", exc)
        pygame.quit()
        raise
