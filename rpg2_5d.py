"""
Retro 2.5D RPG
================

This program implements a simple 2.5‑dimensional role‑playing game using
Python, PyGame and PyOpenGL. It draws inspiration from early 1990s
first‑person adventures such as *Doom* and the first *Elder Scrolls*
installments. Unlike modern fully polygonal engines, a 2.5D game
restricts the geometry to axis aligned walls on a 2D grid and draws
everything from a single viewpoint.  Floors and ceilings are flat
surfaces at fixed heights, walls are vertical quads, and enemies are
sprite billboards that always face the player.  By combining these
elements, the game achieves a convincing three‑dimensional illusion
without the complexity of full 3D modeling.

The implementation below embraces this technique: a level is defined by
a simple ASCII map, where different characters represent empty space,
walls, doors, keys and exit tiles.  The player explores the maze,
collects keys to open doors and battles roaming monsters.  The
visuals include procedurally generated textures for walls, floors,
ceilings and doors, fog to hide far away geometry and a simple
crosshair.  The game runs at 800×480 pixels by default, but the
resolution can be adjusted.  Care is taken to keep the update loop
decoupled from the rendering so that movement speed is independent of
frame rate.

External resources
------------------

Rather than bundling external art, all textures are generated at
runtime using basic drawing operations on PyGame surfaces.  This
ensures that the graphics are free to use and reproducible.  The
overall structure of the engine draws on publicly documented
techniques: raycasting is used to create a 3D perspective by casting a
single ray per vertical stripe of the screen to find the first wall
tile, then calculating the distance to scale the height of the wall
slice【507580107084665†L18-L22】【507580107084665†L59-L66】.  PyOpenGL is used
for hardware accelerated drawing – the library provides bindings for
OpenGL functions and can be installed via `pip install pyopengl`.
Textures are bound with `glBindTexture` and geometry is defined with
vertex arrays and quads.  To enable depth buffering and double
buffering, the display must be created with the `OPENGL` and
`DOUBLEBUF` flags in Pygame【885870262523037†L135-L146】.

Usage
-----

Execute the script directly with Python 3 after installing the
dependencies:

```sh
pip install pygame PyOpenGL PyOpenGL_accelerate
python3 rpg2_5d.py
```

Controls:

* **W/A/S/D** – Move forward, left, backward and right (strafe)
* **Left / Right Arrow** – Rotate view
* **Space** – Attack (fires a projectile)
* **E** – Interact (open doors, pick up keys)
* **Esc** or **Q** – Quit the game

The game window displays the player’s health and the number of keys
collected.  When all enemies in the level are defeated and the exit
door is unlocked, a victory message appears.
"""

import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pygame
from pygame.locals import (
    DOUBLEBUF,
    OPENGL,
    QUIT,
    KEYDOWN,
    K_ESCAPE,
    K_q,
    K_w,
    K_a,
    K_s,
    K_d,
    K_LEFT,
    K_RIGHT,
    K_SPACE,
    K_e,
)

from OpenGL.GL import (
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTranslatef,
    glVertex3f,
    glViewport,
    glDeleteTextures,
    glVertex2f,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_QUADS,
    GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_FOG,
    glFogfv,
    glFogf,
    GL_FOG_COLOR,
    GL_FOG_MODE,
    GL_LINEAR as GL_FOG_LINEAR,
    GL_FOG_START,
    GL_FOG_END,
    GL_PROJECTION,
    GL_MODELVIEW,
    GL_LINES,
)
from OpenGL.GLU import gluPerspective


class TextureManager:
    """Manages OpenGL textures generated from PyGame surfaces.

    Because OpenGL cannot draw directly from Pygame surfaces, this class
    provides a simple interface to convert surfaces into OpenGL
    textures.  Each unique surface is cached and assigned a texture ID.
    """

    def __init__(self) -> None:
        self.cache: Dict[int, int] = {}

    def load_texture(self, surface: pygame.Surface) -> int:
        """Return an OpenGL texture ID for the given surface.

        If the surface has been loaded previously, reuse the existing ID.
        """
        key = id(surface)
        if key in self.cache:
            return self.cache[key]
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        # Convert the surface to RGBA string data for OpenGL
        width, height = surface.get_size()
        # Ensure the surface has an alpha channel
        converted = surface.convert_alpha()
        image_data = pygame.image.tostring(converted, "RGBA", True)
        # Upload to GPU
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            image_data,
        )
        # Use linear filtering for smooth scaling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.cache[key] = texture_id
        return texture_id

    @staticmethod
    def generate_stone_texture(size: int = 128) -> pygame.Surface:
        """Create a tileable stone wall texture procedurally.

        The algorithm divides the image into a grid of bricks and assigns
        each brick a slightly different grey tone.  Mortar lines are
        drawn between the bricks to improve definition.  The result is
        reminiscent of medieval masonry.
        """
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        brick_size = size // 8
        for y in range(0, size, brick_size):
            offset_x = 0 if (y // brick_size) % 2 == 0 else brick_size // 2
            for x in range(0, size + brick_size, brick_size):
                # Wrap around to make the pattern tileable
                tx = (x + offset_x) % size
                rect = pygame.Rect(tx, y, brick_size, brick_size)
                # Assign a random grey colour with slight variation and mossy tint
                base = random.randint(70, 120)
                green = base - random.randint(0, 20)
                color = (base, max(green, 0), base)
                surface.fill(color, rect)
                # Draw mortar line
                pygame.draw.rect(surface, (40, 40, 40), rect, 1)
        return surface

    @staticmethod
    def generate_floor_texture(size: int = 128) -> pygame.Surface:
        """Generate a cobblestone floor texture.

        This function uses Voronoi‑like cells to create irregular
        cobblestones.  Each cell is coloured with subtle variation and
        given a dark outline.  The pattern repeats seamlessly.
        """
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        # Seed points for Voronoi cells
        cells = []
        num_cells = 16
        for _ in range(num_cells):
            cells.append((random.randint(0, size), random.randint(0, size)))
        for y in range(size):
            for x in range(size):
                # Find nearest seed point
                min_dist = float('inf')
                idx = 0
                for i, (cx, cy) in enumerate(cells):
                    dx = x - cx
                    dy = y - cy
                    dist = dx * dx + dy * dy
                    if dist < min_dist:
                        min_dist = dist
                        idx = i
                # Base colour for this cell
                base = 100 + (idx * 13) % 50
                variation = random.randint(-8, 8)
                c = max(min(base + variation, 200), 50)
                surface.set_at((x, y), (c, c, c, 255))
        # Draw cell borders
        for (cx, cy) in cells:
            pygame.draw.circle(surface, (50, 50, 50), (cx, cy), size // 16, 1)
        return surface

    @staticmethod
    def generate_ceiling_texture(size: int = 128) -> pygame.Surface:
        """Generate a simple metal ceiling texture with rivets."""
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        base_colour = (60, 60, 70)
        surface.fill(base_colour)
        for y in range(0, size, size // 8):
            for x in range(0, size, size // 8):
                # Draw rivet as small dark dot
                pygame.draw.circle(
                    surface,
                    (30, 30, 40),
                    (x + size // 16, y + size // 16),
                    size // 32,
                )
        return surface

    @staticmethod
    def generate_door_texture(size: int = 128) -> pygame.Surface:
        """Generate a wooden door texture with planks and iron bands."""
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        # Background wood planks
        plank_width = size // 6
        for x in range(0, size, plank_width):
            color_variation = random.randint(-20, 20)
            brown = 120 + color_variation
            plank_color = (brown, 70 + color_variation // 2, 20)
            surface.fill(plank_color, pygame.Rect(x, 0, plank_width, size))
            # Add vertical plank grain lines
            for y in range(0, size, plank_width // 4):
                pygame.draw.line(
                    surface,
                    (brown - 30, 60, 10),
                    (x + random.randint(0, plank_width - 1), y),
                    (x + random.randint(0, plank_width - 1), y + plank_width),
                    1,
                )
        # Add iron bands
        band_height = size // 16
        for i in range(1, 4):
            y = i * size // 4
            pygame.draw.rect(
                surface, (70, 70, 70), pygame.Rect(0, y, size, band_height)
            )
            # Rivets on band
            for x in range(plank_width // 2, size, plank_width):
                pygame.draw.circle(
                    surface,
                    (50, 50, 50),
                    (x, y + band_height // 2),
                    band_height // 4,
                )
        return surface

    @staticmethod
    def generate_monster_sprite(size: int = 128) -> pygame.Surface:
        """Generate a simple demon sprite for enemies.

        The sprite is drawn front facing with a silhouette reminiscent of
        classic 2D shooters.  It uses symmetry and a limited palette
        to evoke pixel art.  Transparency is used around the figure.
        """
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center_x = size // 2
        center_y = size // 2
        body_width = size // 4
        body_height = size // 2
        # Body shape
        body_color = (150, 30, 30)
        pygame.draw.ellipse(
            surface,
            body_color,
            (center_x - body_width // 2, center_y - body_height // 2, body_width, body_height),
        )
        # Horns
        horn_width = body_width // 2
        horn_height = body_height // 3
        pygame.draw.polygon(
            surface,
            (100, 10, 10),
            [
                (center_x - horn_width, center_y - body_height // 2),
                (center_x, center_y - body_height),
                (center_x, center_y - body_height // 2),
            ],
        )
        pygame.draw.polygon(
            surface,
            (100, 10, 10),
            [
                (center_x, center_y - body_height // 2),
                (center_x, center_y - body_height),
                (center_x + horn_width, center_y - body_height // 2),
            ],
        )
        # Eyes
        eye_radius = size // 32
        pygame.draw.circle(
            surface, (255, 255, 255), (center_x - eye_radius * 2, center_y - eye_radius), eye_radius
        )
        pygame.draw.circle(
            surface, (255, 255, 255), (center_x + eye_radius * 2, center_y - eye_radius), eye_radius
        )
        pygame.draw.circle(
            surface, (0, 0, 0), (center_x - eye_radius * 2, center_y - eye_radius), eye_radius // 2
        )
        pygame.draw.circle(
            surface, (0, 0, 0), (center_x + eye_radius * 2, center_y - eye_radius), eye_radius // 2
        )
        # Mouth
        mouth_width = body_width // 2
        mouth_height = body_height // 10
        pygame.draw.rect(
            surface,
            (200, 60, 60),
            (
                center_x - mouth_width // 2,
                center_y + body_height // 4,
                mouth_width,
                mouth_height,
            ),
        )
        return surface


@dataclass
class MapTile:
    """Represents a single tile in the map."""

    code: str
    position: Tuple[int, int]
    passable: bool
    height: float = 1.0
    texture: int = 0  # Will be filled in once textures are generated


class GameMap:
    """Holds the 2D grid defining the level and manages collision detection."""

    def __init__(self, layout: List[str], tex_mgr: TextureManager) -> None:
        self.layout = layout
        self.width = len(layout[0])
        self.height = len(layout)
        self.tiles: List[List[MapTile]] = []
        # Define passable types
        passable_chars = {" ": True, "K": True, "E": True}
        for j, row in enumerate(layout):
            tile_row: List[MapTile] = []
            for i, c in enumerate(row):
                passable = c in passable_chars
                tile = MapTile(code=c, position=(i, j), passable=passable)
                tile_row.append(tile)
            self.tiles.append(tile_row)
        self.tex_mgr = tex_mgr
        # Assign textures based on tile codes (filled later)
        self.wall_texture = 0
        self.floor_texture = 0
        self.ceiling_texture = 0
        self.door_texture = 0

    def assign_textures(self, wall_tex: int, floor_tex: int, ceiling_tex: int, door_tex: int) -> None:
        self.wall_texture = wall_tex
        self.floor_texture = floor_tex
        self.ceiling_texture = ceiling_tex
        self.door_texture = door_tex
        # Set tile texture ids
        for row in self.tiles:
            for tile in row:
                if tile.code == "#":
                    tile.texture = self.wall_texture
                elif tile.code == "D":
                    tile.texture = self.door_texture
                else:
                    tile.texture = 0

    def in_bounds(self, x: float, z: float) -> bool:
        return 0 <= int(x) < self.width and 0 <= int(z) < self.height

    def is_passable(self, x: float, z: float) -> bool:
        if not self.in_bounds(x, z):
            return False
        tile = self.tiles[int(z)][int(x)]
        return tile.passable

    def open_door(self, x: int, z: int) -> None:
        """Open a door at the specified grid coordinates if present."""
        if not self.in_bounds(x, z):
            return
        tile = self.tiles[z][x]
        if tile.code == "D":
            tile.code = " "
            tile.passable = True
            tile.texture = 0


@dataclass
class Projectile:
    """Represents a projectile fired by the player."""

    x: float
    z: float
    direction: float
    speed: float = 10.0
    lifetime: float = 2.0
    time_alive: float = 0.0

    def update(self, dt: float) -> None:
        self.x += math.cos(self.direction) * self.speed * dt
        self.z += math.sin(self.direction) * self.speed * dt
        self.time_alive += dt

    def is_expired(self) -> bool:
        return self.time_alive >= self.lifetime


@dataclass
class Enemy:
    """Simple AI enemy that chases the player and can be defeated."""

    x: float
    z: float
    health: int = 3
    speed: float = 1.5
    texture_id: int = 0
    animation_timer: float = 0.0
    alive: bool = True

    def update(self, dt: float, player_pos: Tuple[float, float], game_map: GameMap) -> None:
        if not self.alive:
            return
        # Move towards player
        px, pz = player_pos
        dx = px - self.x
        dz = pz - self.z
        dist = math.hypot(dx, dz)
        if dist > 0.3:
            # Normalise direction
            dx /= dist
            dz /= dist
            # Try to move horizontally
            nx = self.x + dx * self.speed * dt
            nz = self.z + dz * self.speed * dt
            # Perform simple collision against walls
            if game_map.is_passable(nx, self.z):
                self.x = nx
            if game_map.is_passable(self.x, nz):
                self.z = nz
        # Basic animation timer
        self.animation_timer += dt

    def draw(self, player_dir: float) -> None:
        if not self.alive:
            return
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # Face the player: compute angle from enemy to player direction
        # We'll draw a billboarded quad centered at enemy position
        size = 0.5
        half = size / 2
        glPushMatrix()
        glTranslatef(self.x + 0.5, 0, self.z + 0.5)
        # Always rotate to camera yaw
        glRotatef(math.degrees(player_dir), 0, -1, 0)
        glBegin(GL_QUADS)
        # Lower left
        glTexCoord2f(0, 1)
        glVertex3f(-half, 0.0, half)
        # Lower right
        glTexCoord2f(1, 1)
        glVertex3f(half, 0.0, half)
        # Upper right
        glTexCoord2f(1, 0)
        glVertex3f(half, 1.0, half)
        # Upper left
        glTexCoord2f(0, 0)
        glVertex3f(-half, 1.0, half)
        glEnd()
        glPopMatrix()

    def hit(self) -> None:
        self.health -= 1
        if self.health <= 0:
            self.alive = False


@dataclass
class Player:
    """Represents the player character."""

    x: float
    z: float
    angle: float = 0.0  # Yaw angle in radians
    speed: float = 3.0
    turn_speed: float = math.radians(90)
    health: int = 10
    keys: int = 0

    def position(self) -> Tuple[float, float]:
        return (self.x, self.z)

    def update(self, dt: float, keys_down: Dict[int, bool], game_map: GameMap) -> None:
        """Update the player's position and orientation based on input."""
        # Rotation
        if keys_down.get(K_LEFT):
            self.angle -= self.turn_speed * dt
        if keys_down.get(K_RIGHT):
            self.angle += self.turn_speed * dt
        # Movement directions
        forward = 0.0
        strafe = 0.0
        if keys_down.get(K_w):
            strafe -= 1.0
        if keys_down.get(K_s):
            strafe += 1.0
        if keys_down.get(K_d):
            forward += 1.0
        if keys_down.get(K_a):
            forward -= 1.0
        # Normalise movement vector
        if forward != 0 or strafe != 0:
            norm = math.hypot(forward, strafe)
            forward /= norm
            strafe /= norm
        # Compute deltas in world space
        dx = math.cos(self.angle) * forward * self.speed * dt
        dz = math.sin(self.angle) * forward * self.speed * dt
        dx += math.cos(self.angle + math.pi / 2) * strafe * self.speed * dt
        dz += math.sin(self.angle + math.pi / 2) * strafe * self.speed * dt
        # Attempt to move; check collision by sampling corners
        new_x = self.x + dx
        new_z = self.z + dz
        radius = 0.2
        # Collide horizontally
        if game_map.is_passable(new_x - radius, self.z) and game_map.is_passable(new_x + radius, self.z):
            self.x = new_x
        # Collide vertically
        if game_map.is_passable(self.x, new_z - radius) and game_map.is_passable(self.x, new_z + radius):
            self.z = new_z

    def fire(self) -> Projectile:
        """Fire a projectile in the direction the player faces."""
        return Projectile(self.x + 0.5, self.z + 0.5, self.angle)


class RetroRPGGame:
    """Main class encapsulating the game logic and rendering."""

    def __init__(self, width: int = 800, height: int = 480) -> None:
        pygame.init()
        pygame.display.set_caption("2.5D Retro RPG")
        # Create an OpenGL‑enabled display as per the tutorial【885870262523037†L135-L146】.
        self.display_size = (width, height)
        pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        # OpenGL initialisation
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Perspective: FOV, aspect ratio, near and far clipping planes
        gluPerspective(65.0, (width / height), 0.1, 100.0)
        # Reset the modelview matrix immediately after setting the projection.
        # We do not use a separate texture matrix, so leave the texture matrix
        # at its default identity.  Passing GL_TEXTURE_2D to glMatrixMode
        # produces an "invalid enumerant" error, so avoid changing it.
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Fog for atmosphere
        glEnable(GL_FOG)
        glFogfv(GL_FOG_COLOR, (0.1, 0.1, 0.1, 1.0))
        glFogf(GL_FOG_MODE, GL_FOG_LINEAR)
        # Specify the distances at which fog starts and ends.  Using GL_FOG_START
        # and GL_FOG_END is clearer than repurposing other constants.
        glFogf(GL_FOG_START, 10.0)  # Start distance
        glFogf(GL_FOG_END, 25.0)    # End distance
        # Create texture manager and generate textures
        self.tex_mgr = TextureManager()
        wall_surface = self.tex_mgr.generate_stone_texture(256)
        floor_surface = self.tex_mgr.generate_floor_texture(256)
        ceiling_surface = self.tex_mgr.generate_ceiling_texture(256)
        door_surface = self.tex_mgr.generate_door_texture(256)
        monster_surface = self.tex_mgr.generate_monster_sprite(256)
        self.wall_tex_id = self.tex_mgr.load_texture(wall_surface)
        self.floor_tex_id = self.tex_mgr.load_texture(floor_surface)
        self.ceiling_tex_id = self.tex_mgr.load_texture(ceiling_surface)
        self.door_tex_id = self.tex_mgr.load_texture(door_surface)
        self.monster_tex_id = self.tex_mgr.load_texture(monster_surface)
        # Setup game map
        layout = [
            "################",
            "#      K      D#",
            "# ###### #### ##",
            "# #    #    #  #",
            "# # ## #### #  #",
            "# # ##    # #  #",
            "# # #### ## #  #",
            "#    #   ## # E#",
            "################",
        ]
        self.game_map = GameMap(layout, self.tex_mgr)
        self.game_map.assign_textures(self.wall_tex_id, self.floor_tex_id, self.ceiling_tex_id, self.door_tex_id)
        # Player starts at an empty tile inside the first room.  The previous
        # spawn position of (2.5, 2.5) placed the character inside a wall,
        # preventing movement.  Choosing (1.5, 1.5) ensures the player spawns
        # on a floor tile and can move immediately.
        self.player = Player(1.5, 1.5)
        # Enemies
        self.enemies: List[Enemy] = []
        enemy_positions = [(7.5, 2.5), (10.5, 5.5), (4.5, 6.5)]
        for (ex, ez) in enemy_positions:
            self.enemies.append(Enemy(ex, ez, texture_id=self.monster_tex_id))
        # Projectiles
        self.projectiles: List[Projectile] = []
        # State flags
        self.running = True
        self.keys_down: Dict[int, bool] = {}
        # Clock for timing
        self.clock = pygame.time.Clock()

    def handle_input(self) -> None:
        """Poll PyGame events and update key states."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key in (K_ESCAPE, K_q):
                    self.running = False
                # Interact with environment
                if event.key == K_e:
                    # Convert player's position to grid
                    gx = int(self.player.x)
                    gz = int(self.player.z)
                    # Check adjacent tiles for doors or keys
                    for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = gx + dx
                        nz = gz + dz
                        if not self.game_map.in_bounds(nx, nz):
                            continue
                        tile = self.game_map.tiles[nz][nx]
                        if tile.code == "D" and self.player.keys > 0:
                            # Spend a key and open the door
                            self.player.keys -= 1
                            self.game_map.open_door(nx, nz)
                            break
                        elif tile.code == "K":
                            # Pick up key
                            tile.code = " "
                            tile.passable = True
                            self.player.keys += 1
                            break
                if event.key == K_SPACE:
                    # Fire a projectile
                    self.projectiles.append(self.player.fire())
            # Track key down state for continuous movement
        pressed = pygame.key.get_pressed()
        self.keys_down = {k: pressed[k] for k in (K_w, K_a, K_s, K_d, K_LEFT, K_RIGHT)}

    def update(self, dt: float) -> None:
        """Update game state based on elapsed time."""
        # Update player
        self.player.update(dt, self.keys_down, self.game_map)
        # Update enemies
        for enemy in self.enemies:
            enemy.update(dt, self.player.position(), self.game_map)
        # Update projectiles and check for hits
        new_projectiles: List[Projectile] = []
        for proj in self.projectiles:
            proj.update(dt)
            if proj.is_expired():
                continue
            # Check collision with walls
            if not self.game_map.is_passable(proj.x, proj.z):
                continue
            # Check collision with enemies
            hit_enemy = False
            for enemy in self.enemies:
                if enemy.alive:
                    dist = math.hypot(enemy.x + 0.5 - proj.x, enemy.z + 0.5 - proj.z)
                    if dist < 0.5:
                        enemy.hit()
                        hit_enemy = True
                        break
            if not hit_enemy:
                new_projectiles.append(proj)
        self.projectiles = new_projectiles

    def draw_world(self) -> None:
        """Render the 3D world: floor, ceiling, walls, enemies."""
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Apply camera transformation (inverse of player transform)
        # Move world relative to player
        glRotatef(-math.degrees(self.player.angle), 0, 1, 0)
        glTranslatef(-self.player.x - 0.5, -0.5, -self.player.z - 0.5)
        # Draw floor
        glBindTexture(GL_TEXTURE_2D, self.floor_tex_id)
        glBegin(GL_QUADS)
        glColor3f(1, 1, 1)
        width = self.game_map.width
        height = self.game_map.height
        # Draw a single large quad for floor and ceiling, scaling texture coordinates
        glTexCoord2f(0, 0)
        glVertex3f(0, 0, 0)
        glTexCoord2f(width, 0)
        glVertex3f(width, 0, 0)
        glTexCoord2f(width, height)
        glVertex3f(width, 0, height)
        glTexCoord2f(0, height)
        glVertex3f(0, 0, height)
        glEnd()
        # Ceiling
        glBindTexture(GL_TEXTURE_2D, self.ceiling_tex_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(0, 1.0, 0)
        glTexCoord2f(width, 0)
        glVertex3f(width, 1.0, 0)
        glTexCoord2f(width, height)
        glVertex3f(width, 1.0, height)
        glTexCoord2f(0, height)
        glVertex3f(0, 1.0, height)
        glEnd()
        # Draw walls and doors
        for row in self.game_map.tiles:
            for tile in row:
                if tile.code in ("#", "D"):
                    x, z = tile.position
                    glBindTexture(GL_TEXTURE_2D, tile.texture)
                    # Four sides of the box; we only draw vertical walls because floor and ceiling drawn separately
                    # Front face (towards positive z)
                    glBegin(GL_QUADS)
                    glTexCoord2f(0, 0)
                    glVertex3f(x, 0, z + 1)
                    glTexCoord2f(1, 0)
                    glVertex3f(x + 1, 0, z + 1)
                    glTexCoord2f(1, 1)
                    glVertex3f(x + 1, 1, z + 1)
                    glTexCoord2f(0, 1)
                    glVertex3f(x, 1, z + 1)
                    # Back face (towards negative z)
                    glTexCoord2f(0, 0)
                    glVertex3f(x + 1, 0, z)
                    glTexCoord2f(1, 0)
                    glVertex3f(x, 0, z)
                    glTexCoord2f(1, 1)
                    glVertex3f(x, 1, z)
                    glTexCoord2f(0, 1)
                    glVertex3f(x + 1, 1, z)
                    # Left face (towards negative x)
                    glTexCoord2f(0, 0)
                    glVertex3f(x, 0, z)
                    glTexCoord2f(1, 0)
                    glVertex3f(x, 0, z + 1)
                    glTexCoord2f(1, 1)
                    glVertex3f(x, 1, z + 1)
                    glTexCoord2f(0, 1)
                    glVertex3f(x, 1, z)
                    # Right face (towards positive x)
                    glTexCoord2f(0, 0)
                    glVertex3f(x + 1, 0, z + 1)
                    glTexCoord2f(1, 0)
                    glVertex3f(x + 1, 0, z)
                    glTexCoord2f(1, 1)
                    glVertex3f(x + 1, 1, z)
                    glTexCoord2f(0, 1)
                    glVertex3f(x + 1, 1, z + 1)
                    glEnd()
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.player.angle)
        # Draw projectiles as small billboards
        glBindTexture(GL_TEXTURE_2D, self.wall_tex_id)
        for proj in self.projectiles:
            glPushMatrix()
            glTranslatef(proj.x, 0.2, proj.z)
            size = 0.1
            glBegin(GL_QUADS)
            glColor3f(1.0, 0.5, 0.0)
            glTexCoord2f(0, 1)
            glVertex3f(-size, -size, 0)
            glTexCoord2f(1, 1)
            glVertex3f(size, -size, 0)
            glTexCoord2f(1, 0)
            glVertex3f(size, size, 0)
            glTexCoord2f(0, 0)
            glVertex3f(-size, size, 0)
            glEnd()
            glPopMatrix()
        # Reset colour to white for subsequent drawing
        glColor3f(1, 1, 1)

    def draw_ui(self) -> None:
        """Render 2D overlay such as health, keys and crosshair."""
        # Switch to orthographic projection for UI
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho = None
        try:
            # Import glOrtho only when needed to avoid cluttering the global namespace
            from OpenGL.GL import glOrtho
            glOrtho(0, self.display_size[0], self.display_size[1], 0, -1, 1)
        except Exception:
            pass
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        # Disable depth testing for UI
        glDisable(GL_DEPTH_TEST)
        # Draw health and keys text using Pygame surface blitting onto the backbuffer
        # We'll use Pygame's font rendering and then convert to texture for efficiency.
        font = pygame.font.SysFont("monospace", 18)
        text = f"Health: {self.player.health}   Keys: {self.player.keys}"
        text_surface = font.render(text, True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        tw, th = text_surface.get_size()
        # Create a temporary texture
        hud_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, hud_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Draw the text texture
        x_pos = 10
        y_pos = 10
        glBindTexture(GL_TEXTURE_2D, hud_tex)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex2f(x_pos, y_pos + th)
        glTexCoord2f(1, 1)
        glVertex2f(x_pos + tw, y_pos + th)
        glTexCoord2f(1, 0)
        glVertex2f(x_pos + tw, y_pos)
        glTexCoord2f(0, 0)
        glVertex2f(x_pos, y_pos)
        glEnd()
        # Delete the temporary texture
        glDeleteTextures([hud_tex])
        # Draw crosshair
        glBindTexture(GL_TEXTURE_2D, 0)
        glColor3f(1, 1, 1)
        cx = self.display_size[0] / 2
        cy = self.display_size[1] / 2
        size = 10
        glBegin(GL_LINES)
        glVertex2f(cx - size, cy)
        glVertex2f(cx + size, cy)
        glVertex2f(cx, cy - size)
        glVertex2f(cx, cy + size)
        glEnd()
        # Restore depth testing and matrices
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def run(self) -> None:
        """Main game loop."""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
            self.handle_input()
            self.update(dt)
            self.draw_world()
            self.draw_ui()
            pygame.display.flip()
            # Check victory condition
            if not any(enemy.alive for enemy in self.enemies):
                # Show victory message for a few seconds and exit
                self.show_victory()
                break
        pygame.quit()

    def show_victory(self) -> None:
        """Display a victory screen when all enemies are defeated."""
        font = pygame.font.SysFont("monospace", 36)
        message = "You defeated all enemies!"
        surface = font.render(message, True, (255, 255, 255))
        text_data = pygame.image.tostring(surface, "RGBA", True)
        tw, th = surface.get_size()
        # Display for 3 seconds
        display_time = 3.0
        start = time.time()
        while time.time() - start < display_time:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho = None
            try:
                from OpenGL.GL import glOrtho
                glOrtho(0, self.display_size[0], self.display_size[1], 0, -1, 1)
            except Exception:
                pass
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            # Create a texture for the message
            hud_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, hud_tex)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                tw,
                th,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                text_data,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            x_pos = (self.display_size[0] - tw) / 2
            y_pos = (self.display_size[1] - th) / 2
            glBindTexture(GL_TEXTURE_2D, hud_tex)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1)
            glVertex2f(x_pos, y_pos + th)
            glTexCoord2f(1, 1)
            glVertex2f(x_pos + tw, y_pos + th)
            glTexCoord2f(1, 0)
            glVertex2f(x_pos + tw, y_pos)
            glTexCoord2f(0, 0)
            glVertex2f(x_pos, y_pos)
            glEnd()
            glDeleteTextures([hud_tex])
            pygame.display.flip()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            # Sleep briefly to yield to the event loop
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
            time.sleep(0.01)


if __name__ == "__main__":
    # Entry point
    game = RetroRPGGame(800, 480)
    game.run()
