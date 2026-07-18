#!/usr/bin/env python3
"""
Newton's Echo Chamber -- Alpha

A first-person SI-unit rigid-body playground built for Raspberry Pi 5-class
hardware with pygame and the fixed-function PyOpenGL pipeline.  The room is
100 m x 100 m x 10 m, the player is 2 m tall, and all seven objects use their
documented mass and representative real-world dimensions.

Controls
========
  W / S                 Move forward / backward
  A / D                 Strafe left / right
  Mouse                 Look (moving upward looks upward; never inverted)
  Space                 Jump to a one-metre ballistic apex
  Right mouse           Pick up / drop the object under the crosshair
  Left mouse            Throw the held object
  R / F                 Raise / lower gravity
  T / G                 Raise / lower throw force (1-2-5 logarithmic steps)
  Y / H                 Raise / lower room surface friction
  Tab                    Release / recapture the mouse
  F1                     Toggle the controls overlay
  F5                     Reset the room and all seven objects
  M                      Mute / unmute audio
  Esc                    Quit

Physics uses metres, kilograms, seconds, Newtons, semi-implicit integration at
120 Hz, swept room contacts, sequential impulses, equal-and-opposite reaction
impulses, Coulomb friction, restitution, angular inertia, quadratic air drag,
rolling resistance, and sleeping.  A throw's selected force acts through a
0.75 m arm stroke, making F and work physically meaningful while allowing the
full requested 1,000,000 N range to remain numerically tractable.

Dependencies:
  pip install pygame PyOpenGL PyOpenGL_accelerate

The impact and exertion sounds are original procedural PCM generated at start.
Audio-device failure is non-fatal.  Run ``--check`` for a headless deterministic
physics/content validation, or ``--capture PATH`` for a rendered smoke frame.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


pygame = None
GL = None
GLU = None

TITLE = "Newton's Echo Chamber -- Alpha"
RELEASE = "ALPHA"
CREDIT_WATERMARK = "Made by OpenAI ChatGPT Codex 5.6 Sol Ultra"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
PHYSICS_HZ = 120
FIXED_DT = 1.0 / PHYSICS_HZ
MAX_FRAME_DT = 0.10
MAX_PHYSICS_STEPS = 12
SOLVER_ITERATIONS = 8

ROOM_WIDTH = 100.0
ROOM_LENGTH = 100.0
ROOM_HEIGHT = 10.0
LIGHT_SPACING = 10.0
LIGHT_POSITIONS = tuple(
    (5.0 + 10.0 * x, ROOM_HEIGHT - 0.035, 5.0 + 10.0 * z)
    for z in range(10)
    for x in range(10)
)

PLAYER_HEIGHT = 2.0
PLAYER_EYE_HEIGHT = 1.76
PLAYER_RADIUS = 0.35
PLAYER_MASS = 80.0
PLAYER_START = (10.0, 0.0, 10.0)
PLAYER_WALK_SPEED = 5.0
PLAYER_GROUND_ACCEL = 24.0
PLAYER_AIR_ACCEL = 5.5
JUMP_HEIGHT = 1.0

EARTH_GRAVITY = 9.80665
GRAVITY_MIN = 0.10
GRAVITY_MAX = 50.0
GRAVITY_STEP = 0.25
DEFAULT_ROOM_FRICTION = 0.65
FRICTION_MIN = 0.0
FRICTION_MAX = 1.50
FRICTION_STEP = 0.05
THROW_FORCE_MIN = 1.0
THROW_FORCE_MAX = 1_000_000.0
THROW_STROKE = 0.75
THROW_FORCE_STEPS = tuple(
    float(mantissa * (10 ** exponent))
    for exponent in range(7)
    for mantissa in (1, 2, 5)
    if mantissa * (10 ** exponent) <= int(THROW_FORCE_MAX)
)
PICKUP_REACH = 6.0
HOLD_DISTANCE = 1.35
AIR_DENSITY = 1.229
MAX_CCD_BOUNCES = 12
MAX_LINEAR_SPEED = 5_000.0
MAX_ANGULAR_SPEED = 500.0

AUDIO_RATE = 44_100
AUDIO_CHANNELS = 2


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def approach(value: float, target: float, amount: float) -> float:
    if value < target:
        return min(target, value + amount)
    return max(target, value - amount)


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def copy(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self, fallback: Optional["Vec3"] = None) -> "Vec3":
        magnitude = self.length()
        if magnitude > 1.0e-12:
            return self / magnitude
        return fallback.copy() if fallback is not None else Vec3()

    def horizontal(self) -> "Vec3":
        return Vec3(self.x, 0.0, self.z)

    def finite(self) -> bool:
        return math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.z)

    def tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass
class Quat:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def normalized(self) -> "Quat":
        length = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if length <= 1.0e-12:
            return Quat()
        return Quat(self.w / length, self.x / length, self.y / length, self.z / length)

    def conjugate(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: "Quat") -> "Quat":
        return Quat(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate(self, vector: Vec3) -> Vec3:
        qv = Quat(0.0, vector.x, vector.y, vector.z)
        result = self * qv * self.conjugate()
        return Vec3(result.x, result.y, result.z)

    def integrate(self, angular_velocity: Vec3, dt: float) -> "Quat":
        omega = Quat(0.0, angular_velocity.x, angular_velocity.y, angular_velocity.z)
        derivative = omega * self
        return Quat(
            self.w + 0.5 * derivative.w * dt,
            self.x + 0.5 * derivative.x * dt,
            self.y + 0.5 * derivative.y * dt,
            self.z + 0.5 * derivative.z * dt,
        ).normalized()

    def matrix(self) -> Tuple[float, ...]:
        q = self.normalized()
        xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
        xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
        wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
        return (
            1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
            2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
            2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
            0, 0, 0, 1,
        )


@dataclass(frozen=True)
class BodySpec:
    key: str
    name: str
    shape: str
    mass: float
    diameter: float = 0.0
    dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float = 0.0
    restitution: float = 0.3
    friction: float = 0.6
    rolling_resistance: float = 0.02
    drag_coefficient: float = 0.5
    inertia_mode: str = "solid_sphere"
    sound_family: str = "medicine"
    rigidity: str = ""
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)

    @property
    def radius(self) -> float:
        return self.diameter * 0.5 if self.shape == "sphere" else 0.5 * max(self.dimensions)

    @property
    def half_extents(self) -> Vec3:
        return Vec3(*(dimension * 0.5 for dimension in self.dimensions))

    @property
    def volume(self) -> float:
        if self.shape == "sphere":
            return 4.0 * math.pi * self.radius ** 3 / 3.0
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    @property
    def frontal_area(self) -> float:
        if self.shape == "sphere":
            return math.pi * self.radius * self.radius
        width, height, _depth = self.dimensions
        return width * height

    @property
    def inertia_diagonal(self) -> Vec3:
        if self.shape == "sphere":
            factor = 2.0 / 3.0 if self.inertia_mode == "thin_shell" else 2.0 / 5.0
            inertia = factor * self.mass * self.radius * self.radius
            return Vec3(inertia, inertia, inertia)
        width, height, depth = self.dimensions
        return Vec3(
            self.mass * (height * height + depth * depth) / 12.0,
            self.mass * (width * width + depth * depth) / 12.0,
            self.mass * (width * width + height * height) / 12.0,
        )

    @property
    def dimension_label(self) -> str:
        if self.shape == "sphere":
            return f"diameter {self.diameter:.3f} m"
        return " x ".join(f"{value:.3f}" for value in self.dimensions) + " m"


def make_item_specs() -> Tuple[BodySpec, ...]:
    # Calibration basis (accessed 2026-07-10): rubber dodgeball rules use a
    # 65-67 cm, 300-320 g range; BOXPT publishes 190/230/280 mm medicine-ball
    # families; FIBA size 7 uses 750-770 mm circumference. Solid masses derive
    # from 7,850 kg/m^3 steel, 2,400 kg/m^3 normal concrete, and 920 kg/m^3
    # natural rubber. URLs are kept here so future revisions can audit values.
    # https://asiadodgeball.com/wp-content/uploads/2020/03/Rules_English.pdf
    # https://eu.boxpt.com/en/products/rebound-medicine-balls-rebmed
    # https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf
    # https://nvlpubs.nist.gov/nistpubs/technicalnotes/nist.tn.1681.pdf
    # https://www.cement.org/wp-content/uploads/2024/08/eb225.pdf
    # https://pml.nist.gov/cgi-bin/Star/compos.pl?matno=243
    basketball_diameter = 0.760 / math.pi
    basketball_volume = 4.0 * math.pi * (basketball_diameter * 0.5) ** 3 / 3.0
    brick_dimensions = (0.194, 0.092, 0.057)
    return (
        BodySpec(
            "dodge", "Inflated Rubber Dodgeball", "sphere", 0.310,
            diameter=0.660 / math.pi, density=0.310 / (4.0 * math.pi * (0.330 / math.pi) ** 3 / 3.0),
            restitution=0.72, friction=0.80, rolling_resistance=0.025,
            drag_coefficient=0.50, inertia_mode="thin_shell", sound_family="dodge",
            rigidity="inflated compliant shell", color=(0.96, 0.22, 0.18),
        ),
        BodySpec(
            "medicine_1", "Medicine Ball 1 kg", "sphere", 1.0,
            diameter=0.190, density=1.0 / (4.0 * math.pi * 0.095 ** 3 / 3.0),
            restitution=0.40, friction=0.75, rolling_resistance=0.040,
            sound_family="medicine", rigidity="compliant filled rubber", color=(0.15, 0.54, 0.94),
        ),
        BodySpec(
            "medicine_3", "Medicine Ball 3 kg", "sphere", 3.0,
            diameter=0.230, density=3.0 / (4.0 * math.pi * 0.115 ** 3 / 3.0),
            restitution=0.32, friction=0.75, rolling_resistance=0.045,
            sound_family="medicine", rigidity="semicompliant filled rubber", color=(0.20, 0.75, 0.42),
        ),
        BodySpec(
            "medicine_10", "Medicine Ball 10 kg", "sphere", 10.0,
            diameter=0.280, density=10.0 / (4.0 * math.pi * 0.140 ** 3 / 3.0),
            restitution=0.22, friction=0.75, rolling_resistance=0.060,
            sound_family="medicine", rigidity="dense damped fill", color=(0.69, 0.30, 0.86),
        ),
        BodySpec(
            "steel", "Solid Smooth-Steel Ball", "sphere", basketball_volume * 7_850.0,
            diameter=basketball_diameter, density=7_850.0, restitution=0.55,
            friction=0.35, rolling_resistance=0.003, drag_coefficient=0.47,
            sound_family="steel", rigidity="rigid steel (~200 GPa)", color=(0.70, 0.76, 0.82),
        ),
        BodySpec(
            "concrete", "Grainy Concrete Ball", "sphere", basketball_volume * 2_400.0,
            diameter=basketball_diameter, density=2_400.0, restitution=0.18,
            friction=0.65, rolling_resistance=0.020, sound_family="concrete",
            rigidity="rigid brittle concrete", color=(0.48, 0.45, 0.40),
        ),
        BodySpec(
            "rubber_brick", "Solid Natural-Rubber Brick", "box",
            brick_dimensions[0] * brick_dimensions[1] * brick_dimensions[2] * 920.0,
            dimensions=brick_dimensions, density=920.0, restitution=0.30,
            friction=0.85, rolling_resistance=0.0, drag_coefficient=1.05,
            inertia_mode="box", sound_family="rubber_brick", rigidity="solid compliant rubber",
            color=(0.90, 0.48, 0.09),
        ),
    )


ITEM_SPECS = make_item_specs()
ITEM_SPEC_BY_KEY = {spec.key: spec for spec in ITEM_SPECS}
TELEMETRY_NAMES = {
    "dodge": "Dodgeball",
    "medicine_1": "Medicine 1 kg",
    "medicine_3": "Medicine 3 kg",
    "medicine_10": "Medicine 10 kg",
    "steel": "Steel ball",
    "concrete": "Concrete ball",
    "rubber_brick": "Rubber brick",
}


@dataclass
class RigidBody:
    spec: BodySpec
    position: Vec3
    velocity: Vec3 = field(default_factory=Vec3)
    orientation: Quat = field(default_factory=Quat)
    angular_velocity: Vec3 = field(default_factory=Vec3)
    force: Vec3 = field(default_factory=Vec3)
    torque: Vec3 = field(default_factory=Vec3)
    previous_position: Vec3 = field(default_factory=Vec3)
    previous_orientation: Quat = field(default_factory=Quat)
    asleep: bool = False
    sleep_time: float = 0.0
    grounded: bool = False
    held: bool = False
    impact_cooldown: float = 0.0
    last_impulse: float = 0.0

    def __post_init__(self) -> None:
        self.previous_position = self.position.copy()
        self.previous_orientation = Quat(
            self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z
        )

    @property
    def inv_mass(self) -> float:
        return 1.0 / self.spec.mass

    @property
    def bounding_radius(self) -> float:
        if self.spec.shape == "sphere":
            return self.spec.radius
        half = self.spec.half_extents
        return math.sqrt(half.x * half.x + half.y * half.y + half.z * half.z)

    def support_extent(self, axis: Vec3) -> float:
        if self.spec.shape == "sphere":
            return self.spec.radius
        half = self.spec.half_extents
        local_axis = self.orientation.conjugate().rotate(axis)
        return abs(local_axis.x) * half.x + abs(local_axis.y) * half.y + abs(local_axis.z) * half.z

    def support_point(self, direction: Vec3) -> Vec3:
        """Furthest world-space material point along direction."""
        if self.spec.shape == "sphere":
            return self.position + direction.normalized(Vec3(1.0, 0.0, 0.0)) * self.spec.radius
        half = self.spec.half_extents
        local_direction = self.orientation.conjugate().rotate(direction)

        def coordinate(component: float, extent: float) -> float:
            if component > 1.0e-10:
                return extent
            if component < -1.0e-10:
                return -extent
            # A face-centre contact avoids inventing torque when an entire
            # unrotated face lands flush on a plane.
            return 0.0

        local = Vec3(
            coordinate(local_direction.x, half.x),
            coordinate(local_direction.y, half.y),
            coordinate(local_direction.z, half.z),
        )
        return self.position + self.orientation.rotate(local)

    def inverse_inertia_world(self, vector: Vec3) -> Vec3:
        local = self.orientation.conjugate().rotate(vector)
        inertia = self.spec.inertia_diagonal
        transformed = Vec3(
            local.x / max(inertia.x, 1.0e-9),
            local.y / max(inertia.y, 1.0e-9),
            local.z / max(inertia.z, 1.0e-9),
        )
        return self.orientation.rotate(transformed)

    def velocity_at(self, world_point: Vec3) -> Vec3:
        return self.velocity + self.angular_velocity.cross(world_point - self.position)

    def apply_force(self, force: Vec3, point: Optional[Vec3] = None) -> None:
        self.force = self.force + force
        if point is not None:
            self.torque = self.torque + (point - self.position).cross(force)
        if force.length_squared() > 1.0e-12:
            self.wake()

    def apply_impulse(
        self, impulse: Vec3, point: Optional[Vec3] = None, wake: bool = True
    ) -> None:
        self.velocity = self.velocity + impulse * self.inv_mass
        if point is not None:
            angular_impulse = (point - self.position).cross(impulse)
            self.angular_velocity = self.angular_velocity + self.inverse_inertia_world(angular_impulse)
        if wake and impulse.length_squared() > 1.0e-12:
            self.wake()

    def wake(self) -> None:
        self.asleep = False
        self.sleep_time = 0.0


@dataclass
class Player:
    position: Vec3 = field(default_factory=lambda: Vec3(*PLAYER_START))
    velocity: Vec3 = field(default_factory=Vec3)
    yaw: float = 0.0
    pitch: float = -0.30
    grounded: bool = True
    move_forward: float = 0.0
    move_strafe: float = 0.0
    previous_position: Vec3 = field(default_factory=lambda: Vec3(*PLAYER_START))
    landing_speed: float = 0.0

    @property
    def eye(self) -> Vec3:
        return self.position + Vec3(0.0, PLAYER_EYE_HEIGHT, 0.0)

    def forward(self, include_pitch: bool = True) -> Vec3:
        pitch = self.pitch if include_pitch else 0.0
        cosine = math.cos(pitch)
        return Vec3(-math.sin(self.yaw) * cosine, math.sin(pitch), math.cos(self.yaw) * cosine)

    def right(self) -> Vec3:
        return Vec3(-math.cos(self.yaw), 0.0, -math.sin(self.yaw))


@dataclass(frozen=True)
class ImpactEvent:
    family: str
    position: Vec3
    impulse: float
    speed: float
    mass: float


class PhysicsWorld:
    """Deterministic seven-body Newtonian room simulation in SI units."""

    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.gravity = EARTH_GRAVITY
        self.room_friction = DEFAULT_ROOM_FRICTION
        self.throw_force = THROW_FORCE_MIN
        self.player = Player()
        self.bodies: List[RigidBody] = []
        self.held_body: Optional[RigidBody] = None
        self.impacts: List[ImpactEvent] = []
        self.messages: List[str] = []
        self.simulation_time = 0.0
        self.reset()

    def reset(self) -> None:
        self.player = Player()
        self.bodies = []
        for index, spec in enumerate(ITEM_SPECS):
            x = 7.0 + index
            if spec.shape == "sphere":
                y = spec.radius
            else:
                y = spec.dimensions[1] * 0.5
            body = RigidBody(spec, Vec3(x, y, 15.0))
            if spec.shape == "sphere":
                body.angular_velocity = Vec3(0.0, 0.0, 0.0)
            self.bodies.append(body)
        self.held_body = None
        self.impacts.clear()
        self.messages = ["Room reset: seven calibrated bodies await five metres ahead."]
        self.simulation_time = 0.0

    def wake_all(self) -> None:
        for body in self.bodies:
            body.wake()

    def set_move_input(self, forward: float, strafe: float) -> None:
        self.player.move_forward = clamp(forward, -1.0, 1.0)
        self.player.move_strafe = clamp(strafe, -1.0, 1.0)

    def adjust_gravity(self, direction: int) -> None:
        old = self.gravity
        self.gravity = clamp(round(self.gravity + direction * GRAVITY_STEP, 5), GRAVITY_MIN, GRAVITY_MAX)
        if abs(old - self.gravity) > 1.0e-9:
            self.wake_all()
            self.messages.append(f"Gravity set to {self.gravity:.5g} m/s^2.")

    def adjust_friction(self, direction: int) -> None:
        old = self.room_friction
        self.room_friction = clamp(round(self.room_friction + direction * FRICTION_STEP, 2), FRICTION_MIN, FRICTION_MAX)
        if abs(old - self.room_friction) > 1.0e-9:
            self.wake_all()
            self.messages.append(f"Room friction coefficient set to {self.room_friction:.2f}.")

    def adjust_throw_force(self, direction: int) -> None:
        nearest = min(range(len(THROW_FORCE_STEPS)), key=lambda i: abs(THROW_FORCE_STEPS[i] - self.throw_force))
        index = int(clamp(nearest + direction, 0, len(THROW_FORCE_STEPS) - 1))
        self.throw_force = THROW_FORCE_STEPS[index]
        self.messages.append(f"Throw force set to {self.throw_force:,.0f} N.")

    def jump(self) -> bool:
        if not self.player.grounded:
            return False
        self.player.velocity.y = math.sqrt(2.0 * self.gravity * JUMP_HEIGHT)
        self.player.grounded = False
        self.messages.append(f"Jump impulse set for a {JUMP_HEIGHT:.1f} m apex at current gravity.")
        return True

    def raycast_body(self, origin: Vec3, direction: Vec3, reach: float = PICKUP_REACH) -> Optional[RigidBody]:
        direction = direction.normalized(Vec3(0.0, 0.0, 1.0))
        best_body: Optional[RigidBody] = None
        best_t = reach + 1.0
        for body in self.bodies:
            if body.held:
                continue
            radius = body.bounding_radius * (1.0 if body.spec.shape == "sphere" else 1.15)
            offset = origin - body.position
            b = offset.dot(direction)
            c = offset.dot(offset) - radius * radius
            discriminant = b * b - c
            if discriminant < 0.0:
                continue
            root = math.sqrt(discriminant)
            t = -b - root
            if t < 0.0:
                t = -b + root
            if 0.0 <= t <= reach and t < best_t:
                best_t = t
                best_body = body
        return best_body

    def pickup_or_drop(self) -> Optional[str]:
        if self.held_body is not None:
            name = self.held_body.spec.name
            self.held_body.held = False
            self.held_body.wake()
            self.held_body = None
            self.messages.append(f"Released {name} without a throw.")
            return "drop"
        body = self.raycast_body(self.player.eye, self.player.forward())
        if body is None:
            self.messages.append("No item is within the crosshair's 6 m pickup reach.")
            return None
        body.held = True
        body.wake()
        self.held_body = body
        self.messages.append(
            f"Holding {body.spec.name}: {body.spec.mass:.3f} kg, {body.spec.dimension_label}."
        )
        return "pickup"

    def throw_held(self) -> Optional[Tuple[RigidBody, float, float]]:
        body = self.held_body
        if body is None:
            self.messages.append("Pick up an item with right mouse before throwing.")
            return None
        direction = self.player.forward().normalized(Vec3(0.0, 0.0, 1.0))
        # The arm stroke is relative separation between player and item. With
        # equal recoil, reduced mass ensures the complete two-body kinetic-
        # energy gain is exactly work W=F*s rather than double-counting recoil.
        reduced_mass = body.spec.mass * PLAYER_MASS / (body.spec.mass + PLAYER_MASS)
        impulse_magnitude = math.sqrt(2.0 * self.throw_force * THROW_STROKE * reduced_mass)
        launch_speed = impulse_magnitude / body.spec.mass
        relative_release_speed = impulse_magnitude / reduced_mass
        impulse = direction * impulse_magnitude
        body.held = False
        body.velocity = self.player.velocity.copy()
        body.apply_impulse(impulse, body.position)
        # Newton's third law: the player receives the exact opposite impulse.
        self.player.velocity = self.player.velocity - impulse / PLAYER_MASS
        self.held_body = None
        self.messages.append(
            f"Threw {body.spec.name}: {self.throw_force:,.0f} N across {THROW_STROKE:.2f} m, "
            f"item delta-v {launch_speed:,.2f} m/s, relative release {relative_release_speed:,.2f} m/s, "
            f"impulse {impulse_magnitude:,.2f} N*s."
        )
        return body, launch_speed, impulse_magnitude

    def _effective_room_friction(self, body: RigidBody) -> float:
        if DEFAULT_ROOM_FRICTION <= 0.0:
            return body.spec.friction
        return body.spec.friction * self.room_friction / DEFAULT_ROOM_FRICTION

    def _body_drag(self, body: RigidBody) -> Vec3:
        speed = body.velocity.length()
        if speed < 1.0e-5:
            return Vec3()
        magnitude = 0.5 * AIR_DENSITY * body.spec.drag_coefficient * body.spec.frontal_area * speed * speed
        return body.velocity * (-magnitude / speed)

    def _integrate_player(self, dt: float) -> None:
        player = self.player
        player.previous_position = player.position.copy()
        was_grounded = player.grounded
        desired = player.forward(False) * player.move_forward + player.right() * player.move_strafe
        if desired.length_squared() > 1.0:
            desired = desired.normalized()
        desired = desired * PLAYER_WALK_SPEED
        acceleration = PLAYER_GROUND_ACCEL if player.grounded else PLAYER_AIR_ACCEL
        if desired.length_squared() > 1.0e-8:
            player.velocity.x = approach(player.velocity.x, desired.x, acceleration * dt)
            player.velocity.z = approach(player.velocity.z, desired.z, acceleration * dt)
        elif player.grounded:
            speed = player.velocity.horizontal().length()
            if speed > 1.0e-8:
                loss = self.room_friction * self.gravity * dt
                next_speed = max(0.0, speed - loss)
                scale = next_speed / speed
                player.velocity.x *= scale
                player.velocity.z *= scale

        # Exact constant-acceleration displacement keeps the requested jump
        # apex at one metre throughout the complete gravity control range.
        vertical_velocity_before = player.velocity.y
        player.position.x += player.velocity.x * dt
        player.position.z += player.velocity.z * dt
        player.position.y += vertical_velocity_before * dt - 0.5 * self.gravity * dt * dt
        player.velocity.y = vertical_velocity_before - self.gravity * dt
        if player.position.x < PLAYER_RADIUS:
            player.position.x = PLAYER_RADIUS
            if player.velocity.x < 0.0:
                player.velocity.x = 0.0
        elif player.position.x > ROOM_WIDTH - PLAYER_RADIUS:
            player.position.x = ROOM_WIDTH - PLAYER_RADIUS
            if player.velocity.x > 0.0:
                player.velocity.x = 0.0
        if player.position.z < PLAYER_RADIUS:
            player.position.z = PLAYER_RADIUS
            if player.velocity.z < 0.0:
                player.velocity.z = 0.0
        elif player.position.z > ROOM_LENGTH - PLAYER_RADIUS:
            player.position.z = ROOM_LENGTH - PLAYER_RADIUS
            if player.velocity.z > 0.0:
                player.velocity.z = 0.0
        maximum_feet_y = ROOM_HEIGHT - PLAYER_HEIGHT
        if player.position.y <= 0.0:
            if not was_grounded and player.velocity.y < 0.0:
                player.landing_speed = -player.velocity.y
            player.position.y = 0.0
            player.velocity.y = 0.0
            player.grounded = True
        else:
            player.grounded = False
        if player.position.y > maximum_feet_y:
            player.position.y = maximum_feet_y
            if player.velocity.y > 0.0:
                player.velocity.y = -player.velocity.y * 0.05

    def _hold_constraint(self, body: RigidBody) -> None:
        target = self.player.eye + self.player.forward() * HOLD_DISTANCE
        radius = body.bounding_radius
        target.y = clamp(target.y, radius + 0.02, ROOM_HEIGHT - radius - 0.02)
        target.x = clamp(target.x, radius + 0.02, ROOM_WIDTH - radius - 0.02)
        target.z = clamp(target.z, radius + 0.02, ROOM_LENGTH - radius - 0.02)
        error = target - body.position
        target_velocity = self.player.velocity
        relative_velocity = body.velocity - target_velocity
        omega = 18.0
        desired_acceleration = error * (omega * omega) - relative_velocity * (2.0 * omega)
        grip_force = desired_acceleration * body.spec.mass
        maximum_grip = max(2_500.0, body.spec.mass * 250.0)
        magnitude = grip_force.length()
        if magnitude > maximum_grip:
            grip_force = grip_force * (maximum_grip / magnitude)
        body.apply_force(grip_force)
        # The hand constraint reacts on the player, preserving system momentum.
        self.player.velocity = self.player.velocity - grip_force * (FIXED_DT / PLAYER_MASS)

    def _integrate_body_forces(self, body: RigidBody, dt: float) -> None:
        if body.asleep and not body.held:
            body.force = Vec3()
            body.torque = Vec3()
            return
        body.previous_position = body.position.copy()
        body.previous_orientation = Quat(
            body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z
        )
        gravity_force = Vec3(0.0, -body.spec.mass * self.gravity, 0.0)
        body.force = body.force + gravity_force + self._body_drag(body)
        body.velocity = body.velocity + body.force * (body.inv_mass * dt)
        body.angular_velocity = body.angular_velocity + body.inverse_inertia_world(body.torque) * dt
        speed = body.velocity.length()
        if speed > MAX_LINEAR_SPEED:
            body.velocity = body.velocity * (MAX_LINEAR_SPEED / speed)
        angular_speed = body.angular_velocity.length()
        if angular_speed > MAX_ANGULAR_SPEED:
            body.angular_velocity = body.angular_velocity * (MAX_ANGULAR_SPEED / angular_speed)
        body.orientation = body.orientation.integrate(body.angular_velocity, dt)
        body.force = Vec3()
        body.torque = Vec3()

    def _room_impact(self, body: RigidBody, normal: Vec3, contact: Vec3) -> float:
        relative = body.velocity_at(contact)
        normal_speed = relative.dot(normal)
        if normal_speed >= 0.0:
            return 0.0
        restitution = body.spec.restitution if -normal_speed >= 0.6 else 0.0
        lever = contact - body.position
        lever_cross_normal = lever.cross(normal)
        angular_term = body.inverse_inertia_world(lever_cross_normal).cross(lever).dot(normal)
        denominator = body.inv_mass + max(0.0, angular_term)
        normal_impulse_magnitude = -(1.0 + restitution) * normal_speed / max(denominator, 1.0e-9)
        normal_impulse = normal * normal_impulse_magnitude
        meaningful_impact = -normal_speed > 0.20
        body.apply_impulse(normal_impulse, contact, wake=meaningful_impact)

        tangent_velocity = relative - normal * normal_speed
        tangent_speed = tangent_velocity.length()
        if tangent_speed > 1.0e-8:
            tangent = tangent_velocity / tangent_speed
            lever_cross_tangent = lever.cross(tangent)
            tangent_angular = body.inverse_inertia_world(lever_cross_tangent).cross(lever).dot(tangent)
            tangent_denominator = body.inv_mass + max(0.0, tangent_angular)
            desired = tangent_speed / max(tangent_denominator, 1.0e-9)
            friction_limit = self._effective_room_friction(body) * normal_impulse_magnitude
            friction_impulse = tangent * (-min(desired, friction_limit))
            body.apply_impulse(friction_impulse, contact, wake=meaningful_impact)

        body.last_impulse = max(body.last_impulse, normal_impulse_magnitude)
        if body.impact_cooldown <= 0.0 and -normal_speed > 0.55:
            self.impacts.append(
                ImpactEvent(body.spec.sound_family, body.position.copy(), normal_impulse_magnitude, -normal_speed, body.spec.mass)
            )
            body.impact_cooldown = 0.075
        return normal_impulse_magnitude

    def _project_body_inside_room(self, body: RigidBody) -> None:
        """Recover pair-solver penetration and resolve all inward wall velocity."""
        axes = (
            ("x", Vec3(1.0, 0.0, 0.0), 0.0, ROOM_WIDTH),
            ("y", Vec3(0.0, 1.0, 0.0), 0.0, ROOM_HEIGHT),
            ("z", Vec3(0.0, 0.0, 1.0), 0.0, ROOM_LENGTH),
        )
        for attribute, positive, room_minimum, room_maximum in axes:
            extent = body.support_extent(positive)
            coordinate = getattr(body.position, attribute)
            minimum = room_minimum + extent
            maximum = room_maximum - extent
            if coordinate < minimum:
                setattr(body.position, attribute, minimum)
                contact = body.support_point(-positive)
                self._room_impact(body, positive, contact)
            elif coordinate > maximum:
                setattr(body.position, attribute, maximum)
                contact = body.support_point(positive)
                self._room_impact(body, -positive, contact)
            else:
                # A body exactly on a plane may still carry velocity into it
                # after a pair/player impulse. Resolve that t=0 contact now.
                epsilon = 2.0e-6
                if coordinate <= minimum + epsilon and body.velocity_at(body.support_point(-positive)).dot(positive) < 0.0:
                    self._room_impact(body, positive, body.support_point(-positive))
                elif coordinate >= maximum - epsilon and body.velocity_at(body.support_point(positive)).dot(-positive) < 0.0:
                    self._room_impact(body, -positive, body.support_point(positive))
        vertical_extent = body.support_extent(Vec3(0.0, 1.0, 0.0))
        if body.position.y <= vertical_extent + 2.0e-5:
            body.grounded = True

    def _advance_body_swept(self, body: RigidBody, dt: float) -> None:
        if body.asleep:
            return
        remaining = dt
        body.grounded = False
        self._project_body_inside_room(body)
        for _bounce in range(MAX_CCD_BOUNCES):
            if remaining <= 1.0e-8:
                break
            velocity = body.velocity
            earliest = remaining + 1.0
            hit_normal: Optional[Vec3] = None
            hit_axis = ""
            extents = {
                "xmin": body.support_extent(Vec3(1.0, 0.0, 0.0)),
                "xmax": body.support_extent(Vec3(1.0, 0.0, 0.0)),
                "ymin": body.support_extent(Vec3(0.0, 1.0, 0.0)),
                "ymax": body.support_extent(Vec3(0.0, 1.0, 0.0)),
                "zmin": body.support_extent(Vec3(0.0, 0.0, 1.0)),
                "zmax": body.support_extent(Vec3(0.0, 0.0, 1.0)),
            }
            candidates = (
                (velocity.x, extents["xmin"], body.position.x, Vec3(1.0, 0.0, 0.0), "xmin"),
                (velocity.x, ROOM_WIDTH - extents["xmax"], body.position.x, Vec3(-1.0, 0.0, 0.0), "xmax"),
                (velocity.y, extents["ymin"], body.position.y, Vec3(0.0, 1.0, 0.0), "floor"),
                (velocity.y, ROOM_HEIGHT - extents["ymax"], body.position.y, Vec3(0.0, -1.0, 0.0), "ceiling"),
                (velocity.z, extents["zmin"], body.position.z, Vec3(0.0, 0.0, 1.0), "zmin"),
                (velocity.z, ROOM_LENGTH - extents["zmax"], body.position.z, Vec3(0.0, 0.0, -1.0), "zmax"),
            )
            for component, boundary, coordinate, normal, axis_name in candidates:
                moving_toward = component < -1.0e-12 if normal.dot(Vec3(1.0, 1.0, 1.0)) > 0 else component > 1.0e-12
                if not moving_toward:
                    continue
                collision_time = (boundary - coordinate) / component
                if -1.0e-9 <= collision_time <= remaining and collision_time < earliest:
                    earliest = max(0.0, collision_time)
                    hit_normal = normal
                    hit_axis = axis_name
            if hit_normal is None:
                body.position = body.position + body.velocity * remaining
                remaining = 0.0
                break
            body.position = body.position + body.velocity * earliest
            remaining -= earliest
            contact = body.support_point(-hit_normal)
            self._room_impact(body, hit_normal, contact)
            if hit_axis == "floor":
                body.grounded = True
            body.position = body.position + hit_normal * 1.0e-6
            remaining = max(0.0, remaining - 1.0e-8)

        self._project_body_inside_room(body)

    @staticmethod
    def _sphere_sweep_fraction(a: RigidBody, b: RigidBody) -> Optional[float]:
        start = b.previous_position - a.previous_position
        motion = (b.position - b.previous_position) - (a.position - a.previous_position)
        radius = a.bounding_radius + b.bounding_radius
        c = start.dot(start) - radius * radius
        if c <= 0.0:
            return 0.0
        aa = motion.dot(motion)
        if aa <= 1.0e-14:
            return None
        bb = 2.0 * start.dot(motion)
        discriminant = bb * bb - 4.0 * aa * c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        fraction = (-bb - root) / (2.0 * aa)
        if 0.0 <= fraction <= 1.0:
            return fraction
        return None

    def _body_pair_impulse(self, a: RigidBody, b: RigidBody, normal: Vec3, contact: Vec3) -> float:
        ra = contact - a.position
        rb = contact - b.position
        relative = b.velocity_at(contact) - a.velocity_at(contact)
        normal_speed = relative.dot(normal)
        if normal_speed >= 0.0:
            return 0.0
        if normal.y > 0.55:
            b.grounded = True
        elif normal.y < -0.55:
            a.grounded = True
        restitution = math.sqrt(a.spec.restitution * b.spec.restitution) if -normal_speed >= 0.6 else 0.0
        ra_cross_n = ra.cross(normal)
        rb_cross_n = rb.cross(normal)
        angular_a = a.inverse_inertia_world(ra_cross_n).cross(ra).dot(normal)
        angular_b = b.inverse_inertia_world(rb_cross_n).cross(rb).dot(normal)
        denominator = a.inv_mass + b.inv_mass + max(0.0, angular_a) + max(0.0, angular_b)
        magnitude = -(1.0 + restitution) * normal_speed / max(denominator, 1.0e-9)
        impulse = normal * magnitude
        meaningful_impact = -normal_speed > 0.20
        a.apply_impulse(-impulse, contact, wake=meaningful_impact)
        b.apply_impulse(impulse, contact, wake=meaningful_impact)

        post_relative = b.velocity_at(contact) - a.velocity_at(contact)
        tangent_velocity = post_relative - normal * post_relative.dot(normal)
        tangent_speed = tangent_velocity.length()
        if tangent_speed > 1.0e-8:
            tangent = tangent_velocity / tangent_speed
            ra_cross_t = ra.cross(tangent)
            rb_cross_t = rb.cross(tangent)
            tangent_denominator = (
                a.inv_mass + b.inv_mass
                + max(0.0, a.inverse_inertia_world(ra_cross_t).cross(ra).dot(tangent))
                + max(0.0, b.inverse_inertia_world(rb_cross_t).cross(rb).dot(tangent))
            )
            desired = tangent_speed / max(tangent_denominator, 1.0e-9)
            mu = math.sqrt(a.spec.friction * b.spec.friction)
            friction_impulse = tangent * (-min(desired, mu * magnitude))
            a.apply_impulse(-friction_impulse, contact, wake=meaningful_impact)
            b.apply_impulse(friction_impulse, contact, wake=meaningful_impact)

        louder = a if a.spec.mass <= b.spec.mass else b
        if louder.impact_cooldown <= 0.0 and -normal_speed > 0.55:
            self.impacts.append(
                ImpactEvent(louder.spec.sound_family, contact.copy(), magnitude, -normal_speed, louder.spec.mass)
            )
            louder.impact_cooldown = 0.075
        return magnitude

    @staticmethod
    def _sphere_box_contact(
        sphere: RigidBody, box: RigidBody
    ) -> Optional[Tuple[Vec3, float, Vec3]]:
        """Return sphere-to-box normal, penetration, and world contact for an OBB."""
        inverse_orientation = box.orientation.conjugate()
        local_center = inverse_orientation.rotate(sphere.position - box.position)
        half = box.spec.half_extents
        closest = Vec3(
            clamp(local_center.x, -half.x, half.x),
            clamp(local_center.y, -half.y, half.y),
            clamp(local_center.z, -half.z, half.z),
        )
        toward_box = closest - local_center
        distance_squared = toward_box.length_squared()
        radius = sphere.spec.radius
        if distance_squared > radius * radius:
            return None
        if distance_squared > 1.0e-14:
            distance = math.sqrt(distance_squared)
            normal_local = toward_box / distance
            penetration = radius - distance
            contact_local = closest
        else:
            # The centre is inside/on the cuboid. Pick the shortest exit face;
            # normal points sphere -> box so -normal expels the sphere.
            face_distances = (
                (half.x - abs(local_center.x), Vec3(1.0 if local_center.x >= 0.0 else -1.0, 0.0, 0.0)),
                (half.y - abs(local_center.y), Vec3(0.0, 1.0 if local_center.y >= 0.0 else -1.0, 0.0)),
                (half.z - abs(local_center.z), Vec3(0.0, 0.0, 1.0 if local_center.z >= 0.0 else -1.0)),
            )
            face_distance, outward = min(face_distances, key=lambda pair: pair[0])
            normal_local = -outward
            penetration = radius + face_distance
            contact_local = local_center + outward * face_distance
        normal_world = box.orientation.rotate(normal_local).normalized(Vec3(1.0, 0.0, 0.0))
        contact_world = box.position + box.orientation.rotate(contact_local)
        return normal_world, max(0.0, penetration), contact_world

    @staticmethod
    def _swept_sphere_box_fraction(sphere: RigidBody, box: RigidBody) -> Optional[float]:
        """Conservative advancement against the rounded radius-expanded OBB."""
        inverse_orientation = box.orientation.conjugate()
        start = inverse_orientation.rotate(sphere.previous_position - box.previous_position)
        end = inverse_orientation.rotate(sphere.position - box.position)
        motion = end - start
        half = box.spec.half_extents
        radius = sphere.spec.radius
        lower = Vec3(-half.x - radius, -half.y - radius, -half.z - radius)
        upper = Vec3(half.x + radius, half.y + radius, half.z + radius)
        entry, exit_fraction = 0.0, 1.0
        for start_component, motion_component, low, high in (
            (start.x, motion.x, lower.x, upper.x),
            (start.y, motion.y, lower.y, upper.y),
            (start.z, motion.z, lower.z, upper.z),
        ):
            if abs(motion_component) <= 1.0e-14:
                if start_component < low or start_component > high:
                    return None
                continue
            first = (low - start_component) / motion_component
            second = (high - start_component) / motion_component
            if first > second:
                first, second = second, first
            entry = max(entry, first)
            exit_fraction = min(exit_fraction, second)
            if entry > exit_fraction:
                return None
        if not (0.0 <= entry <= 1.0):
            return None
        # A plain expanded AABB fabricates contacts around its square corners.
        # Advance by the exact point-to-box separation; distance is Lipschitz,
        # so this cannot step over a real rounded-edge/corner contact.
        motion_length = motion.length()
        if motion_length <= 1.0e-14:
            return None
        fraction = entry
        for _iteration in range(24):
            point = start + motion * fraction
            closest = Vec3(
                clamp(point.x, -half.x, half.x),
                clamp(point.y, -half.y, half.y),
                clamp(point.z, -half.z, half.z),
            )
            separation = (point - closest).length() - radius
            if separation <= 1.0e-7:
                return fraction
            fraction += max(1.0e-7, separation / motion_length)
            if fraction > exit_fraction + 1.0e-9 or fraction > 1.0:
                return None
        return None

    def _resolve_sphere_box(self, sphere: RigidBody, box: RigidBody, dt: float) -> None:
        contact_data = self._sphere_box_contact(sphere, box)
        swept_fraction: Optional[float] = None
        if contact_data is None:
            if dt <= 0.0:
                return
            swept_fraction = self._swept_sphere_box_fraction(sphere, box)
            if swept_fraction is None:
                return
            sphere.position = sphere.previous_position + (sphere.position - sphere.previous_position) * swept_fraction
            box.position = box.previous_position + (box.position - box.previous_position) * swept_fraction
            contact_data = self._sphere_box_contact(sphere, box)
            if contact_data is None:
                # Expanded-box corners are conservative; use the centre direction
                # at the time of impact when the rounded-corner contact is outside.
                normal = (box.position - sphere.position).normalized(Vec3(1.0, 0.0, 0.0))
                contact = sphere.position + normal * sphere.spec.radius
                contact_data = (normal, 0.0, contact)
        normal, penetration, contact = contact_data
        if penetration > 0.0:
            correction = normal * (max(0.0, penetration - 1.0e-5) * 0.66 / (sphere.inv_mass + box.inv_mass))
            sphere.position = sphere.position - correction * sphere.inv_mass
            box.position = box.position + correction * box.inv_mass
        self._body_pair_impulse(sphere, box, normal, contact)
        if swept_fraction is not None and swept_fraction < 1.0:
            remaining = dt * (1.0 - swept_fraction)
            self._advance_body_swept(sphere, remaining)
            self._advance_body_swept(box, remaining)

    def _resolve_body_pair(self, a: RigidBody, b: RigidBody, dt: float) -> None:
        if a.held and b.held:
            return
        if a.spec.shape == "sphere" and b.spec.shape == "box":
            self._resolve_sphere_box(a, b, dt)
            return
        if a.spec.shape == "box" and b.spec.shape == "sphere":
            self._resolve_sphere_box(b, a, dt)
            return
        radius_a = a.bounding_radius
        radius_b = b.bounding_radius
        delta = b.position - a.position
        distance_squared = delta.length_squared()
        radius_sum = radius_a + radius_b
        swept_fraction: Optional[float] = None
        if distance_squared > radius_sum * radius_sum:
            if dt <= 0.0:
                return
            swept_fraction = self._sphere_sweep_fraction(a, b)
            if swept_fraction is None:
                return
            a.position = a.previous_position + (a.position - a.previous_position) * swept_fraction
            b.position = b.previous_position + (b.position - b.previous_position) * swept_fraction
            delta = b.position - a.position
            distance_squared = delta.length_squared()

        distance = math.sqrt(max(distance_squared, 1.0e-16))
        normal = delta / distance if distance > 1.0e-8 else Vec3(1.0, 0.0, 0.0)
        penetration = radius_sum - distance
        if penetration > 0.0:
            correction = normal * (max(0.0, penetration - 1.0e-5) * 0.62 / (a.inv_mass + b.inv_mass))
            a.position = a.position - correction * a.inv_mass
            b.position = b.position + correction * b.inv_mass
        contact = a.position + normal * radius_a
        self._body_pair_impulse(a, b, normal, contact)

        if swept_fraction is not None and swept_fraction < 1.0:
            remaining = dt * (1.0 - swept_fraction)
            self._advance_body_swept(a, remaining)
            self._advance_body_swept(b, remaining)

    def _player_body_contact(self, body: RigidBody) -> Optional[Tuple[Vec3, float, Vec3]]:
        """Return player-to-body normal, penetration, contact; capsule vs sphere/OBB."""
        player = self.player
        capsule_low = player.position.y + PLAYER_RADIUS
        capsule_high = player.position.y + PLAYER_HEIGHT - PLAYER_RADIUS
        if body.spec.shape == "sphere":
            center_y = clamp(body.position.y, capsule_low, capsule_high)
            closest = Vec3(player.position.x, center_y, player.position.z)
            delta = body.position - closest
            combined = PLAYER_RADIUS + body.spec.radius
            distance_squared = delta.length_squared()
            if distance_squared >= combined * combined:
                return None
            distance = math.sqrt(max(distance_squared, 1.0e-16))
            normal = delta / distance if distance > 1.0e-8 else Vec3(1.0, 0.0, 0.0)
            contact = body.position - normal * body.spec.radius
            return normal, combined - distance, contact

        inverse_orientation = body.orientation.conjugate()
        half = body.spec.half_extents
        best_distance_squared = float("inf")
        best_delta = Vec3()
        best_contact = body.position.copy()
        # The cuboid is small; nine capsule-axis samples are cheaper and more
        # faithful than treating it as a large bounding sphere.
        for index in range(9):
            fraction = index / 8.0
            axis_point = Vec3(
                player.position.x,
                capsule_low + (capsule_high - capsule_low) * fraction,
                player.position.z,
            )
            local_point = inverse_orientation.rotate(axis_point - body.position)
            closest_local = Vec3(
                clamp(local_point.x, -half.x, half.x),
                clamp(local_point.y, -half.y, half.y),
                clamp(local_point.z, -half.z, half.z),
            )
            closest_world = body.position + body.orientation.rotate(closest_local)
            delta = closest_world - axis_point
            distance_squared = delta.length_squared()
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_delta = delta
                best_contact = closest_world
        if best_distance_squared >= PLAYER_RADIUS * PLAYER_RADIUS:
            return None
        distance = math.sqrt(max(best_distance_squared, 1.0e-16))
        if distance > 1.0e-8:
            normal = best_delta / distance
        else:
            normal = (body.position - player.position).horizontal().normalized(Vec3(1.0, 0.0, 0.0))
        return normal, PLAYER_RADIUS - distance, best_contact

    @staticmethod
    def _swept_body_player_fraction(body: RigidBody, player: Player) -> Optional[Tuple[float, Vec3]]:
        """Continuous bounding-sphere vs vertical player capsule time of impact."""
        start = body.previous_position - player.previous_position
        end = body.position - player.position
        motion = end - start
        combined = PLAYER_RADIUS + body.bounding_radius
        low, high = PLAYER_RADIUS, PLAYER_HEIGHT - PLAYER_RADIUS
        candidates: List[float] = []

        # Infinite vertical cylinder, filtered to its finite axial span.
        aa = motion.x * motion.x + motion.z * motion.z
        bb = 2.0 * (start.x * motion.x + start.z * motion.z)
        cc = start.x * start.x + start.z * start.z - combined * combined
        if aa > 1.0e-14:
            discriminant = bb * bb - 4.0 * aa * cc
            if discriminant >= 0.0:
                root = math.sqrt(discriminant)
                for fraction in ((-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)):
                    axial = start.y + motion.y * fraction
                    if 0.0 <= fraction <= 1.0 and low <= axial <= high:
                        candidates.append(fraction)

        # Hemispherical capsule ends.
        for cap_y in (low, high):
            offset = start - Vec3(0.0, cap_y, 0.0)
            aa = motion.dot(motion)
            bb = 2.0 * offset.dot(motion)
            cc = offset.dot(offset) - combined * combined
            if cc <= 0.0:
                candidates.append(0.0)
            elif aa > 1.0e-14:
                discriminant = bb * bb - 4.0 * aa * cc
                if discriminant >= 0.0:
                    fraction = (-bb - math.sqrt(discriminant)) / (2.0 * aa)
                    if 0.0 <= fraction <= 1.0:
                        candidates.append(fraction)
        if not candidates:
            return None
        fraction = min(candidates)
        relative = start + motion * fraction
        closest = Vec3(0.0, clamp(relative.y, low, high), 0.0)
        normal = (relative - closest).normalized(Vec3(1.0, 0.0, 0.0))
        return fraction, normal

    def _player_body_impulse(self, body: RigidBody, normal: Vec3, contact: Vec3) -> float:
        relative = body.velocity_at(contact) - self.player.velocity
        normal_speed = relative.dot(normal)
        if normal_speed >= 0.0:
            return 0.0
        lever = contact - body.position
        lever_cross_normal = lever.cross(normal)
        angular = body.inverse_inertia_world(lever_cross_normal).cross(lever).dot(normal)
        denominator = 1.0 / PLAYER_MASS + body.inv_mass + max(0.0, angular)
        restitution = min(0.18, body.spec.restitution) if -normal_speed >= 0.6 else 0.0
        magnitude = -(1.0 + restitution) * normal_speed / max(denominator, 1.0e-9)
        impulse = normal * magnitude
        body.apply_impulse(impulse, contact, wake=-normal_speed > 0.20)
        self.player.velocity = self.player.velocity - impulse / PLAYER_MASS
        return magnitude

    def _project_player_inside_room(self) -> None:
        player = self.player
        if player.position.x < PLAYER_RADIUS:
            player.position.x = PLAYER_RADIUS
            player.velocity.x = max(0.0, player.velocity.x)
        elif player.position.x > ROOM_WIDTH - PLAYER_RADIUS:
            player.position.x = ROOM_WIDTH - PLAYER_RADIUS
            player.velocity.x = min(0.0, player.velocity.x)
        if player.position.z < PLAYER_RADIUS:
            player.position.z = PLAYER_RADIUS
            player.velocity.z = max(0.0, player.velocity.z)
        elif player.position.z > ROOM_LENGTH - PLAYER_RADIUS:
            player.position.z = ROOM_LENGTH - PLAYER_RADIUS
            player.velocity.z = min(0.0, player.velocity.z)
        if player.position.y < 0.0:
            player.position.y = 0.0
            player.velocity.y = max(0.0, player.velocity.y)
        elif player.position.y > ROOM_HEIGHT - PLAYER_HEIGHT:
            player.position.y = ROOM_HEIGHT - PLAYER_HEIGHT
            player.velocity.y = min(0.0, player.velocity.y)

    def _resolve_player_body(self, body: RigidBody, dt: float) -> None:
        if body.held:
            return
        contact_data = self._player_body_contact(body)
        if contact_data is not None:
            normal, penetration, contact = contact_data
            total_inverse_mass = 1.0 / PLAYER_MASS + body.inv_mass
            correction = normal * (max(0.0, penetration - 1.0e-5) * 0.72 / total_inverse_mass)
            self.player.position = self.player.position - correction * (1.0 / PLAYER_MASS)
            body.position = body.position + correction * body.inv_mass
            self._player_body_impulse(body, normal, contact)
            return
        if dt <= 0.0:
            return
        swept = self._swept_body_player_fraction(body, self.player)
        if swept is None:
            return
        fraction, normal = swept
        body.position = body.previous_position + (body.position - body.previous_position) * fraction
        contact = body.position - normal * body.bounding_radius
        old_player_velocity = self.player.velocity.copy()
        impulse = self._player_body_impulse(body, normal, contact)
        if impulse <= 0.0:
            return
        remaining = dt * (1.0 - fraction)
        player_delta_velocity = self.player.velocity - old_player_velocity
        self.player.position = self.player.position + player_delta_velocity * remaining
        self._advance_body_swept(body, remaining)
        self._project_player_inside_room()

    def _apply_ground_resistance_and_sleep(self, body: RigidBody, dt: float) -> None:
        body.impact_cooldown = max(0.0, body.impact_cooldown - dt)
        if body.grounded and not body.held:
            horizontal = body.velocity.horizontal()
            speed = horizontal.length()
            scale = self.room_friction / DEFAULT_ROOM_FRICTION if DEFAULT_ROOM_FRICTION else 1.0
            if speed > 1.0e-8:
                loss = body.spec.rolling_resistance * scale * self.gravity * dt
                next_speed = max(0.0, speed - loss)
                velocity_scale = next_speed / speed
                body.velocity.x *= velocity_scale
                body.velocity.z *= velocity_scale
            angular_decay = max(0.0, 1.0 - body.spec.rolling_resistance * 2.0 * scale * dt)
            body.angular_velocity = body.angular_velocity * angular_decay

        quiet = body.velocity.length() < 0.025 and body.angular_velocity.length() < 0.05 and body.grounded
        if quiet and not body.held:
            body.sleep_time += dt
            if body.sleep_time >= 0.55:
                body.velocity = Vec3()
                body.angular_velocity = Vec3()
                body.asleep = True
        else:
            body.sleep_time = 0.0
            body.asleep = False

    def step(self, dt: float = FIXED_DT) -> None:
        self.impacts.clear()
        self._integrate_player(dt)
        if self.held_body is not None:
            self._hold_constraint(self.held_body)
        for body in self.bodies:
            self._integrate_body_forces(body, dt)
            self._advance_body_swept(body, dt)

        # A few inexpensive sequential-impulse passes stabilize the seven bodies.
        for _iteration in range(SOLVER_ITERATIONS):
            for index, first in enumerate(self.bodies):
                for second in self.bodies[index + 1:]:
                    self._resolve_body_pair(first, second, dt if _iteration == 0 else -1.0)
            for body in self.bodies:
                self._resolve_player_body(body, dt if _iteration == 0 else -1.0)
            for body in self.bodies:
                self._project_body_inside_room(body)
            self._project_player_inside_room()

        for body in self.bodies:
            self._apply_ground_resistance_and_sleep(body, dt)
            if not (body.position.finite() and body.velocity.finite() and body.angular_velocity.finite()):
                raise FloatingPointError(f"non-finite rigid-body state for {body.spec.name}")
        self.simulation_time += dt

    def selected_body(self) -> Optional[RigidBody]:
        if self.held_body is not None:
            return self.held_body
        return self.raycast_body(self.player.eye, self.player.forward())

    def status_lines(self) -> List[str]:
        return [
            f"Gravity       {self.gravity:8.5g} m/s^2   [{GRAVITY_MIN:.2f} - {GRAVITY_MAX:.2f}]  R/F",
            f"Room friction mu {self.room_friction:5.2f}       [{FRICTION_MIN:.2f} - {FRICTION_MAX:.2f}]  Y/H",
            f"Throw force {self.throw_force:10,.0f} N       [{THROW_FORCE_MIN:,.0f} - {THROW_FORCE_MAX:,.0f}]  T/G",
        ]

    def velocity_telemetry_rows(self) -> List[str]:
        rows: List[str] = []
        for index, body in enumerate(self.bodies, 1):
            marker = "[HELD]" if body.held else "[SLEEP]" if body.asleep else ""
            speed = body.velocity.length()
            rows.append(
                f"{index}. {TELEMETRY_NAMES[body.spec.key]:14} "
                f"v=({body.velocity.x:+7.2f},{body.velocity.y:+7.2f},{body.velocity.z:+7.2f}) "
                f"|v|={speed:7.2f} m/s {marker}"
            )
        return rows


def load_dependencies() -> None:
    global pygame, GL, GLU
    if pygame is not None:
        return
    try:
        import pygame as pygame_module
        from OpenGL import GL as gl_module
        from OpenGL import GLU as glu_module
    except ImportError as error:
        raise SystemExit(
            "Newton's Echo Chamber requires pygame and PyOpenGL. "
            "Install with: pip install pygame PyOpenGL PyOpenGL_accelerate"
        ) from error
    pygame = pygame_module
    GL = gl_module
    GLU = glu_module


# ---------------------------------------------------------------------------
# Original procedural sound design
# ---------------------------------------------------------------------------


def _stereo_pcm(duration: float, sample_function: Any) -> array:
    sample_count = max(1, int(AUDIO_RATE * duration))
    output = array("h")
    for index in range(sample_count):
        t = index / AUDIO_RATE
        value = clamp(float(sample_function(t, index)), -1.0, 1.0)
        shaped = int(value * 27_500)
        output.append(shaped)
        output.append(shaped)
    return output


def build_pcm_bank() -> Dict[str, array]:
    """Return deterministic, stereo, signed-16-bit procedural effects."""
    noises: Dict[str, random.Random] = {
        key: random.Random(0xEC40 + index * 7919)
        for index, key in enumerate(
            ("dodge", "medicine", "steel", "concrete", "rubber_brick", "throw", "jump", "land")
        )
    }

    def dodge(t: float, _i: int) -> float:
        envelope = math.exp(-8.2 * t)
        phase = 2.0 * math.pi * (138.0 * t - 52.0 * t * t)
        return envelope * (0.70 * math.sin(phase) + 0.22 * math.sin(phase * 2.03))

    def medicine(t: float, i: int) -> float:
        envelope = math.exp(-18.0 * t)
        noise = noises["medicine"].uniform(-1.0, 1.0) * math.exp(-42.0 * t)
        click = 0.2 * math.sin(2.0 * math.pi * 61.0 * t)
        return envelope * (0.72 * math.sin(2.0 * math.pi * 47.0 * t) + click) + 0.25 * noise

    def steel(t: float, _i: int) -> float:
        onset = min(1.0, t * 380.0)
        envelope = onset * math.exp(-4.9 * t)
        partials = (
            0.46 * math.sin(2.0 * math.pi * 713.0 * t)
            + 0.28 * math.sin(2.0 * math.pi * 1_147.0 * t + 0.4)
            + 0.18 * math.sin(2.0 * math.pi * 1_933.0 * t + 1.1)
            + 0.10 * math.sin(2.0 * math.pi * 2_887.0 * t)
        )
        return envelope * partials

    def concrete(t: float, _i: int) -> float:
        grain = noises["concrete"].uniform(-1.0, 1.0)
        envelope = math.exp(-25.0 * t)
        knock = math.sin(2.0 * math.pi * 128.0 * t) * math.exp(-15.0 * t)
        return 0.52 * grain * envelope + 0.45 * knock

    def rubber_brick(t: float, _i: int) -> float:
        slap = noises["rubber_brick"].uniform(-1.0, 1.0) * math.exp(-55.0 * t)
        body = math.sin(2.0 * math.pi * (82.0 - 30.0 * t) * t) * math.exp(-13.0 * t)
        return 0.42 * slap + 0.70 * body

    def exert_throw(t: float, _i: int) -> float:
        envelope = math.sin(math.pi * clamp(t / 0.30, 0.0, 1.0)) ** 0.75
        voice = math.sin(2.0 * math.pi * (96.0 - 18.0 * t) * t)
        grit = noises["throw"].uniform(-1.0, 1.0) * 0.20
        return envelope * (0.55 * voice + grit)

    def exert_jump(t: float, _i: int) -> float:
        envelope = math.sin(math.pi * clamp(t / 0.22, 0.0, 1.0))
        phase = 2.0 * math.pi * (102.0 * t + 110.0 * t * t)
        breath = noises["jump"].uniform(-1.0, 1.0) * 0.12
        return envelope * (0.58 * math.sin(phase) + breath)

    def landing(t: float, _i: int) -> float:
        kick_phase = 2.0 * math.pi * (74.0 * t - 90.0 * t * t)
        kick = math.sin(kick_phase) * math.exp(-18.0 * t)
        dust = noises["land"].uniform(-1.0, 1.0) * math.exp(-38.0 * t)
        return 0.72 * kick + 0.30 * dust

    return {
        "dodge": _stereo_pcm(0.42, dodge),
        "medicine": _stereo_pcm(0.30, medicine),
        "steel": _stereo_pcm(0.82, steel),
        "concrete": _stereo_pcm(0.28, concrete),
        "rubber_brick": _stereo_pcm(0.34, rubber_brick),
        "throw": _stereo_pcm(0.30, exert_throw),
        "jump": _stereo_pcm(0.22, exert_jump),
        "land": _stereo_pcm(0.27, landing),
    }


class AudioEngine:
    def __init__(self, enabled: bool = True) -> None:
        self.available = False
        self.muted = False
        self.error = ""
        self.sounds: Dict[str, Any] = {}
        if not enabled:
            self.error = "disabled by command line"
            return
        try:
            desired = (AUDIO_RATE, -16, AUDIO_CHANNELS)
            if pygame.mixer.get_init() != desired:
                if pygame.mixer.get_init() is not None:
                    pygame.mixer.quit()
                pygame.mixer.init(AUDIO_RATE, -16, AUDIO_CHANNELS, 512)
            for key, pcm in build_pcm_bank().items():
                self.sounds[key] = pygame.mixer.Sound(buffer=pcm.tobytes())
            pygame.mixer.set_num_channels(24)
            self.available = True
        except pygame.error as error:
            self.error = str(error)
            self.available = False

    def toggle_mute(self) -> None:
        self.muted = not self.muted

    def play(self, key: str, volume: float = 1.0, pan: float = 0.0) -> None:
        if not self.available or self.muted or key not in self.sounds:
            return
        channel = pygame.mixer.find_channel(True)
        if channel is None:
            return
        volume = clamp(volume, 0.0, 1.0)
        pan = clamp(pan, -1.0, 1.0)
        left = volume * math.sqrt(0.5 * (1.0 - pan))
        right = volume * math.sqrt(0.5 * (1.0 + pan))
        channel.set_volume(left, right)
        channel.play(self.sounds[key])

    def play_impact(self, event: ImpactEvent, player: Player) -> None:
        offset = event.position - player.eye
        distance = offset.length()
        attenuation = 1.0 / (1.0 + (distance / 18.0) ** 2)
        physical_level = clamp(math.log10(1.0 + max(event.impulse, event.mass * event.speed)) / 3.0, 0.08, 1.0)
        right_amount = offset.normalized().dot(player.right()) if distance > 1.0e-5 else 0.0
        self.play(event.family, physical_level * attenuation, right_amount)


# ---------------------------------------------------------------------------
# Lightweight fixed-pipeline rendering helpers
# ---------------------------------------------------------------------------


class GLText:
    def __init__(self) -> None:
        pygame.font.init()
        self.fonts = {
            "tiny": pygame.font.SysFont("dejavusansmono", 12),
            "small": pygame.font.SysFont("dejavusansmono", 15),
            "normal": pygame.font.SysFont("dejavusansmono", 18),
            "heading": pygame.font.SysFont("dejavusans", 24, bold=True),
            "title": pygame.font.SysFont("dejavusans", 31, bold=True),
        }
        self.cache: OrderedDict[Tuple[str, Tuple[int, int, int], str], Tuple[int, int, int]] = OrderedDict()
        self.dynamic: Dict[Tuple[str, Tuple[int, int, int], str], Tuple[str, int, int, int]] = {}
        self.cache_limit = 420

    def _texture(self, text: str, color: Tuple[int, int, int], size: str) -> Tuple[int, int, int]:
        key = (str(text), color, size)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        surface = self.fonts[size].render(str(text), True, color)
        width, height = surface.get_size()
        pixels = pygame.image.tostring(surface, "RGBA", True)
        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels,
        )
        result = (texture, width, height)
        self.cache[key] = result
        if len(self.cache) > self.cache_limit:
            _old_key, (old_texture, _w, _h) = self.cache.popitem(last=False)
            GL.glDeleteTextures([old_texture])
        return result

    def _dynamic_texture(
        self, slot: str, text: str, color: Tuple[int, int, int], size: str
    ) -> Tuple[int, int, int]:
        key = (slot, color, size)
        cached = self.dynamic.get(key)
        if cached is not None and cached[0] == str(text):
            return cached[1], cached[2], cached[3]
        surface = self.fonts[size].render(str(text), True, color)
        width, height = surface.get_size()
        pixels = pygame.image.tostring(surface, "RGBA", True)
        texture = cached[1] if cached is not None else GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels,
        )
        self.dynamic[key] = (str(text), texture, width, height)
        return texture, width, height

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int] = (235, 244, 250),
        size: str = "normal",
        center: bool = False,
        right: bool = False,
        dynamic_slot: Optional[str] = None,
    ) -> Tuple[int, int]:
        if dynamic_slot is None:
            texture, width, height = self._texture(str(text), color, size)
        else:
            texture, width, height = self._dynamic_texture(dynamic_slot, str(text), color, size)
        if center:
            x -= width * 0.5
        elif right:
            x -= width
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(x, y)
        GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(x + width, y)
        GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(x + width, y + height)
        GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(x, y + height)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        return width, height

    def cleanup(self) -> None:
        if self.cache:
            GL.glDeleteTextures([value[0] for value in self.cache.values()])
            self.cache.clear()
        if self.dynamic:
            GL.glDeleteTextures([value[1] for value in self.dynamic.values()])
            self.dynamic.clear()


def draw_panel(x: float, y: float, width: float, height: float, alpha: float = 0.78) -> None:
    GL.glColor4f(0.018, 0.035, 0.052, alpha)
    GL.glBegin(GL.GL_QUADS)
    GL.glVertex2f(x, y); GL.glVertex2f(x + width, y)
    GL.glVertex2f(x + width, y + height); GL.glVertex2f(x, y + height)
    GL.glEnd()
    GL.glColor4f(0.18, 0.73, 0.92, min(1.0, alpha + 0.12))
    GL.glBegin(GL.GL_LINE_LOOP)
    GL.glVertex2f(x, y); GL.glVertex2f(x + width, y)
    GL.glVertex2f(x + width, y + height); GL.glVertex2f(x, y + height)
    GL.glEnd()


def format_force(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g} MN"
    if value >= 1_000:
        return f"{value / 1_000:.3g} kN"
    return f"{value:.0f} N"


CONTROL_BINDINGS = {
    "forward": "W",
    "backward": "S",
    "left": "A",
    "right": "D",
    "jump": "SPACE",
    "look": "MOUSE",
    "throw": "LEFT MOUSE",
    "pickup": "RIGHT MOUSE",
    "gravity_up": "R",
    "gravity_down": "F",
    "force_up": "T",
    "force_down": "G",
    "friction_up": "Y",
    "friction_down": "H",
}


class GameApp:
    def __init__(
        self,
        seed: int = 1337,
        audio_enabled: bool = True,
        capture_mode: bool = False,
    ) -> None:
        pygame.display.init()
        pygame.font.init()
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        # A fixed 16:9 canvas keeps both telemetry columns non-overlapping and
        # avoids expensive OpenGL context recreation on Raspberry Pi drivers.
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        try:
            pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags, vsync=1)
        except (TypeError, pygame.error):
            pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        pygame.display.set_caption(TITLE)
        self.width, self.height = pygame.display.get_surface().get_size()
        self.clock = pygame.time.Clock()
        self.world = PhysicsWorld(seed)
        self.audio = AudioEngine(audio_enabled)
        self.text = GLText()
        self.running = True
        self.mouse_captured = not capture_mode
        self.show_help = True
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        self.message_until = 0.0
        self.last_message = self.world.messages[-1]
        self.room_list = 0
        self.sphere_list = 0
        self.concrete_points: List[Vec3] = []
        self.active_light_cell: Optional[Tuple[int, int]] = None
        self.active_lights: List[Tuple[float, float, float]] = []
        self.telemetry_next_update = 0.0
        self.telemetry_snapshot: List[Tuple[str, Tuple[int, int, int]]] = []
        self._init_gl()
        self._build_geometry()
        self._set_mouse_capture(self.mouse_captured)

    def _init_gl(self) -> None:
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.025, 0.035, 0.047, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_FOG)
        GL.glFogfv(GL.GL_FOG_COLOR, (0.045, 0.060, 0.074, 1.0))
        GL.glFogi(GL.GL_FOG_MODE, GL.GL_LINEAR)
        GL.glFogf(GL.GL_FOG_START, 78.0)
        GL.glFogf(GL.GL_FOG_END, 155.0)
        GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, (0.54, 0.56, 0.59, 1.0))
        GL.glLightModeli(GL.GL_LIGHT_MODEL_LOCAL_VIEWER, GL.GL_TRUE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

    def _build_geometry(self) -> None:
        self.sphere_list = GL.glGenLists(1)
        GL.glNewList(self.sphere_list, GL.GL_COMPILE)
        slices, stacks = 32, 20
        for stack in range(stacks):
            latitude0 = -math.pi * 0.5 + math.pi * stack / stacks
            latitude1 = -math.pi * 0.5 + math.pi * (stack + 1) / stacks
            GL.glBegin(GL.GL_TRIANGLE_STRIP)
            for slice_index in range(slices + 1):
                longitude = 2.0 * math.pi * slice_index / slices
                for latitude in (latitude1, latitude0):
                    cosine = math.cos(latitude)
                    normal = Vec3(cosine * math.sin(longitude), math.sin(latitude), cosine * math.cos(longitude))
                    GL.glNormal3f(normal.x, normal.y, normal.z)
                    GL.glVertex3f(normal.x, normal.y, normal.z)
            GL.glEnd()
        GL.glEndList()

        self.room_list = GL.glGenLists(1)
        GL.glNewList(self.room_list, GL.GL_COMPILE)
        self._draw_static_room_geometry()
        GL.glEndList()

        rng = random.Random(0xC0C0A)
        for _ in range(150):
            y = rng.uniform(-1.0, 1.0)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = math.sqrt(max(0.0, 1.0 - y * y)) * 1.006
            self.concrete_points.append(Vec3(radius * math.sin(angle), y * 1.006, radius * math.cos(angle)))

    def _draw_static_room_geometry(self) -> None:
        # Floor: a 10 m calibration grid aligned with the 100 ceiling fixtures.
        GL.glNormal3f(0.0, 1.0, 0.0)
        GL.glBegin(GL.GL_QUADS)
        for z in range(10):
            for x in range(10):
                shade = 0.225 + 0.025 * ((x + z) & 1)
                GL.glColor3f(shade, shade + 0.025, shade + 0.035)
                x0, x1 = x * 10.0, (x + 1) * 10.0
                z0, z1 = z * 10.0, (z + 1) * 10.0
                GL.glVertex3f(x0, 0.0, z0); GL.glVertex3f(x0, 0.0, z1)
                GL.glVertex3f(x1, 0.0, z1); GL.glVertex3f(x1, 0.0, z0)
        GL.glEnd()

        # Ceiling and four interior wall faces.
        GL.glColor3f(0.31, 0.34, 0.38)
        GL.glBegin(GL.GL_QUADS)
        GL.glNormal3f(0.0, -1.0, 0.0)
        GL.glVertex3f(0.0, ROOM_HEIGHT, 0.0); GL.glVertex3f(ROOM_WIDTH, ROOM_HEIGHT, 0.0)
        GL.glVertex3f(ROOM_WIDTH, ROOM_HEIGHT, ROOM_LENGTH); GL.glVertex3f(0.0, ROOM_HEIGHT, ROOM_LENGTH)
        GL.glNormal3f(1.0, 0.0, 0.0)
        GL.glVertex3f(0.0, 0.0, 0.0); GL.glVertex3f(0.0, ROOM_HEIGHT, 0.0)
        GL.glVertex3f(0.0, ROOM_HEIGHT, ROOM_LENGTH); GL.glVertex3f(0.0, 0.0, ROOM_LENGTH)
        GL.glNormal3f(-1.0, 0.0, 0.0)
        GL.glVertex3f(ROOM_WIDTH, 0.0, ROOM_LENGTH); GL.glVertex3f(ROOM_WIDTH, ROOM_HEIGHT, ROOM_LENGTH)
        GL.glVertex3f(ROOM_WIDTH, ROOM_HEIGHT, 0.0); GL.glVertex3f(ROOM_WIDTH, 0.0, 0.0)
        GL.glNormal3f(0.0, 0.0, 1.0)
        GL.glVertex3f(ROOM_WIDTH, 0.0, 0.0); GL.glVertex3f(ROOM_WIDTH, ROOM_HEIGHT, 0.0)
        GL.glVertex3f(0.0, ROOM_HEIGHT, 0.0); GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glNormal3f(0.0, 0.0, -1.0)
        GL.glVertex3f(0.0, 0.0, ROOM_LENGTH); GL.glVertex3f(0.0, ROOM_HEIGHT, ROOM_LENGTH)
        GL.glVertex3f(ROOM_WIDTH, ROOM_HEIGHT, ROOM_LENGTH); GL.glVertex3f(ROOM_WIDTH, 0.0, ROOM_LENGTH)
        GL.glEnd()

        # Ten-metre wall rulers make scale legible without distant-line shimmer.
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.0)
        GL.glBegin(GL.GL_LINES)
        for metre in range(0, 101, 10):
            GL.glColor4f(0.18, 0.68, 0.82, 0.34)
            GL.glVertex3f(metre, 0.012, 0.015); GL.glVertex3f(metre, ROOM_HEIGHT, 0.015)
            GL.glVertex3f(metre, 0.012, ROOM_LENGTH - 0.015); GL.glVertex3f(metre, ROOM_HEIGHT, ROOM_LENGTH - 0.015)
            GL.glVertex3f(0.015, 0.012, metre); GL.glVertex3f(0.015, ROOM_HEIGHT, metre)
            GL.glVertex3f(ROOM_WIDTH - 0.015, 0.012, metre); GL.glVertex3f(ROOM_WIDTH - 0.015, ROOM_HEIGHT, metre)
        for metre in range(0, 11, 5):
            GL.glColor4f(0.18, 0.68, 0.82, 0.28)
            GL.glVertex3f(0.015, metre, 0.0); GL.glVertex3f(0.015, metre, ROOM_LENGTH)
            GL.glVertex3f(ROOM_WIDTH - 0.015, metre, 0.0); GL.glVertex3f(ROOM_WIDTH - 0.015, metre, ROOM_LENGTH)
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def _set_mouse_capture(self, capture: bool) -> None:
        self.mouse_captured = capture
        pygame.event.set_grab(capture)
        pygame.mouse.set_visible(not capture)
        pygame.mouse.get_rel()

    def _configure_camera_and_lights(self) -> None:
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(72.0, self.width / max(1.0, float(self.height)), 0.035, 175.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        eye = self.world.player.eye
        forward = self.world.player.forward()
        target = eye + forward
        GLU.gluLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, 0.0, 1.0, 0.0)

        cell = (int(eye.x // LIGHT_SPACING), int(eye.z // LIGHT_SPACING))
        if cell != self.active_light_cell:
            self.active_light_cell = cell
            self.active_lights = sorted(
                LIGHT_POSITIONS,
                key=lambda position: (position[0] - eye.x) ** 2 + (position[2] - eye.z) ** 2,
            )[:4]
        for index, position in enumerate(self.active_lights):
            light = GL.GL_LIGHT0 + index
            GL.glEnable(light)
            GL.glLightfv(light, GL.GL_POSITION, (position[0], position[1] - 0.12, position[2], 1.0))
            GL.glLightfv(light, GL.GL_DIFFUSE, (1.0, 0.985, 0.91, 1.0))
            GL.glLightfv(light, GL.GL_SPECULAR, (1.0, 0.98, 0.88, 1.0))
            GL.glLightf(light, GL.GL_CONSTANT_ATTENUATION, 0.42)
            GL.glLightf(light, GL.GL_LINEAR_ATTENUATION, 0.035)
            GL.glLightf(light, GL.GL_QUADRATIC_ATTENUATION, 0.010)
        for index in range(len(self.active_lights), 8):
            GL.glDisable(GL.GL_LIGHT0 + index)

    def _draw_ceiling_fixtures(self) -> None:
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glColor4f(0.35, 0.72, 1.0, 0.11)
        GL.glBegin(GL.GL_QUADS)
        for x, y, z in LIGHT_POSITIONS:
            GL.glVertex3f(x - 3.2, y - 0.015, z - 3.2); GL.glVertex3f(x + 3.2, y - 0.015, z - 3.2)
            GL.glVertex3f(x + 3.2, y - 0.015, z + 3.2); GL.glVertex3f(x - 3.2, y - 0.015, z + 3.2)
        GL.glEnd()
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glColor4f(1.0, 0.99, 0.91, 1.0)
        GL.glBegin(GL.GL_QUADS)
        for x, y, z in LIGHT_POSITIONS:
            GL.glVertex3f(x - 2.35, y - 0.025, z - 2.35); GL.glVertex3f(x + 2.35, y - 0.025, z - 2.35)
            GL.glVertex3f(x + 2.35, y - 0.025, z + 2.35); GL.glVertex3f(x - 2.35, y - 0.025, z + 2.35)
        GL.glEnd()
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPopAttrib()

    def _draw_shadow(self, body: RigidBody, position: Vec3) -> None:
        radius = body.bounding_radius * 1.15
        height = max(0.0, position.y - body.support_extent(Vec3(0.0, 1.0, 0.0)))
        alpha = 0.28 / (1.0 + height * 1.7)
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glColor4f(0.0, 0.0, 0.0, alpha)
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(position.x, 0.008, position.z)
        for index in range(25):
            angle = 2.0 * math.pi * index / 24.0
            GL.glVertex3f(position.x + math.cos(angle) * radius, 0.008, position.z + math.sin(angle) * radius)
        GL.glEnd()
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPopAttrib()

    def _draw_box(self, dimensions: Tuple[float, float, float]) -> None:
        hx, hy, hz = (value * 0.5 for value in dimensions)
        faces = (
            (Vec3(1, 0, 0), ((hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz))),
            (Vec3(-1, 0, 0), ((-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-hx, -hy, -hz))),
            (Vec3(0, 1, 0), ((-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz))),
            (Vec3(0, -1, 0), ((-hx, -hy, hz), (-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz))),
            (Vec3(0, 0, 1), ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz))),
            (Vec3(0, 0, -1), ((hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz))),
        )
        GL.glBegin(GL.GL_QUADS)
        for normal, vertices in faces:
            GL.glNormal3f(normal.x, normal.y, normal.z)
            for vertex in vertices:
                GL.glVertex3f(*vertex)
        GL.glEnd()

    def _draw_surface_details(self, body: RigidBody) -> None:
        if body.spec.key == "concrete":
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_POINT_BIT | GL.GL_CURRENT_BIT)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glPointSize(2.0)
            GL.glColor4f(0.23, 0.20, 0.17, 0.82)
            GL.glBegin(GL.GL_POINTS)
            for point in self.concrete_points:
                GL.glVertex3f(point.x, point.y, point.z)
            GL.glEnd()
            GL.glPopAttrib()
        elif body.spec.shape == "sphere":
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_LINE_BIT | GL.GL_CURRENT_BIT)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glLineWidth(1.5)
            if body.spec.key == "steel":
                color = (0.88, 0.96, 1.0, 0.46)
            elif body.spec.key == "dodge":
                color = (0.18, 0.03, 0.02, 0.70)
            else:
                color = (0.04, 0.07, 0.10, 0.54)
            GL.glColor4f(*color)
            for axis in range(2 if body.spec.key == "dodge" else 1):
                GL.glBegin(GL.GL_LINE_LOOP)
                for index in range(49):
                    angle = 2.0 * math.pi * index / 48.0
                    if axis == 0:
                        GL.glVertex3f(math.sin(angle) * 1.003, math.cos(angle) * 1.003, 0.0)
                    else:
                        GL.glVertex3f(0.0, math.cos(angle) * 1.003, math.sin(angle) * 1.003)
                GL.glEnd()
            GL.glPopAttrib()

    def _draw_body(self, body: RigidBody, alpha: float) -> None:
        position = body.previous_position * (1.0 - alpha) + body.position * alpha
        self._draw_shadow(body, position)
        speed = body.velocity.length()
        if speed > 12.0:
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_LINE_BIT)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glLineWidth(clamp(1.0 + math.log10(speed), 1.0, 4.0))
            GL.glColor4f(*body.spec.color, clamp(math.log10(speed) / 5.0, 0.12, 0.65))
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(position.x, position.y, position.z)
            tail = position - body.velocity.normalized() * min(4.0, 0.05 * speed)
            GL.glVertex3f(tail.x, tail.y, tail.z)
            GL.glEnd()
            GL.glPopAttrib()

        GL.glPushMatrix()
        GL.glTranslatef(position.x, position.y, position.z)
        GL.glMultMatrixf(body.orientation.matrix())
        GL.glColor3f(*body.spec.color)
        if body.spec.key == "steel":
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 112.0)
        elif body.spec.key == "concrete":
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.05, 0.05, 0.05, 1.0))
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 4.0)
        else:
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.28, 0.30, 0.31, 1.0))
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 28.0)
        if body.spec.shape == "sphere":
            squash = 1.0
            if body.spec.key == "dodge" and body.last_impulse > 0.0:
                squash = 1.0 - min(0.10, body.last_impulse / 900.0)
            GL.glScalef(body.spec.radius / math.sqrt(squash), body.spec.radius * squash, body.spec.radius / math.sqrt(squash))
            GL.glCallList(self.sphere_list)
            self._draw_surface_details(body)
        else:
            self._draw_box(body.spec.dimensions)
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_LINE_BIT)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glLineWidth(1.2)
            GL.glColor4f(0.20, 0.07, 0.015, 0.75)
            GL.glScalef(1.012, 1.012, 1.012)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            self._draw_box(body.spec.dimensions)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glPopAttrib()
        GL.glPopMatrix()

    def _begin_hud(self) -> None:
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_FOG)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0, self.width, self.height, 0, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()

    def _end_hud(self) -> None:
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopAttrib()

    def _draw_hud(self) -> None:
        self._begin_hud()
        now = time.monotonic()
        cyan = (82, 220, 248)
        white = (235, 244, 250)
        muted = (157, 179, 193)
        orange = (255, 183, 72)

        draw_panel(18, 18, 480, 126)
        self.text.draw(TITLE, 32, 27, white, "heading")
        self.text.draw(
            f"GRAVITY  {self.world.gravity:8.5f} m/s^2   RANGE {GRAVITY_MIN:.2f}-{GRAVITY_MAX:.2f}   [R/F]",
            32, 61, cyan, "small",
        )
        self.text.draw(
            f"FRICTION {self.world.room_friction:8.2f} mu      RANGE {FRICTION_MIN:.2f}-{FRICTION_MAX:.2f}    [Y/H]",
            32, 85, cyan, "small",
        )
        self.text.draw(
            f"FORCE    {format_force(self.world.throw_force):>8}       RANGE 1 N-1 MN      [T/G]",
            32, 109, orange, "small",
        )

        telemetry_width = 562
        telemetry_x = self.width - telemetry_width - 18
        draw_panel(telemetry_x, 18, telemetry_width, 230)
        self.text.draw("RIGID-BODY VELOCITY TELEMETRY", telemetry_x + 14, 28, white, "small")
        y = 57
        if now >= self.telemetry_next_update or not self.telemetry_snapshot:
            self.telemetry_next_update = now + 0.10
            self.telemetry_snapshot = []
            for body, label in zip(self.world.bodies, self.world.velocity_telemetry_rows()):
                velocity = body.velocity.length()
                color = orange if body.held else cyan if velocity >= 0.05 else muted
                self.telemetry_snapshot.append((label, color))
        for index, (label, color) in enumerate(self.telemetry_snapshot):
            self.text.draw(label, telemetry_x + 14, y, color, "tiny", dynamic_slot=f"telemetry-{index}")
            y += 25

        selected = self.world.selected_body()
        if selected is not None:
            info_width = min(520, self.width - 36)
            info_y = self.height - 154
            draw_panel(18, info_y, info_width, 92, 0.72)
            spec = selected.spec
            self.text.draw(spec.name, 31, info_y + 10, orange, "normal")
            self.text.draw(
                f"{spec.mass:.3f} kg | {spec.dimension_label} | "
                f"{'effective bulk density' if spec.key.startswith('medicine') or spec.key == 'dodge' else 'density'} "
                f"{spec.density:,.1f} kg/m^3",
                31, info_y + 37, white, "tiny",
            )
            self.text.draw(
                f"restitution {spec.restitution:.2f} | material mu {spec.friction:.2f} | {spec.rigidity}",
                31, info_y + 59, muted, "tiny",
            )

        # Crosshair and contextual pickup/throw prompt.
        cx, cy = self.width * 0.5, self.height * 0.5
        GL.glColor4f(0.86, 0.97, 1.0, 0.95)
        GL.glLineWidth(2.0)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex2f(cx - 9, cy); GL.glVertex2f(cx - 2, cy)
        GL.glVertex2f(cx + 2, cy); GL.glVertex2f(cx + 9, cy)
        GL.glVertex2f(cx, cy - 9); GL.glVertex2f(cx, cy - 2)
        GL.glVertex2f(cx, cy + 2); GL.glVertex2f(cx, cy + 9)
        GL.glEnd()
        if self.world.held_body is not None:
            prompt = f"LMB THROW {self.world.held_body.spec.name.upper()}  |  RMB DROP"
        elif selected is not None:
            prompt = f"RMB PICK UP {selected.spec.name.upper()}"
        else:
            prompt = "AIM AT AN ITEM | RMB PICK UP"
        self.text.draw(prompt, cx, cy + 18, white, "small", center=True)

        if self.world.messages and self.world.messages[-1] != self.last_message:
            self.last_message = self.world.messages[-1]
            self.message_until = now + 4.5
        if now < self.message_until:
            message_width = min(self.width - 80, self.text.fonts["small"].size(self.last_message)[0] + 30)
            draw_panel((self.width - message_width) * 0.5, self.height - 94, message_width, 34, 0.74)
            self.text.draw(self.last_message, self.width * 0.5, self.height - 86, white, "small", center=True)

        if self.show_help:
            help_width = min(650, self.width - 36)
            draw_panel(18, 158, help_width, 77, 0.69)
            self.text.draw("WASD move | SPACE 1 m jump | Mouse look | RMB pick/drop | LMB throw", 31, 168, white, "small")
            self.text.draw("R/F gravity | T/G force | Y/H friction | TAB mouse | M mute | F5 reset", 31, 191, cyan, "small")
            self.text.draw(
                "PLAYER 2.00 m / 80 kg | ROOM 100 x 100 x 10 m | force uses a 0.75 m arm stroke",
                31, 214, muted, "tiny",
            )

        audio_state = "MUTED" if self.audio.muted else "AUDIO" if self.audio.available else "SILENT FALLBACK"
        measured_fps = self.clock.get_fps()
        measured_label = f"{measured_fps:4.1f}" if measured_fps > 0.0 else "--.-"
        self.text.draw(
            f"{audio_state} | physics {PHYSICS_HZ} Hz | target {TARGET_FPS} fps | actual {measured_label}",
            18, self.height - 29, muted, "tiny",
        )
        watermark_width, _watermark_height = self.text.fonts["tiny"].size(CREDIT_WATERMARK)
        watermark_x = self.width - watermark_width - 18
        GL.glColor4f(0.015, 0.025, 0.035, 0.52)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex2f(watermark_x - 6, self.height - 32)
        GL.glVertex2f(self.width - 12, self.height - 32)
        GL.glVertex2f(self.width - 12, self.height - 12)
        GL.glVertex2f(watermark_x - 6, self.height - 12)
        GL.glEnd()
        self.text.draw(CREDIT_WATERMARK, self.width - 18, self.height - 29, (190, 204, 214), "tiny", right=True)
        self._end_hud()

    def render(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._configure_camera_and_lights()
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_FOG)
        GL.glCallList(self.room_list)
        self._draw_ceiling_fixtures()
        alpha = clamp(self.accumulator / FIXED_DT, 0.0, 1.0)
        for body in self.world.bodies:
            self._draw_body(body, alpha)
            body.last_impulse *= 0.82
        self._draw_hud()

    def _handle_keydown(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_TAB:
            self._set_mouse_capture(not self.mouse_captured)
        elif key == pygame.K_F1:
            self.show_help = not self.show_help
        elif key == pygame.K_F5:
            self.world.reset()
        elif key == pygame.K_m:
            self.audio.toggle_mute()
        elif key == pygame.K_SPACE:
            if self.world.jump():
                self.audio.play("jump", 0.72)
        elif key == pygame.K_r:
            self.world.adjust_gravity(1)
        elif key == pygame.K_f:
            self.world.adjust_gravity(-1)
        elif key == pygame.K_t:
            self.world.adjust_throw_force(1)
        elif key == pygame.K_g:
            self.world.adjust_throw_force(-1)
        elif key == pygame.K_y:
            self.world.adjust_friction(1)
        elif key == pygame.K_h:
            self.world.adjust_friction(-1)

    def _handle_event(self, event: Any) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self._handle_keydown(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not self.mouse_captured:
                self._set_mouse_capture(True)
                return
            if event.button == 1:
                result = self.world.throw_held()
                if result is not None:
                    _body, speed, _impulse = result
                    self.audio.play("throw", clamp(0.42 + math.log10(1.0 + speed) * 0.12, 0.42, 0.95))
            elif event.button == 3:
                self.world.pickup_or_drop()
        elif event.type == pygame.MOUSEMOTION and self.mouse_captured:
            dx, dy = event.rel
            sensitivity = 0.00225
            self.world.player.yaw = (self.world.player.yaw + dx * sensitivity) % (2.0 * math.pi)
            # Positive relative y means the mouse moved down, so pitch decreases.
            self.world.player.pitch = clamp(
                self.world.player.pitch - dy * sensitivity,
                math.radians(-88.0), math.radians(88.0),
            )

    def _update_movement_input(self) -> None:
        keys = pygame.key.get_pressed()
        forward = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        strafe = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        self.world.set_move_input(forward, strafe)

    def tick(self, frame_dt: float) -> None:
        self._update_movement_input()
        self.accumulator += min(MAX_FRAME_DT, max(0.0, frame_dt))
        steps = 0
        while self.accumulator >= FIXED_DT and steps < MAX_PHYSICS_STEPS:
            self.world.step(FIXED_DT)
            for impact in self.world.impacts:
                self.audio.play_impact(impact, self.world.player)
            if self.world.player.landing_speed > 0.0:
                volume = clamp((self.world.player.landing_speed - 0.5) / 7.0, 0.18, 1.0)
                self.audio.play("land", volume)
                self.world.player.landing_speed = 0.0
            self.accumulator -= FIXED_DT
            steps += 1
        if steps >= MAX_PHYSICS_STEPS and self.accumulator >= FIXED_DT:
            self.accumulator = 0.0

    def capture_frame(self, path: str) -> None:
        pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        surface = pygame.image.fromstring(pixels, (self.width, self.height), "RGB", True)
        pygame.image.save(surface, path)

    def run(self, capture_path: Optional[str] = None, maximum_frames: int = 0) -> int:
        frame_count = 0
        try:
            while self.running:
                frame_dt = self.clock.tick(TARGET_FPS) / 1000.0
                for event in pygame.event.get():
                    self._handle_event(event)
                self.tick(frame_dt)
                self.render()
                pygame.display.flip()
                frame_count += 1
                if capture_path and frame_count >= 3:
                    self.render()
                    self.capture_frame(capture_path)
                    break
                if maximum_frames and frame_count >= maximum_frames:
                    break
            return 0
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        self._set_mouse_capture(False)
        if self.text is not None:
            self.text.cleanup()
        if self.sphere_list:
            GL.glDeleteLists(self.sphere_list, 1)
            self.sphere_list = 0
        if self.room_list:
            GL.glDeleteLists(self.room_list, 1)
            self.room_list = 0
        if pygame.mixer.get_init() is not None:
            pygame.mixer.quit()
        pygame.quit()


# ---------------------------------------------------------------------------
# Deterministic validation and command-line entry point
# ---------------------------------------------------------------------------


def run_self_check(verbose: bool = True) -> bool:
    passed: List[str] = []
    failed: List[str] = []

    def verify(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            passed.append(label)
        else:
            failed.append(f"{label}{': ' + detail if detail else ''}")

    verify("room dimensions", (ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT) == (100.0, 100.0, 10.0))
    verify("two-metre player", PLAYER_HEIGHT == 2.0 and 0.0 < PLAYER_EYE_HEIGHT < PLAYER_HEIGHT)
    verify("Earth gravity default", abs(EARTH_GRAVITY - 9.80665) < 1.0e-10)
    verify("requested control ranges", (
        GRAVITY_MIN <= EARTH_GRAVITY <= GRAVITY_MAX
        and FRICTION_MIN <= DEFAULT_ROOM_FRICTION <= FRICTION_MAX
        and THROW_FORCE_MIN == 1.0 and THROW_FORCE_MAX == 1_000_000.0
    ))
    verify("eight-pass sequential impulse solver", SOLVER_ITERATIONS >= 8)
    verify("one-hundred ceiling fixtures", len(LIGHT_POSITIONS) == 100 and len(set(LIGHT_POSITIONS)) == 100)
    verify("ten-metre fixture pitch", all(
        abs((x - 5.0) % 10.0) < 1.0e-9 and abs((z - 5.0) % 10.0) < 1.0e-9
        for x, _y, z in LIGHT_POSITIONS
    ))
    verify("six balls plus one brick", (
        len(ITEM_SPECS) == 7
        and sum(spec.shape == "sphere" for spec in ITEM_SPECS) == 6
        and sum(spec.shape == "box" for spec in ITEM_SPECS) == 1
    ))
    verify("three medicine masses", [ITEM_SPEC_BY_KEY[key].mass for key in ("medicine_1", "medicine_3", "medicine_10")] == [1.0, 3.0, 10.0])
    verify("documented dodgeball midpoint", (
        abs(ITEM_SPEC_BY_KEY["dodge"].diameter - 0.660 / math.pi) < 1.0e-12
        and abs(ITEM_SPEC_BY_KEY["dodge"].mass - 0.310) < 1.0e-12
    ))
    verify("solid steel density-derived mass", abs(ITEM_SPEC_BY_KEY["steel"].mass - 58.19) < 0.03)
    verify("solid concrete density-derived mass", abs(ITEM_SPEC_BY_KEY["concrete"].mass - 17.79) < 0.03)
    verify("solid rubber brick density-derived mass", abs(ITEM_SPEC_BY_KEY["rubber_brick"].mass - 0.936) < 0.002)
    verify("positive rigid-body inertia", all(
        min(spec.inertia_diagonal.tuple()) > 0.0 for spec in ITEM_SPECS
    ))
    verify("inflated shell inertia model", (
        abs(ITEM_SPEC_BY_KEY["dodge"].inertia_diagonal.x - (2.0 / 3.0) * 0.310 * ITEM_SPEC_BY_KEY["dodge"].radius ** 2) < 1.0e-12
    ))

    world = PhysicsWorld(2468)
    verify("player starts at floor coordinate 10 x 10", world.player.position.tuple() == PLAYER_START and world.player.grounded)
    initial_line = all(
        abs(body.position.x - (7.0 + index)) < 1.0e-12
        and abs(body.position.z - 15.0) < 1.0e-12
        for index, body in enumerate(world.bodies)
    )
    verify("items five metres ahead and one metre apart", initial_line)
    verify("initial bodies non-overlapping", all(
        (second.position - first.position).length() > first.bounding_radius + second.bounding_radius
        for index, first in enumerate(world.bodies)
        for second in world.bodies[index + 1:]
    ))

    # Newton I: absent force or contact, constant velocity produces x(t)=x0+vt.
    inertial_body = RigidBody(ITEM_SPEC_BY_KEY["medicine_1"], Vec3(50.0, 5.0, 50.0), velocity=Vec3(3.0, 0.0, -2.0))
    inertial_start = inertial_body.position.copy()
    world._advance_body_swept(inertial_body, 0.25)
    expected_position = inertial_start + Vec3(3.0, 0.0, -2.0) * 0.25
    verify("Newton first law drift", (inertial_body.position - expected_position).length() < 1.0e-10)

    # Newton II: isolate a known x force; gravity only affects y and zero initial
    # velocity means the first drag evaluation is zero.
    force_world = PhysicsWorld(1)
    force_world.gravity = 0.10
    forced = RigidBody(ITEM_SPEC_BY_KEY["medicine_3"], Vec3(50.0, 5.0, 50.0))
    forced.apply_force(Vec3(90.0, 0.0, 0.0))
    force_world._integrate_body_forces(forced, 0.2)
    verify("Newton second law F equals ma", abs(forced.velocity.x - (90.0 / 3.0) * 0.2) < 1.0e-10)

    # Newton III plus linear-momentum conservation for a central two-body impact.
    collision_world = PhysicsWorld(2)
    a = RigidBody(ITEM_SPEC_BY_KEY["medicine_1"], Vec3(40.0, 5.0, 50.0), velocity=Vec3(3.0, 0.0, 0.0))
    b = RigidBody(ITEM_SPEC_BY_KEY["medicine_3"], Vec3(40.21, 5.0, 50.0), velocity=Vec3(-1.0, 0.0, 0.0))
    normal = Vec3(1.0, 0.0, 0.0)
    contact = a.position + normal * a.spec.radius
    before_momentum = a.velocity * a.spec.mass + b.velocity * b.spec.mass
    collision_world._body_pair_impulse(a, b, normal, contact)
    after_momentum = a.velocity * a.spec.mass + b.velocity * b.spec.mass
    verify("Newton third law equal-opposite impulses", (after_momentum - before_momentum).length() < 1.0e-9)

    verify("material elasticity ordering", (
        ITEM_SPEC_BY_KEY["dodge"].restitution > ITEM_SPEC_BY_KEY["steel"].restitution
        > ITEM_SPEC_BY_KEY["medicine_1"].restitution > ITEM_SPEC_BY_KEY["concrete"].restitution
    ))
    friction_body = RigidBody(ITEM_SPEC_BY_KEY["dodge"], Vec3(50.0, ITEM_SPEC_BY_KEY["dodge"].radius, 50.0), velocity=Vec3(5.0, 0.0, 0.0))
    friction_body.grounded = True
    before_speed = friction_body.velocity.length()
    world._apply_ground_resistance_and_sleep(friction_body, 0.5)
    verify("rolling resistance dissipates motion", 0.0 <= friction_body.velocity.length() < before_speed)

    jump_results: List[Tuple[float, float, float]] = []
    for gravity in (GRAVITY_MIN, EARTH_GRAVITY, GRAVITY_MAX):
        jump_world = PhysicsWorld(3)
        jump_world.bodies = []
        jump_world.gravity = gravity
        jump_world.jump()
        peak = jump_world.player.position.y
        for _ in range(12_000):
            jump_world._integrate_player(FIXED_DT)
            peak = max(peak, jump_world.player.position.y)
            if jump_world.player.grounded:
                break
        jump_results.append((gravity, peak, jump_world.player.landing_speed))
    verify("one-metre jump across gravity range", all(
        abs(peak - JUMP_HEIGHT) < 0.001 for _gravity, peak, _landing in jump_results
    ), ", ".join(f"g={gravity:g}: {peak:.5f}m" for gravity, peak, _landing in jump_results))
    verify("landing event survives minimum gravity", jump_results[0][2] > 0.0)

    throw_world = PhysicsWorld(4)
    thrown = throw_world.bodies[0]
    thrown.held = True
    throw_world.held_body = thrown
    throw_world.throw_force = THROW_FORCE_MAX
    throw_result = throw_world.throw_held()
    assert throw_result is not None
    _thrown_body, launch_speed, impulse_magnitude = throw_result
    total_throw_energy = (
        0.5 * thrown.spec.mass * thrown.velocity.length_squared()
        + 0.5 * PLAYER_MASS * throw_world.player.velocity.length_squared()
    )
    verify("force-through-distance total work energy", abs(total_throw_energy - THROW_FORCE_MAX * THROW_STROKE) < 1.0e-5)
    total_throw_momentum = thrown.velocity * thrown.spec.mass + throw_world.player.velocity * PLAYER_MASS
    verify("throw has equal recoil", total_throw_momentum.length() < 1.0e-7, f"residual {total_throw_momentum.length():.3g}")
    verify("maximum force remains under solver safety speed", launch_speed < MAX_LINEAR_SPEED)

    ccd_world = PhysicsWorld(5)
    fast = RigidBody(ITEM_SPEC_BY_KEY["dodge"], Vec3(99.0, 5.0, 50.0), velocity=Vec3(launch_speed, 0.0, 0.0))
    ccd_world._advance_body_swept(fast, FIXED_DT)
    verify("maximum-force swept wall collision", (
        fast.spec.radius <= fast.position.x <= ROOM_WIDTH - fast.spec.radius
        and fast.velocity.x < 0.0
    ), f"x={fast.position.x:.3f}, vx={fast.velocity.x:.3f}")

    # Fast object/player CCD transfers momentum instead of crossing the capsule.
    player_ccd_world = PhysicsWorld(51)
    player_ccd_body = RigidBody(
        ITEM_SPEC_BY_KEY["dodge"], Vec3(10.0, 1.0, 5.0), velocity=Vec3(0.0, 0.0, 1_000.0)
    )
    player_ccd_world.bodies = [player_ccd_body]
    player_ccd_world.step(FIXED_DT)
    verify("swept body versus player capsule", (
        player_ccd_world.player.velocity.z > 0.0 and player_ccd_body.velocity.z < 1_000.0
    ))

    player_wall_world = PhysicsWorld(52)
    player_wall_world.player.position.x = PLAYER_RADIUS
    player_wall_world.player.velocity.x = -5.0
    player_wall_world._integrate_player(FIXED_DT)
    verify("player wall resolves outward momentum", (
        player_wall_world.player.position.x == PLAYER_RADIUS
        and player_wall_world.player.velocity.x >= 0.0
    ))

    zero_friction_world = PhysicsWorld(53)
    zero_friction_world.room_friction = 0.0
    spinning = zero_friction_world.bodies[0]
    zero_friction_world.bodies = [spinning]
    spinning.angular_velocity = Vec3(0.0, 10.0, 0.0)
    for _ in range(PHYSICS_HZ):
        zero_friction_world.step(FIXED_DT)
    verify("zero room friction preserves contact spin", abs(spinning.angular_velocity.y - 10.0) < 1.0e-8)

    tilted_world = PhysicsWorld(54)
    tilted = RigidBody(
        ITEM_SPEC_BY_KEY["rubber_brick"], Vec3(50.0, 1.0, 50.0),
        velocity=Vec3(0.0, -10.0, 0.0),
        orientation=Quat(math.cos(math.pi / 8.0), 0.0, 0.0, math.sin(math.pi / 8.0)),
    )
    tilted_world.bodies = [tilted]
    for _ in range(30):
        tilted_world.step(FIXED_DT)
    verify("tilted brick floor contact creates torque", tilted.angular_velocity.length() > 0.1)

    stacked_world = PhysicsWorld(55)
    lower = RigidBody(ITEM_SPEC_BY_KEY["medicine_1"], Vec3(50.0, 0.60, 50.0))
    upper = RigidBody(ITEM_SPEC_BY_KEY["medicine_3"], Vec3(50.0, 1.20, 50.0))
    stacked_world.bodies = [lower, upper]
    stack_stayed_inside = True
    for _ in range(1_200):
        stacked_world.step(FIXED_DT)
        stack_stayed_inside = stack_stayed_inside and all(
            body.position.y + 1.0e-8 >= body.support_extent(Vec3(0.0, 1.0, 0.0))
            for body in stacked_world.bodies
        )
    verify("stacked contacts never escape floor", stack_stayed_inside)
    verify("resting stack reaches low residual speed", max(body.velocity.length() for body in stacked_world.bodies) < 0.03)

    pcm = build_pcm_bank()
    required_sounds = {"dodge", "medicine", "steel", "concrete", "rubber_brick", "throw", "jump", "land"}
    hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in pcm.items()}
    verify("five material plus three exertion sounds", set(pcm) == required_sounds and all(len(value) > 1_000 for value in pcm.values()))
    verify("procedural effects are sonically distinct", len(set(hashes.values())) == len(required_sounds))
    verify("medicine balls share one impact family", all(
        ITEM_SPEC_BY_KEY[key].sound_family == "medicine" for key in ("medicine_1", "medicine_3", "medicine_10")
    ))

    verify("complete requested input map", set(CONTROL_BINDINGS) == {
        "forward", "backward", "left", "right", "jump", "look", "throw", "pickup",
        "gravity_up", "gravity_down", "force_up", "force_down", "friction_up", "friction_down",
    })
    camera_probe = Player(yaw=0.0, pitch=0.0)
    camera_probe.yaw = 0.1
    verify("mouse-right and D follow screen-right", (
        camera_probe.forward(False).x < 0.0
        and camera_probe.right().x < 0.0
        and abs(camera_probe.forward(False).dot(camera_probe.right())) < 1.0e-12
    ))
    verify("exact watermark credit", CREDIT_WATERMARK == "Made by OpenAI ChatGPT Codex 5.6 Sol Ultra")
    telemetry_rows = world.velocity_telemetry_rows()
    verify("HUD exposes seven vector velocities", (
        len(telemetry_rows) == 7
        and all("v=(" in row and "|v|=" in row and "m/s" in row for row in telemetry_rows)
        and all(name in " ".join(telemetry_rows) for name in TELEMETRY_NAMES.values())
    ))
    verify("HUD exposes current values and ranges", (
        len(world.status_lines()) == 3
        and "m/s^2" in world.status_lines()[0]
        and "friction" in world.status_lines()[1].lower()
        and "N" in world.status_lines()[2]
    ))

    stable_world = PhysicsWorld(999)
    for _ in range(360):
        stable_world.step(FIXED_DT)
    verify("settling simulation remains finite and bounded", all(
        body.position.finite()
        and body.velocity.finite()
        and 0.0 <= body.position.x <= ROOM_WIDTH
        and 0.0 <= body.position.y <= ROOM_HEIGHT
        and 0.0 <= body.position.z <= ROOM_LENGTH
        for body in stable_world.bodies
    ))
    verify("settled bodies enter sleep state", all(body.asleep for body in stable_world.bodies))
    stable_world.adjust_gravity(1)
    verify("setting changes wake sleeping bodies", all(not body.asleep for body in stable_world.bodies))
    twin_a, twin_b = PhysicsWorld(777), PhysicsWorld(777)
    twin_a.set_move_input(1.0, 0.25)
    twin_b.set_move_input(1.0, 0.25)
    for _ in range(120):
        twin_a.step(FIXED_DT)
        twin_b.step(FIXED_DT)
    deterministic = (
        twin_a.player.position.tuple() == twin_b.player.position.tuple()
        and all(a_body.position.tuple() == b_body.position.tuple() for a_body, b_body in zip(twin_a.bodies, twin_b.bodies))
    )
    verify("fixed-step simulation is deterministic", deterministic)

    if verbose:
        print(f"{TITLE} deterministic self-check")
        for label in passed:
            print(f"  PASS  {label}")
        for failure in failed:
            print(f"  FAIL  {failure}")
        print(f"Result: {len(passed)} passed, {len(failed)} failed")
    return not failed


def run_gl_smoke(seed: int) -> bool:
    load_dependencies()
    pygame.mixer.pre_init(AUDIO_RATE, -16, AUDIO_CHANNELS, 512)
    app = GameApp(seed, audio_enabled=False, capture_mode=True)
    try:
        app.tick(FIXED_DT)
        app.render()
        error = GL.glGetError()
        pygame.display.flip()
        if error == GL.GL_NO_ERROR:
            print(f"{TITLE} OpenGL smoke: PASS (GL_NO_ERROR)")
            return True
        print(f"{TITLE} OpenGL smoke: FAIL (error {error})")
        return False
    finally:
        app.cleanup()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337, help="deterministic room seed")
    parser.add_argument("--check", action="store_true", help="run headless content and physics checks")
    parser.add_argument("--gl-check", action="store_true", help="open one frame and check OpenGL errors")
    parser.add_argument("--capture", metavar="PATH", help="render three frames, save a PNG, and exit")
    parser.add_argument("--frames", type=int, default=0, help="exit after this many rendered frames (0 = interactive)")
    parser.add_argument("--no-audio", action="store_true", help="use the silent audio fallback")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.check:
        return 0 if run_self_check(True) else 1
    if args.gl_check:
        return 0 if run_gl_smoke(args.seed) else 1
    load_dependencies()
    pygame.mixer.pre_init(AUDIO_RATE, -16, AUDIO_CHANNELS, 512)
    app = GameApp(args.seed, audio_enabled=not args.no_audio, capture_mode=bool(args.capture))
    return app.run(args.capture, max(0, args.frames))


if __name__ == "__main__":
    raise SystemExit(main())
