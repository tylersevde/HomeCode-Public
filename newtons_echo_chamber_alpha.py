#!/usr/bin/env python3
"""
Newton's Echo Chamber -- Alpha

A first-person SI-unit rigid-body playground built for Raspberry Pi 5-class
hardware with pygame and the fixed-function PyOpenGL pipeline.  The room is
100 m x 100 m x 10 m, the player is 2 m tall, and all 551 objects use
documented or explicitly representative real-world dimensions and masses.
The original seven calibration pieces now share the room with an ash bat,
functional one-wheel wheelbarrow, giant helium balloon, five PE-foam pool
noodles, ten adhesive goo blobs, twenty oversized ceramic marbles, five
randomly selected stuffed animals, and 500 modular clay bricks on an EPAL
wooden pallet.

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
  F2                     Toggle group / original-seven telemetry
  F3                     Toggle live four-core / memory telemetry
  F5                     Reset the room and all 551 objects
  M                      Mute / unmute audio
  Esc                    Quit

Physics uses metres, kilograms, seconds, Newtons, semi-implicit integration at
120 Hz, swept room contacts, sequential impulses, equal-and-opposite reaction
impulses, Coulomb friction, restitution, angular inertia, quadratic air drag,
rolling resistance, and sleeping.  A throw's selected force acts through a
0.75 m arm stroke, making F and work physically meaningful while allowing the
full requested 1,000,000 N range to remain numerically tractable.

Raspberry Pi performance mode makes all four CPU cores scheduler-visible: the
ordered solver, pygame, and OpenGL submission stay on one core while three
persistent workers parallelize deterministic swept spatial-hash and immutable
mesh work when it is large enough to beat IPC overhead.  NumPy's AArch64/NEON
kernel handles the dense swept-sphere broadphase.  VideoCore VII rendering uses
cached driver-side geometry, view culling, static exhibit batching, and a
transform-feedback micro-debris simulation whose gravity, momentum, drag,
friction, and restitution remain GPU-resident without per-frame readback.
The HUD reports frame timing, all four logical-core loads, memory headroom,
worker mode, GPU physics work, and the live GL renderer.  AUTO targets 60 FPS
and aggressively protects a 30 FPS floor; ``--maximum-throughput`` removes the
frame cap and requests the maximum useful GPU workload for profiling while
retaining an emergency 30 FPS load shed.

Dependencies:
  pip install pygame PyOpenGL PyOpenGL_accelerate

The impact and exertion sounds are original procedural PCM generated at start.
Audio-device failure is non-fatal.  Run ``--check`` for a headless deterministic
physics/content validation, or ``--capture PATH`` for a rendered smoke frame.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import multiprocessing
import os
import random
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Keep every process single-threaded inside numeric libraries; the game owns
# CPU placement explicitly (main core plus three worker processes).
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

try:
    import numpy as np
except ImportError:  # Scalar physics remains a supported dependency fallback.
    np = None


pygame = None
GL = None
GLU = None
RAW_GL_INTERLEAVED_ARRAYS = None
RAW_GL_VERTEX_ATTRIB_POINTER = None
RAW_GL_TRANSFORM_FEEDBACK_VARYINGS = None
GL_SHADERS = None

TITLE = "Newton's Echo Chamber -- Alpha"
RELEASE = "ALPHA"
CREDIT_WATERMARK = "Made by OpenAI ChatGPT Codex 5.6 Sol Ultra"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
MINIMUM_RENDER_FPS = 30
PHYSICS_HZ = 120
FIXED_DT = 1.0 / PHYSICS_HZ
MAX_FRAME_DT = 0.10
MAX_PHYSICS_STEPS = 12
SOLVER_ITERATIONS = 8
BROADPHASE_CELL_SIZE = 0.15
BROADPHASE_SKIN = 0.006
MAX_ACTIVE_BRICKS = 4
MAX_NEW_BRICK_WAKES_PER_TICK = 2
MAX_BULLET_CONTACTS_PER_TICK = 2
MAX_BRICK_NEIGHBORS_PER_TICK = 10
CLAY_BRICK_COUNT = 500
NOODLE_COUNT = 5
GOO_COUNT = 10
CERAMIC_MARBLE_COUNT = 20
STUFFED_ANIMAL_COUNT = 5
EXPECTED_BODY_COUNT = 551
MAX_IMPACT_SOUNDS_PER_FRAME = 8
CPU_CORE_BUDGET = 4
SPATIAL_WORKER_COUNT = CPU_CORE_BUDGET - 1
# IPC costs more than a short serial spatial-hash pass.  This approximate
# cell-visit threshold only wakes the three helper processes for genuinely dense
# or fast-moving scenes, while the ordinary sleeping room remains latency-low.
PARALLEL_SPATIAL_WORK_MIN = 120_000

# VideoCore VII/Mesa friendly render tiers.  Geometry is compiled once into
# driver-side display lists and the automatic tier only sheds cosmetic work;
# simulation frequency and physical results never change with frame rate.
RENDER_QUALITY_NAMES = ("SAFE", "BALANCED", "ULTRA")
SPHERE_LOD = ((12, 8), (20, 12), (32, 20))
PERFORMANCE_SAMPLE_HZ = 4.0
QUALITY_RECOVERY_SECONDS = 12.0
SOFTWARE_GL_MARKERS = ("llvmpipe", "softpipe", "swrast", "swiftshader", "software")

# Tier-aligned catch-up and sound budgets.  The fixed 120 Hz physics frequency
# is unchanged; a late frame simply cannot monopolize the following frame and
# start a persistent "spiral of death".  Five SAFE steps still cover the four
# simulation ticks needed to maintain real time at the requested 30 FPS floor.
QUALITY_MAX_PHYSICS_STEPS = (5, 8, MAX_PHYSICS_STEPS)
QUALITY_IMPACT_SOUND_LIMITS = (2, 5, MAX_IMPACT_SOUNDS_PER_FRAME)
QUALITY_PHYSICS_BUDGET_MS = (11.0, 18.0, 28.0)

# Transform-feedback physics is intentionally bounded.  Two 8-float state
# buffers at the maximum tier occupy 16 MiB and remain in GPU/shared memory;
# increasing this into gigabytes would consume unified memory bandwidth and
# make the game slower rather than more capable.
GPU_PHYSICS_TIER_COUNTS = (32_768, 98_304, 196_608)
GPU_PHYSICS_MAX_PARTICLES = 262_144
GPU_PARTICLE_STATE_FLOATS = 8

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

# Counter-clockwise unit-cube faces shared by the compatibility display-list
# path and the GPU-resident brick-course VBO builder.
UNIT_BOX_FACES = (
    ((1.0, 0.0, 0.0), ((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5))),
    ((-1.0, 0.0, 0.0), ((-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5))),
    ((0.0, 1.0, 0.0), ((-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5))),
    ((0.0, -1.0, 0.0), ((-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5))),
    ((0.0, 0.0, 1.0), ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5))),
    ((0.0, 0.0, -1.0), ((0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5))),
)


_ORIGINAL_CPU_AFFINITY: Optional[Tuple[int, ...]] = None


def configure_cpu_core_budget(requested: int = CPU_CORE_BUDGET) -> Tuple[int, ...]:
    """Limit the game process tree to the requested scheduler-visible cores.

    The main process owns one Cortex-A76 and three persistent helpers share the
    remaining three-core mask.  Linux can then rebalance replacement workers
    without ever moving pygame/OpenGL off the main core.  The original allowed
    mask is cached because a later call would otherwise only see the main
    process's already-narrowed affinity.  Unsupported platforms simply retain
    their normal scheduler policy.
    """
    global _ORIGINAL_CPU_AFFINITY
    try:
        if _ORIGINAL_CPU_AFFINITY is None:
            _ORIGINAL_CPU_AFFINITY = tuple(sorted(os.sched_getaffinity(0)))
        available = _ORIGINAL_CPU_AFFINITY
        selected = available[:max(1, min(int(requested), len(available)))]
        # pygame/OpenGL and the ordered solver exclusively own the first core.
        os.sched_setaffinity(0, {selected[0]})
        return selected
    except (AttributeError, OSError, ValueError):
        count = max(1, os.cpu_count() or 1)
        return tuple(range(min(max(1, int(requested)), count)))


SpatialState = Tuple[int, float, float, float, float, float, float, float]
BoxRenderState = Tuple[
    float, float, float,  # dimensions
    float, float, float,  # color
    float, float, float,  # position
    float, float, float, float,  # quaternion w/x/y/z
]
BoxBatchTask = Tuple[int, Tuple[BoxRenderState, ...]]


def _initialize_spatial_worker(worker_cores: Tuple[int, ...]) -> None:
    """Keep every helper on the three non-render Cortex-A76 cores."""
    if not worker_cores:
        return
    try:
        # A shared mask lets Linux spread the three single-threaded workers and
        # also handles a Pool replacement process without duplicate slot IDs.
        os.sched_setaffinity(0, set(worker_cores))
    except (AttributeError, OSError, ValueError):
        pass


def _spatial_state_work(state: SpatialState) -> int:
    """Estimate hash-cell visits for deterministic worker load balancing."""
    _index, x, y, z, old_x, old_y, old_z, radius = state
    longest_axis = max(abs(x - old_x), abs(y - old_y), abs(z - old_z))
    samples = max(1, math.ceil(longest_axis / (BROADPHASE_CELL_SIZE * 0.50))) + 1
    diameter_cells = max(
        1, math.ceil((2.0 * (radius + BROADPHASE_SKIN)) / BROADPHASE_CELL_SIZE) + 1
    )
    return samples * diameter_cells * diameter_cells * diameter_cells


def _spatial_cell_chunk(
    states: Sequence[SpatialState],
) -> List[Tuple[int, Tuple[Tuple[int, int, int], ...]]]:
    """Pure spawn-worker kernel for swept spatial-hash rasterization."""
    inverse_cell = 1.0 / BROADPHASE_CELL_SIZE
    results: List[Tuple[int, Tuple[Tuple[int, int, int], ...]]] = []
    for index, x, y, z, old_x, old_y, old_z, body_radius in states:
        motion_x, motion_y, motion_z = x - old_x, y - old_y, z - old_z
        longest_axis = max(abs(motion_x), abs(motion_y), abs(motion_z))
        steps = max(1, math.ceil(longest_axis / (BROADPHASE_CELL_SIZE * 0.50)))
        radius = body_radius + BROADPHASE_SKIN
        keys: set[Tuple[int, int, int]] = set()
        for step_index in range(steps + 1):
            fraction = step_index / steps
            center_x = old_x + motion_x * fraction
            center_y = old_y + motion_y * fraction
            center_z = old_z + motion_z * fraction
            x0 = math.floor((center_x - radius) * inverse_cell)
            x1 = math.floor((center_x + radius) * inverse_cell)
            y0 = math.floor((center_y - radius) * inverse_cell)
            y1 = math.floor((center_y + radius) * inverse_cell)
            z0 = math.floor((center_z - radius) * inverse_cell)
            z1 = math.floor((center_z + radius) * inverse_cell)
            for cell_x in range(x0, x1 + 1):
                for cell_y in range(y0, y1 + 1):
                    for cell_z in range(z0, z1 + 1):
                        keys.add((cell_x, cell_y, cell_z))
        # Pair generation is sorted after hash lookup, so key order cannot
        # affect physics.  Avoiding a large tuple sort materially cuts worker
        # latency for long high-speed sweeps.
        results.append((index, tuple(keys)))
    return results


def _quaternion_rotate_components(
    qw: float, qx: float, qy: float, qz: float,
    vx: float, vy: float, vz: float,
) -> Tuple[float, float, float]:
    """Allocation-free unit-quaternion vector rotation for worker geometry."""
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _static_box_batch_payload(
    task: BoxBatchTask,
) -> Tuple[int, bytes, int]:
    """Pure worker kernel: bake one course into C4F/N3F/V3F bytes."""
    chunk_index, states = task
    output = array("f")
    triangle_order = (0, 1, 2, 0, 2, 3)
    for state in states:
        (
            dimension_x, dimension_y, dimension_z,
            color_r, color_g, color_b,
            position_x, position_y, position_z,
            qw, qx, qy, qz,
        ) = state
        for normal, corners in UNIT_BOX_FACES:
            normal_x, normal_y, normal_z = _quaternion_rotate_components(
                qw, qx, qy, qz, normal[0], normal[1], normal[2]
            )
            for corner_index in triangle_order:
                corner = corners[corner_index]
                local_x = corner[0] * dimension_x
                local_y = corner[1] * dimension_y
                local_z = corner[2] * dimension_z
                world_x, world_y, world_z = _quaternion_rotate_components(
                    qw, qx, qy, qz, local_x, local_y, local_z
                )
                output.extend((
                    color_r, color_g, color_b, 1.0,
                    normal_x, normal_y, normal_z,
                    position_x + world_x,
                    position_y + world_y,
                    position_z + world_z,
                ))
    return chunk_index, output.tobytes(), len(output) // 10


class SpatialHashWorkerPool:
    """Persistent processes for pure, deterministic broadphase work.

    pygame event handling, OpenGL calls, state mutation, and the ordered
    sequential-impulse solver stay on the main process.  Workers receive only
    compact numeric snapshots and return cell sets which the main process
    merges before its sorted pair pass, so scheduling cannot alter the result.
    """

    def __init__(self, worker_cores: Sequence[int] = ()) -> None:
        self.worker_cores = tuple(int(core) for core in worker_cores)
        self.worker_count = len(self.worker_cores)
        self.failure = ""
        self.parallel_dispatches = 0
        self.serial_fallbacks = 0
        self.render_dispatches = 0
        self.render_batches = 0
        self.last_render_build_ms = 0.0
        self.last_work_estimate = 0
        self.last_mode = "serial"
        self._pool: Any = None
        if self.worker_count <= 0:
            return
        try:
            context = multiprocessing.get_context("spawn")
            self._pool = context.Pool(
                processes=self.worker_count,
                initializer=_initialize_spatial_worker,
                initargs=(self.worker_cores,),
            )
        except Exception as error:  # Spawn/profiler failures degrade safely.
            self.failure = str(error)
            self.worker_count = 0

    @property
    def available(self) -> bool:
        return self._pool is not None and not self.failure

    @property
    def status_label(self) -> str:
        if not self.worker_count:
            return "CPU serial fallback"
        cores = "/".join(str(core) for core in self.worker_cores)
        state = "FAULT" if self.failure else self.last_mode.upper()
        return f"{self.worker_count} spatial workers cores {cores} {state}"

    def submit(self, states: Sequence[SpatialState]) -> Any:
        if not self.available or len(states) < 2:
            self.last_mode = "serial"
            self.serial_fallbacks += 1
            return None
        weighted = sorted(
            ((_spatial_state_work(state), state) for state in states),
            key=lambda entry: (-entry[0], entry[1][0]),
        )
        self.last_work_estimate = sum(weight for weight, _state in weighted)
        if self.last_work_estimate < PARALLEL_SPATIAL_WORK_MIN:
            self.last_mode = "serial"
            self.serial_fallbacks += 1
            return None
        active_worker_count = min(self.worker_count, len(weighted))
        if active_worker_count < 2:
            self.last_mode = "serial"
            self.serial_fallbacks += 1
            return None
        chunks: List[List[SpatialState]] = [list() for _ in range(active_worker_count)]
        loads = [0] * active_worker_count
        for weight, state in weighted:
            target = min(
                range(active_worker_count),
                key=lambda slot: (loads[slot], slot),
            )
            chunks[target].append(state)
            loads[target] += weight
        chunks = [chunk for chunk in chunks if chunk]
        try:
            job = self._pool.map_async(_spatial_cell_chunk, chunks)
            self.last_mode = "parallel"
            self.parallel_dispatches += 1
            return job
        except (OSError, RuntimeError, ValueError) as error:
            self.failure = str(error)
            return None

    def collect(
        self, job: Any,
    ) -> Optional[Dict[int, Tuple[Tuple[int, int, int], ...]]]:
        if job is None:
            return None
        try:
            batches = job.get()
        except Exception as error:  # Worker failure must degrade to serial physics.
            self.failure = str(error)
            return None
        merged: Dict[int, Tuple[Tuple[int, int, int], ...]] = {}
        for batch in batches:
            for index, keys in batch:
                merged[index] = keys
        return merged

    def build_render_batches(
        self, tasks: Sequence[BoxBatchTask],
    ) -> Optional[Dict[int, Tuple[bytes, int]]]:
        """Use all helpers for two or more dirty immutable render courses."""
        if not self.available or len(tasks) < 2:
            return None
        started = time.perf_counter()
        try:
            results = self._pool.map(_static_box_batch_payload, list(tasks), chunksize=1)
        except Exception as error:
            self.failure = str(error)
            return None
        self.last_render_build_ms = (time.perf_counter() - started) * 1000.0
        self.render_dispatches += 1
        self.render_batches += len(results)
        return {
            chunk_index: (payload, vertex_count)
            for chunk_index, payload, vertex_count in results
        }

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        try:
            if self.failure:
                pool.terminate()
            else:
                pool.close()
            pool.join()
        except (OSError, RuntimeError, ValueError):
            pool.terminate()
            pool.join()


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def approach(value: float, target: float, amount: float) -> float:
    if value < target:
        return min(target, value + amount)
    return max(target, value - amount)


def is_hardware_gl_renderer(renderer: str) -> bool:
    normalized = str(renderer).strip().lower()
    return bool(normalized) and not any(
        marker in normalized for marker in SOFTWARE_GL_MARKERS
    )


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
    category: str = "calibration"
    render_kind: str = "default"
    buoyancy_volume: float = 0.0
    added_mass: float = 0.0
    adhesion_strength: float = 0.0
    linear_damping: float = 0.0

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
        inertial_mass = self.mass + self.added_mass
        if self.shape == "sphere":
            factor = 2.0 / 3.0 if self.inertia_mode == "thin_shell" else 2.0 / 5.0
            inertia = factor * inertial_mass * self.radius * self.radius
            return Vec3(inertia, inertia, inertia)
        width, height, depth = self.dimensions
        return Vec3(
            inertial_mass * (height * height + depth * depth) / 12.0,
            inertial_mass * (width * width + depth * depth) / 12.0,
            inertial_mass * (width * width + height * height) / 12.0,
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
    base_specs = (
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

    # Expansion calibration basis (accessed 2026-07-11):
    # Rawlings 34 in / 31 oz ash bat, True Temper R625 wheelbarrow,
    # 3 ft helium chart, PE-foam noodle, 8 fl oz PVA slime, 50 mm alumina
    # ceramic media, Belden modular brick, EPAL 1 pallet, IKEA 15 in plush.
    # https://www.rawlings.com/product/R271V-34%2F31.html
    # https://www.homebyames.com/en-us/lawn-garden/wheelbarrows-carts/steel-wheelbarrows/R6STPTEC.html
    # https://www.acmetools.com/true-temper-6-cu-ft-steel-wheelbarrow-with-single-tubed-tire-r625/049206111103.html
    # https://www.burtonandburton.com/_images/Helium_Balloon_Chart.pdf
    # https://www.andymark.com/products/solid-pool-noodle
    # https://www.elmers.com/slime-glue/premade-slime/elmers-gue-glassy-clear-premade-slime/SP_2752252.html
    # https://www.magotteaux.com/wp-content/uploads/2023/01/magotteaux_ceramic_balls_datasheet_EN_LR-1.pdf
    # https://www.beldenbrick.com/resources/brick-dimensions-guide
    # https://www.epal-pallets.org/eu-en/load-carriers/epal-euro-pallet
    # https://www.ikea.com/us/en/p/grejsimojs-soft-toy-bear-off-white-60607053/
    expansion_specs = (
        BodySpec(
            "wood_bat", "Ash Wooden Baseball Bat", "box", 0.8788,
            dimensions=(0.8636, 0.0663, 0.0663), density=678.0,
            restitution=0.32, friction=0.55, rolling_resistance=0.010,
            drag_coefficient=1.20, sound_family="wood_bat", rigidity="hard tapered ash",
            color=(0.73, 0.46, 0.20), category="bat", render_kind="bat",
        ),
        BodySpec(
            "wheelbarrow", "Single-Wheel Steel Wheelbarrow", "box", 17.28,
            dimensions=(0.648, 0.686, 1.492), density=0.0,
            restitution=0.12, friction=0.62, rolling_resistance=0.018,
            drag_coefficient=1.15, sound_family="wheelbarrow",
            rigidity="steel tray, hardwood handles, pneumatic tire",
            color=(0.22, 0.48, 0.32), category="wheelbarrow", render_kind="wheelbarrow",
        ),
        BodySpec(
            "helium_balloon", "Three-Foot Helium Latex Balloon", "sphere", 0.183,
            diameter=0.914, density=0.431, restitution=0.55, friction=0.80,
            rolling_resistance=0.0, drag_coefficient=0.50, inertia_mode="thin_shell",
            sound_family="balloon", rigidity="compliant inflated latex",
            color=(0.96, 0.35, 0.78), category="balloon", render_kind="balloon",
            buoyancy_volume=0.4248, added_mass=0.261, linear_damping=0.04,
        ),
        BodySpec(
            "foam_noodle", "Closed-Cell PE Foam Noodle", "box", 0.1134,
            dimensions=(1.1938, 0.0635, 0.0635), density=30.0,
            restitution=0.25, friction=0.65, rolling_resistance=0.060,
            drag_coefficient=1.20, sound_family="foam_noodle",
            rigidity="flexible PE foam (damped rigid proxy)",
            color=(0.22, 0.88, 0.92), category="noodles", render_kind="noodle",
            linear_damping=0.11,
        ),
        BodySpec(
            "sticky_goo", "Sticky PVA Goo Blob", "sphere", 0.248,
            diameter=0.0767, density=1_050.0, restitution=0.02,
            friction=1.10, rolling_resistance=0.20, drag_coefficient=0.60,
            sound_family="goo", rigidity="viscoelastic adhesive hydrogel",
            color=(0.22, 0.96, 0.36), category="goo", render_kind="goo",
            adhesion_strength=25.0, linear_damping=1.80,
        ),
        BodySpec(
            "ceramic_marble", "Oversized Alumina Ceramic Marble", "sphere", 0.2356,
            diameter=0.050, density=3_600.0, restitution=0.55,
            friction=0.28, rolling_resistance=0.003, drag_coefficient=0.47,
            sound_family="ceramic", rigidity="hard 92% alumina ceramic",
            color=(0.94, 0.94, 0.89), category="marbles", render_kind="ceramic",
        ),
        BodySpec(
            "clay_brick", "Modular Fired-Clay Brick", "box", 1.90,
            dimensions=(0.194, 0.057, 0.092), density=1_868.0,
            restitution=0.12, friction=0.68, rolling_resistance=0.0,
            drag_coefficient=1.05, sound_family="clay_brick",
            rigidity="rigid brittle fired clay", color=(0.57, 0.19, 0.09),
            category="bricks", render_kind="clay_brick",
        ),
        BodySpec(
            "wood_pallet", "EPAL 1 Wooden Pallet", "box", 25.0,
            dimensions=(1.200, 0.144, 0.800), density=181.0,
            restitution=0.10, friction=0.62, rolling_resistance=0.0,
            drag_coefficient=1.20, sound_family="pallet",
            rigidity="spruce/pine pallet rated 1,500 kg",
            color=(0.52, 0.33, 0.16), category="pallet", render_kind="pallet",
        ),
    )
    # The bear follows the cited 15-inch product; the other species are
    # deliberately representative soft-toy variants randomized per reset.
    plush_data = (
        ("bear", "Bear", 0.381, 0.2835, (0.72, 0.54, 0.36)),
        ("rabbit", "Rabbit", 0.410, 0.310, (0.91, 0.86, 0.78)),
        ("fox", "Fox", 0.390, 0.295, (0.92, 0.39, 0.16)),
        ("penguin", "Penguin", 0.360, 0.270, (0.18, 0.22, 0.29)),
        ("dinosaur", "Dinosaur", 0.460, 0.345, (0.28, 0.67, 0.36)),
        ("octopus", "Octopus", 0.350, 0.255, (0.68, 0.35, 0.86)),
        ("axolotl", "Axolotl", 0.400, 0.285, (0.98, 0.55, 0.66)),
        ("elephant", "Elephant", 0.430, 0.335, (0.55, 0.61, 0.68)),
        ("raccoon", "Raccoon", 0.370, 0.280, (0.42, 0.45, 0.49)),
        ("sloth", "Sloth", 0.420, 0.320, (0.48, 0.34, 0.25)),
    )
    plush_specs = tuple(
        BodySpec(
            f"plush_{key}", f"Stuffed {label}", "sphere", mass,
            diameter=diameter, density=mass / (4.0 * math.pi * (diameter * 0.5) ** 3 / 3.0),
            restitution=0.08, friction=0.75, rolling_resistance=0.12,
            drag_coefficient=1.10, sound_family="plush",
            rigidity="soft polyester shell and hollow-fiber fill", color=color,
            category="plush", render_kind="plush", linear_damping=0.42,
        )
        for key, label, diameter, mass, color in plush_data
    )
    return base_specs + expansion_specs + plush_specs


ITEM_SPECS = make_item_specs()
ITEM_SPEC_BY_KEY = {spec.key: spec for spec in ITEM_SPECS}
ORIGINAL_ITEM_KEYS = (
    "dodge", "medicine_1", "medicine_3", "medicine_10",
    "steel", "concrete", "rubber_brick",
)
PLUSH_KEYS = tuple(spec.key for spec in ITEM_SPECS if spec.category == "plush")
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
    instance_label: str = ""
    group_index: int = 0
    stuck_surface: bool = False
    stuck_normal: Vec3 = field(default_factory=Vec3)
    stuck_to: Optional["RigidBody"] = None
    stuck_local_position: Vec3 = field(default_factory=Vec3)
    pristine: bool = True
    wheel_angle: float = 0.0
    color_override: Optional[Tuple[float, float, float]] = None
    attached_payload_mass: float = 0.0
    mass_carried_by_host: bool = False
    _cached_bounding_radius: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.previous_position = self.position.copy()
        self.previous_orientation = Quat(
            self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z
        )
        if self.spec.shape == "sphere":
            self._cached_bounding_radius = self.spec.radius
        else:
            half = self.spec.half_extents
            self._cached_bounding_radius = math.sqrt(
                half.x * half.x + half.y * half.y + half.z * half.z
            )

    @property
    def inv_mass(self) -> float:
        mass = self.dynamic_mass
        return 0.0 if mass <= 1.0e-12 else 1.0 / mass

    @property
    def base_dynamic_mass(self) -> float:
        return self.spec.mass + self.spec.added_mass

    @property
    def dynamic_mass(self) -> float:
        if self.mass_carried_by_host:
            return 0.0
        return self.base_dynamic_mass + self.attached_payload_mass

    @property
    def display_name(self) -> str:
        return self.instance_label or self.spec.name

    @property
    def bounding_radius(self) -> float:
        return self._cached_bounding_radius

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

    def wake(self, mark_dirty: bool = True) -> None:
        self.asleep = False
        self.sleep_time = 0.0
        if mark_dirty:
            self.pristine = False


def box_render_state(body: RigidBody) -> BoxRenderState:
    dimensions = body.spec.dimensions
    color = body.color_override or body.spec.color
    return (
        dimensions[0], dimensions[1], dimensions[2],
        color[0], color[1], color[2],
        body.position.x, body.position.y, body.position.z,
        body.orientation.w, body.orientation.x,
        body.orientation.y, body.orientation.z,
    )


def build_static_box_batch_vertices(bodies: Sequence[RigidBody]) -> array:
    """Bake world-space C4F/N3F/V3F triangles for immutable box bodies.

    A 48-brick course becomes one roughly 62 KiB static buffer and one draw.
    Physics remains authoritative; a course is rebuilt whenever its sleeping
    membership or transform signature changes.
    """
    _chunk_index, payload, _vertex_count = _static_box_batch_payload((
        0, tuple(box_render_state(body) for body in bodies)
    ))
    vertices = array("f")
    vertices.frombytes(payload)
    return vertices


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
    """Deterministic 551-body Newtonian room simulation in SI units."""

    def __init__(
        self,
        seed: int = 1337,
        spatial_workers: Optional[SpatialHashWorkerPool] = None,
        cpu_backend: str = "auto",
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.spatial_workers = spatial_workers
        requested_backend = str(cpu_backend).lower()
        self.cpu_backend = (
            "neon" if requested_backend != "scalar" and np is not None
            else "scalar"
        )
        self.last_broadphase_backend = self.cpu_backend.upper()
        self.last_broadphase_ms = 0.0
        self._soa_owner = 0
        self._soa_positions: Any = None
        self._soa_previous: Any = None
        self._soa_radii: Any = None
        self._soa_active: Any = None
        self.gravity = EARTH_GRAVITY
        self.room_friction = DEFAULT_ROOM_FRICTION
        self.throw_force = THROW_FORCE_MIN
        self.player = Player()
        self.bodies: List[RigidBody] = []
        self._wheelbarrow_index: Optional[int] = None
        self._cargo_candidate_indices: Tuple[int, ...] = ()
        self._cargo_cache_owner = 0
        self.held_body: Optional[RigidBody] = None
        self.impacts: List[ImpactEvent] = []
        self.messages: List[str] = []
        self.simulation_time = 0.0
        self.last_broadphase_candidates = 0
        self.last_active_contacts = 0
        self.last_solver_iterations = SOLVER_ITERATIONS
        self._static_cells: Dict[Tuple[int, int, int], set[int]] = {}
        self._static_body_cells: Dict[int, Tuple[Tuple[int, int, int], ...]] = {}
        self._static_signature: Tuple[Tuple[float, ...], ...] = ()
        self._solver_static_ids: set[int] = set()
        self._pending_wakes: Dict[int, Tuple[RigidBody, Vec3, Vec3, float]] = {}
        self.reset()

    def reset(self) -> None:
        self.player = Player()
        self.bodies = []
        self._pending_wakes = {}

        def spawn(
            spec: BodySpec,
            position: Vec3,
            label: str = "",
            group_index: int = 0,
            asleep: bool = False,
            grounded: bool = False,
            orientation: Optional[Quat] = None,
            color: Optional[Tuple[float, float, float]] = None,
        ) -> RigidBody:
            body = RigidBody(
                spec, position,
                orientation=orientation if orientation is not None else Quat(),
                asleep=asleep, grounded=grounded, instance_label=label,
                group_index=group_index, color_override=color,
            )
            self.bodies.append(body)
            return body

        for index, key in enumerate(ORIGINAL_ITEM_KEYS):
            spec = ITEM_SPEC_BY_KEY[key]
            x = 7.0 + index
            if spec.shape == "sphere":
                y = spec.radius
            else:
                y = spec.dimensions[1] * 0.5
            spawn(spec, Vec3(x, y, 15.0), grounded=True)

        bat = ITEM_SPEC_BY_KEY["wood_bat"]
        spawn(bat, Vec3(15.0, bat.dimensions[1] * 0.5, 15.5), grounded=True)

        barrow = ITEM_SPEC_BY_KEY["wheelbarrow"]
        spawn(
            barrow, Vec3(18.5, barrow.dimensions[1] * 0.5, 18.0),
            asleep=True, grounded=True,
        )

        balloon = ITEM_SPEC_BY_KEY["helium_balloon"]
        spawn(balloon, Vec3(21.5, 2.7, 17.0))

        noodle_colors = (
            (0.18, 0.90, 0.96), (0.98, 0.30, 0.35), (0.98, 0.82, 0.18),
            (0.44, 0.95, 0.32), (0.74, 0.38, 0.96),
        )
        noodle = ITEM_SPEC_BY_KEY["foam_noodle"]
        for index in range(NOODLE_COUNT):
            spawn(
                noodle,
                Vec3(24.5, noodle.dimensions[1] * 0.5, 15.0 + index * 0.24),
                f"Foam Noodle {index + 1}/{NOODLE_COUNT}", index + 1,
                asleep=True, grounded=True, color=noodle_colors[index],
            )

        goo_colors = ((0.20, 0.98, 0.35), (0.20, 0.78, 1.0), (0.92, 0.30, 0.98))
        goo = ITEM_SPEC_BY_KEY["sticky_goo"]
        for index in range(GOO_COUNT):
            spawn(
                goo,
                Vec3(27.0 + (index % 5) * 0.14, goo.radius, 15.0 + (index // 5) * 0.16),
                f"Sticky Goo {index + 1}/{GOO_COUNT}", index + 1,
                asleep=True, grounded=True, color=goo_colors[index % len(goo_colors)],
            )

        ceramic_colors = ((0.96, 0.95, 0.89), (0.54, 0.78, 0.98), (0.98, 0.70, 0.35))
        marble = ITEM_SPEC_BY_KEY["ceramic_marble"]
        for index in range(CERAMIC_MARBLE_COUNT):
            spawn(
                marble,
                Vec3(30.0 + (index % 5) * 0.075, marble.radius, 15.0 + (index // 5) * 0.075),
                f"Ceramic Marble {index + 1}/{CERAMIC_MARBLE_COUNT}", index + 1,
                asleep=True, grounded=True, color=ceramic_colors[index % len(ceramic_colors)],
            )

        pallet = ITEM_SPEC_BY_KEY["wood_pallet"]
        pallet_center = Vec3(50.0, pallet.dimensions[1] * 0.5, 28.0)
        spawn(pallet, pallet_center, asleep=True, grounded=True)

        brick = ITEM_SPEC_BY_KEY["clay_brick"]
        brick_number = 0
        pallet_top = pallet.dimensions[1]
        layer_pitch = brick.dimensions[1] + 0.001
        yaw_quarter = Quat(math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0)
        for layer in range(10):
            odd = bool(layer & 1)
            across, deep = (12, 4) if odd else (6, 8)
            x_pitch = (brick.dimensions[2] if odd else brick.dimensions[0]) + 0.001
            z_pitch = (brick.dimensions[0] if odd else brick.dimensions[2]) + 0.001
            orientation = yaw_quarter if odd else Quat()
            for row in range(deep):
                for column in range(across):
                    brick_number += 1
                    spawn(
                        brick,
                        Vec3(
                            pallet_center.x + (column - (across - 1) * 0.5) * x_pitch,
                            pallet_top + brick.dimensions[1] * 0.5 + layer * layer_pitch,
                            pallet_center.z + (row - (deep - 1) * 0.5) * z_pitch,
                        ),
                        f"Clay Brick {brick_number:03d}/{CLAY_BRICK_COUNT}", brick_number,
                        asleep=True, grounded=True, orientation=orientation,
                        color=(0.50 + 0.018 * ((brick_number * 7) % 5), 0.15, 0.065),
                    )
        # Twenty bricks crown the eleventh course: 5 x 4, centered.
        for row in range(4):
            for column in range(5):
                brick_number += 1
                spawn(
                    brick,
                    Vec3(
                        pallet_center.x + (column - 2.0) * (brick.dimensions[0] + 0.001),
                        pallet_top + brick.dimensions[1] * 0.5 + 10 * layer_pitch,
                        pallet_center.z + (row - 1.5) * (brick.dimensions[2] + 0.001),
                    ),
                    f"Clay Brick {brick_number:03d}/{CLAY_BRICK_COUNT}", brick_number,
                    asleep=True, grounded=True,
                    color=(0.50 + 0.018 * ((brick_number * 7) % 5), 0.15, 0.065),
                )

        selected_plush = self.rng.sample(list(PLUSH_KEYS), STUFFED_ANIMAL_COUNT)
        for index, key in enumerate(selected_plush):
            spec = ITEM_SPEC_BY_KEY[key]
            spawn(
                spec, Vec3(35.0 + index * 0.62, spec.radius, 16.0),
                f"{spec.name} {index + 1}/{STUFFED_ANIMAL_COUNT}", index + 1,
                asleep=True, grounded=True,
            )

        if brick_number != CLAY_BRICK_COUNT or len(self.bodies) != EXPECTED_BODY_COUNT:
            raise AssertionError(
                f"scene population mismatch: bricks={brick_number}, bodies={len(self.bodies)}"
            )
        self._refresh_wheelbarrow_cache()
        self._rebuild_broadphase_soa()
        self._rebuild_static_broadphase_index()
        self.held_body = None
        self.impacts.clear()
        self.messages = [
            "Room reset: 551 bodies, including 500 palletized clay bricks, are ready."
        ]
        self.simulation_time = 0.0
        self.last_broadphase_candidates = 0
        self.last_active_contacts = 0
        self.last_solver_iterations = SOLVER_ITERATIONS

    def wake_all(self) -> None:
        brick_slots = max(
            0,
            MAX_ACTIVE_BRICKS - sum(
                body.spec.key == "clay_brick" and not body.asleep for body in self.bodies
            ),
        )
        for body in self.bodies:
            if body.pristine and body.asleep and body.spec.key in {"clay_brick", "wood_pallet"}:
                continue
            if body.spec.key == "clay_brick" and body.asleep:
                if brick_slots <= 0:
                    continue
                brick_slots -= 1
            body.wake()

    def wake_nearby(self, position: Vec3, radius: float = 0.34, limit: int = 16) -> None:
        radius_squared = radius * radius
        nearby = sorted(
            (
                ((body.position - position).length_squared(), body)
                for body in self.bodies
                if (body.position - position).length_squared() <= radius_squared
            ),
            key=lambda pair: pair[0],
        )
        for _distance, body in nearby[:max(1, limit)]:
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
            name = self.held_body.display_name
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
        body.stuck_surface = False
        if body.stuck_to is not None:
            self._detach_goo_from_body(body)
        body.wake()
        if body.spec.key in {"clay_brick", "wood_pallet"}:
            # A local breakaway cluster gives the picked course room to move
            # without activating hundreds of densely packed bricks at once.
            self.wake_nearby(body.position, 0.16, MAX_ACTIVE_BRICKS)
        self.held_body = body
        self.messages.append(
            f"Holding {body.display_name}: {body.spec.mass:.3f} kg, {body.spec.dimension_label}."
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
        reduced_mass = body.dynamic_mass * PLAYER_MASS / (body.dynamic_mass + PLAYER_MASS)
        impulse_magnitude = math.sqrt(2.0 * self.throw_force * THROW_STROKE * reduced_mass)
        launch_speed = impulse_magnitude / body.dynamic_mass
        relative_release_speed = impulse_magnitude / reduced_mass
        impulse = direction * impulse_magnitude
        body.held = False
        body.velocity = self.player.velocity.copy()
        body.apply_impulse(impulse, body.position)
        # Newton's third law: the player receives the exact opposite impulse.
        self.player.velocity = self.player.velocity - impulse / PLAYER_MASS
        self.held_body = None
        self.messages.append(
            f"Threw {body.display_name}: {self.throw_force:,.0f} N across {THROW_STROKE:.2f} m, "
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
        if body.spec.key == "wheelbarrow":
            forward = self.player.forward(False).normalized(Vec3(0.0, 0.0, 1.0))
            target = self.player.position + forward * 1.38
            target.y = body.spec.dimensions[1] * 0.5
            error = target - body.position
            relative_velocity = body.velocity - self.player.velocity
            desired_acceleration = error * 25.0 - relative_velocity * 10.0
            desired_acceleration.y = 0.0
            grip_force = desired_acceleration * body.dynamic_mass
            magnitude = grip_force.length()
            if magnitude > 600.0:
                grip_force = grip_force * (600.0 / magnitude)
            body.apply_force(grip_force)
            current_forward = body.orientation.rotate(Vec3(0.0, 0.0, 1.0)).horizontal().normalized(Vec3(0.0, 0.0, 1.0))
            yaw_error = current_forward.cross(forward).y
            body.torque.y += (yaw_error * 420.0 - body.angular_velocity.y * 58.0)
            self.player.velocity = self.player.velocity - grip_force * (FIXED_DT / PLAYER_MASS)
            return
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
        grip_force = desired_acceleration * body.dynamic_mass
        maximum_grip = max(2_500.0, body.dynamic_mass * 250.0)
        magnitude = grip_force.length()
        if magnitude > maximum_grip:
            grip_force = grip_force * (maximum_grip / magnitude)
        body.apply_force(grip_force)
        # The hand constraint reacts on the player, preserving system momentum.
        self.player.velocity = self.player.velocity - grip_force * (FIXED_DT / PLAYER_MASS)

    def _integrate_body_forces(self, body: RigidBody, dt: float) -> None:
        if body.stuck_surface:
            separating_force = max(0.0, body.force.dot(body.stuck_normal))
            if separating_force <= body.spec.adhesion_strength:
                body.force = Vec3()
                body.torque = Vec3()
                return
            body.stuck_surface = False
            body.asleep = False
            self.messages.append(
                f"{body.display_name} peeled free at {separating_force:.1f} N "
                f"(adhesion {body.spec.adhesion_strength:.1f} N)."
            )
        if body.stuck_to is not None:
            separating_force = body.force.length()
            if separating_force <= body.spec.adhesion_strength:
                body.force = Vec3()
                body.torque = Vec3()
                return
            host_name = body.stuck_to.display_name
            self._detach_goo_from_body(body)
            body.asleep = False
            self.messages.append(
                f"{body.display_name} pulled free from {host_name} at "
                f"{separating_force:.1f} N."
            )
        if body.asleep and not body.held:
            body.force = Vec3()
            body.torque = Vec3()
            return
        body.previous_position = body.position.copy()
        body.previous_orientation = Quat(
            body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z
        )
        gravity_force = Vec3(0.0, -body.spec.mass * self.gravity, 0.0)
        buoyancy_force = Vec3(
            0.0, AIR_DENSITY * body.spec.buoyancy_volume * self.gravity, 0.0
        )
        damping_force = body.velocity * (-body.spec.linear_damping * body.dynamic_mass)
        body.force = body.force + gravity_force + buoyancy_force + self._body_drag(body) + damping_force
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

    def _update_attachments(self) -> None:
        for body in self.bodies:
            host = body.stuck_to
            if host is None:
                continue
            body.previous_position = body.position.copy()
            target = host.position + host.orientation.rotate(body.stuck_local_position)
            body.position = target
            body.velocity = host.velocity_at(target)
            body.angular_velocity = host.angular_velocity.copy()
            body.asleep = host.asleep

    @staticmethod
    def _detach_goo_from_body(goo: RigidBody) -> Optional[RigidBody]:
        host = goo.stuck_to
        if host is None:
            return None
        transferred_mass = goo.base_dynamic_mass + goo.attached_payload_mass
        host.attached_payload_mass = max(
            0.0, host.attached_payload_mass - transferred_mass
        )
        goo.mass_carried_by_host = False
        goo.stuck_to = None
        return host

    def _attach_goo_to_body(self, goo: RigidBody, host: RigidBody) -> None:
        if goo.held or host is goo or goo.stuck_to is not None or goo.stuck_surface:
            return
        goo_mass = goo.dynamic_mass
        host_mass = host.dynamic_mass
        combined_mass = host_mass + goo_mass
        if combined_mass > 1.0e-12:
            host.velocity = (
                host.velocity * host_mass + goo.velocity * goo_mass
            ) / combined_mass
        host.attached_payload_mass += goo_mass
        goo.stuck_to = host
        goo.mass_carried_by_host = True
        goo.stuck_local_position = host.orientation.conjugate().rotate(goo.position - host.position)
        goo.velocity = host.velocity_at(goo.position)
        goo.angular_velocity = Vec3()
        goo.pristine = False
        self.messages.append(f"{goo.display_name} adhered to {host.display_name}.")

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
                ImpactEvent(body.spec.sound_family, body.position.copy(), normal_impulse_magnitude, -normal_speed, body.dynamic_mass)
            )
            body.impact_cooldown = 0.075
        if body.spec.adhesion_strength > 0.0 and -normal_speed > 0.35 and not body.held:
            body.stuck_surface = True
            body.stuck_normal = normal.copy()
            body.velocity = Vec3()
            body.angular_velocity = Vec3()
            body.asleep = True
            body.pristine = False
            self.messages.append(f"{body.display_name} splatted and adhered to the room surface.")
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
        if body.asleep or body.stuck_surface or body.stuck_to is not None:
            return
        remaining = dt
        body.grounded = False
        self._project_body_inside_room(body)
        positive_x = Vec3(1.0, 0.0, 0.0)
        positive_y = Vec3(0.0, 1.0, 0.0)
        positive_z = Vec3(0.0, 0.0, 1.0)
        extent_x = body.support_extent(positive_x)
        extent_y = body.support_extent(positive_y)
        extent_z = body.support_extent(positive_z)
        for _bounce in range(MAX_CCD_BOUNCES):
            if remaining <= 1.0e-8:
                break
            velocity = body.velocity
            earliest = remaining + 1.0
            hit_normal: Optional[Vec3] = None
            hit_axis = ""
            candidates = (
                (velocity.x, extent_x, body.position.x, positive_x, "xmin", -1.0),
                (velocity.x, ROOM_WIDTH - extent_x, body.position.x, -positive_x, "xmax", 1.0),
                (velocity.y, extent_y, body.position.y, positive_y, "floor", -1.0),
                (velocity.y, ROOM_HEIGHT - extent_y, body.position.y, -positive_y, "ceiling", 1.0),
                (velocity.z, extent_z, body.position.z, positive_z, "zmin", -1.0),
                (velocity.z, ROOM_LENGTH - extent_z, body.position.z, -positive_z, "zmax", 1.0),
            )
            for component, boundary, coordinate, normal, axis_name, toward_sign in candidates:
                if component * toward_sign <= 1.0e-12:
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

        if remaining > 1.0e-8:
            # Repeated zero-time resting contacts can exhaust the bounce budget
            # for a wide box. Preserve its tangential motion, then project it
            # safely back against the room planes.
            body.position = body.position + body.velocity * remaining
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

    def _solver_inv_mass(self, body: RigidBody) -> float:
        """Sleeping bodies are immovable for one solver tick, then wake.

        Deferring the wake prevents one impact from propagating through every
        course during the eight iterations of a single 1/120-second step.
        """
        return 0.0 if id(body) in self._solver_static_ids else body.inv_mass

    def _solver_inverse_inertia(self, body: RigidBody, vector: Vec3) -> Vec3:
        if id(body) in self._solver_static_ids:
            return Vec3()
        return body.inverse_inertia_world(vector)

    def _apply_solver_impulse(
        self,
        body: RigidBody,
        impulse: Vec3,
        point: Vec3,
        wake: bool,
        priority: float,
    ) -> None:
        if id(body) not in self._solver_static_ids:
            body.apply_impulse(impulse, point, wake=wake)
            return
        if not wake or impulse.length_squared() <= 1.0e-12:
            return
        angular_impulse = (point - body.position).cross(impulse)
        existing = self._pending_wakes.get(id(body))
        if existing is None:
            self._pending_wakes[id(body)] = (
                body, impulse.copy(), angular_impulse, max(priority, impulse.length())
            )
        else:
            queued_body, linear, angular, old_priority = existing
            self._pending_wakes[id(body)] = (
                queued_body,
                linear + impulse,
                angular + angular_impulse,
                max(old_priority, priority, impulse.length()),
            )

    def _drain_pending_wakes(self) -> None:
        """Admit the strongest queued breakaways to the bounded active pool."""
        active_bricks = sum(
            body.spec.key == "clay_brick" and not body.asleep for body in self.bodies
        )
        new_brick_wakes = 0
        for key, (body, linear, angular, priority) in sorted(
            tuple(self._pending_wakes.items()), key=lambda item: item[1][3], reverse=True
        ):
            is_brick = body.spec.key == "clay_brick"
            if body.asleep and is_brick:
                if (
                    active_bricks >= MAX_ACTIVE_BRICKS
                    or new_brick_wakes >= MAX_NEW_BRICK_WAKES_PER_TICK
                ):
                    continue
                active_bricks += 1
                new_brick_wakes += 1
            body.wake()
            body.velocity = body.velocity + linear * body.inv_mass
            body.angular_velocity = (
                body.angular_velocity + body.inverse_inertia_world(angular)
            )
            body.last_impulse = max(body.last_impulse, priority)
            del self._pending_wakes[key]

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
        inverse_mass_a = self._solver_inv_mass(a)
        inverse_mass_b = self._solver_inv_mass(b)
        angular_a = self._solver_inverse_inertia(a, ra_cross_n).cross(ra).dot(normal)
        angular_b = self._solver_inverse_inertia(b, rb_cross_n).cross(rb).dot(normal)
        denominator = inverse_mass_a + inverse_mass_b + max(0.0, angular_a) + max(0.0, angular_b)
        if denominator <= 1.0e-12:
            return 0.0
        magnitude = -(1.0 + restitution) * normal_speed / max(denominator, 1.0e-9)
        impulse = normal * magnitude
        meaningful_impact = -normal_speed > 0.20
        self._apply_solver_impulse(a, -impulse, contact, meaningful_impact, magnitude)
        self._apply_solver_impulse(b, impulse, contact, meaningful_impact, magnitude)

        post_relative = b.velocity_at(contact) - a.velocity_at(contact)
        tangent_velocity = post_relative - normal * post_relative.dot(normal)
        tangent_speed = tangent_velocity.length()
        if tangent_speed > 1.0e-8:
            tangent = tangent_velocity / tangent_speed
            ra_cross_t = ra.cross(tangent)
            rb_cross_t = rb.cross(tangent)
            tangent_denominator = (
                inverse_mass_a + inverse_mass_b
                + max(0.0, self._solver_inverse_inertia(a, ra_cross_t).cross(ra).dot(tangent))
                + max(0.0, self._solver_inverse_inertia(b, rb_cross_t).cross(rb).dot(tangent))
            )
            desired = tangent_speed / max(tangent_denominator, 1.0e-9)
            mu = math.sqrt(a.spec.friction * b.spec.friction)
            friction_impulse = tangent * (-min(desired, mu * magnitude))
            self._apply_solver_impulse(a, -friction_impulse, contact, meaningful_impact, magnitude)
            self._apply_solver_impulse(b, friction_impulse, contact, meaningful_impact, magnitude)

        louder = a if a.dynamic_mass <= b.dynamic_mass else b
        if louder.impact_cooldown <= 0.0 and -normal_speed > 0.55:
            self.impacts.append(
                ImpactEvent(louder.spec.sound_family, contact.copy(), magnitude, -normal_speed, louder.dynamic_mass)
            )
            louder.impact_cooldown = 0.075
        if -normal_speed > 0.35:
            if a.spec.adhesion_strength > 0.0:
                self._attach_goo_to_body(a, b)
            elif b.spec.adhesion_strength > 0.0:
                self._attach_goo_to_body(b, a)
        return magnitude

    @staticmethod
    def _box_axes(body: RigidBody) -> Tuple[Vec3, Vec3, Vec3]:
        return (
            body.orientation.rotate(Vec3(1.0, 0.0, 0.0)).normalized(Vec3(1.0, 0.0, 0.0)),
            body.orientation.rotate(Vec3(0.0, 1.0, 0.0)).normalized(Vec3(0.0, 1.0, 0.0)),
            body.orientation.rotate(Vec3(0.0, 0.0, 1.0)).normalized(Vec3(0.0, 0.0, 1.0)),
        )

    @classmethod
    def _box_box_contact(
        cls, a: RigidBody, b: RigidBody
    ) -> Optional[Tuple[Vec3, float, Vec3]]:
        """Fifteen-axis OBB SAT with a stable single-point manifold."""
        axes_a = cls._box_axes(a)
        axes_b = cls._box_axes(b)
        half_a = a.spec.half_extents
        half_b = b.spec.half_extents
        extents_a = (half_a.x, half_a.y, half_a.z)
        extents_b = (half_b.x, half_b.y, half_b.z)
        delta = b.position - a.position
        candidate_axes: List[Vec3] = list(axes_a) + list(axes_b)
        for axis_a in axes_a:
            for axis_b in axes_b:
                cross = axis_a.cross(axis_b)
                if cross.length_squared() > 1.0e-12:
                    candidate_axes.append(cross.normalized())
        minimum_overlap = float("inf")
        minimum_axis = Vec3(1.0, 0.0, 0.0)
        for axis in candidate_axes:
            radius_a = sum(extent * abs(axis.dot(box_axis)) for extent, box_axis in zip(extents_a, axes_a))
            radius_b = sum(extent * abs(axis.dot(box_axis)) for extent, box_axis in zip(extents_b, axes_b))
            signed_distance = delta.dot(axis)
            overlap = radius_a + radius_b - abs(signed_distance)
            if overlap <= 0.0:
                return None
            if overlap < minimum_overlap:
                minimum_overlap = overlap
                minimum_axis = axis if signed_distance >= 0.0 else -axis
        point_a = a.support_point(minimum_axis)
        point_b = b.support_point(-minimum_axis)
        contact = (point_a + point_b) * 0.5
        return minimum_axis, minimum_overlap, contact

    @classmethod
    def _swept_box_box_fraction(cls, a: RigidBody, b: RigidBody) -> Optional[float]:
        """Continuous SAT for the boxes' relative linear translation.

        Orientation is frozen at the integrated pose for this substep; angular
        sweep is conservatively covered by the broadphase padding. This exact
        translation interval is both faster and safer than a circumscribed-
        sphere hit followed by dozens of speculative SAT samples.
        """
        axes_a = cls._box_axes(a)
        axes_b = cls._box_axes(b)
        half_a, half_b = a.spec.half_extents, b.spec.half_extents
        extents_a = (half_a.x, half_a.y, half_a.z)
        extents_b = (half_b.x, half_b.y, half_b.z)
        axes: List[Vec3] = list(axes_a) + list(axes_b)
        for axis_a in axes_a:
            for axis_b in axes_b:
                cross = axis_a.cross(axis_b)
                if cross.length_squared() > 1.0e-12:
                    axes.append(cross.normalized())
        start_delta = b.previous_position - a.previous_position
        relative_motion = (
            (b.position - b.previous_position) - (a.position - a.previous_position)
        )
        entry, exit_fraction = 0.0, 1.0
        for axis in axes:
            radius = (
                sum(extent * abs(axis.dot(box_axis)) for extent, box_axis in zip(extents_a, axes_a))
                + sum(extent * abs(axis.dot(box_axis)) for extent, box_axis in zip(extents_b, axes_b))
            )
            start = start_delta.dot(axis)
            speed = relative_motion.dot(axis)
            if abs(speed) <= 1.0e-14:
                if abs(start) > radius:
                    return None
                continue
            first = (-radius - start) / speed
            second = (radius - start) / speed
            if first > second:
                first, second = second, first
            entry = max(entry, first)
            exit_fraction = min(exit_fraction, second)
            if entry > exit_fraction:
                return None
        if exit_fraction < 0.0 or entry > 1.0:
            return None
        return clamp(entry, 0.0, 1.0)

    def _resolve_box_box(
        self, a: RigidBody, b: RigidBody, dt: float
    ) -> Optional[Tuple[Vec3, Vec3]]:
        contact_data = self._box_box_contact(a, b)
        swept_fraction: Optional[float] = None
        if contact_data is None and dt > 0.0:
            swept_fraction = self._swept_box_box_fraction(a, b)
            if swept_fraction is not None:
                end_a, end_b = a.position.copy(), b.position.copy()
                motion_length = (
                    (end_a - a.previous_position) - (end_b - b.previous_position)
                ).length()
                contact_fraction = min(
                    1.0, swept_fraction + max(1.0e-7, 2.0e-6 / max(motion_length, 1.0e-9))
                )
                a.position = a.previous_position + (end_a - a.previous_position) * contact_fraction
                b.position = b.previous_position + (end_b - b.previous_position) * contact_fraction
                contact_data = self._box_box_contact(a, b)
                if contact_data is None:
                    a.position, b.position = end_a, end_b
                    return None
                swept_fraction = contact_fraction
        if contact_data is None:
            return None
        normal, penetration, contact = contact_data
        if penetration > 0.0:
            inverse_mass_a = self._solver_inv_mass(a)
            inverse_mass_b = self._solver_inv_mass(b)
            total_inverse_mass = inverse_mass_a + inverse_mass_b
            if total_inverse_mass > 1.0e-12:
                correction = normal * (
                    max(0.0, penetration - 1.0e-5) * 0.64 / total_inverse_mass
                )
                a.position = a.position - correction * inverse_mass_a
                b.position = b.position + correction * inverse_mass_b
        self._body_pair_impulse(a, b, normal, contact)
        if swept_fraction is not None and swept_fraction < 1.0:
            remaining = dt * (1.0 - swept_fraction)
            self._advance_body_swept(a, remaining)
            self._advance_body_swept(b, remaining)
        return normal, contact

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

    def _resolve_sphere_box(
        self, sphere: RigidBody, box: RigidBody, dt: float
    ) -> Optional[Tuple[Vec3, Vec3]]:
        contact_data = self._sphere_box_contact(sphere, box)
        swept_fraction: Optional[float] = None
        if contact_data is None:
            if dt <= 0.0:
                return
            swept_fraction = self._swept_sphere_box_fraction(sphere, box)
            if swept_fraction is None:
                return None
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
            inverse_mass_sphere = self._solver_inv_mass(sphere)
            inverse_mass_box = self._solver_inv_mass(box)
            total_inverse_mass = inverse_mass_sphere + inverse_mass_box
            if total_inverse_mass > 1.0e-12:
                correction = normal * (
                    max(0.0, penetration - 1.0e-5) * 0.66 / total_inverse_mass
                )
                sphere.position = sphere.position - correction * inverse_mass_sphere
                box.position = box.position + correction * inverse_mass_box
        self._body_pair_impulse(sphere, box, normal, contact)
        if swept_fraction is not None and swept_fraction < 1.0:
            remaining = dt * (1.0 - swept_fraction)
            self._advance_body_swept(sphere, remaining)
            self._advance_body_swept(box, remaining)
        return normal, contact

    def _resolve_body_pair(
        self, a: RigidBody, b: RigidBody, dt: float
    ) -> Optional[Tuple[Vec3, Vec3]]:
        if a.held and b.held:
            return None
        if a.stuck_surface or b.stuck_surface or a.stuck_to is not None or b.stuck_to is not None:
            return None
        if a.spec.shape == "sphere" and b.spec.shape == "box":
            return self._resolve_sphere_box(a, b, dt)
        if a.spec.shape == "box" and b.spec.shape == "sphere":
            result = self._resolve_sphere_box(b, a, dt)
            if result is None:
                return None
            normal, contact = result
            return -normal, contact
        if a.spec.shape == "box" and b.spec.shape == "box":
            return self._resolve_box_box(a, b, dt)
        radius_a = a.bounding_radius
        radius_b = b.bounding_radius
        delta = b.position - a.position
        distance_squared = delta.length_squared()
        radius_sum = radius_a + radius_b
        swept_fraction: Optional[float] = None
        if distance_squared > radius_sum * radius_sum:
            if dt <= 0.0:
                return None
            swept_fraction = self._sphere_sweep_fraction(a, b)
            if swept_fraction is None:
                return None
            a.position = a.previous_position + (a.position - a.previous_position) * swept_fraction
            b.position = b.previous_position + (b.position - b.previous_position) * swept_fraction
            delta = b.position - a.position
            distance_squared = delta.length_squared()

        distance = math.sqrt(max(distance_squared, 1.0e-16))
        normal = delta / distance if distance > 1.0e-8 else Vec3(1.0, 0.0, 0.0)
        penetration = radius_sum - distance
        if penetration > 0.0:
            inverse_mass_a = self._solver_inv_mass(a)
            inverse_mass_b = self._solver_inv_mass(b)
            total_inverse_mass = inverse_mass_a + inverse_mass_b
            if total_inverse_mass > 1.0e-12:
                correction = normal * (
                    max(0.0, penetration - 1.0e-5) * 0.62 / total_inverse_mass
                )
                a.position = a.position - correction * inverse_mass_a
                b.position = b.position + correction * inverse_mass_b
        contact = a.position + normal * radius_a
        self._body_pair_impulse(a, b, normal, contact)

        if swept_fraction is not None and swept_fraction < 1.0:
            remaining = dt * (1.0 - swept_fraction)
            self._advance_body_swept(a, remaining)
            self._advance_body_swept(b, remaining)
        return normal, contact

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
        inverse_mass = self._solver_inv_mass(body)
        angular = self._solver_inverse_inertia(body, lever_cross_normal).cross(lever).dot(normal)
        denominator = 1.0 / PLAYER_MASS + inverse_mass + max(0.0, angular)
        restitution = min(0.18, body.spec.restitution) if -normal_speed >= 0.6 else 0.0
        magnitude = -(1.0 + restitution) * normal_speed / max(denominator, 1.0e-9)
        impulse = normal * magnitude
        self._apply_solver_impulse(
            body, impulse, contact, -normal_speed > 0.20, magnitude
        )
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
            body_inverse_mass = self._solver_inv_mass(body)
            total_inverse_mass = 1.0 / PLAYER_MASS + body_inverse_mass
            correction = normal * (max(0.0, penetration - 1.0e-5) * 0.72 / total_inverse_mass)
            self.player.position = self.player.position - correction * (1.0 / PLAYER_MASS)
            body.position = body.position + correction * body_inverse_mass
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
        # A sleeping rigid body is immutable until apply_force/apply_impulse or
        # an explicit setting change wakes it.  Avoiding vector math for the
        # 500-brick pallet removes the largest steady-state 120 Hz hotspot.
        if body.asleep and not body.held:
            return
        if body.stuck_surface or body.stuck_to is not None:
            return
        if body.spec.key == "wheelbarrow":
            rolling_forward = body.orientation.rotate(Vec3(0.0, 0.0, 1.0)).horizontal().normalized(Vec3(0.0, 0.0, 1.0))
            body.wheel_angle = (
                body.wheel_angle + body.velocity.horizontal().dot(rolling_forward) * dt / 0.2032
            ) % (2.0 * math.pi)
        if body.grounded and not body.held:
            horizontal = body.velocity.horizontal()
            speed = horizontal.length()
            scale = self.room_friction / DEFAULT_ROOM_FRICTION if DEFAULT_ROOM_FRICTION else 1.0
            if body.spec.key == "wheelbarrow":
                forward = body.orientation.rotate(Vec3(0.0, 0.0, 1.0)).horizontal().normalized(Vec3(0.0, 0.0, 1.0))
                side = Vec3(forward.z, 0.0, -forward.x)
                forward_speed = horizontal.dot(forward)
                side_speed = horizontal.dot(side)
                forward_loss = body.spec.rolling_resistance * scale * self.gravity * dt
                side_loss = 0.90 * scale * self.gravity * dt
                forward_speed = math.copysign(max(0.0, abs(forward_speed) - forward_loss), forward_speed)
                side_speed = math.copysign(max(0.0, abs(side_speed) - side_loss), side_speed)
                body.velocity.x = forward.x * forward_speed + side.x * side_speed
                body.velocity.z = forward.z * forward_speed + side.z * side_speed
            elif speed > 1.0e-8:
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

    @staticmethod
    def _body_swept_aabb(body: RigidBody) -> Tuple[Vec3, Vec3]:
        def oriented_extent(orientation: Quat, axis: Vec3) -> float:
            if body.spec.shape == "sphere":
                return body.spec.radius
            half = body.spec.half_extents
            local_axis = orientation.conjugate().rotate(axis)
            return (
                abs(local_axis.x) * half.x
                + abs(local_axis.y) * half.y
                + abs(local_axis.z) * half.z
            )

        axes = (Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0))
        current_extents = tuple(oriented_extent(body.orientation, axis) for axis in axes)
        previous_extents = tuple(oriented_extent(body.previous_orientation, axis) for axis in axes)
        angular_padding = min(
            body.bounding_radius,
            body.angular_velocity.length() * body.bounding_radius * FIXED_DT,
        )
        ex, ey, ez = (
            max(current, previous) + BROADPHASE_SKIN + angular_padding
            for current, previous in zip(current_extents, previous_extents)
        )
        minimum = Vec3(
            min(body.previous_position.x, body.position.x) - ex,
            min(body.previous_position.y, body.position.y) - ey,
            min(body.previous_position.z, body.position.z) - ez,
        )
        maximum = Vec3(
            max(body.previous_position.x, body.position.x) + ex,
            max(body.previous_position.y, body.position.y) + ey,
            max(body.previous_position.z, body.position.z) + ez,
        )
        return minimum, maximum

    @staticmethod
    def _aabb_cell_keys(minimum: Vec3, maximum: Vec3) -> List[Tuple[int, int, int]]:
        inverse_cell = 1.0 / BROADPHASE_CELL_SIZE
        x0, y0, z0 = (math.floor(value * inverse_cell) for value in minimum.tuple())
        x1, y1, z1 = (math.floor(value * inverse_cell) for value in maximum.tuple())
        return [
            (cell_x, cell_y, cell_z)
            for cell_x in range(x0, x1 + 1)
            for cell_y in range(y0, y1 + 1)
            for cell_z in range(z0, z1 + 1)
        ]

    @classmethod
    def _body_current_aabb(cls, body: RigidBody) -> Tuple[Vec3, Vec3]:
        ex = body.support_extent(Vec3(1.0, 0.0, 0.0)) + BROADPHASE_SKIN
        ey = body.support_extent(Vec3(0.0, 1.0, 0.0)) + BROADPHASE_SKIN
        ez = body.support_extent(Vec3(0.0, 0.0, 1.0)) + BROADPHASE_SKIN
        return (
            Vec3(body.position.x - ex, body.position.y - ey, body.position.z - ez),
            Vec3(body.position.x + ex, body.position.y + ey, body.position.z + ez),
        )

    @classmethod
    def _body_swept_cell_keys(cls, body: RigidBody) -> List[Tuple[int, int, int]]:
        """Rasterize a swept bounding sphere without filling its whole AABB.

        A fast diagonal throw can span tens of metres in one safety-clamped
        substep. Enumerating the Cartesian volume of that diagonal AABB creates
        hundreds of thousands of empty hash cells. Sampling a conservative
        swept sphere scales with path length instead of swept volume.
        """
        motion = body.position - body.previous_position
        longest_axis = max(abs(motion.x), abs(motion.y), abs(motion.z))
        steps = max(1, math.ceil(longest_axis / (BROADPHASE_CELL_SIZE * 0.50)))
        radius = body.bounding_radius + BROADPHASE_SKIN
        inverse_cell = 1.0 / BROADPHASE_CELL_SIZE
        keys: set[Tuple[int, int, int]] = set()
        for step_index in range(steps + 1):
            center = body.previous_position + motion * (step_index / steps)
            x0 = math.floor((center.x - radius) * inverse_cell)
            x1 = math.floor((center.x + radius) * inverse_cell)
            y0 = math.floor((center.y - radius) * inverse_cell)
            y1 = math.floor((center.y + radius) * inverse_cell)
            z0 = math.floor((center.z - radius) * inverse_cell)
            z1 = math.floor((center.z + radius) * inverse_cell)
            for cell_x in range(x0, x1 + 1):
                for cell_y in range(y0, y1 + 1):
                    for cell_z in range(z0, z1 + 1):
                        keys.add((cell_x, cell_y, cell_z))
        return list(keys)

    def _sleeping_static_signature(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            (
                float(index),
                body.position.x, body.position.y, body.position.z,
                body.orientation.w, body.orientation.x,
                body.orientation.y, body.orientation.z,
            )
            for index, body in enumerate(self.bodies)
            if body.asleep
        )

    def _rebuild_static_broadphase_index(self) -> None:
        cells: Dict[Tuple[int, int, int], set[int]] = {}
        body_cells: Dict[int, Tuple[Tuple[int, int, int], ...]] = {}
        for index, body in enumerate(self.bodies):
            if not body.asleep:
                continue
            minimum, maximum = self._body_current_aabb(body)
            keys = tuple(self._aabb_cell_keys(minimum, maximum))
            body_cells[index] = keys
            for key in keys:
                cells.setdefault(key, set()).add(index)
        self._static_cells = cells
        self._static_body_cells = body_cells
        self._static_signature = self._sleeping_static_signature()

    def _sync_static_broadphase_index(
        self, signature: Tuple[Tuple[float, ...], ...]
    ) -> None:
        previous_states = {
            int(row[0]): row[1:] for row in self._static_signature
        }
        current_states = {int(row[0]): row[1:] for row in signature}
        changed = {
            index for index in set(previous_states) | set(current_states)
            if previous_states.get(index) != current_states.get(index)
        }
        for index in changed:
            for key in self._static_body_cells.pop(index, ()):
                members = self._static_cells.get(key)
                if members is None:
                    continue
                members.discard(index)
                if not members:
                    del self._static_cells[key]
        for index in changed:
            if index not in current_states:
                continue
            body = self.bodies[index]
            minimum, maximum = self._body_current_aabb(body)
            keys = tuple(self._aabb_cell_keys(minimum, maximum))
            self._static_body_cells[index] = keys
            for key in keys:
                self._static_cells.setdefault(key, set()).add(index)
        self._static_signature = signature

    def _rebuild_broadphase_soa(self) -> None:
        """Create the persistent packed state used by NumPy's NEON kernel."""
        self._soa_owner = id(self.bodies)
        if self.cpu_backend != "neon" or np is None:
            self._soa_positions = None
            self._soa_previous = None
            self._soa_radii = None
            self._soa_active = None
            return
        count = len(self.bodies)
        self._soa_positions = np.empty((count, 3), dtype=np.float64)
        self._soa_previous = np.empty((count, 3), dtype=np.float64)
        self._soa_radii = np.empty(count, dtype=np.float64)
        self._soa_active = np.zeros(count, dtype=np.bool_)
        for index, body in enumerate(self.bodies):
            self._soa_positions[index] = body.position.tuple()
            self._soa_previous[index] = body.previous_position.tuple()
            self._soa_radii[index] = body.bounding_radius + BROADPHASE_SKIN * 0.5

    def _numpy_broadphase_pairs(
        self, active_indices: Sequence[int],
    ) -> List[Tuple[int, int]]:
        """NEON-vectorized swept-sphere broadphase over persistent SoA state."""
        started = time.perf_counter()
        if self._soa_owner != id(self.bodies) or (
            self._soa_positions is not None
            and len(self._soa_positions) != len(self.bodies)
        ):
            self._rebuild_broadphase_soa()
        if self._soa_positions is None or np is None:
            return self._scalar_broadphase_pairs(active_indices)

        signature = self._sleeping_static_signature()
        if signature != self._static_signature:
            self._sync_static_broadphase_index(signature)
            for row in signature:
                index = int(row[0])
                body = self.bodies[index]
                self._soa_positions[index] = body.position.tuple()
                self._soa_previous[index] = body.previous_position.tuple()

        self._soa_active.fill(False)
        for index in active_indices:
            body = self.bodies[index]
            self._soa_positions[index] = body.position.tuple()
            self._soa_previous[index] = body.previous_position.tuple()
            self._soa_radii[index] = body.bounding_radius + BROADPHASE_SKIN * 0.5
            self._soa_active[index] = True

        if not active_indices:
            self.last_broadphase_backend = "NEON"
            self.last_broadphase_ms = (time.perf_counter() - started) * 1000.0
            return []

        active = np.asarray(active_indices, dtype=np.intp)
        current_delta = (
            self._soa_positions[np.newaxis, :, :]
            - self._soa_positions[active, np.newaxis, :]
        )
        previous_delta = (
            self._soa_previous[np.newaxis, :, :]
            - self._soa_previous[active, np.newaxis, :]
        )
        combined_radius = (
            self._soa_radii[active, np.newaxis]
            + self._soa_radii[np.newaxis, :]
        )
        # Exact closest approach between linearly swept bounding spheres.  The
        # previous radius-plus-path-length inequality was safe but excessively
        # conservative for the 122 m/s steel-ball pallet test, creating many
        # candidate pairs that the exact sweep immediately rejected.
        relative_delta = current_delta - previous_delta
        relative_motion2 = np.einsum(
            "ijk,ijk->ij", relative_delta, relative_delta, optimize=True
        )
        projection_numerator = -np.einsum(
            "ijk,ijk->ij", previous_delta, relative_delta, optimize=True
        )
        closest_fraction = np.divide(
            projection_numerator,
            relative_motion2,
            out=np.zeros_like(projection_numerator),
            where=relative_motion2 > 1.0e-14,
        )
        np.clip(closest_fraction, 0.0, 1.0, out=closest_fraction)
        closest_delta = (
            previous_delta + relative_delta * closest_fraction[:, :, np.newaxis]
        )
        closest_distance2 = np.einsum(
            "ijk,ijk->ij", closest_delta, closest_delta, optimize=True
        )
        candidates = closest_distance2 <= combined_radius * combined_radius

        all_indices = np.arange(len(self.bodies), dtype=np.intp)
        candidates &= active[:, np.newaxis] != all_indices[np.newaxis, :]
        # Keep an active/active pair once; active/sleeping pairs remain valid in
        # either numeric index order and are canonicalized below.
        candidates &= (
            ~self._soa_active[np.newaxis, :]
            | (active[:, np.newaxis] < all_indices[np.newaxis, :])
        )
        active_rows, other_indices = np.nonzero(candidates)
        active_body_indices = active[active_rows]
        first = np.minimum(active_body_indices, other_indices)
        second = np.maximum(active_body_indices, other_indices)
        if len(first):
            order = np.lexsort((second, first))
            result = [
                (int(first[index]), int(second[index])) for index in order
            ]
        else:
            result = []
        self.last_broadphase_backend = "NEON"
        self.last_broadphase_ms = (time.perf_counter() - started) * 1000.0
        return result

    def _broadphase_pairs(
        self, active_indices: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, int]]:
        if active_indices is None:
            active_indices = [
                index for index, body in enumerate(self.bodies)
                if not body.asleep
            ]
        if self.cpu_backend == "neon" and np is not None:
            return self._numpy_broadphase_pairs(active_indices)
        return self._scalar_broadphase_pairs(active_indices)

    def _scalar_broadphase_pairs(
        self, active_indices: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, int]]:
        started = time.perf_counter()
        if active_indices is None:
            active_indices = [
                index for index, body in enumerate(self.bodies)
                if not body.asleep
            ]
        spatial_states: List[SpatialState] = [
            (
                index,
                self.bodies[index].position.x,
                self.bodies[index].position.y,
                self.bodies[index].position.z,
                self.bodies[index].previous_position.x,
                self.bodies[index].previous_position.y,
                self.bodies[index].previous_position.z,
                self.bodies[index].bounding_radius,
            )
            for index in active_indices
        ]
        spatial_job = (
            self.spatial_workers.submit(spatial_states)
            if self.spatial_workers is not None else None
        )

        # Sleeping bodies live in a current-transform grid. A disturbed body
        # rejoins it after settling instead of remaining a permanent hot-path
        # participant merely because it is no longer pristine.  This main-core
        # work intentionally overlaps the three worker processes above.
        signature = self._sleeping_static_signature()
        if signature != self._static_signature:
            self._sync_static_broadphase_index(signature)
        parallel_cells = (
            self.spatial_workers.collect(spatial_job)
            if self.spatial_workers is not None else None
        )
        active_cells: Dict[Tuple[int, int, int], List[int]] = {}
        pair_set: set[Tuple[int, int]] = set()
        for index in active_indices:
            body = self.bodies[index]
            cell_keys = (
                parallel_cells[index]
                if parallel_cells is not None and index in parallel_cells
                else self._body_swept_cell_keys(body)
            )
            # A large brick overlaps many neighboring hash cells.  Gather its
            # neighbors in one local set, then emit each pair once instead of
            # performing thousands of duplicate Python pair-set insertions.
            neighbors: set[int] = set()
            for key in cell_keys:
                neighbors.update(self._static_cells.get(key, ()))
                neighbors.update(active_cells.get(key, ()))
                active_cells.setdefault(key, []).append(index)
            neighbors.discard(index)
            pair_set.update(
                (index, other) if index < other else (other, index)
                for other in neighbors
            )
        # Remove stale reset-cell false positives with a cheap bounding-sphere
        # sweep test before the more expensive SAT/narrow phase.
        filtered: List[Tuple[int, int]] = []
        for first, second in sorted(pair_set):
            a, b = self.bodies[first], self.bodies[second]
            radius = a.bounding_radius + b.bounding_radius + BROADPHASE_SKIN
            delta_now = b.position - a.position
            delta_before = b.previous_position - a.previous_position
            if min(delta_now.length_squared(), delta_before.length_squared()) <= radius * radius:
                filtered.append((first, second))
                continue
            relative_motion = (delta_now - delta_before).length()
            if min(delta_now.length(), delta_before.length()) <= radius + relative_motion:
                filtered.append((first, second))
        self.last_broadphase_backend = "SCALAR"
        self.last_broadphase_ms = (time.perf_counter() - started) * 1000.0
        return filtered

    def _limit_bullet_candidates(
        self, pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Keep each fast body on its earliest plausible contact path.

        A swept AABB through a dense pallet can contain hundreds of bricks.
        Resolving all of them against the body's *pre-impact* trajectory would
        be both slow and physically wrong. The first impact must change that
        trajectory before later contacts can become valid.
        """
        bullets: set[int] = set()
        for index, body in enumerate(self.bodies):
            displacement = (body.position - body.previous_position).length()
            minimum_dimension = min(body.spec.dimensions)
            if not body.asleep and displacement > max(0.04, minimum_dimension * 0.45):
                bullets.add(index)
        if not bullets:
            return pairs

        retained: set[Tuple[int, int]] = {
            pair for pair in pairs if pair[0] not in bullets and pair[1] not in bullets
        }
        ranked: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {
            index: [] for index in bullets
        }
        for pair in pairs:
            first, second = pair
            if first not in bullets and second not in bullets:
                continue
            body_a, body_b = self.bodies[first], self.bodies[second]
            approximate = self._sphere_sweep_fraction(body_a, body_b)
            if approximate is None:
                continue
            if first in bullets:
                ranked[first].append((approximate, pair))
            if second in bullets:
                ranked[second].append((approximate, pair))

        exact_cache: Dict[Tuple[int, int], Optional[float]] = {}

        def exact_fraction(pair: Tuple[int, int]) -> Optional[float]:
            if pair in exact_cache:
                return exact_cache[pair]
            first, second = pair
            body_a, body_b = self.bodies[first], self.bodies[second]
            endpoint_contact: Any = None
            if body_a.spec.shape == "sphere" and body_b.spec.shape == "box":
                fraction = self._swept_sphere_box_fraction(body_a, body_b)
                endpoint_contact = self._sphere_box_contact(body_a, body_b)
            elif body_a.spec.shape == "box" and body_b.spec.shape == "sphere":
                fraction = self._swept_sphere_box_fraction(body_b, body_a)
                endpoint_contact = self._sphere_box_contact(body_b, body_a)
            elif body_a.spec.shape == "box" and body_b.spec.shape == "box":
                fraction = self._swept_box_box_fraction(body_a, body_b)
                endpoint_contact = self._box_box_contact(body_a, body_b)
            else:
                fraction = self._sphere_sweep_fraction(body_a, body_b)
                combined = body_a.bounding_radius + body_b.bounding_radius
                endpoint_contact = (
                    True
                    if (body_b.position - body_a.position).length_squared() <= combined * combined
                    else None
                )
            if fraction is None and endpoint_contact is not None:
                fraction = 1.0
            exact_cache[pair] = fraction
            return fraction

        order_score: Dict[Tuple[int, int], float] = {}
        for candidates in ranked.values():
            admitted = 0
            for _approximate, pair in sorted(candidates, key=lambda entry: (entry[0], entry[1])):
                score = exact_fraction(pair)
                if score is None:
                    continue
                retained.add(pair)
                order_score[pair] = min(order_score.get(pair, score), score)
                admitted += 1
                if admitted >= MAX_BULLET_CONTACTS_PER_TICK:
                    break
        return sorted(retained, key=lambda pair: (order_score.get(pair, 0.0), pair))

    def _limit_dense_brick_candidates(
        self, pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        active_bricks = {
            index for index, body in enumerate(self.bodies)
            if body.spec.key in {"clay_brick", "wood_pallet"} and not body.asleep
        }
        if not active_bricks:
            return pairs
        retained: set[Tuple[int, int]] = set()
        ranked: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {
            index: [] for index in active_bricks
        }
        for pair in pairs:
            first, second = pair
            involved = [index for index in pair if index in active_bricks]
            if not involved:
                retained.add(pair)
                continue
            other_keys = {self.bodies[index].spec.key for index in pair}
            if not other_keys.issubset({"clay_brick", "wood_pallet"}):
                retained.add(pair)
                continue
            distance = (
                self.bodies[second].position - self.bodies[first].position
            ).length_squared()
            for index in involved:
                ranked[index].append((distance, pair))
        for candidates in ranked.values():
            retained.update(
                pair for _distance, pair in sorted(candidates, key=lambda entry: (entry[0], entry[1]))[
                    :MAX_BRICK_NEIGHBORS_PER_TICK
                ]
            )
        return sorted(retained)

    def _body_near_player(self, body: RigidBody) -> bool:
        reach = body.bounding_radius + PLAYER_RADIUS + 0.35
        vertical_low = self.player.position.y - body.bounding_radius
        vertical_high = (
            self.player.position.y + PLAYER_HEIGHT + body.bounding_radius
        )
        if (
            max(body.position.y, body.previous_position.y) < vertical_low
            or min(body.position.y, body.previous_position.y) > vertical_high
        ):
            return False
        start_x = body.previous_position.x - self.player.position.x
        start_z = body.previous_position.z - self.player.position.z
        motion_x = body.position.x - body.previous_position.x
        motion_z = body.position.z - body.previous_position.z
        motion_squared = motion_x * motion_x + motion_z * motion_z
        if motion_squared > 1.0e-14:
            fraction = clamp(
                -(start_x * motion_x + start_z * motion_z) / motion_squared,
                0.0, 1.0,
            )
            start_x += motion_x * fraction
            start_z += motion_z * fraction
        return start_x * start_x + start_z * start_z <= reach * reach

    def _refresh_wheelbarrow_cache(self) -> None:
        self._wheelbarrow_index = next(
            (
                index for index, body in enumerate(self.bodies)
                if body.spec.key == "wheelbarrow"
            ),
            None,
        )
        self._cargo_candidate_indices = tuple(
            index for index, body in enumerate(self.bodies)
            if index != self._wheelbarrow_index and body.bounding_radius <= 0.26
        )
        self._cargo_cache_owner = id(self.bodies)

    def _apply_wheelbarrow_cargo(self, dt: float) -> None:
        if self._cargo_cache_owner != id(self.bodies):
            self._refresh_wheelbarrow_cache()
        if self._wheelbarrow_index is None:
            return
        barrow = self.bodies[self._wheelbarrow_index]
        if barrow.asleep and not any(
            not self.bodies[index].asleep for index in self._cargo_candidate_indices
        ):
            return
        inverse = barrow.orientation.conjugate()
        tray_top = barrow.spec.dimensions[1] * 0.5
        for index in self._cargo_candidate_indices:
            body = self.bodies[index]
            if body.stuck_surface or body.stuck_to is not None:
                continue
            cargo_radius = body.bounding_radius
            if abs(body.position.x - barrow.position.x) > 0.8 or abs(body.position.z - barrow.position.z) > 1.1:
                continue
            local = inverse.rotate(body.position - barrow.position)
            x_limit = max(0.02, 0.275 - cargo_radius)
            z_min = -0.255 + cargo_radius
            z_max = 0.485 - cargo_radius
            if not (
                abs(local.x) <= 0.40
                and -0.42 <= local.z <= 0.65
                and tray_top - 0.08 <= local.y - cargo_radius <= tray_top + 0.55
            ):
                continue

            original_local = local.copy()
            local.x = clamp(local.x, -x_limit, x_limit)
            local.z = clamp(local.z, z_min, z_max)
            local.y = max(local.y, tray_top + cargo_radius)
            if (local - original_local).length_squared() > 1.0e-12:
                body.position = barrow.position + barrow.orientation.rotate(local)

            relative_world = body.velocity - barrow.velocity_at(body.position)
            relative_local = inverse.rotate(relative_world)
            delta_local = Vec3()
            if original_local.x < -x_limit and relative_local.x < 0.0:
                delta_local.x = -relative_local.x
            elif original_local.x > x_limit and relative_local.x > 0.0:
                delta_local.x = -relative_local.x
            if original_local.z < z_min and relative_local.z < 0.0:
                delta_local.z = -relative_local.z
            elif original_local.z > z_max and relative_local.z > 0.0:
                delta_local.z = -relative_local.z
            if original_local.y < tray_top + cargo_radius and relative_local.y < 0.0:
                delta_local.y = -relative_local.y
            wall_delta = barrow.orientation.rotate(delta_local)
            if wall_delta.length_squared() > 1.0e-12:
                body.velocity = body.velocity + wall_delta
                barrow.velocity = (
                    barrow.velocity - wall_delta * (body.dynamic_mass / barrow.dynamic_mass)
                )

            # Rolling cargo approaches the tray velocity through finite static
            # friction, with the equal and opposite momentum sent to the barrow.
            desired_delta = barrow.velocity.horizontal() - body.velocity.horizontal()
            maximum_delta = 4.0 * dt
            magnitude = desired_delta.length()
            if magnitude > maximum_delta:
                desired_delta = desired_delta * (maximum_delta / magnitude)
            body.velocity = body.velocity + desired_delta
            barrow.velocity = (
                barrow.velocity - desired_delta * (body.dynamic_mass / barrow.dynamic_mass)
            )
            body.grounded = True
            body.wake()

    def category_summary_rows(self) -> List[str]:
        ordered = (
            ("calibration", "CALIBRATION"), ("bat", "BAT"),
            ("wheelbarrow", "WHEELBARROW"), ("balloon", "BALLOON"),
            ("noodles", "NOODLES"), ("goo", "GOO"),
            ("marbles", "MARBLES"), ("bricks", "BRICKS"),
            ("pallet", "PALLET"), ("plush", "PLUSH"),
        )
        rows: List[str] = []
        for category, label in ordered:
            members = [body for body in self.bodies if body.spec.category == category]
            if not members:
                continue
            speeds = [body.velocity.length() for body in members]
            awake = sum(not body.asleep for body in members)
            rows.append(
                f"{label:11} {len(members):3d}  awake {awake:3d}  "
                f"avg {sum(speeds) / len(speeds):6.2f}  max {max(speeds):7.2f} m/s"
            )
        return rows

    def step(self, dt: float = FIXED_DT) -> None:
        self.impacts.clear()
        self._solver_static_ids = {id(body) for body in self.bodies if body.asleep}
        self._update_attachments()
        self._integrate_player(dt)
        if self.held_body is not None:
            self._hold_constraint(self.held_body)
        active_indices = [
            index for index, body in enumerate(self.bodies)
            if not body.asleep or body.held
        ]
        for index in active_indices:
            body = self.bodies[index]
            self._integrate_body_forces(body, dt)
            self._advance_body_swept(body, dt)

        candidate_pairs = self._limit_dense_brick_candidates(
            self._limit_bullet_candidates(self._broadphase_pairs(active_indices))
        )
        self.last_broadphase_candidates = len(candidate_pairs)
        active_pairs = 0
        active_index_set = set(active_indices)
        # Awake does not imply near the player.  The former unconditional path
        # ran capsule contact math up to eight times for every distant moving
        # exhibit during each pallet-impact tick.
        player_candidates = [
            index for index, body in enumerate(self.bodies)
            if self._body_near_player(body)
        ]
        solver_indices = set(player_candidates)
        for first_index, second_index in candidate_pairs:
            solver_indices.add(first_index)
            solver_indices.add(second_index)
        active_bricks = sum(
            body.spec.key == "clay_brick" and not body.asleep for body in self.bodies
        )
        if len(candidate_pairs) > 60 or active_bricks > 8:
            solver_iterations = 3
        elif len(candidate_pairs) > 30 or active_bricks > 2:
            solver_iterations = 4
        else:
            solver_iterations = SOLVER_ITERATIONS
        self.last_solver_iterations = solver_iterations
        contact_constraints: List[Tuple[int, int, Vec3, Vec3]] = []
        for first_index, second_index in candidate_pairs:
            first = self.bodies[first_index]
            second = self.bodies[second_index]
            if first.asleep and second.asleep:
                continue
            resolved = self._resolve_body_pair(first, second, dt)
            if resolved is not None:
                normal, contact = resolved
                contact_constraints.append((first_index, second_index, normal, contact))
                active_pairs += 1
        for index in player_candidates:
            self._resolve_player_body(self.bodies[index], dt)
        for index in solver_indices:
            body = self.bodies[index]
            if (
                id(body) not in self._solver_static_ids
                and body.stuck_to is None and not body.stuck_surface
            ):
                self._project_body_inside_room(body)
        self._project_player_inside_room()

        # Narrow phase and positional correction are generated once. Reusing
        # those contacts keeps the expensive 15-axis OBB SAT out of the inner
        # sequential-impulse loop, a large win on Raspberry Pi-class CPUs.
        for _iteration in range(1, solver_iterations):
            for first_index, second_index, normal, contact in contact_constraints:
                first = self.bodies[first_index]
                second = self.bodies[second_index]
                self._body_pair_impulse(first, second, normal, contact)
            for index in player_candidates:
                body = self.bodies[index]
                self._resolve_player_body(body, -1.0)
            for index in solver_indices:
                body = self.bodies[index]
                if (
                    id(body) not in self._solver_static_ids
                    and body.stuck_to is None and not body.stuck_surface
                ):
                    self._project_body_inside_room(body)
            self._project_player_inside_room()

        self.last_active_contacts = active_pairs
        self._apply_wheelbarrow_cargo(dt)
        self._update_attachments()
        settling_indices = solver_indices | active_index_set
        settling_indices.update(
            index for index in self._cargo_candidate_indices
            if not self.bodies[index].asleep
        )
        for index in sorted(settling_indices):
            body = self.bodies[index]
            self._apply_ground_resistance_and_sleep(body, dt)
            if not body.asleep and not (
                body.position.finite()
                and body.velocity.finite()
                and body.angular_velocity.finite()
            ):
                raise FloatingPointError(f"non-finite rigid-body state for {body.spec.name}")
        self._drain_pending_wakes()
        self._solver_static_ids = set()
        if len(self.messages) > 256:
            del self.messages[:-256]
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
        members = [
            next(body for body in self.bodies if body.spec.key == key)
            for key in ORIGINAL_ITEM_KEYS
        ]
        for index, body in enumerate(members, 1):
            marker = "[HELD]" if body.held else "[SLEEP]" if body.asleep else ""
            speed = body.velocity.length()
            rows.append(
                f"{index}. {TELEMETRY_NAMES[body.spec.key]:14} "
                f"v=({body.velocity.x:+7.2f},{body.velocity.y:+7.2f},{body.velocity.z:+7.2f}) "
                f"|v|={speed:7.2f} m/s {marker}"
            )
        return rows


def load_dependencies() -> None:
    global pygame, GL, GLU, GL_SHADERS
    global RAW_GL_INTERLEAVED_ARRAYS, RAW_GL_VERTEX_ATTRIB_POINTER
    global RAW_GL_TRANSFORM_FEEDBACK_VARYINGS
    if pygame is not None:
        return
    try:
        import pygame as pygame_module
        from OpenGL import GL as gl_module
        from OpenGL import GLU as glu_module
        from OpenGL.GL import shaders as gl_shaders_module
        from OpenGL.raw.GL.VERSION.GL_1_1 import (
            glInterleavedArrays as raw_gl_interleaved_arrays,
        )
        from OpenGL.raw.GL.VERSION.GL_2_0 import (
            glVertexAttribPointer as raw_gl_vertex_attrib_pointer,
        )
        from OpenGL.raw.GL.VERSION.GL_3_0 import (
            glTransformFeedbackVaryings as raw_gl_transform_feedback_varyings,
        )
    except ImportError as error:
        raise SystemExit(
            "Newton's Echo Chamber requires pygame and PyOpenGL. "
            "Install with: pip install pygame PyOpenGL PyOpenGL_accelerate"
        ) from error
    pygame = pygame_module
    GL = gl_module
    GLU = glu_module
    GL_SHADERS = gl_shaders_module
    RAW_GL_INTERLEAVED_ARRAYS = raw_gl_interleaved_arrays
    RAW_GL_VERTEX_ATTRIB_POINTER = raw_gl_vertex_attrib_pointer
    RAW_GL_TRANSFORM_FEEDBACK_VARYINGS = raw_gl_transform_feedback_varyings


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
            (
                "dodge", "medicine", "steel", "concrete", "rubber_brick",
                "wood_bat", "wheelbarrow", "balloon", "foam_noodle", "goo",
                "ceramic", "clay_brick", "pallet", "plush",
                "throw", "jump", "land",
            )
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

    def wood_bat(t: float, _i: int) -> float:
        crack = noises["wood_bat"].uniform(-1.0, 1.0) * math.exp(-70.0 * t)
        body = (
            0.58 * math.sin(2.0 * math.pi * 183.0 * t)
            + 0.29 * math.sin(2.0 * math.pi * 721.0 * t + 0.3)
        ) * math.exp(-11.0 * t)
        return 0.52 * crack + body

    def wheelbarrow(t: float, _i: int) -> float:
        clang = (
            0.42 * math.sin(2.0 * math.pi * 414.0 * t)
            + 0.31 * math.sin(2.0 * math.pi * 937.0 * t)
            + 0.17 * math.sin(2.0 * math.pi * 1_663.0 * t)
        ) * math.exp(-6.3 * t)
        tire = 0.28 * math.sin(2.0 * math.pi * 73.0 * t) * math.exp(-15.0 * t)
        return clang + tire

    def balloon(t: float, _i: int) -> float:
        envelope = math.exp(-8.5 * t)
        phase = 2.0 * math.pi * (205.0 * t - 115.0 * t * t)
        squeak = noises["balloon"].uniform(-1.0, 1.0) * math.exp(-32.0 * t)
        return envelope * (0.70 * math.sin(phase) + 0.16 * math.sin(phase * 3.1)) + 0.12 * squeak

    def foam_noodle(t: float, _i: int) -> float:
        slap = noises["foam_noodle"].uniform(-1.0, 1.0) * math.exp(-48.0 * t)
        hollow = math.sin(2.0 * math.pi * 118.0 * t) * math.exp(-18.0 * t)
        return 0.44 * slap + 0.50 * hollow

    def goo(t: float, _i: int) -> float:
        wet = noises["goo"].uniform(-1.0, 1.0) * math.exp(-24.0 * t)
        bubble = math.sin(2.0 * math.pi * (94.0 * t - 78.0 * t * t)) * math.exp(-10.0 * t)
        return 0.25 * wet + 0.72 * bubble

    def ceramic(t: float, _i: int) -> float:
        envelope = math.exp(-8.0 * t)
        return envelope * (
            0.45 * math.sin(2.0 * math.pi * 2_409.0 * t)
            + 0.29 * math.sin(2.0 * math.pi * 3_811.0 * t + 0.4)
            + 0.17 * math.sin(2.0 * math.pi * 6_097.0 * t + 1.1)
        )

    def clay_brick(t: float, _i: int) -> float:
        grit = noises["clay_brick"].uniform(-1.0, 1.0) * math.exp(-58.0 * t)
        knock = (
            0.55 * math.sin(2.0 * math.pi * 229.0 * t)
            + 0.24 * math.sin(2.0 * math.pi * 727.0 * t)
        ) * math.exp(-15.0 * t)
        return 0.34 * grit + knock

    def pallet(t: float, _i: int) -> float:
        knock = (
            0.56 * math.sin(2.0 * math.pi * 139.0 * t)
            + 0.28 * math.sin(2.0 * math.pi * 359.0 * t)
            + 0.12 * math.sin(2.0 * math.pi * 821.0 * t)
        ) * math.exp(-12.0 * t)
        return knock + noises["pallet"].uniform(-0.12, 0.12) * math.exp(-40.0 * t)

    def plush(t: float, _i: int) -> float:
        air = noises["plush"].uniform(-1.0, 1.0) * math.exp(-29.0 * t)
        muffled = math.sin(2.0 * math.pi * 67.0 * t) * math.exp(-17.0 * t)
        squeak = 0.13 * math.sin(2.0 * math.pi * 512.0 * t) * math.exp(-12.0 * t)
        return 0.30 * air + 0.55 * muffled + squeak

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
        "wood_bat": _stereo_pcm(0.38, wood_bat),
        "wheelbarrow": _stereo_pcm(0.62, wheelbarrow),
        "balloon": _stereo_pcm(0.48, balloon),
        "foam_noodle": _stereo_pcm(0.25, foam_noodle),
        "goo": _stereo_pcm(0.42, goo),
        "ceramic": _stereo_pcm(0.55, ceramic),
        "clay_brick": _stereo_pcm(0.34, clay_brick),
        "pallet": _stereo_pcm(0.40, pallet),
        "plush": _stereo_pcm(0.32, plush),
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
        self.family_next_play: Dict[str, float] = {}
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
        now = time.monotonic()
        family_cooldown = {
            "clay_brick": 0.055,
            "ceramic": 0.035,
            "pallet": 0.045,
        }.get(event.family, 0.015)
        if now < self.family_next_play.get(event.family, 0.0):
            return
        self.family_next_play[event.family] = now + family_cooldown
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
        # Numeric HUD fields change frequently.  Reuse the allocation whenever
        # glyph dimensions match instead of making Mesa allocate a fresh GPU
        # texture on every update.
        if cached is not None and cached[2] == width and cached[3] == height:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, width, height,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels,
            )
        else:
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


class AdaptivePerformanceController:
    """Small, deterministic frame-budget governor with recovery hysteresis.

    A short exponential average ignores harmless single-frame jitter while a
    genuine sub-30 FPS interval immediately selects the least expensive tier.
    Recovery is intentionally gradual, preventing quality oscillation around a
    threshold.  The class has no pygame/OpenGL dependency so ``--check`` can
    validate the exact policy on development hosts and on a Raspberry Pi.
    """

    def __init__(self) -> None:
        # Performance-first startup avoids a several-second low-FPS burst on
        # the Pi. Sustained frame headroom promotes quality tier by tier.
        self.quality_level = 0
        self.ema_frame_seconds = 1.0 / TARGET_FPS
        self.display_fps = float(TARGET_FPS)
        self.display_frame_ms = 1000.0 / TARGET_FPS
        self._sample_age = 0.0
        self._slow_age = 0.0
        self._recovery_age = 0.0
        self.frames_observed = 0
        self.backlog_drops = 0

    @property
    def quality_name(self) -> str:
        return RENDER_QUALITY_NAMES[self.quality_level]

    @property
    def max_physics_steps(self) -> int:
        return QUALITY_MAX_PHYSICS_STEPS[self.quality_level]

    @property
    def impact_sound_limit(self) -> int:
        return QUALITY_IMPACT_SOUND_LIMITS[self.quality_level]

    @property
    def below_floor(self) -> bool:
        return self.display_fps < MINIMUM_RENDER_FPS

    def observe(self, frame_seconds: float) -> None:
        sample = clamp(float(frame_seconds), 1.0 / 500.0, MAX_FRAME_DT)
        self.frames_observed += 1
        # Roughly 350 ms response time: fast enough to catch a collapse, slow
        # enough not to punish a single texture upload or window-system hiccup.
        alpha = 1.0 - math.exp(-sample / 0.35)
        self.ema_frame_seconds += (sample - self.ema_frame_seconds) * alpha
        fps = 1.0 / max(1.0e-6, self.ema_frame_seconds)

        if fps < 55.0:
            desired_level = 0
        elif fps < 59.0:
            desired_level = 1
        else:
            desired_level = 2

        if desired_level < self.quality_level:
            self._recovery_age = 0.0
            self._slow_age += sample
            downgrade_delay = 0.10 if desired_level == 0 else 0.55
            if self._slow_age >= downgrade_delay:
                # A floor violation is an emergency; intermediate pressure
                # sheds one tier at a time to preserve presentation quality.
                self.quality_level = (
                    0 if desired_level == 0 else max(desired_level, self.quality_level - 1)
                )
                self._slow_age = 0.0
        elif desired_level > self.quality_level:
            self._slow_age = 0.0
            self._recovery_age += sample
            if self._recovery_age >= QUALITY_RECOVERY_SECONDS:
                self.quality_level += 1
                self._recovery_age = 0.0
        else:
            self._slow_age = 0.0
            self._recovery_age = 0.0

        self._sample_age += sample
        if self._sample_age >= 1.0 / PERFORMANCE_SAMPLE_HZ or self.frames_observed == 1:
            self.display_fps = fps
            self.display_frame_ms = self.ema_frame_seconds * 1000.0
            self._sample_age = 0.0

    def note_backlog_drop(self) -> None:
        self.backlog_drops += 1


class ResourceMonitor:
    """Low-overhead Linux telemetry sampled once per second.

    Per-core percentages intentionally use the same aggregate counters exposed
    by ``/proc/stat`` that desktop task managers consume.  Sampling, parsing,
    and HUD uploads stay far below the frame-rate hot path.
    """

    def __init__(self, cpu_cores: Sequence[int]) -> None:
        self.cpu_cores = tuple(int(core) for core in cpu_cores)
        self.core_loads = {core: 0.0 for core in self.cpu_cores}
        self.rss_mib = 0.0
        self.memory_available_mib = 0.0
        self.memory_total_mib = 0.0
        self.next_sample = 0.0
        self._previous_cpu = self._read_cpu_times()
        self.sample(time.monotonic(), force=True)

    def _read_cpu_times(self) -> Dict[int, Tuple[int, int]]:
        values: Dict[int, Tuple[int, int]] = {}
        try:
            with open("/proc/stat", "r", encoding="ascii") as handle:
                for line in handle:
                    if not line.startswith("cpu") or not line[3:4].isdigit():
                        continue
                    fields = line.split()
                    core = int(fields[0][3:])
                    if core not in self.core_loads:
                        continue
                    counters = [int(value) for value in fields[1:]]
                    total = sum(counters)
                    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
                    values[core] = (total, idle)
        except (OSError, ValueError):
            pass
        return values

    @staticmethod
    def _status_value(path: str, key: str) -> float:
        try:
            with open(path, "r", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith(key + ":"):
                        return float(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return 0.0

    @staticmethod
    def _memory_values() -> Tuple[float, float]:
        total = available = 0.0
        try:
            with open("/proc/meminfo", "r", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        total = float(line.split()[1]) / 1024.0
                    elif line.startswith("MemAvailable:"):
                        available = float(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return total, available

    def sample(self, now: float, force: bool = False) -> None:
        if not force and now < self.next_sample:
            return
        current = self._read_cpu_times()
        for core in self.cpu_cores:
            before = self._previous_cpu.get(core)
            after = current.get(core)
            if before is None or after is None:
                continue
            elapsed = after[0] - before[0]
            idle = after[1] - before[1]
            if elapsed > 0:
                self.core_loads[core] = clamp(
                    100.0 * (elapsed - idle) / elapsed, 0.0, 100.0
                )
        self._previous_cpu = current
        self.rss_mib = self._status_value("/proc/self/status", "VmRSS")
        self.memory_total_mib, self.memory_available_mib = self._memory_values()
        self.next_sample = now + 1.0


class GPUPhysicsParticles:
    """GPU-resident secondary rigid-mote physics via transform feedback.

    The authoritative 551-object solver remains deterministic on the CPU.
    These independent micro-debris bodies are a useful asynchronous physics
    workload: the GPU integrates gravity, momentum, air drag, surface friction,
    wall/floor restitution, and lifetime respawn, then renders the same buffer.
    No state is copied back to Python, so graphics and physics share one V3D
    command stream without a synchronization stall.
    """

    UPDATE_VERTEX_SHADER = r"""
#version 140
in vec4 in_position_life;
in vec4 in_velocity_seed;
out vec4 out_position_life;
out vec4 out_velocity_seed;
uniform float u_dt;
uniform float u_gravity;
uniform float u_friction;
uniform float u_time;
uniform vec3 u_room;

float hash11(float value) {
    return fract(sin(value * 12.9898 + 78.233) * 43758.5453);
}

void main() {
    vec3 position = in_position_life.xyz;
    float life = in_position_life.w - u_dt;
    vec3 velocity = in_velocity_seed.xyz;
    float seed = in_velocity_seed.w;

    velocity.y -= u_gravity * u_dt;
    velocity *= 1.0 / (1.0 + 0.018 * u_dt);
    position += velocity * u_dt;

    float bounce = 0.34 + 0.40 * hash11(seed + 4.0);
    if (position.x < 0.035) {
        position.x = 0.035;
        if (velocity.x < 0.0) velocity.x = -velocity.x * bounce;
    } else if (position.x > u_room.x - 0.035) {
        position.x = u_room.x - 0.035;
        if (velocity.x > 0.0) velocity.x = -velocity.x * bounce;
    }
    if (position.z < 0.035) {
        position.z = 0.035;
        if (velocity.z < 0.0) velocity.z = -velocity.z * bounce;
    } else if (position.z > u_room.z - 0.035) {
        position.z = u_room.z - 0.035;
        if (velocity.z > 0.0) velocity.z = -velocity.z * bounce;
    }
    if (position.y < 0.035) {
        position.y = 0.035;
        if (velocity.y < 0.0) velocity.y = -velocity.y * bounce;
        float resistance = clamp(1.0 - u_friction * 4.5 * u_dt, 0.0, 1.0);
        velocity.xz *= resistance;
    } else if (position.y > u_room.y - 0.035) {
        position.y = u_room.y - 0.035;
        if (velocity.y > 0.0) velocity.y = -velocity.y * bounce;
    }

    if (life <= 0.0) {
        float epoch = floor(u_time * 0.2);
        position = vec3(
            0.5 + hash11(seed + epoch * 3.1) * (u_room.x - 1.0),
            2.0 + hash11(seed + epoch * 5.7) * (u_room.y - 2.5),
            0.5 + hash11(seed + epoch * 7.9) * (u_room.z - 1.0)
        );
        velocity = vec3(
            (hash11(seed + epoch * 11.3) - 0.5) * 3.0,
            hash11(seed + epoch * 13.7) * 2.0,
            (hash11(seed + epoch * 17.1) - 0.5) * 3.0
        );
        life = 8.0 + 24.0 * hash11(seed + epoch * 19.9);
    }

    out_position_life = vec4(position, life);
    out_velocity_seed = vec4(velocity, seed);
    gl_Position = vec4(0.0);
}
"""

    RENDER_VERTEX_SHADER = r"""
#version 140
in vec4 in_position_life;
in vec4 in_velocity_seed;
out float particle_speed;
out float particle_seed;
void main() {
    vec4 eye_position = gl_ModelViewMatrix * vec4(in_position_life.xyz, 1.0);
    gl_Position = gl_ProjectionMatrix * eye_position;
    float distance_to_eye = max(0.25, -eye_position.z);
    gl_PointSize = clamp(17.0 / distance_to_eye, 1.0, 3.2);
    particle_speed = length(in_velocity_seed.xyz);
    particle_seed = in_velocity_seed.w;
}
"""

    RENDER_FRAGMENT_SHADER = r"""
#version 140
in float particle_speed;
in float particle_seed;
out vec4 fragment_color;
void main() {
    vec2 offset = gl_PointCoord - vec2(0.5);
    float radius2 = dot(offset, offset);
    if (radius2 > 0.25) discard;
    float hot = clamp(particle_speed * 0.09, 0.0, 1.0);
    float family = fract(particle_seed * 0.61803398875);
    vec3 cold_color = mix(vec3(0.12, 0.38, 0.64), vec3(0.36, 0.74, 0.91), family);
    vec3 color = mix(cold_color, vec3(1.00, 0.47, 0.12), hot);
    float alpha = (1.0 - radius2 * 4.0) * (0.055 + 0.075 * hot);
    fragment_color = vec4(color, alpha);
}
"""

    def __init__(self, seed: int, mode: str = "auto") -> None:
        self.mode = str(mode).lower()
        self.max_particles = GPU_PHYSICS_MAX_PARTICLES
        self.active_particles = 0
        self.buffer_bytes = (
            self.max_particles * GPU_PARTICLE_STATE_FLOATS * 4 * 2
        )
        self.update_program = 0
        self.render_program = 0
        self.buffers: List[int] = []
        self.source_index = 0
        self.simulated_frames = 0
        self.enabled = False
        self.failure = ""
        self.floor_guard_active = False
        self._floor_recovery_seconds = 0.0
        self.timer_queries: List[int] = []
        self.timer_pending: List[int] = []
        self.timer_active_query = 0
        self.gpu_frame_ms = 0.0
        self.timer_failure = ""
        self.timer_extension = ""
        self._update_uniforms: Dict[str, int] = {}
        try:
            self.update_program = self._link_program(
                self.UPDATE_VERTEX_SHADER,
                varyings=("out_position_life", "out_velocity_seed"),
            )
            self.render_program = self._link_program(
                self.RENDER_VERTEX_SHADER, self.RENDER_FRAGMENT_SHADER
            )
            generated = GL.glGenBuffers(2)
            if isinstance(generated, (int, np.integer if np is not None else int)):
                self.buffers = [int(generated)]
            else:
                self.buffers = [int(value) for value in generated]
            if len(self.buffers) != 2 or not all(self.buffers):
                raise RuntimeError(f"expected two GPU buffers, received {self.buffers}")
            payload = self._initial_payload(seed)
            for buffer_id in self.buffers:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
                GL.glBufferData(
                    GL.GL_ARRAY_BUFFER, len(payload), payload, GL.GL_STREAM_DRAW
                )
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            self._update_uniforms = {
                name: int(GL.glGetUniformLocation(self.update_program, name))
                for name in ("u_dt", "u_gravity", "u_friction", "u_time", "u_room")
            }
            try:
                extension_value = GL.glGetString(GL.GL_EXTENSIONS) or b""
                extension_text = (
                    extension_value.decode("ascii", "ignore")
                    if isinstance(extension_value, bytes) else str(extension_value)
                )
                self.timer_extension = next((
                    extension for extension in (
                        "GL_ARB_timer_query", "GL_EXT_timer_query"
                    ) if extension in extension_text.split()
                ), "")
                if self.timer_extension and all(getattr(GL, name, None) is not None for name in (
                    "glGenQueries", "glBeginQuery", "glEndQuery",
                    "glGetQueryObjectiv", "glGetQueryObjectui64v",
                )) and all(hasattr(GL, name) for name in (
                    "GL_TIME_ELAPSED", "GL_QUERY_RESULT_AVAILABLE", "GL_QUERY_RESULT",
                )):
                    generated_queries = GL.glGenQueries(4)
                    if isinstance(
                        generated_queries,
                        (int, np.integer if np is not None else int),
                    ):
                        self.timer_queries = [int(generated_queries)]
                    else:
                        self.timer_queries = [
                            int(value) for value in generated_queries
                        ]
            except Exception:
                self.timer_queries = []
            self.set_quality(0)
            self.enabled = True
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.cleanup()
            raise RuntimeError(self.failure) from error

    @staticmethod
    def _decode_log(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return str(value or "")

    @classmethod
    def _link_program(
        cls,
        vertex_source: str,
        fragment_source: Optional[str] = None,
        varyings: Sequence[str] = (),
    ) -> int:
        shaders: List[int] = []
        program = int(GL.glCreateProgram())
        if not program:
            raise RuntimeError("OpenGL returned program id 0")
        try:
            vertex = int(GL_SHADERS.compileShader(vertex_source, GL.GL_VERTEX_SHADER))
            shaders.append(vertex)
            GL.glAttachShader(program, vertex)
            if fragment_source is not None:
                fragment = int(GL_SHADERS.compileShader(fragment_source, GL.GL_FRAGMENT_SHADER))
                shaders.append(fragment)
                GL.glAttachShader(program, fragment)
            GL.glBindAttribLocation(program, 0, "in_position_life")
            GL.glBindAttribLocation(program, 1, "in_velocity_seed")
            if varyings:
                encoded = [ctypes.create_string_buffer(name.encode("ascii")) for name in varyings]
                names = (ctypes.POINTER(ctypes.c_char) * len(encoded))(*(
                    ctypes.cast(value, ctypes.POINTER(ctypes.c_char))
                    for value in encoded
                ))
                RAW_GL_TRANSFORM_FEEDBACK_VARYINGS(
                    program, len(encoded), names, GL.GL_INTERLEAVED_ATTRIBS
                )
            GL.glLinkProgram(program)
            if not int(GL.glGetProgramiv(program, GL.GL_LINK_STATUS)):
                raise RuntimeError(cls._decode_log(GL.glGetProgramInfoLog(program)))
            return program
        except Exception:
            GL.glDeleteProgram(program)
            raise
        finally:
            for shader in shaders:
                try:
                    GL.glDetachShader(program, shader)
                except Exception:
                    pass
                GL.glDeleteShader(shader)

    def _initial_payload(self, seed: int) -> bytes:
        if np is not None:
            rng = np.random.default_rng(int(seed) ^ 0x51A6D3)
            state = np.empty(
                (self.max_particles, GPU_PARTICLE_STATE_FLOATS), dtype=np.float32
            )
            state[:, 0] = rng.uniform(0.5, ROOM_WIDTH - 0.5, self.max_particles)
            state[:, 1] = rng.uniform(0.05, ROOM_HEIGHT - 0.2, self.max_particles)
            state[:, 2] = rng.uniform(0.5, ROOM_LENGTH - 0.5, self.max_particles)
            state[:, 3] = rng.uniform(0.1, 32.0, self.max_particles)
            state[:, 4] = rng.normal(0.0, 0.65, self.max_particles)
            state[:, 5] = rng.uniform(-0.25, 1.35, self.max_particles)
            state[:, 6] = rng.normal(0.0, 0.65, self.max_particles)
            state[:, 7] = np.arange(self.max_particles, dtype=np.float32) * 0.0137 + 1.0
            return state.tobytes(order="C")
        rng = random.Random(int(seed) ^ 0x51A6D3)
        state = array("f")
        for index in range(self.max_particles):
            state.extend((
                rng.uniform(0.5, ROOM_WIDTH - 0.5),
                rng.uniform(0.05, ROOM_HEIGHT - 0.2),
                rng.uniform(0.1, ROOM_LENGTH - 0.5),
                rng.uniform(0.1, 32.0),
                rng.gauss(0.0, 0.65), rng.uniform(-0.25, 1.35),
                rng.gauss(0.0, 0.65), index * 0.0137 + 1.0,
            ))
        return state.tobytes()

    def set_quality(
        self,
        quality: int,
        observed_fps: float = float(TARGET_FPS),
        frame_dt: float = 0.0,
    ) -> None:
        if self.mode == "maximum":
            # Maximum means "use the whole GPU physics stream while the 30 FPS
            # contract survives."  A sustained emergency drops to the SAFE
            # stream, then requires three seconds of healthy headroom before
            # restoring maximum work; this prevents utilization oscillation.
            # A ten-FPS buffer is needed because the displayed EMA precedes
            # the 1% low; shedding only after 29.9 FPS is already too late.
            if not self.floor_guard_active and observed_fps < MINIMUM_RENDER_FPS + 10.0:
                self.floor_guard_active = True
                self._floor_recovery_seconds = 0.0
            elif self.floor_guard_active:
                if observed_fps >= 45.0:
                    self._floor_recovery_seconds += max(0.0, float(frame_dt))
                    if self._floor_recovery_seconds >= 3.0:
                        self.floor_guard_active = False
                        self._floor_recovery_seconds = 0.0
                else:
                    self._floor_recovery_seconds = 0.0
            self.active_particles = (
                GPU_PHYSICS_TIER_COUNTS[0]
                if self.floor_guard_active else self.max_particles
            )
        else:
            index = int(clamp(float(quality), 0.0, 2.0))
            self.active_particles = GPU_PHYSICS_TIER_COUNTS[index]

    @staticmethod
    def _bind_state_attributes(buffer_id: int) -> None:
        stride = GPU_PARTICLE_STATE_FLOATS * 4
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)
        RAW_GL_VERTEX_ATTRIB_POINTER(
            0, 4, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0)
        )
        RAW_GL_VERTEX_ATTRIB_POINTER(
            1, 4, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(16)
        )

    @staticmethod
    def _unbind_state_attributes() -> None:
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _poll_gpu_timer(self) -> None:
        if not self.timer_pending:
            return
        query = self.timer_pending[0]
        try:
            available = int(GL.glGetQueryObjectiv(
                query, GL.GL_QUERY_RESULT_AVAILABLE
            ))
            if available:
                nanoseconds = int(GL.glGetQueryObjectui64v(
                    query, GL.GL_QUERY_RESULT
                ))
                self.gpu_frame_ms = nanoseconds / 1_000_000.0
                self.timer_pending.pop(0)
        except Exception as error:
            self.timer_failure = f"poll {type(error).__name__}: {error}"
            self.timer_queries = []
            self.timer_pending = []
            self.timer_active_query = 0

    def _begin_gpu_timer(self) -> None:
        if self.timer_active_query or not self.timer_queries:
            return
        busy = set(self.timer_pending)
        query = next(
            (candidate for candidate in self.timer_queries if candidate not in busy),
            0,
        )
        if not query:
            return
        try:
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
            self.timer_active_query = query
        except Exception as error:
            self.timer_failure = f"begin {type(error).__name__}: {error}"
            self.timer_queries = []
            self.timer_pending = []
            self.timer_active_query = 0

    def _end_gpu_timer(self) -> None:
        if not self.timer_active_query:
            return
        query = self.timer_active_query
        self.timer_active_query = 0
        try:
            GL.glEndQuery(GL.GL_TIME_ELAPSED)
            self.timer_pending.append(query)
        except Exception as error:
            self.timer_failure = f"end {type(error).__name__}: {error}"
            self.timer_queries = []
            self.timer_pending = []

    def step(
        self,
        dt: float,
        gravity: float,
        friction: float,
        quality: int,
        observed_fps: float = float(TARGET_FPS),
    ) -> None:
        if not self.enabled:
            return
        self.set_quality(quality, observed_fps, dt)
        self._poll_gpu_timer()
        self._begin_gpu_timer()
        source = self.buffers[self.source_index]
        destination_index = 1 - self.source_index
        destination = self.buffers[destination_index]
        try:
            GL.glUseProgram(self.update_program)
            self._bind_state_attributes(source)
            GL.glUniform1f(self._update_uniforms["u_dt"], clamp(float(dt), 0.0, 1.0 / 20.0))
            GL.glUniform1f(self._update_uniforms["u_gravity"], float(gravity))
            GL.glUniform1f(self._update_uniforms["u_friction"], float(friction))
            GL.glUniform1f(self._update_uniforms["u_time"], float(time.monotonic() % 4096.0))
            GL.glUniform3f(
                self._update_uniforms["u_room"], ROOM_WIDTH, ROOM_HEIGHT, ROOM_LENGTH
            )
            GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, destination)
            GL.glEnable(GL.GL_RASTERIZER_DISCARD)
            GL.glBeginTransformFeedback(GL.GL_POINTS)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.active_particles)
            GL.glEndTransformFeedback()
            GL.glDisable(GL.GL_RASTERIZER_DISCARD)
            GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0)
            self._unbind_state_attributes()
            GL.glUseProgram(0)
            self.source_index = destination_index
            self.simulated_frames += 1
        except Exception as error:
            try:
                GL.glEndTransformFeedback()
            except Exception:
                pass
            GL.glDisable(GL.GL_RASTERIZER_DISCARD)
            GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0)
            self._unbind_state_attributes()
            GL.glUseProgram(0)
            self.failure = f"{type(error).__name__}: {error}"
            self.enabled = False
            print(f"{TITLE} GPU physics disabled: {self.failure}")

    def draw(self) -> None:
        if not self.enabled or not self.simulated_frames:
            return
        GL.glPushAttrib(
            GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT | GL.GL_POINT_BIT
        )
        try:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_FOG)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glUseProgram(self.render_program)
            self._bind_state_attributes(self.buffers[self.source_index])
            GL.glDrawArrays(GL.GL_POINTS, 0, self.active_particles)
            self._unbind_state_attributes()
            GL.glUseProgram(0)
        finally:
            self._end_gpu_timer()
            GL.glDepthMask(GL.GL_TRUE)
            GL.glPopAttrib()

    def cleanup(self) -> None:
        self.enabled = False
        try:
            self._end_gpu_timer()
            GL.glUseProgram(0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            if self.buffers:
                GL.glDeleteBuffers(len(self.buffers), self.buffers)
                self.buffers.clear()
            if self.update_program:
                GL.glDeleteProgram(self.update_program)
                self.update_program = 0
            if self.render_program:
                GL.glDeleteProgram(self.render_program)
                self.render_program = 0
            if self.timer_queries:
                GL.glDeleteQueries(len(self.timer_queries), self.timer_queries)
                self.timer_queries.clear()
                self.timer_pending.clear()
        except Exception:
            pass


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
        cpu_cores: Optional[Sequence[int]] = None,
        cpu_backend: str = "auto",
        quality_mode: str = "auto",
        gpu_physics_mode: str = "auto",
        maximum_throughput: bool = False,
    ) -> None:
        self.maximum_throughput = bool(maximum_throughput)
        self.frame_cap = 0 if self.maximum_throughput else TARGET_FPS
        self.cpu_cores = tuple(cpu_cores) if cpu_cores is not None else configure_cpu_core_budget()
        worker_cores = self.cpu_cores[1:]
        self.spatial_workers = SpatialHashWorkerPool(worker_cores)
        pygame.display.init()
        pygame.font.init()
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        # A fixed 16:9 canvas keeps both telemetry columns non-overlapping and
        # avoids expensive OpenGL context recreation on Raspberry Pi drivers.
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        try:
            pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                flags,
                vsync=0 if self.maximum_throughput else 1,
            )
        except (TypeError, pygame.error):
            pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        pygame.display.set_caption(TITLE)
        self.width, self.height = pygame.display.get_surface().get_size()
        renderer_value = GL.glGetString(GL.GL_RENDERER)
        vendor_value = GL.glGetString(GL.GL_VENDOR)
        version_value = GL.glGetString(GL.GL_VERSION)

        def decode_gl(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", "replace")
            return str(value or "unknown")

        self.gl_renderer = decode_gl(renderer_value)
        self.gl_vendor = decode_gl(vendor_value)
        self.gl_version = decode_gl(version_value)
        self.hardware_gl = is_hardware_gl_renderer(self.gl_renderer)
        required_vbo_calls = (
            "glGenBuffers", "glBindBuffer", "glBufferData",
            "glDeleteBuffers", "glDrawArrays", "glDisableClientState",
        )
        self.gpu_vbo_batches = (
            self.hardware_gl and RAW_GL_INTERLEAVED_ARRAYS is not None and all(
            getattr(GL, name, None) is not None for name in required_vbo_calls
            )
        )
        self.gpu_vbo_error = ""
        self.clock = pygame.time.Clock()
        self.performance = AdaptivePerformanceController()
        quality_lookup = {name.lower(): index for index, name in enumerate(RENDER_QUALITY_NAMES)}
        self.quality_override = quality_lookup.get(str(quality_mode).lower())
        self.world = PhysicsWorld(
            seed,
            spatial_workers=self.spatial_workers,
            cpu_backend=cpu_backend,
        )
        self.audio = AudioEngine(audio_enabled)
        self.text = GLText()
        self.resource_monitor = ResourceMonitor(self.cpu_cores)
        self.running = True
        self.mouse_captured = not capture_mode
        self.show_help = True
        self.show_resources = True
        self.telemetry_groups = True
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        self.message_until = 0.0
        self.last_message = self.world.messages[-1]
        self.room_list = 0
        self.sphere_list = 0
        self.sphere_lists: Dict[int, int] = {}
        self.box_list = 0
        self.shadow_list = 0
        self.ceiling_glow_list = 0
        self.ceiling_panel_list = 0
        self.static_exhibit_list = 0
        self.static_exhibit_signature: Tuple[Any, ...] = ()
        self.brick_batch_lists: Dict[int, int] = {}
        self.brick_batch_vbos: Dict[int, Tuple[int, int]] = {}
        self.brick_batch_signatures: Dict[int, Tuple[Any, ...]] = {}
        self.concrete_points: List[Vec3] = []
        self.active_light_cell: Optional[Tuple[int, int, int]] = None
        self.active_lights: List[Tuple[float, float, float]] = []
        self.telemetry_next_update = 0.0
        self.telemetry_snapshot: List[Tuple[str, Tuple[int, int, int]]] = []
        self.selected_body_next_update = 0.0
        self.selected_body_snapshot: Optional[RigidBody] = None
        self._cull_eye = self.world.player.eye
        self._cull_forward = self.world.player.forward()
        self._cull_right = self.world.player.right()
        self.render_quality = (
            self.performance.quality_level
            if self.quality_override is None else self.quality_override
        )
        self.gpu_physics_mode = (
            "maximum" if self.maximum_throughput else str(gpu_physics_mode).lower()
        )
        self.gpu_physics: Optional[GPUPhysicsParticles] = None
        self.gpu_physics_error = ""
        core_label = "/".join(str(core) for core in self.cpu_cores)
        acceleration = "hardware" if self.hardware_gl else "SOFTWARE FALLBACK"
        print(
            f"{TITLE} performance: CPU cores {core_label}; "
            f"{self.spatial_workers.worker_count} workers; "
            f"GPU {self.gl_renderer} ({acceleration}, "
            f"{'course VBOs' if self.gpu_vbo_batches else 'display-list fallback'}); "
            f"OpenGL {self.gl_version}"
        )
        self._init_gl()
        if self.gpu_physics_mode != "off" and self.hardware_gl:
            required_gpu_physics = (
                RAW_GL_VERTEX_ATTRIB_POINTER,
                RAW_GL_TRANSFORM_FEEDBACK_VARYINGS,
                GL_SHADERS,
                getattr(GL, "glBindBufferBase", None),
                getattr(GL, "glBeginTransformFeedback", None),
                getattr(GL, "glEndTransformFeedback", None),
            )
            if all(required_gpu_physics):
                try:
                    self.gpu_physics = GPUPhysicsParticles(
                        seed, mode=self.gpu_physics_mode
                    )
                    print(
                        f"{TITLE} GPU physics: transform feedback ACTIVE; "
                        f"{self.gpu_physics.max_particles:,} capacity; "
                        f"{self.gpu_physics.buffer_bytes / (1024.0 * 1024.0):.1f} MiB "
                        "ping-pong state; zero CPU readback; "
                        f"timer queries "
                        f"{'ACTIVE' if self.gpu_physics.timer_queries else 'unavailable'}"
                    )
                except Exception as error:
                    self.gpu_physics_error = str(error)
                    print(f"{TITLE} GPU physics fallback: {self.gpu_physics_error}")
            else:
                self.gpu_physics_error = "transform-feedback entry points unavailable"
        elif self.gpu_physics_mode != "off":
            self.gpu_physics_error = "software OpenGL renderer"
        self._build_geometry()
        self._set_mouse_capture(self.mouse_captured)
        self.last_time = time.perf_counter()

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
        # The compatibility renderer keeps meshes in driver-side command
        # storage.  This avoids thousands of Python -> OpenGL calls per frame
        # on the Pi while retaining broad Mesa/VideoCore VII compatibility.
        self.box_list = GL.glGenLists(1)
        GL.glNewList(self.box_list, GL.GL_COMPILE)
        self._emit_unit_box()
        GL.glEndList()

        for quality, (slices, stacks) in enumerate(SPHERE_LOD):
            display_list = GL.glGenLists(1)
            GL.glNewList(display_list, GL.GL_COMPILE)
            for stack in range(stacks):
                latitude0 = -math.pi * 0.5 + math.pi * stack / stacks
                latitude1 = -math.pi * 0.5 + math.pi * (stack + 1) / stacks
                GL.glBegin(GL.GL_TRIANGLE_STRIP)
                for slice_index in range(slices + 1):
                    longitude = 2.0 * math.pi * slice_index / slices
                    for latitude in (latitude1, latitude0):
                        cosine = math.cos(latitude)
                        normal = Vec3(
                            cosine * math.sin(longitude), math.sin(latitude),
                            cosine * math.cos(longitude),
                        )
                        GL.glNormal3f(normal.x, normal.y, normal.z)
                        GL.glVertex3f(normal.x, normal.y, normal.z)
                GL.glEnd()
            GL.glEndList()
            self.sphere_lists[quality] = display_list
        self.sphere_list = self.sphere_lists[2]

        self.shadow_list = GL.glGenLists(1)
        GL.glNewList(self.shadow_list, GL.GL_COMPILE)
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(0.0, 0.0, 0.0)
        for index in range(25):
            angle = 2.0 * math.pi * index / 24.0
            GL.glVertex3f(math.cos(angle), 0.0, math.sin(angle))
        GL.glEnd()
        GL.glEndList()

        self.ceiling_glow_list = self._build_ceiling_quad_list(3.2, -0.015)
        self.ceiling_panel_list = self._build_ceiling_quad_list(2.35, -0.025)

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

    def _build_ceiling_quad_list(self, half_extent: float, y_offset: float) -> int:
        display_list = GL.glGenLists(1)
        GL.glNewList(display_list, GL.GL_COMPILE)
        GL.glBegin(GL.GL_QUADS)
        for x, y, z in LIGHT_POSITIONS:
            GL.glVertex3f(x - half_extent, y + y_offset, z - half_extent)
            GL.glVertex3f(x + half_extent, y + y_offset, z - half_extent)
            GL.glVertex3f(x + half_extent, y + y_offset, z + half_extent)
            GL.glVertex3f(x - half_extent, y + y_offset, z + half_extent)
        GL.glEnd()
        GL.glEndList()
        return display_list

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

        light_count = (2, 3, 4)[self.render_quality]
        cell = (
            int(eye.x // LIGHT_SPACING), int(eye.z // LIGHT_SPACING),
            light_count,
        )
        light_layout_changed = cell != self.active_light_cell
        if light_layout_changed:
            self.active_light_cell = cell
            self.active_lights = sorted(
                LIGHT_POSITIONS,
                key=lambda position: (position[0] - eye.x) ** 2 + (position[2] - eye.z) ** 2,
            )[:light_count]
            for index, _position in enumerate(self.active_lights):
                light = GL.GL_LIGHT0 + index
                GL.glEnable(light)
                GL.glLightfv(light, GL.GL_DIFFUSE, (1.0, 0.985, 0.91, 1.0))
                GL.glLightfv(light, GL.GL_SPECULAR, (1.0, 0.98, 0.88, 1.0))
                GL.glLightf(light, GL.GL_CONSTANT_ATTENUATION, 0.42)
                GL.glLightf(light, GL.GL_LINEAR_ATTENUATION, 0.035)
                GL.glLightf(light, GL.GL_QUADRATIC_ATTENUATION, 0.010)
            for index in range(len(self.active_lights), 8):
                GL.glDisable(GL.GL_LIGHT0 + index)
        # Positions are transformed by the current camera modelview and must
        # therefore be resent each frame; all other light state is invariant.
        for index, position in enumerate(self.active_lights):
            GL.glLightfv(
                GL.GL_LIGHT0 + index, GL.GL_POSITION,
                (position[0], position[1] - 0.12, position[2], 1.0),
            )

    def _draw_ceiling_fixtures(self) -> None:
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        if self.render_quality > 0:
            GL.glColor4f(0.35, 0.72, 1.0, 0.11)
            GL.glCallList(self.ceiling_glow_list)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glColor4f(1.0, 0.99, 0.91, 1.0)
        GL.glCallList(self.ceiling_panel_list)
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
        GL.glPushMatrix()
        GL.glTranslatef(position.x, 0.008, position.z)
        GL.glScalef(radius, 1.0, radius)
        GL.glCallList(self.shadow_list)
        GL.glPopMatrix()
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPopAttrib()

    def _emit_unit_box(self) -> None:
        GL.glBegin(GL.GL_QUADS)
        for normal, vertices in UNIT_BOX_FACES:
            GL.glNormal3f(*normal)
            for vertex in vertices:
                GL.glVertex3f(*vertex)
        GL.glEnd()

    def _draw_box(self, dimensions: Tuple[float, float, float]) -> None:
        GL.glPushMatrix()
        GL.glScalef(*dimensions)
        GL.glCallList(self.box_list)
        GL.glPopMatrix()

    def _draw_tapered_cylinder_x(
        self, length: float, radius_left: float, radius_right: float, segments: int = 16
    ) -> None:
        segments = min(segments, (8, 12, segments)[self.render_quality])
        half = length * 0.5
        GL.glBegin(GL.GL_QUAD_STRIP)
        for index in range(segments + 1):
            angle = 2.0 * math.pi * index / segments
            cosine, sine = math.cos(angle), math.sin(angle)
            GL.glNormal3f(0.0, cosine, sine)
            GL.glVertex3f(-half, cosine * radius_left, sine * radius_left)
            GL.glVertex3f(half, cosine * radius_right, sine * radius_right)
        GL.glEnd()
        for x, radius, normal_x in ((-half, radius_left, -1.0), (half, radius_right, 1.0)):
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glNormal3f(normal_x, 0.0, 0.0)
            GL.glVertex3f(x, 0.0, 0.0)
            for index in range(segments + 1):
                angle = 2.0 * math.pi * index / segments
                if normal_x < 0.0:
                    angle = -angle
                GL.glVertex3f(x, math.cos(angle) * radius, math.sin(angle) * radius)
            GL.glEnd()

    def _draw_torus_x(self, major_radius: float, tube_radius: float, rings: int = 18, sides: int = 10) -> None:
        quality_scale = (0.55, 0.78, 1.0)[self.render_quality]
        rings = max(8, int(rings * quality_scale))
        sides = max(6, int(sides * quality_scale))
        for ring in range(rings):
            u0 = 2.0 * math.pi * ring / rings
            u1 = 2.0 * math.pi * (ring + 1) / rings
            GL.glBegin(GL.GL_QUAD_STRIP)
            for side in range(sides + 1):
                v = 2.0 * math.pi * side / sides
                for u in (u0, u1):
                    cosine_v, sine_v = math.cos(v), math.sin(v)
                    radial = major_radius + tube_radius * cosine_v
                    x = tube_radius * sine_v
                    y = radial * math.cos(u)
                    z = radial * math.sin(u)
                    GL.glNormal3f(sine_v, cosine_v * math.cos(u), cosine_v * math.sin(u))
                    GL.glVertex3f(x, y, z)
            GL.glEnd()

    def _draw_local_box(
        self,
        position: Tuple[float, float, float],
        dimensions: Tuple[float, float, float],
        color: Tuple[float, float, float],
    ) -> None:
        GL.glPushMatrix()
        GL.glTranslatef(*position)
        GL.glColor3f(*color)
        self._draw_box(dimensions)
        GL.glPopMatrix()

    def _draw_bat_model(self, body: RigidBody) -> None:
        length = body.spec.dimensions[0]
        GL.glColor3f(*(body.color_override or body.spec.color))
        # Three frustums approximate the knob, narrow handle, taper, and barrel.
        GL.glPushMatrix(); GL.glTranslatef(-length * 0.42, 0.0, 0.0)
        self._draw_tapered_cylinder_x(length * 0.16, 0.020, 0.012, 16); GL.glPopMatrix()
        GL.glPushMatrix(); GL.glTranslatef(-length * 0.14, 0.0, 0.0)
        self._draw_tapered_cylinder_x(length * 0.42, 0.012, 0.019, 16); GL.glPopMatrix()
        GL.glPushMatrix(); GL.glTranslatef(length * 0.25, 0.0, 0.0)
        self._draw_tapered_cylinder_x(length * 0.46, 0.019, 0.03315, 18); GL.glPopMatrix()

    def _draw_noodle_model(self, body: RigidBody) -> None:
        color = body.color_override or body.spec.color
        segment_length = body.spec.dimensions[0] / 5.0
        phase = self.world.simulation_time * 2.1 + body.group_index * 0.8
        for index in range(5):
            local_x = (index - 2.0) * segment_length
            flex = math.sin(phase + index * 0.72) * min(0.055, body.velocity.length() * 0.004)
            GL.glPushMatrix()
            GL.glTranslatef(local_x, flex, 0.0)
            GL.glRotatef(math.degrees(flex * 2.2), 0.0, 0.0, 1.0)
            GL.glColor3f(*color)
            self._draw_tapered_cylinder_x(segment_length * 1.03, 0.03175, 0.03175, 12)
            GL.glPopMatrix()

    def _draw_wheelbarrow_model(self, body: RigidBody) -> None:
        # Open steel tray, hardwood handles/frame, two rear feet, one live wheel.
        tray = (0.21, 0.48, 0.29)
        wood = (0.48, 0.29, 0.12)
        steel = (0.30, 0.34, 0.31)
        # The tray floor coincides with the overall OBB's physical top, so
        # carried objects visibly rest on steel rather than floating above it.
        self._draw_local_box((0.0, 0.293, 0.12), (0.62, 0.10, 0.78), tray)
        self._draw_local_box((-0.30, 0.413, 0.12), (0.035, 0.28, 0.78), tray)
        self._draw_local_box((0.30, 0.413, 0.12), (0.035, 0.28, 0.78), tray)
        self._draw_local_box((0.0, 0.413, 0.49), (0.62, 0.28, 0.035), tray)
        self._draw_local_box((0.0, 0.383, -0.27), (0.62, 0.18, 0.035), tray)
        for x in (-0.245, 0.245):
            self._draw_local_box((x, -0.12, -0.25), (0.045, 0.045, 1.10), wood)
            self._draw_local_box((x, -0.203, -0.25), (0.035, 0.28, 0.035), steel)
        GL.glPushMatrix()
        GL.glTranslatef(0.0, -0.14, 0.55)
        GL.glRotatef(math.degrees(body.wheel_angle), 1.0, 0.0, 0.0)
        GL.glColor3f(0.055, 0.060, 0.065)
        self._draw_torus_x(0.154, 0.049, 20, 10)
        GL.glColor3f(0.58, 0.61, 0.62)
        self._draw_tapered_cylinder_x(0.22, 0.035, 0.035, 12)
        GL.glPopMatrix()

    def _draw_pallet_model(self) -> None:
        wood = (0.54, 0.35, 0.17)
        for z in (-0.34, -0.17, 0.0, 0.17, 0.34):
            self._draw_local_box((0.0, 0.045, z), (1.20, 0.040, 0.105), wood)
        for z in (-0.32, 0.0, 0.32):
            self._draw_local_box((0.0, -0.015, z), (1.12, 0.080, 0.075), (0.44, 0.27, 0.12))
        for x in (-0.48, 0.0, 0.48):
            for z in (-0.31, 0.0, 0.31):
                self._draw_local_box((x, -0.055, z), (0.15, 0.07, 0.12), (0.38, 0.23, 0.11))

    def _draw_scaled_sphere(
        self,
        position: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        color: Tuple[float, float, float],
    ) -> None:
        GL.glPushMatrix(); GL.glTranslatef(*position); GL.glScalef(*scale)
        GL.glColor3f(*color); GL.glCallList(self.sphere_lists[self.render_quality]); GL.glPopMatrix()

    def _draw_plush_model(self, body: RigidBody) -> None:
        radius = body.spec.radius
        color = body.color_override or body.spec.color
        dark = tuple(component * 0.56 for component in color)
        self._draw_scaled_sphere((0.0, -0.03, 0.0), (radius * 0.72, radius, radius * 0.62), color)
        self._draw_scaled_sphere((0.0, radius * 0.76, 0.02), (radius * 0.62, radius * 0.57, radius * 0.58), color)
        key = body.spec.key
        if "rabbit" in key:
            ears = ((-0.20, 1.30), (0.20, 1.30))
            for x_factor, y_factor in ears:
                self._draw_scaled_sphere((radius * x_factor, radius * y_factor, 0.0), (radius * 0.16, radius * 0.48, radius * 0.13), color)
        elif "octopus" in key:
            for index in range(6):
                angle = 2.0 * math.pi * index / 6.0
                self._draw_scaled_sphere((math.cos(angle) * radius * 0.62, -radius * 0.65, math.sin(angle) * radius * 0.62), (radius * 0.14, radius * 0.45, radius * 0.14), color)
        else:
            for sign in (-1.0, 1.0):
                self._draw_scaled_sphere((sign * radius * 0.40, radius * 1.15, 0.0), (radius * 0.22, radius * 0.25, radius * 0.18), color)
        for sign in (-1.0, 1.0):
            self._draw_scaled_sphere((sign * radius * 0.68, 0.0, 0.0), (radius * 0.20, radius * 0.62, radius * 0.20), color)
            self._draw_scaled_sphere((sign * radius * 0.32, -radius * 0.77, 0.0), (radius * 0.24, radius * 0.52, radius * 0.25), dark)
        self._draw_scaled_sphere((-radius * 0.22, radius * 0.86, radius * 0.50), (radius * 0.065,) * 3, (0.03, 0.03, 0.035))
        self._draw_scaled_sphere((radius * 0.22, radius * 0.86, radius * 0.50), (radius * 0.065,) * 3, (0.03, 0.03, 0.035))

    @staticmethod
    def _brick_chunk_signature(bodies: Sequence[RigidBody]) -> Tuple[Any, ...]:
        return tuple(
            (
                body.group_index,
                body.position.x, body.position.y, body.position.z,
                body.orientation.w, body.orientation.x,
                body.orientation.y, body.orientation.z,
                body.color_override or body.spec.color,
            )
            for body in bodies
        )

    def _delete_brick_vbo(self, chunk_index: int) -> None:
        batch = self.brick_batch_vbos.pop(chunk_index, None)
        if batch is not None:
            GL.glDeleteBuffers(1, [batch[0]])

    def _upload_brick_vbo(
        self,
        chunk_index: int,
        bodies: Sequence[RigidBody],
        prepared: Optional[Tuple[bytes, int]] = None,
    ) -> None:
        if prepared is None:
            vertex_data = build_static_box_batch_vertices(bodies)
            payload = vertex_data.tobytes()
            vertex_count = len(vertex_data) // 10
        else:
            payload, vertex_count = prepared
        buffer_id = 0
        try:
            buffer_id = int(GL.glGenBuffers(1))
            if not buffer_id:
                raise RuntimeError("OpenGL returned buffer id 0")
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                len(payload),
                payload,
                GL.GL_STATIC_DRAW,
            )
            self.brick_batch_vbos[chunk_index] = (buffer_id, vertex_count)
        except Exception:
            if buffer_id:
                GL.glDeleteBuffers(1, [buffer_id])
            raise
        finally:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _disable_brick_vbos(self, error: Exception) -> None:
        traceback_locations: List[str] = []
        traceback_cursor = error.__traceback__
        while traceback_cursor is not None:
            traceback_locations.append(
                f"{traceback_cursor.tb_frame.f_code.co_name}:{traceback_cursor.tb_lineno}"
            )
            traceback_cursor = traceback_cursor.tb_next
        location = " <- ".join(traceback_locations[-6:])
        self.gpu_vbo_error = (
            f"{type(error).__name__}: {error}"
            f"{f' at {location}' if location else ''}"
        )
        for chunk_index in list(self.brick_batch_vbos):
            self._delete_brick_vbo(chunk_index)
        self.gpu_vbo_batches = False
        self.brick_batch_signatures.clear()
        print(f"{TITLE} GPU course-VBO fallback: {self.gpu_vbo_error}")

    def _draw_brick_vbos(self, chunks: Dict[int, List[RigidBody]]) -> None:
        GL.glMaterialfv(
            GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.04, 0.025, 0.02, 1.0)
        )
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 3.0)
        try:
            for chunk_index in sorted(chunks):
                batch = self.brick_batch_vbos.get(chunk_index)
                if batch is None:
                    continue
                buffer_id, vertex_count = batch
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
                RAW_GL_INTERLEAVED_ARRAYS(
                    GL.GL_C4F_N3F_V3F, 0, ctypes.c_void_p(0)
                )
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, vertex_count)
        finally:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            for client_state in (
                GL.GL_COLOR_ARRAY, GL.GL_NORMAL_ARRAY, GL.GL_VERTEX_ARRAY
            ):
                GL.glDisableClientState(client_state)

    def _ensure_brick_batch(self, draw_batches: bool = True) -> set[int]:
        static_bricks = [
            body for body in self.world.bodies
            if body.spec.key == "clay_brick" and body.asleep
        ]
        # The pallet uses ten complete 48-brick courses and one 20-brick
        # crown.  Each course is one GPU-resident interleaved buffer on V3D;
        # one dirty course costs at most 48 boxes to rebuild.  Legacy display
        # lists remain a runtime fallback for compatibility/software contexts.
        chunks: Dict[int, List[RigidBody]] = {}
        for body in static_bricks:
            chunk_index = max(0, (body.group_index - 1) // 48)
            chunks.setdefault(chunk_index, []).append(body)

        chunk_indices = (
            set(chunks) | set(self.brick_batch_lists) | set(self.brick_batch_vbos)
        )
        signatures = {
            chunk_index: self._brick_chunk_signature(chunks.get(chunk_index, []))
            for chunk_index in chunk_indices
        }
        dirty_indices = [
            chunk_index for chunk_index in sorted(chunk_indices)
            if (
                chunk_index not in self.brick_batch_signatures
                or signatures[chunk_index]
                != self.brick_batch_signatures[chunk_index]
            )
        ]
        prepared_batches: Dict[int, Tuple[bytes, int]] = {}
        if self.gpu_vbo_batches:
            tasks: List[BoxBatchTask] = [
                (
                    chunk_index,
                    tuple(box_render_state(body) for body in chunks[chunk_index]),
                )
                for chunk_index in dirty_indices if chunks.get(chunk_index)
            ]
            worker_batches = self.spatial_workers.build_render_batches(tasks)
            if worker_batches is not None:
                prepared_batches = worker_batches
        try:
            for chunk_index in dirty_indices:
                bodies = chunks.get(chunk_index, [])
                signature = signatures[chunk_index]
                old_list = self.brick_batch_lists.pop(chunk_index, 0)
                if old_list:
                    GL.glDeleteLists(old_list, 1)
                self._delete_brick_vbo(chunk_index)
                if bodies and self.gpu_vbo_batches:
                    self._upload_brick_vbo(
                        chunk_index, bodies, prepared_batches.get(chunk_index)
                    )
                elif bodies:
                    display_list = GL.glGenLists(1)
                    GL.glNewList(display_list, GL.GL_COMPILE)
                    GL.glMaterialfv(
                        GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR,
                        (0.04, 0.025, 0.02, 1.0),
                    )
                    GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 3.0)
                    for body in bodies:
                        GL.glPushMatrix()
                        GL.glTranslatef(*body.position.tuple())
                        GL.glMultMatrixf(body.orientation.matrix())
                        GL.glColor3f(*(body.color_override or body.spec.color))
                        self._draw_box(body.spec.dimensions)
                        GL.glPopMatrix()
                    GL.glEndList()
                    self.brick_batch_lists[chunk_index] = display_list
                self.brick_batch_signatures[chunk_index] = signature
        except Exception as error:
            if self.gpu_vbo_batches:
                self._disable_brick_vbos(error)
                return self._ensure_brick_batch(draw_batches)
            raise

        if draw_batches:
            if self.gpu_vbo_batches:
                try:
                    self._draw_brick_vbos(chunks)
                except Exception as error:
                    self._disable_brick_vbos(error)
                    return self._ensure_brick_batch(draw_batches)
            else:
                for chunk_index in sorted(chunks):
                    display_list = self.brick_batch_lists.get(chunk_index, 0)
                    if display_list:
                        GL.glCallList(display_list)
        return {id(body) for body in static_bricks}

    def _ensure_static_exhibit_batch(self) -> set[int]:
        """Submit sleeping non-brick exhibits through one GPU command list."""
        static_bodies = [
            body for body in self.world.bodies
            if body.asleep
            and not body.held
            and body.spec.key != "clay_brick"
            and body.spec.category != "balloon"
        ]
        signature: Tuple[Any, ...] = (
            self.render_quality,
            tuple(
                (
                    id(body), body.spec.key,
                    body.position.x, body.position.y, body.position.z,
                    body.orientation.w, body.orientation.x,
                    body.orientation.y, body.orientation.z,
                )
                for body in static_bodies
            ),
        )
        if signature != self.static_exhibit_signature:
            if self.static_exhibit_list:
                GL.glDeleteLists(self.static_exhibit_list, 1)
                self.static_exhibit_list = 0
            if static_bodies:
                self.static_exhibit_list = GL.glGenLists(1)
                GL.glNewList(self.static_exhibit_list, GL.GL_COMPILE)
                for body in static_bodies:
                    self._draw_body(body, 1.0)
                GL.glEndList()
            self.static_exhibit_signature = signature
        if self.static_exhibit_list:
            GL.glCallList(self.static_exhibit_list)
        return {id(body) for body in static_bodies}

    def _sphere_visible(self, position: Vec3, radius: float) -> bool:
        """Conservative camera-space cull before submitting a model to Mesa."""
        offset = position - self._cull_eye
        depth = offset.dot(self._cull_forward)
        if depth < -radius or depth - radius > 175.0:
            return False
        if depth <= 0.0:
            return offset.length_squared() <= (radius + 0.5) ** 2
        # 72-degree vertical FOV at 16:9 is about 104 degrees horizontally.
        # Generous padding prevents visible edge popping for elongated bodies.
        lateral = abs(offset.dot(self._cull_right))
        vertical = abs(offset.y - self._cull_forward.y * depth)
        return (
            lateral <= depth * 1.45 + radius * 2.0
            and vertical <= depth * 0.86 + radius * 2.0
        )

    def _draw_surface_details(self, body: RigidBody) -> None:
        if self.render_quality == 0:
            return
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
        draw_shadow = (
            self.render_quality == 2
            or (self.render_quality == 1 and body.bounding_radius >= 0.18)
        )
        if draw_shadow and body.spec.category not in {"bricks", "marbles", "goo"}:
            self._draw_shadow(body, position)
        speed = body.velocity.length()
        if speed > 12.0:
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_LINE_BIT)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glLineWidth(clamp(1.0 + math.log10(speed), 1.0, 4.0))
            GL.glColor4f(*(body.color_override or body.spec.color), clamp(math.log10(speed) / 5.0, 0.12, 0.65))
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(position.x, position.y, position.z)
            tail = position - body.velocity.normalized() * min(4.0, 0.05 * speed)
            GL.glVertex3f(tail.x, tail.y, tail.z)
            GL.glEnd()
            GL.glPopAttrib()

        GL.glPushMatrix()
        GL.glTranslatef(position.x, position.y, position.z)
        GL.glMultMatrixf(body.orientation.matrix())
        color = body.color_override or body.spec.color
        GL.glColor3f(*color)
        if body.spec.key in {"steel", "ceramic_marble"}:
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 112.0 if body.spec.key == "steel" else 94.0)
        elif body.spec.key == "concrete":
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.05, 0.05, 0.05, 1.0))
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 4.0)
        else:
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.28, 0.30, 0.31, 1.0))
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 28.0)
        kind = body.spec.render_kind
        if kind == "bat":
            self._draw_bat_model(body)
        elif kind == "wheelbarrow":
            self._draw_wheelbarrow_model(body)
        elif kind == "noodle":
            self._draw_noodle_model(body)
        elif kind == "pallet":
            self._draw_pallet_model()
        elif kind == "plush":
            self._draw_plush_model(body)
        elif kind == "balloon":
            radius = body.spec.radius
            wobble = 1.0 + 0.025 * math.sin(self.world.simulation_time * 3.2 + body.group_index)
            self._draw_scaled_sphere((0.0, 0.0, 0.0), (radius * 0.96 / wobble, radius * 1.08 * wobble, radius * 0.96 / wobble), color)
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_LINE_BIT)
            GL.glDisable(GL.GL_LIGHTING); GL.glColor4f(0.82, 0.82, 0.86, 0.8); GL.glLineWidth(1.0)
            GL.glBegin(GL.GL_LINE_STRIP)
            for index in range(10):
                y = -radius * 1.04 - index * 0.085
                GL.glVertex3f(math.sin(index * 0.9 + self.world.simulation_time) * 0.018, y, 0.0)
            GL.glEnd(); GL.glPopAttrib()
        elif kind == "goo":
            radius = body.spec.radius
            flatten = 0.32 if body.stuck_surface or body.stuck_to is not None else 0.72
            pulse = 1.0 + 0.10 * math.sin(self.world.simulation_time * 4.0 + body.group_index)
            self._draw_scaled_sphere((0.0, 0.0, 0.0), (radius * 1.28 * pulse, radius * flatten, radius * 1.18 / pulse), color)
        elif body.spec.shape == "sphere":
            squash = 1.0
            if body.spec.key == "dodge" and body.last_impulse > 0.0:
                squash = 1.0 - min(0.10, body.last_impulse / 900.0)
            GL.glScalef(body.spec.radius / math.sqrt(squash), body.spec.radius * squash, body.spec.radius / math.sqrt(squash))
            GL.glCallList(self.sphere_lists[self.render_quality])
            self._draw_surface_details(body)
        else:
            self._draw_box(body.spec.dimensions)
            if kind != "clay_brick" and self.render_quality == 2:
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
        telemetry_height = 270 if self.telemetry_groups else 230
        draw_panel(telemetry_x, 18, telemetry_width, telemetry_height)
        telemetry_title = (
            "551-BODY GROUP TELEMETRY  [F2: CORE 7]"
            if self.telemetry_groups else "CORE 7 VECTOR TELEMETRY  [F2: GROUPS]"
        )
        self.text.draw(telemetry_title, telemetry_x + 14, 28, white, "small")
        y = 57
        if now >= self.telemetry_next_update or not self.telemetry_snapshot:
            self.telemetry_next_update = now + (0.50, 0.22, 0.10)[self.render_quality]
            self.telemetry_snapshot = []
            if self.telemetry_groups:
                for label in self.world.category_summary_rows():
                    self.telemetry_snapshot.append((label, cyan))
            else:
                core_bodies = [
                    next(body for body in self.world.bodies if body.spec.key == key)
                    for key in ORIGINAL_ITEM_KEYS
                ]
                for body, label in zip(core_bodies, self.world.velocity_telemetry_rows()):
                    velocity = body.velocity.length()
                    color = orange if body.held else cyan if velocity >= 0.05 else muted
                    self.telemetry_snapshot.append((label, color))
        for index, (label, color) in enumerate(self.telemetry_snapshot):
            self.text.draw(label, telemetry_x + 14, y, color, "tiny", dynamic_slot=f"telemetry-{index}")
            y += 21 if self.telemetry_groups else 25

        if self.world.held_body is not None:
            self.selected_body_snapshot = self.world.held_body
        elif now >= self.selected_body_next_update:
            # Right-click performs its own immediate raycast.  The HUD only
            # needs a responsive visual sample, not a 551-body query at 60 Hz.
            self.selected_body_snapshot = self.world.selected_body()
            self.selected_body_next_update = now + (0.12, 0.065, 0.034)[self.render_quality]
        selected = self.selected_body_snapshot
        if selected is not None:
            info_width = min(520, self.width - 36)
            info_y = self.height - 178
            draw_panel(18, info_y, info_width, 116, 0.72)
            spec = selected.spec
            self.text.draw(selected.display_name, 31, info_y + 10, orange, "normal")
            mass_label = f"physical {spec.mass:.3f} kg"
            if spec.added_mass > 0.0:
                mass_label += f" | effective inertia {selected.dynamic_mass:.3f} kg"
            self.text.draw(
                f"{mass_label} | {spec.dimension_label} | "
                f"{'effective bulk density' if spec.category in {'balloon', 'plush', 'noodles'} or spec.key.startswith('medicine') or spec.key == 'dodge' else 'density'} "
                f"{spec.density:,.1f} kg/m^3" if spec.density > 0.0 else f"{mass_label} | {spec.dimension_label}",
                31, info_y + 37, white, "tiny",
            )
            self.text.draw(
                f"v=({selected.velocity.x:+.3f}, {selected.velocity.y:+.3f}, {selected.velocity.z:+.3f}) "
                f"|v|={selected.velocity.length():.3f} m/s",
                31, info_y + 59, cyan, "tiny", dynamic_slot="selected-velocity",
            )
            adhesion = " | ADHERED" if selected.stuck_surface or selected.stuck_to is not None else ""
            self.text.draw(
                f"restitution {spec.restitution:.2f} | material mu {spec.friction:.2f} | {spec.rigidity}{adhesion}",
                31, info_y + 81, muted, "tiny",
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
            prompt = f"LMB THROW {self.world.held_body.display_name.upper()}  |  RMB DROP"
        elif selected is not None:
            prompt = f"RMB PICK UP {selected.display_name.upper()}"
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
            self.text.draw("R/F gravity | T/G force | Y/H friction | F2 data | F3 resources | TAB mouse | M mute | F5 reset", 31, 193, cyan, "tiny")
            self.text.draw(
                "PLAYER 2.00 m / 80 kg | ROOM 100 x 100 x 10 m | force uses a 0.75 m arm stroke",
                31, 214, muted, "tiny",
            )

        self.resource_monitor.sample(now)
        if self.show_resources:
            resource_y = 246 if self.show_help else 158
            draw_panel(18, resource_y, 480, 77, 0.69)
            core_values = "  ".join(
                f"C{core} {self.resource_monitor.core_loads.get(core, 0.0):4.0f}%"
                for core in self.cpu_cores
            )
            self.text.draw(
                f"LIVE FOUR-CORE LOAD  {core_values}",
                31, resource_y + 10, white, "tiny",
                dynamic_slot="resource-cores",
            )
            used_mib = max(
                0.0,
                self.resource_monitor.memory_total_mib
                - self.resource_monitor.memory_available_mib,
            )
            self.text.draw(
                f"RAM APP RSS {self.resource_monitor.rss_mib:,.0f} MiB | "
                f"SYSTEM ACTIVE/CACHE {used_mib:,.0f} MiB | "
                f"AVAILABLE {self.resource_monitor.memory_available_mib:,.0f} MiB",
                31, resource_y + 32, cyan, "tiny",
                dynamic_slot="resource-memory",
            )
            if self.gpu_physics is not None and self.gpu_physics.enabled:
                gpu_time_label = (
                    f"{self.gpu_physics.gpu_frame_ms:.2f} ms"
                    if self.gpu_physics.timer_queries else "driver N/A"
                )
                gpu_physics_line = (
                    f"GPU PHYSICS TF {self.gpu_physics.active_particles:,}/"
                    f"{self.gpu_physics.max_particles:,} | "
                    f"STATE {self.gpu_physics.buffer_bytes / (1024.0 * 1024.0):.1f} MiB | "
                    f"GPU FRAME {gpu_time_label}"
                )
                gpu_line_color = (108, 238, 158)
            else:
                gpu_physics_line = (
                    "GPU PHYSICS OFF"
                    + (f" | {self.gpu_physics_error[:46]}" if self.gpu_physics_error else "")
                )
                gpu_line_color = orange
            self.text.draw(
                gpu_physics_line, 31, resource_y + 54, gpu_line_color, "tiny",
                dynamic_slot="resource-gpu",
            )

        audio_state = "MUTED" if self.audio.muted else "AUDIO" if self.audio.available else "SILENT FALLBACK"
        performance_color = (
            (255, 92, 88) if self.performance.below_floor
            else orange if self.performance.display_fps < 55.0
            else (108, 238, 158)
        )
        drop_label = (
            f" | DROP {self.performance.backlog_drops}"
            if self.performance.backlog_drops else ""
        )
        core_label = "/".join(str(core) for core in self.cpu_cores)
        worker_mode = (
            self.spatial_workers.last_mode.upper()
            if self.spatial_workers.available else "FALLBACK"
        )
        gpu_state = (
            "GPU:HW+VBO+TF" if self.gpu_vbo_batches and self.gpu_physics is not None
            else "GPU:HW+VBO" if self.gpu_vbo_batches
            else "GPU:HW+LIST" if self.hardware_gl
            else "GPU:SOFTWARE"
        )
        gpu_color = (108, 238, 158) if self.hardware_gl else (255, 183, 72)
        renderer_label = self.gl_renderer[:46]
        self.text.draw(
            f"CPU {len(self.cpu_cores)}C [{core_label}] | "
            f"{self.world.last_broadphase_backend} "
            f"{self.world.last_broadphase_ms:.2f}ms | "
            f"WORKERS {self.spatial_workers.worker_count} {worker_mode} | "
            f"JOBS P{self.spatial_workers.parallel_dispatches}/"
            f"G{self.spatial_workers.render_dispatches} | "
            f"{gpu_state} {renderer_label}",
            18, self.height - 47, gpu_color, "tiny", dynamic_slot="hardware-performance",
        )
        self.text.draw(
            f"FPS {self.performance.display_fps:5.1f} [{self.performance.display_frame_ms:4.1f}ms] | "
            f"{'AUTO' if self.quality_override is None else 'LOCK'}:{RENDER_QUALITY_NAMES[self.render_quality]} | "
            f"GOAL {TARGET_FPS} FLOOR {MINIMUM_RENDER_FPS} | "
            f"PHYS {PHYSICS_HZ}Hz | {audio_state}{drop_label}",
            18, self.height - 29, performance_color, "tiny", dynamic_slot="frame-performance",
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

    def _set_render_quality(self, quality: int) -> None:
        quality = int(clamp(float(quality), 0.0, 2.0))
        if quality != self.render_quality:
            self.render_quality = quality
            # Re-select the nearest light set with the new tier's light count.
            self.active_light_cell = None

    def _update_render_quality(self, frame_dt: float) -> None:
        self.performance.observe(frame_dt)
        self._set_render_quality(
            self.performance.quality_level
            if self.quality_override is None else self.quality_override
        )

    def render(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._configure_camera_and_lights()
        self._cull_eye = self.world.player.eye
        self._cull_forward = self.world.player.forward()
        self._cull_right = self.world.player.right()
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_FOG)
        GL.glCallList(self.room_list)
        self._draw_ceiling_fixtures()
        alpha = clamp(self.accumulator / FIXED_DT, 0.0, 1.0)
        stack_visible = self._sphere_visible(Vec3(50.0, 0.75, 28.0), 2.0)
        batched_bricks = self._ensure_brick_batch(stack_visible)
        batched_exhibits = self._ensure_static_exhibit_batch()
        for body in self.world.bodies:
            if id(body) in batched_bricks or id(body) in batched_exhibits:
                continue
            position = body.previous_position * (1.0 - alpha) + body.position * alpha
            if self._sphere_visible(position, body.bounding_radius):
                self._draw_body(body, alpha)
            body.last_impulse *= 0.82
        if self.gpu_physics is not None:
            self.gpu_physics.draw()
        self._draw_hud()

    def _handle_keydown(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_TAB:
            self._set_mouse_capture(not self.mouse_captured)
        elif key == pygame.K_F1:
            self.show_help = not self.show_help
        elif key == pygame.K_F2:
            self.telemetry_groups = not self.telemetry_groups
            self.telemetry_snapshot = []
        elif key == pygame.K_F3:
            self.show_resources = not self.show_resources
        elif key == pygame.K_F5:
            self.world.reset()
            self.telemetry_snapshot = []
            self.selected_body_snapshot = None
            self.selected_body_next_update = 0.0
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
        step_limit = QUALITY_MAX_PHYSICS_STEPS[self.render_quality]
        physics_budget = QUALITY_PHYSICS_BUDGET_MS[self.render_quality] / 1000.0
        physics_started = time.perf_counter()
        impact_buckets: Dict[Tuple[str, int, int], ImpactEvent] = {}
        while self.accumulator >= FIXED_DT and steps < step_limit:
            self.world.step(FIXED_DT)
            for impact in self.world.impacts:
                bucket = (
                    impact.family,
                    int(math.floor(impact.position.x / 3.0)),
                    int(math.floor(impact.position.z / 3.0)),
                )
                previous = impact_buckets.get(bucket)
                if previous is None or impact.impulse > previous.impulse:
                    impact_buckets[bucket] = impact
            if self.world.player.landing_speed > 0.0:
                volume = clamp((self.world.player.landing_speed - 0.5) / 7.0, 0.18, 1.0)
                self.audio.play("land", volume)
                self.world.player.landing_speed = 0.0
            self.accumulator -= FIXED_DT
            steps += 1
            if time.perf_counter() - physics_started >= physics_budget:
                break
        strongest = sorted(
            impact_buckets.values(), key=lambda event: event.impulse, reverse=True
        )[:QUALITY_IMPACT_SOUND_LIMITS[self.render_quality]]
        for impact in strongest:
            self.audio.play_impact(impact, self.world.player)
        if self.accumulator >= FIXED_DT:
            self.accumulator = 0.0
            self.performance.note_backlog_drop()
        if self.gpu_physics is not None:
            self.gpu_physics.step(
                frame_dt,
                self.world.gravity,
                self.world.room_friction,
                self.render_quality,
                self.performance.display_fps,
            )

    def capture_frame(self, path: str) -> None:
        pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        surface = pygame.image.fromstring(pixels, (self.width, self.height), "RGB", True)
        pygame.image.save(surface, path)

    def run(self, capture_path: Optional[str] = None, maximum_frames: int = 0) -> int:
        frame_count = 0
        frame_times: List[float] = []
        run_started = time.perf_counter()
        simulation_started = self.world.simulation_time
        self.last_time = run_started
        try:
            while self.running:
                # Busy-loop pacing avoids SDL_Delay overshoot on Raspberry Pi
                # and holds a materially steadier 60 Hz cadence.  The explicit
                # maximum-throughput profile passes zero to remove the cap for
                # hardware profiling; ordinary play remains thermally sane.
                self.clock.tick_busy_loop(self.frame_cap)
                frame_now = time.perf_counter()
                frame_dt = frame_now - self.last_time
                self.last_time = frame_now
                self._update_render_quality(frame_dt)
                for event in pygame.event.get():
                    self._handle_event(event)
                self.tick(frame_dt)
                self.render()
                pygame.display.flip()
                if maximum_frames:
                    frame_times.append(frame_dt)
                frame_count += 1
                if capture_path and frame_count >= 3:
                    self.render()
                    self.capture_frame(capture_path)
                    break
                if maximum_frames and frame_count >= maximum_frames:
                    break
            if maximum_frames:
                elapsed = max(1.0e-9, time.perf_counter() - run_started)
                warmup = min(120, max(0, len(frame_times) // 5))
                steady = frame_times[warmup:] or frame_times or [elapsed]
                ordered = sorted(steady)

                def percentile(fraction: float) -> float:
                    index = min(
                        len(ordered) - 1,
                        max(0, math.ceil(len(ordered) * fraction) - 1),
                    )
                    return ordered[index]

                worst_count = max(1, math.ceil(len(ordered) * 0.01))
                worst_one_percent = ordered[-worst_count:]
                one_percent_low = worst_count / max(
                    1.0e-9, sum(worst_one_percent)
                )
                over_floor = sum(
                    sample > 1.0 / MINIMUM_RENDER_FPS for sample in steady
                )
                longest_floor_streak = current_floor_streak = 0
                for sample in steady:
                    if sample > 1.0 / MINIMUM_RENDER_FPS:
                        current_floor_streak += 1
                        longest_floor_streak = max(
                            longest_floor_streak, current_floor_streak
                        )
                    else:
                        current_floor_streak = 0
                average_frame = sum(steady) / len(steady)
                simulation_ratio = (
                    self.world.simulation_time - simulation_started
                ) / elapsed
                self.resource_monitor.sample(time.monotonic(), force=True)
                core_load_label = "/".join(
                    f"{self.resource_monitor.core_loads.get(core, 0.0):.0f}%"
                    for core in self.cpu_cores
                )
                print(
                    f"{TITLE} benchmark: {frame_count} frames in {elapsed:.3f}s = "
                    f"{frame_count / elapsed:.2f} FPS; steady {1.0 / average_frame:.2f} FPS; "
                    f"1% low {one_percent_low:.2f} FPS; "
                    f"p95/p99/worst {percentile(0.95) * 1000.0:.2f}/"
                    f"{percentile(0.99) * 1000.0:.2f}/{ordered[-1] * 1000.0:.2f} ms; "
                    f"over 33.3ms {over_floor}/{len(steady)} streak {longest_floor_streak}; "
                    f"sim {simulation_ratio:.3f}x; "
                    f"quality {RENDER_QUALITY_NAMES[self.render_quality]}; "
                    f"broadphase {self.world.last_broadphase_backend} "
                    f"{self.world.last_broadphase_ms:.3f}ms; "
                    f"workers {self.spatial_workers.last_mode}; "
                    f"dispatches {self.spatial_workers.parallel_dispatches}; "
                    f"serial {self.spatial_workers.serial_fallbacks}; "
                    f"work {self.spatial_workers.last_work_estimate}; "
                    f"geometry jobs {self.spatial_workers.render_dispatches}/"
                    f"{self.spatial_workers.render_batches} "
                    f"({self.spatial_workers.last_render_build_ms:.2f}ms); "
                    f"GPU {'VBO' if self.gpu_vbo_batches else 'LIST'}; "
                    f"GPU physics "
                    f"{self.gpu_physics.active_particles if self.gpu_physics is not None and self.gpu_physics.enabled else 0:,} TF "
                    f"{self.gpu_physics.gpu_frame_ms if self.gpu_physics is not None and self.gpu_physics.timer_queries else 0.0:.2f}ms"
                    f"{(('(N/A ' + self.gpu_physics.timer_failure[:36] + ')') if self.gpu_physics.timer_failure else '(driver N/A)') if self.gpu_physics is not None and not self.gpu_physics.timer_queries else ''}; "
                    f"core load {core_load_label}; "
                    f"RSS {self.resource_monitor.rss_mib:.0f}MiB; "
                    f"RAM available {self.resource_monitor.memory_available_mib:.0f}MiB"
                )
            return 0
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        try:
            self._set_mouse_capture(False)
            if self.gpu_physics is not None:
                self.gpu_physics.cleanup()
                self.gpu_physics = None
            if self.text is not None:
                self.text.cleanup()
            for display_list in self.sphere_lists.values():
                GL.glDeleteLists(display_list, 1)
            self.sphere_lists.clear()
            self.sphere_list = 0
            for attribute in (
                "box_list", "shadow_list", "ceiling_glow_list",
                "ceiling_panel_list", "static_exhibit_list",
            ):
                display_list = getattr(self, attribute, 0)
                if display_list:
                    GL.glDeleteLists(display_list, 1)
                    setattr(self, attribute, 0)
            if self.room_list:
                GL.glDeleteLists(self.room_list, 1)
                self.room_list = 0
            for display_list in self.brick_batch_lists.values():
                GL.glDeleteLists(display_list, 1)
            self.brick_batch_lists.clear()
            for chunk_index in list(self.brick_batch_vbos):
                self._delete_brick_vbo(chunk_index)
            self.brick_batch_signatures.clear()
        finally:
            try:
                self.spatial_workers.close()
            finally:
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
    original_specs = [ITEM_SPEC_BY_KEY[key] for key in ORIGINAL_ITEM_KEYS]
    verify("original six balls plus rubber brick preserved", (
        len(original_specs) == 7
        and sum(spec.shape == "sphere" for spec in original_specs) == 6
        and sum(spec.shape == "box" for spec in original_specs) == 1
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
    verify("four-core process architecture", (
        CPU_CORE_BUDGET == 4
        and SPATIAL_WORKER_COUNT == 3
        and PARALLEL_SPATIAL_WORK_MIN >= 100_000
    ))
    verify("bounded GPU physics state budget", (
        GPU_PHYSICS_TIER_COUNTS == (32_768, 98_304, 196_608)
        and GPU_PHYSICS_MAX_PARTICLES * GPU_PARTICLE_STATE_FLOATS * 4 * 2
        == 16 * 1024 * 1024
    ))

    world = PhysicsWorld(2468)
    spatial_probe = world.bodies[0]
    spatial_state: SpatialState = (
        0,
        spatial_probe.position.x,
        spatial_probe.position.y,
        spatial_probe.position.z,
        spatial_probe.previous_position.x,
        spatial_probe.previous_position.y,
        spatial_probe.previous_position.z,
        spatial_probe.bounding_radius,
    )
    worker_probe = dict(_spatial_cell_chunk((spatial_state,)))[0]
    verify("worker spatial kernel exactly matches serial broadphase", (
        set(worker_probe) == set(world._body_swept_cell_keys(spatial_probe))
    ))
    verify("player starts at floor coordinate 10 x 10", world.player.position.tuple() == PLAYER_START and world.player.grounded)
    core_bodies = world.bodies[:7]
    initial_line = all(
        abs(body.position.x - (7.0 + index)) < 1.0e-12
        and abs(body.position.z - 15.0) < 1.0e-12
        for index, body in enumerate(core_bodies)
    )
    verify("original items five metres ahead and one metre apart", initial_line)
    verify("original lineup non-overlapping", all(
        (second.position - first.position).length() > first.bounding_radius + second.bounding_radius
        for index, first in enumerate(core_bodies)
        for second in core_bodies[index + 1:]
    ))
    expected_categories = {
        "calibration": 7, "bat": 1, "wheelbarrow": 1, "balloon": 1,
        "noodles": 5, "goo": 10, "marbles": 20, "bricks": 500,
        "pallet": 1, "plush": 5,
    }
    category_counts = {
        category: sum(body.spec.category == category for body in world.bodies)
        for category in expected_categories
    }
    verify("exact 551-body expansion population", len(world.bodies) == EXPECTED_BODY_COUNT and category_counts == expected_categories)
    brick_batch_probe = build_static_box_batch_vertices([
        body for body in world.bodies if body.spec.key == "clay_brick"
    ][:2])
    verify("GPU-resident brick batch payload", (
        brick_batch_probe.itemsize == 4
        and len(brick_batch_probe) == 2 * 6 * 6 * 10
        and all(math.isfinite(value) for value in brick_batch_probe)
    ))
    verify("five randomly selected unique stuffed animals", len({body.spec.key for body in world.bodies if body.spec.category == "plush"}) == 5)
    verify("realistic bat and wheelbarrow profiles", (
        abs(ITEM_SPEC_BY_KEY["wood_bat"].mass - 0.8788) < 1.0e-9
        and ITEM_SPEC_BY_KEY["wood_bat"].dimensions == (0.8636, 0.0663, 0.0663)
        and abs(ITEM_SPEC_BY_KEY["wheelbarrow"].mass - 17.28) < 1.0e-9
        and ITEM_SPEC_BY_KEY["wheelbarrow"].dimensions == (0.648, 0.686, 1.492)
    ))
    verify("realistic balloon noodle goo and ceramic profiles", (
        abs(ITEM_SPEC_BY_KEY["helium_balloon"].buoyancy_volume - 0.4248) < 1.0e-9
        and abs(ITEM_SPEC_BY_KEY["foam_noodle"].dimensions[0] - 1.1938) < 1.0e-9
        and ITEM_SPEC_BY_KEY["sticky_goo"].adhesion_strength == 25.0
        and abs(ITEM_SPEC_BY_KEY["ceramic_marble"].diameter - 0.050) < 1.0e-12
    ))
    verify("500 bricks fit EPAL working load", (
        abs(ITEM_SPEC_BY_KEY["clay_brick"].mass * CLAY_BRICK_COUNT + ITEM_SPEC_BY_KEY["wood_pallet"].mass - 975.0) < 1.0e-8
        and all(body.asleep and body.pristine for body in world.bodies if body.spec.key == "clay_brick")
    ))
    verify("spatial broadphase avoids all-pairs scene", len(world._broadphase_pairs()) < 1_000)
    if np is not None:
        parity_world = PhysicsWorld(2468, cpu_backend="auto")
        parity_active: List[int] = []
        for index, body in enumerate(parity_world.bodies[:24]):
            body.asleep = False
            body.previous_position = body.position - Vec3(
                0.06 * ((index % 3) - 1),
                0.02 * (index % 2),
                0.04 * ((index % 5) - 2),
            )
            parity_active.append(index)
        neon_pairs = set(parity_world._numpy_broadphase_pairs(parity_active))
        exact_pairs: set[Tuple[int, int]] = set()
        parity_active_set = set(parity_active)
        for first in parity_active:
            for second in range(len(parity_world.bodies)):
                if first == second:
                    continue
                if second in parity_active_set and first > second:
                    continue
                body_a = parity_world.bodies[first]
                body_b = parity_world.bodies[second]
                start = body_b.previous_position - body_a.previous_position
                motion = (
                    (body_b.position - body_b.previous_position)
                    - (body_a.position - body_a.previous_position)
                )
                motion2 = motion.length_squared()
                fraction = (
                    clamp(-start.dot(motion) / motion2, 0.0, 1.0)
                    if motion2 > 1.0e-14 else 0.0
                )
                closest = start + motion * fraction
                radius = (
                    body_a.bounding_radius + body_b.bounding_radius
                    + BROADPHASE_SKIN
                )
                if closest.length_squared() <= radius * radius:
                    exact_pairs.add(
                        (first, second) if first < second else (second, first)
                    )
        verify(
            "NEON swept broadphase matches exact swept-sphere candidates",
            neon_pairs == exact_pairs,
            f"NEON-only {sorted(neon_pairs - exact_pairs)[:3]}, "
            f"exact-only {sorted(exact_pairs - neon_pairs)[:3]}",
        )
    else:
        verify("scalar fallback works without optional NumPy", world.cpu_backend == "scalar")

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
    stacked_world._rebuild_static_broadphase_index()
    stack_stayed_inside = True
    for _ in range(1_200):
        stacked_world.step(FIXED_DT)
        stack_stayed_inside = stack_stayed_inside and all(
            body.position.y + 1.0e-8 >= body.support_extent(Vec3(0.0, 1.0, 0.0))
            for body in stacked_world.bodies
        )
    verify("stacked contacts never escape floor", stack_stayed_inside)
    verify("resting stack reaches low residual speed", max(body.velocity.length() for body in stacked_world.bodies) < 0.03)

    box_probe_world = PhysicsWorld(56)
    box_a = RigidBody(ITEM_SPEC_BY_KEY["clay_brick"], Vec3(50.0, 2.0, 50.0))
    box_b = RigidBody(ITEM_SPEC_BY_KEY["clay_brick"], Vec3(50.15, 2.0, 50.0))
    overlapping_contact = box_probe_world._box_box_contact(box_a, box_b)
    box_b.position.x = 50.40
    separated_contact = box_probe_world._box_box_contact(box_a, box_b)
    verify("OBB clay-brick contact and separation", overlapping_contact is not None and separated_contact is None)

    balloon_world = PhysicsWorld(57)
    floating = RigidBody(ITEM_SPEC_BY_KEY["helium_balloon"], Vec3(50.0, 4.0, 50.0))
    balloon_world.bodies = [floating]
    balloon_world._rebuild_static_broadphase_index()
    balloon_world._integrate_body_forces(floating, FIXED_DT)
    expected_balloon_acceleration = (
        AIR_DENSITY * floating.spec.buoyancy_volume * balloon_world.gravity
        - floating.spec.mass * balloon_world.gravity
    ) / floating.dynamic_mass
    verify("helium buoyancy uses displaced-air force and added mass", (
        floating.velocity.y > 0.0
        and abs(floating.velocity.y / FIXED_DT - expected_balloon_acceleration) < 1.0e-8
    ))

    goo_world = PhysicsWorld(58)
    sticky = RigidBody(
        ITEM_SPEC_BY_KEY["sticky_goo"],
        Vec3(50.0, ITEM_SPEC_BY_KEY["sticky_goo"].radius, 50.0),
        velocity=Vec3(0.0, -2.0, 0.0),
    )
    goo_world._room_impact(sticky, Vec3(0.0, 1.0, 0.0), sticky.position - Vec3(0.0, sticky.spec.radius, 0.0))
    verify("goo adheres on meaningful impact", sticky.stuck_surface and sticky.velocity.length() == 0.0)
    sticky.apply_force(Vec3(0.0, 100.0, 0.0))
    goo_world._integrate_body_forces(sticky, FIXED_DT)
    verify("goo adhesion has a finite break force", (
        not sticky.stuck_surface and sticky.velocity.y > 0.0
    ))
    host = RigidBody(ITEM_SPEC_BY_KEY["medicine_1"], Vec3(50.0, 2.0, 50.0))
    carried_goo = RigidBody(ITEM_SPEC_BY_KEY["sticky_goo"], Vec3(50.0, 2.2, 50.0))
    goo_world._attach_goo_to_body(carried_goo, host)
    host.apply_impulse(Vec3(1.0, 0.0, 0.0), host.position)
    goo_world.bodies = [host, carried_goo]
    goo_world._update_attachments()
    represented_momentum = (
        host.velocity * host.dynamic_mass + carried_goo.velocity * carried_goo.dynamic_mass
    )
    verify("body-adhered goo contributes mass without creating momentum", (
        abs(represented_momentum.x - 1.0) < 1.0e-9
        and represented_momentum.y == 0.0 and represented_momentum.z == 0.0
    ))

    rolling_world = PhysicsWorld(59)
    rolling = next(body for body in rolling_world.bodies if body.spec.key == "wheelbarrow")
    rolling.asleep = False
    rolling.velocity = Vec3(0.0, 0.0, 2.0)
    rolling.grounded = True
    rolling_world._apply_ground_resistance_and_sleep(rolling, 0.25)
    verify("functional wheelbarrow wheel rotates with forward travel", abs(rolling.wheel_angle) > 0.1)
    cargo = RigidBody(
        ITEM_SPEC_BY_KEY["ceramic_marble"],
        rolling.position + Vec3(0.0, rolling.spec.dimensions[1] * 0.5 + ITEM_SPEC_BY_KEY["ceramic_marble"].radius, 0.0),
    )
    rolling_world.bodies = [rolling, cargo]
    rolling.velocity = Vec3(0.0, 0.0, 2.0)
    rolling_world._apply_wheelbarrow_cargo(0.10)
    verify("wheelbarrow tray transfers motion to cargo with reaction", cargo.velocity.z > 0.0 and rolling.velocity.z < 2.0)
    cargo.position = rolling.position + rolling.orientation.rotate(
        Vec3(0.0, rolling.spec.dimensions[1] * 0.5 + cargo.bounding_radius, 0.60)
    )
    cargo.velocity = rolling.orientation.rotate(Vec3(0.0, 0.0, 3.0))
    rolling_world._apply_wheelbarrow_cargo(FIXED_DT)
    contained_local = rolling.orientation.conjugate().rotate(cargo.position - rolling.position)
    verify("wheelbarrow tray walls contain normal-size cargo", (
        contained_local.z <= 0.485 - cargo.bounding_radius + 1.0e-9
    ))

    reindex_world = PhysicsWorld(591)
    settled = RigidBody(
        ITEM_SPEC_BY_KEY["medicine_1"], Vec3(42.0, ITEM_SPEC_BY_KEY["medicine_1"].radius, 42.0),
        asleep=True, grounded=True, pristine=False,
    )
    reindex_world.bodies = [settled]
    reindex_world._rebuild_static_broadphase_index()
    settled_pairs = reindex_world._broadphase_pairs()
    verify("disturbed sleeping bodies rejoin the static spatial index", (
        not settled_pairs and any(0 in indices for indices in reindex_world._static_cells.values())
    ))

    rotational_world = PhysicsWorld(592)
    rotating_noodle = RigidBody(
        ITEM_SPEC_BY_KEY["foam_noodle"], Vec3(50.0, 2.0, 50.0),
        orientation=Quat(math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)),
    )
    rotating_noodle.previous_orientation = Quat()
    nearby_marble = RigidBody(
        ITEM_SPEC_BY_KEY["ceramic_marble"], Vec3(50.52, 2.0, 50.0), asleep=True,
    )
    rotational_world.bodies = [rotating_noodle, nearby_marble]
    rotational_world._rebuild_static_broadphase_index()
    verify("rotational broadphase retains swept long-body contacts", (
        (0, 1) in rotational_world._broadphase_pairs()
    ))

    pass_world = PhysicsWorld(593)
    passing_bat = RigidBody(
        ITEM_SPEC_BY_KEY["wood_bat"], Vec3(51.0, 2.0, 50.5), velocity=Vec3(80.0, 0.0, 0.0),
    )
    passing_bat.previous_position = Vec3(49.0, 2.0, 50.5)
    missed_brick = RigidBody(ITEM_SPEC_BY_KEY["clay_brick"], Vec3(50.0, 2.0, 50.0))
    before_pass = passing_bat.velocity.copy()
    false_contact = pass_world._resolve_box_box(passing_bat, missed_brick, FIXED_DT)
    verify("box CCD rejects separated bounding-sphere near misses", (
        false_contact is None and (passing_bat.velocity - before_pass).length() < 1.0e-12
    ))

    pcm = build_pcm_bank()
    required_sounds = {
        "dodge", "medicine", "steel", "concrete", "rubber_brick",
        "wood_bat", "wheelbarrow", "balloon", "foam_noodle", "goo",
        "ceramic", "clay_brick", "pallet", "plush",
        "throw", "jump", "land",
    }
    hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in pcm.items()}
    verify("fourteen material plus three exertion sounds", set(pcm) == required_sounds and all(len(value) > 1_000 for value in pcm.values()))
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
    group_rows = world.category_summary_rows()
    verify("group telemetry accounts for all 551 bodies", (
        len(group_rows) == 10
        and sum(category_counts.values()) == EXPECTED_BODY_COUNT
        and any("BRICKS      500" in row for row in group_rows)
    ))
    verify("HUD exposes current values and ranges", (
        len(world.status_lines()) == 3
        and "m/s^2" in world.status_lines()[0]
        and "friction" in world.status_lines()[1].lower()
        and "N" in world.status_lines()[2]
    ))
    performance_probe = AdaptivePerformanceController()
    for _ in range(30):
        performance_probe.observe(1.0 / 20.0)
    verify("sub-30 FPS load selects Pi SAFE promptly", (
        performance_probe.quality_level == 0
        and performance_probe.max_physics_steps >= math.ceil(PHYSICS_HZ / MINIMUM_RENDER_FPS)
        and performance_probe.impact_sound_limit == 2
        and QUALITY_PHYSICS_BUDGET_MS[0] < 1_000.0 / MINIMUM_RENDER_FPS
    ))
    for _ in range(1_800):
        performance_probe.observe(1.0 / 60.0)
    verify("adaptive quality recovers gradually to ULTRA", (
        performance_probe.quality_level == 2
        and performance_probe.quality_name == "ULTRA"
        and abs(performance_probe.display_fps - TARGET_FPS) < 0.2
    ))
    performance_probe.note_backlog_drop()
    verify("frame governor accounts for discarded catch-up backlog", performance_probe.backlog_drops == 1)
    verify("GPU hardware detector rejects software Mesa renderers", (
        is_hardware_gl_renderer("V3D 7.1")
        and is_hardware_gl_renderer("VideoCore VII")
        and not is_hardware_gl_renderer("llvmpipe (LLVM 15.0.6, 256 bits)")
        and not is_hardware_gl_renderer("softpipe")
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
    verify("settled bodies sleep while helium balloon remains active", all(
        body.asleep or body.spec.category == "balloon" for body in stable_world.bodies
    ))
    stable_world.adjust_gravity(1)
    verify("setting changes wake non-pallet exhibits", all(
        not body.asleep
        for body in stable_world.bodies
        if body.spec.key not in {"clay_brick", "wood_pallet"}
    ))
    verify("pristine 500-brick load stays sleeping for Pi steady state", all(
        body.asleep for body in stable_world.bodies if body.spec.key == "clay_brick"
    ))
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
    verify(
        "four-core command-line selection",
        parse_args(["--cpu-cores", "4", "--no-audio"]).cpu_cores == 4,
    )
    maximum_args = parse_args(["--maximum-throughput", "--gpu-physics", "maximum"])
    verify("maximum-throughput command-line profile", (
        maximum_args.maximum_throughput
        and maximum_args.gpu_physics == "maximum"
    ))
    resource_probe = ResourceMonitor(tuple(range(min(4, max(1, os.cpu_count() or 1)))))
    verify("bounded one-hertz resource telemetry", (
        set(resource_probe.core_loads) == set(resource_probe.cpu_cores)
        and resource_probe.rss_mib >= 0.0
        and resource_probe.memory_available_mib >= 0.0
    ))

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
        errors: List[int] = []
        for quality in range(len(RENDER_QUALITY_NAMES)):
            app._set_render_quality(quality)
            app.tick(FIXED_DT)
            for position, yaw, pitch in (
                (Vec3(*PLAYER_START), 0.0, -0.30),
                (Vec3(50.0, 0.0, 22.0), 0.0, -0.12),
            ):
                app.world.player.position = position
                app.world.player.previous_position = position.copy()
                app.world.player.yaw = yaw
                app.world.player.pitch = pitch
                app.render()
                errors.append(GL.glGetError())
                pygame.display.flip()
        if all(error == GL.GL_NO_ERROR for error in errors):
            print(
                f"{TITLE} OpenGL smoke: PASS "
                "(SAFE/BALANCED/ULTRA, start/pallet VBO, GL_NO_ERROR)"
            )
            return True
        print(f"{TITLE} OpenGL smoke: FAIL (errors {errors})")
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
    parser.add_argument(
        "--view", choices=("start", "pallet", "exhibits"), default="start",
        help="initial camera station for screenshots or inspection",
    )
    parser.add_argument(
        "--quality", choices=("auto", "safe", "balanced", "ultra"), default="auto",
        help="automatic Pi frame governor or a locked render tier",
    )
    parser.add_argument("--no-audio", action="store_true", help="use the silent audio fallback")
    parser.add_argument(
        "--cpu-cores", type=int, choices=(1, 2, 3, 4), default=CPU_CORE_BUDGET,
        help="scheduler-visible game cores (default 4: main plus three workers)",
    )
    parser.add_argument(
        "--cpu-backend", choices=("scalar", "auto", "maximum"), default="auto",
        help="scalar compatibility, automatic NEON, or maximum NEON/offload mode",
    )
    parser.add_argument(
        "--gpu-physics", choices=("off", "auto", "maximum"), default="auto",
        help="GPU transform-feedback secondary physics workload (default adaptive)",
    )
    parser.add_argument(
        "--maximum-throughput", action="store_true",
        help=(
            "uncap rendering and request maximum GPU physics for thermal/"
            "hardware profiling; the 30 FPS emergency guard remains active"
        ),
    )
    parser.add_argument(
        "--stress-pallet", action="store_true",
        help="launch the steel ball into the brick pallet for a Pi performance test",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.check:
        return 0 if run_self_check(True) else 1
    if args.gl_check:
        return 0 if run_gl_smoke(args.seed) else 1
    load_dependencies()
    pygame.mixer.pre_init(AUDIO_RATE, -16, AUDIO_CHANNELS, 512)
    app = GameApp(
        args.seed,
        audio_enabled=not args.no_audio,
        capture_mode=bool(args.capture),
        cpu_cores=configure_cpu_core_budget(args.cpu_cores),
        cpu_backend="maximum" if args.maximum_throughput else args.cpu_backend,
        quality_mode=args.quality,
        gpu_physics_mode=args.gpu_physics,
        maximum_throughput=args.maximum_throughput,
    )
    if args.view == "pallet":
        app.world.player.position = Vec3(50.0, 0.0, 22.0)
        app.world.player.previous_position = app.world.player.position.copy()
        app.world.player.yaw = 0.0
        app.world.player.pitch = -0.12
    elif args.view == "exhibits":
        app.world.player.position = Vec3(29.0, 0.0, 10.5)
        app.world.player.previous_position = app.world.player.position.copy()
        app.world.player.yaw = 0.0
        app.world.player.pitch = -0.22
    if args.stress_pallet:
        steel = next(body for body in app.world.bodies if body.spec.key == "steel")
        steel.position = Vec3(50.0, 0.60, 25.0)
        steel.previous_position = steel.position.copy()
        steel.velocity = Vec3(0.0, 0.0, 122.0)
        steel.wake()
    return app.run(args.capture, max(0, args.frames))


if __name__ == "__main__":
    raise SystemExit(main())
