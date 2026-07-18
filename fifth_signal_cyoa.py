#!/usr/bin/env python3
"""
The Fifth Signal - sealed-room PyOpenGL social survival full final release

Four animated NPC portraits occupy the upper half of the window. The lower
half contains the story, all five participants' status indexes, the player's
inventory, and clickable choices. NPC inventories exist for trading and
survival simulation but are intentionally never displayed. Every player-led
round begins with exactly one player action, followed by one self-chosen action
from each living NPC; NPC-only rounds continue after player elimination. The
Fifth Signal must be discovered inside the room.
The Full Final Release has NPCs score complete action-target-item plans from all six personal
indexes, all five private supply levels, item effects, visible needs, recent
choices, and escalating survival pressure; stock-safe rationales are archived.
Spirituality now owns a complete portrait-lighting language: -1 raises black
flaming horns, 0 dims only that NPC's avatar window, and +1 emits a glowing halo.

Controls:
  Startup name field  Type a custom name; Enter confirms; Esc keeps Tyler
  Mouse              Select portraits, inventory items, choices, and trade controls
  1-4                Select an NPC portrait
  F1-F3              Choose a room investigation (uses the player's action)
  T / L               Talk / listen
  G / S / B           Give / steal selected item / open the trade builder
  R                   Reflect with the selected NPC
  U / X               Use selected item / rest
  C / V / A / F       Compliment / flirt / antagonize / fight selected NPC
  M                   Mute / unmute the procedural 8-bit voices
  Tab                 Cycle selected inventory item
  PageUp / PageDown   Browse older / newer activity rounds
  Mouse wheel         Browse the activity log during play and on the ending screen
  Trade builder       Type exact quantities; Tab swaps fields; Enter confirms; Esc cancels
  F5                  Restart with newly randomized room occupants
  Esc                 Quit

Dependencies:
  pip install pygame PyOpenGL PyOpenGL_accelerate

CC0 character, particle, interface, and font provenance is documented in
fifth_signal_assets/ASSET_LICENSES.md. Missing images fall back to the original
flat materials instead of preventing the game from starting.

All voice motifs and victory stings are original, deterministic chiptune PCM
generated in code. Audio-device failure falls back to silent play without
changing the simulation.

The medicine and caffeine effects are deliberately abstract game mechanics.
They are not a dosing model or medical guidance.

The interactive build renders at 1920x1080 and leaves one nonblocking second
between each participant's action so outcomes can be read as they happen.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


pygame = None
GL = None
GLU = None

WIDTH = 1920
HEIGHT = 1080
TOP_HEIGHT = HEIGHT // 2
FPS = 60
ACTION_MOMENT_SECONDS = 1.0
RELEASE_LABEL = "FULL FINAL RELEASE"
PRESENTATION_SALT = "full-final-release-presentation"
TITLE = f"The Fifth Signal - {RELEASE_LABEL.title()}"
CREDIT_WATERMARK = "Created by OpenAI ChatGPT Codex 5.6 Sol Ultra"
INVENTORY_CAP = 1_000_000
DEFAULT_PLAYER_NAME = "Tyler"
MAX_PLAYER_NAME_LENGTH = 18
PLAYER_NAME_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '-"
)
ASSET_DIR = Path(__file__).resolve().with_name("fifth_signal_assets")
TEXTURE_ASSET_FILES: Dict[str, str] = {
    "skin": "skin_detail.png",
    "hair": "hair_detail.png",
    "cloth": "cloth_detail.png",
    "iris": "iris_detail.png",
    "particle_star": "particle_star.png",
    "particle_smoke": "particle_smoke.png",
    "particle_light": "particle_light.png",
    "particle_magic": "particle_magic.png",
    "ui_panel": "ui_panel.png",
    "ui_button": "ui_button.png",
}

AUDIO_SAMPLE_RATE = 22_050
AUDIO_CHANNELS = 2
AUDIO_SYNTH_DIVISOR = 3
AUDIO_CACHE_LIMIT = 48
AUDIO_QUEUE_LIMIT = 12

STATUS_KEYS = (
    "physical",
    "emotional",
    "cognitive",
    "social",
    "sentient",
    "spiritual",
)
RANGED_STATUS_KEYS = STATUS_KEYS[:-1]
STATUS_LABELS = {
    "physical": "Physical Health",
    "emotional": "Emotional State",
    "cognitive": "Cognitive",
    "social": "Social",
    "sentient": "Sentient",
    "spiritual": "Spiritual",
}
STATUS_SHORT = {
    "physical": "PHYS",
    "emotional": "EMO",
    "cognitive": "COG",
    "social": "SOC",
    "sentient": "SENT",
    "spiritual": "SP",
}

# Gesture keys are deliberately semantic rather than tied to any one animation.
# This keeps the social simulation deterministic while the portrait renderer is
# free to add secondary motion, hand articulation, and transitions.  The names
# nod toward broad online body-language formats without reproducing a specific
# person's likeness or a copyrighted animation.
GESTURE_LABELS: Dict[str, str] = {
    "idle": "Listening Stillness",
    "open_palms": "Open-Channel Palms",
    "heart_hands": "Heart-Signal Hands",
    "temple_tap": "Thought-Loop Tap",
    "guarded_cross": "Guarded Signal Fold",
    "fist_clench": "Resolve Lock",
    "prayer_pose": "Quiet-Frequency Pose",
    "self_hug": "Shelter Signal",
    "viral_point": "Signal Receipt Point",
    "seesaw_67": "Six-Seven Signal Seesaw",
    "funky_ehh": "Funky Maybe Groove",
    "sprint_pose": "No-Exit Sprint Stance",
    "affirmation": "Absolute Signal Affirmation",
    "dramatic_turn": "Dramatic Channel Turn",
    "play_them_off": "Invisible Air-Keyboard",
    "retro_chacha": "Retro Room Cha-Cha",
    "pixel_wave": "Pixel-Sprite Wave",
    "spitting": "Pixel Spit-Take",
    "yawning": "No-Exit Yawn",
    "chomping": "Inventory Chomp",
    "bowing": "Signal-Court Bow",
    "head_banging": "Fifth-Beat Head Bang",
}
NPC_GESTURE_KEYS = tuple(key for key in GESTURE_LABELS if key != "idle")
ACTION_EMOTE_KEYS = (
    "spitting",
    "yawning",
    "chomping",
    "bowing",
    "head_banging",
)

# Action context supplies a strong expressive nudge, while the status-derived
# terms in _npc_gesture_weights retain the final say.  Every action has several
# plausible readings, so two NPCs need not perform the same response identically.
ACTION_GESTURE_CONTEXT: Dict[str, Tuple[str, ...]] = {
    "talk": ("viral_point", "open_palms", "bowing", "pixel_wave"),
    "listen": ("open_palms", "temple_tap", "yawning", "bowing"),
    "compliment": ("heart_hands", "bowing", "pixel_wave", "retro_chacha"),
    "flirt": ("heart_hands", "retro_chacha", "bowing", "pixel_wave"),
    "antagonize": ("spitting", "guarded_cross", "fist_clench", "head_banging"),
    "reflect": ("prayer_pose", "bowing", "temple_tap", "open_palms"),
    "fight": ("head_banging", "fist_clench", "spitting", "sprint_pose"),
    "give": ("bowing", "open_palms", "heart_hands", "affirmation"),
    "steal": ("dramatic_turn", "temple_tap", "guarded_cross", "sprint_pose"),
    "trade": ("open_palms", "bowing", "seesaw_67", "temple_tap"),
    "use": ("chomping", "yawning", "temple_tap", "head_banging"),
    "rest": ("yawning", "self_hug", "prayer_pose", "bowing"),
}
NPC_ITEM_GESTURE_CONTEXT: Dict[str, str] = {
    "dollars": "bowing",
    "water_liters": "open_palms",
    "food_pounds": "chomping",
    "caffeine_pills": "head_banging",
    "acetaminophen_pills": "yawning",
}

# Idle performances are deliberately separate from action gestures.  They are
# selected from elapsed render time and a mixed integer seed, never from the
# simulation RNG, so leaving the game open cannot change a later NPC decision.
@dataclass(frozen=True)
class IdlePersonality:
    key: str
    label: str
    emotes: Tuple[str, ...]
    slot_seconds: float
    body_sway: float
    head_motion: float
    hand_energy: float


@dataclass(frozen=True)
class IdleEmoteFrame:
    personality_key: str
    personality_label: str
    emote_key: str
    next_emote_key: str
    slot_index: int
    slot_progress: float
    blend: float
    motion_phase: float


IDLE_PERSONALITIES: Tuple[IdlePersonality, ...] = (
    IdlePersonality(
        "sentinel",
        "Watchful Sentinel",
        ("room_scan", "cuff_check", "knuckle_roll"),
        5.15,
        0.55,
        0.72,
        0.58,
    ),
    IdlePersonality(
        "anchor",
        "Compassionate Anchor",
        ("heart_breath", "reassure_palm", "soft_wave"),
        5.65,
        0.88,
        0.66,
        0.72,
    ),
    IdlePersonality(
        "analyst",
        "Restless Analyst",
        ("chin_think", "air_type", "temple_count"),
        4.70,
        0.42,
        0.94,
        0.92,
    ),
    IdlePersonality(
        "oracle",
        "Signal Oracle",
        ("signal_trace", "prayer_breath", "palm_orbit"),
        6.10,
        1.00,
        0.82,
        0.78,
    ),
)
IDLE_PERSONALITY_BY_KEY = {
    personality.key: personality
    for personality in IDLE_PERSONALITIES
}
IDLE_EMOTE_LABELS: Dict[str, str] = {
    "room_scan": "Scanning the Room",
    "cuff_check": "Resetting a Cuff",
    "knuckle_roll": "Rolling Knuckles",
    "heart_breath": "Grounding Heartbeat",
    "reassure_palm": "Quiet Reassurance",
    "soft_wave": "Small Check-In Wave",
    "chin_think": "Chin-Thought Loop",
    "air_type": "Invisible Field Notes",
    "temple_count": "Counting the Pulses",
    "signal_trace": "Tracing the Signal",
    "prayer_breath": "Measured Stillness",
    "palm_orbit": "Orbiting Frequency",
}


def _idle_seed_word(seed: Optional[int], npc_index: int) -> int:
    """Return a stable per-portrait word without consuming any Random state."""
    value = (0x5F17_1D1E if seed is None else int(seed)) & 0xFFFFFFFF
    value ^= ((int(npc_index) + 1) * 0x9E37_79B9) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB_352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846C_A68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def idle_emote_frame(seed: Optional[int], npc_index: int, elapsed: float) -> IdleEmoteFrame:
    """Pure deterministic idle scheduler for one of the four portrait profiles."""
    if not 0 <= npc_index < len(IDLE_PERSONALITIES):
        raise IndexError("NPC idle index must be between zero and three")
    personality = IDLE_PERSONALITIES[npc_index]
    word = _idle_seed_word(seed, npc_index)
    tempo = 0.90 + (word & 0xFF) / 255.0 * 0.20
    slot_seconds = personality.slot_seconds * tempo
    offset = ((word >> 8) & 0xFFFF) / 65535.0 * slot_seconds
    scheduled = max(0.0, float(elapsed)) + offset
    slot_index = int(scheduled // slot_seconds)
    progress = (scheduled / slot_seconds) - slot_index
    sequence_offset = (word >> 24) % len(personality.emotes)
    emote_index = (slot_index + sequence_offset) % len(personality.emotes)
    emote_key = personality.emotes[emote_index]
    next_emote_key = personality.emotes[(emote_index + 1) % len(personality.emotes)]

    # The neutral crossings at slot boundaries make every pose transition
    # smooth even after a long action-gesture override.
    fade_window = 0.18
    fade_in = float(clamp(progress / fade_window, 0.0, 1.0))
    fade_out = float(clamp((1.0 - progress) / fade_window, 0.0, 1.0))
    fade_in = fade_in * fade_in * (3.0 - 2.0 * fade_in)
    fade_out = fade_out * fade_out * (3.0 - 2.0 * fade_out)
    blend = fade_in * fade_out
    phase_offset = ((word >> 4) & 0xFFF) / 4095.0 * math.tau
    return IdleEmoteFrame(
        personality.key,
        personality.label,
        emote_key,
        next_emote_key,
        slot_index,
        progress,
        blend,
        scheduled * (math.tau / slot_seconds) + phase_offset,
    )


def idle_motion_parameters(
    frame: IdleEmoteFrame,
    npc_index: int,
) -> Dict[str, float]:
    """Pure body/head motion envelope shared by checks and the GL renderer."""
    personality = IDLE_PERSONALITIES[npc_index]
    wave = math.sin(frame.motion_phase)
    counter_wave = math.sin(frame.motion_phase * 0.53 + npc_index * 1.17)
    detail_wave = math.sin(frame.motion_phase * 2.15 + npc_index * 0.61)
    emphasis = frame.blend
    params = {
        "body_x": wave * 0.018 * personality.body_sway,
        "body_y": counter_wave * 0.010 * personality.body_sway,
        "body_roll": wave * 0.70 * personality.body_sway,
        "head_yaw": counter_wave * 1.45 * personality.head_motion,
        "head_tilt": wave * 0.75 * personality.head_motion,
        "head_nod": detail_wave * 0.48 * personality.head_motion,
        "gaze_x": counter_wave * 0.035 * personality.head_motion,
        "hand_energy": personality.hand_energy,
    }
    if frame.emote_key == "room_scan":
        params["head_yaw"] += wave * 4.2 * emphasis
        params["gaze_x"] += wave * 0.075 * emphasis
    elif frame.emote_key == "heart_breath":
        params["body_y"] += math.sin(frame.motion_phase * 2.0) * 0.012 * emphasis
        params["head_tilt"] += 1.25 * emphasis
    elif frame.emote_key in ("chin_think", "temple_count"):
        params["head_tilt"] -= 1.05 * emphasis
        params["head_nod"] += detail_wave * 0.72 * emphasis
    elif frame.emote_key in ("signal_trace", "palm_orbit"):
        params["body_roll"] += wave * 1.15 * emphasis
        params["head_yaw"] -= wave * 1.35 * emphasis
    return params


def action_gesture_override_weight(
    gesture_turn: int,
    gesture_age: float,
    idle_enabled: bool,
) -> float:
    """Pure action-to-idle crossfade; action poses briefly own the portrait."""
    if gesture_turn <= 0:
        return 0.0
    if not idle_enabled:
        return 1.0
    hold_seconds = 1.55
    fade_seconds = 0.90
    if gesture_age <= hold_seconds:
        return 1.0
    progress = float(clamp((gesture_age - hold_seconds) / fade_seconds, 0.0, 1.0))
    progress = progress * progress * (3.0 - 2.0 * progress)
    return 1.0 - progress


def action_emote_motion(
    gesture_key: str,
    gesture_age: float,
    npc_index: int,
) -> Dict[str, float]:
    """Return deterministic body/face controls for the five animated emotes.

    The helper is deliberately pure and elapsed-time based. Rendering a pose
    can therefore never advance the simulation RNG or alter a later NPC choice.
    All one-shot motion returns to neutral before the existing action-to-idle
    crossfade finishes.
    """
    motion = {
        "body_y": 0.0,
        "body_pitch": 0.0,
        "body_roll": 0.0,
        "head_yaw": 0.0,
        "head_tilt": 0.0,
        "head_nod": 0.0,
        "gaze_x": 0.0,
        "gaze_y": 0.0,
        "eye_scale": 1.0,
        "jaw_open": 0.0,
        "mouth_width_scale": 1.0,
        "brow_offset": 0.0,
        "spit_strength": 0.0,
        "crumb_strength": 0.0,
        "spark_strength": 0.0,
        "arm_beat": 0.0,
    }
    if gesture_key not in ACTION_EMOTE_KEYS:
        return motion

    age = max(0.0, float(gesture_age))
    direction = -1.0 if int(npc_index) % 2 else 1.0

    if gesture_key == "spitting":
        # A short recoil, cheek-puff and forward snap precede a bounded spray.
        progress = float(clamp(age / 0.78, 0.0, 1.0))
        envelope = math.sin(progress * math.pi) if age <= 0.78 else 0.0
        burst_progress = float(clamp((age - 0.22) / 0.42, 0.0, 1.0))
        burst = math.sin(burst_progress * math.pi) if 0.22 <= age <= 0.64 else 0.0
        motion.update(
            body_pitch=3.0 * envelope,
            body_roll=-direction * 1.8 * envelope,
            head_yaw=direction * 8.0 * envelope,
            head_nod=(-3.0 * envelope + 7.0 * burst),
            gaze_x=direction * 0.045 * envelope,
            eye_scale=1.0 - 0.25 * envelope,
            jaw_open=0.025 * envelope,
            mouth_width_scale=1.0 - 0.45 * envelope,
            brow_offset=-0.025 * envelope,
            spit_strength=burst,
        )
    elif gesture_key == "yawning":
        progress = float(clamp(age / 1.40, 0.0, 1.0))
        envelope = math.sin(progress * math.pi) if age <= 1.40 else 0.0
        motion.update(
            body_y=-0.025 * envelope,
            body_pitch=4.0 * envelope,
            head_tilt=2.0 * direction * envelope,
            head_nod=7.0 * envelope,
            gaze_y=-0.035 * envelope,
            eye_scale=1.0 - 0.48 * envelope,
            jaw_open=0.130 * envelope,
            mouth_width_scale=1.0 - 0.22 * envelope,
            brow_offset=-0.018 * envelope,
        )
    elif gesture_key == "chomping":
        progress = float(clamp(age / 1.35, 0.0, 1.0))
        envelope = math.sin(progress * math.pi) if age <= 1.35 else 0.0
        bite = 0.5 + 0.5 * math.sin(age * 18.0 - 0.65)
        bite_open = envelope * (0.25 + bite * 0.75)
        motion.update(
            body_y=0.008 * envelope * bite,
            head_nod=2.7 * envelope * math.sin(age * 18.0),
            gaze_y=-0.018 * envelope,
            eye_scale=1.0 - 0.10 * envelope,
            jaw_open=0.078 * bite_open,
            mouth_width_scale=1.0 - 0.28 * envelope,
            brow_offset=0.012 * envelope,
            crumb_strength=envelope * max(0.0, math.sin(age * 9.0)),
            arm_beat=envelope * math.sin(age * 18.0),
        )
    elif gesture_key == "bowing":
        progress = float(clamp(age / 1.18, 0.0, 1.0))
        envelope = math.sin(progress * math.pi) if age <= 1.18 else 0.0
        motion.update(
            body_y=-0.030 * envelope,
            body_pitch=13.0 * envelope,
            head_nod=6.0 * envelope,
            gaze_y=-0.045 * envelope,
            eye_scale=1.0 - 0.18 * envelope,
            mouth_width_scale=1.0 - 0.05 * envelope,
        )
    elif gesture_key == "head_banging":
        progress = float(clamp(age / 1.55, 0.0, 1.0))
        envelope = math.sin(progress * math.pi) if age <= 1.55 else 0.0
        beat = math.sin(age * 18.0)
        motion.update(
            body_y=0.022 * envelope * abs(beat),
            body_pitch=3.5 * envelope * beat,
            body_roll=direction * 1.2 * envelope * math.sin(age * 9.0),
            head_tilt=direction * 1.8 * envelope * math.sin(age * 9.0),
            head_nod=12.0 * envelope * beat,
            gaze_y=-0.020 * envelope,
            eye_scale=1.0 - 0.28 * envelope,
            jaw_open=0.038 * envelope * max(0.0, beat),
            brow_offset=-0.022 * envelope,
            spark_strength=envelope * max(0.0, -beat),
            arm_beat=envelope * beat,
        )
    return motion


def npc_idle_enabled(
    phase: str,
    player_alive: bool,
    npc_alive: bool,
    is_ending: bool,
) -> bool:
    """Pure render gate: only the player's decision window performs idles."""
    return bool(
        phase == "awaiting_player"
        and player_alive
        and npc_alive
        and not is_ending
    )


# ---------------------------------------------------------------------------
# Original procedural 8-bit voices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceProfile:
    key: str
    base_hz: float
    waveform: str
    duty: float
    pan: float


@dataclass(frozen=True)
class VoiceMood:
    tone: str
    energy: int
    clarity: int
    spiritual: int
    valence: float
    energy_value: float
    clarity_value: float

    @property
    def cache_key(self) -> Tuple[str, int, int, int, int, int, int]:
        return (
            self.tone,
            self.energy,
            self.clarity,
            self.spiritual,
            int(round(self.valence * 1000.0)),
            int(round(self.energy_value * 1000.0)),
            int(round(self.clarity_value * 100.0)),
        )


@dataclass(frozen=True)
class PCMVoice:
    buffer: bytes
    duration: float
    mouth_windows: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class QueuedVoice:
    speaker_index: int
    profile_index: int
    cue_key: str
    mood: VoiceMood
    earliest_time: float
    priority: int
    variant: int
    caption: str


@dataclass
class ActiveVoice:
    speaker_index: int
    cue_key: str
    start_time: float
    duration: float
    mouth_windows: Tuple[Tuple[float, float], ...]
    caption: str
    priority: int


VOICE_PROFILES: Tuple[VoiceProfile, ...] = (
    VoiceProfile("medic_pulse", 146.8, "pulse", 0.25, -0.65),
    VoiceProfile("anchor_triangle", 207.7, "triangle", 0.38, -0.22),
    VoiceProfile("analyst_modem", 246.9, "saw", 0.125, 0.22),
    VoiceProfile("oracle_hollow", 174.6, "hollow", 0.50, 0.65),
)

# Semitone contours are intentionally short enough for five serialized voices
# to remain readable after the simulation resolves a complete round at once.
VOICE_CUE_CONTOURS: Dict[str, Tuple[int, ...]] = {
    "talk": (0, 2, 0, 5),
    "listen": (2, 0, -2),
    "compliment": (0, 4, 7, 12),
    "flirt": (0, 4, 7, 11, 7),
    "antagonize": (0, -1, -6, -8),
    "reflect": (0, 7, 12, 7),
    "fight": (0, -7, -12),
    "give": (0, 7, 12),
    "steal": (0, -2, 3, -5),
    "trade": (0, 4, 9, 12),
    "investigate": (0, 0, 0, 0, 1),
    "use": (0, 5, 9),
    "rest": (4, 0, -5),
    "elimination": (0, -4, -9, -16, -24),
    "victory": (0, 4, 7, 12, 16, 19),
    "idle_room_scan": (0, 7, 2),
    "idle_cuff_check": (0, 0, 5),
    "idle_knuckle_roll": (-5, 0, -5, 2),
    "idle_heart_breath": (0, 4, 0, 7),
    "idle_reassure_palm": (0, 2, 7),
    "idle_soft_wave": (0, 4, 7, 4),
    "idle_chin_think": (0, 2, -1),
    "idle_air_type": (0, 7, 0, 9, 2),
    "idle_temple_count": (0, 0, 0, 0, 7),
    "idle_signal_trace": (0, 7, 12, 7),
    "idle_prayer_breath": (0, 7, 0),
    "idle_palm_orbit": (0, 4, 7, 11, 7, 4),
}

VOICE_NOTE_SECONDS: Dict[str, float] = {
    "fight": 0.098,
    "elimination": 0.086,
    "victory": 0.082,
    "investigate": 0.052,
}

VOICE_MOOD_CAPTIONS: Tuple[Dict[str, str], ...] = (
    {
        "positive": "VITALS SAY: KEEP COOKING",
        "neutral": "MEDIC BLEEP: NOTED",
        "negative": "SIDE-EYE PROTOCOL ARMED",
    },
    {
        "positive": "VIBE PATCH: DEPLOYED",
        "neutral": "HEARTBEAT CHECK: BEEP",
        "negative": "BAD VIBES: QUARANTINED",
    },
    {
        "positive": "LOGIC: ABSOLUTELY COOKING",
        "neutral": "PROCESSING... DRAMATICALLY",
        "negative": "ERROR: VIBES NOT FOUND",
    },
    {
        "positive": "SIGNAL SAYS: WE ASCEND",
        "neutral": "COSMIC HOLD MUSIC",
        "negative": "OMINOUS BLEEP DETECTED",
    },
)

IDLE_VOICE_CAPTIONS: Dict[str, str] = {
    "room_scan": "ROOM CHECK: STILL A ROOM",
    "cuff_check": "CUFF PATCH INSTALLED",
    "knuckle_roll": "KNUCKLES: BUFFERING",
    "heart_breath": "HEART.EXE: STEADY",
    "reassure_palm": "TINY PALM, HUGE VIBES",
    "soft_wave": "HELLO FROM THE VOID",
    "chin_think": "THOUGHT LOADING...",
    "air_type": "TYPING INTO THE AIR",
    "temple_count": "FIVE PULSES. SUSPICIOUS.",
    "signal_trace": "TRACING THE WEIRD",
    "prayer_breath": "COSMIC DEEP BREATH",
    "palm_orbit": "ORBIT MODE: EXTREMELY ON",
}

VICTORY_VOICE_CAPTIONS: Tuple[str, ...] = (
    "CLINICALLY SPEAKING: STILL HERE",
    "CONSENSUS REACHED: I WON",
    "NO BUGS. ONLY FINAL FEATURES.",
    "THE VIBES SURVIVED THE ROOM",
)


def _mix_u32(value: int) -> int:
    """Stable presentation-only integer mixer; never use Python's hash()."""
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB_352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846C_A68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def _stable_text_word(value: str) -> int:
    word = 0x811C_9DC5
    for byte in value.encode("utf-8"):
        word ^= byte
        word = (word * 0x0100_0193) & 0xFFFFFFFF
    return word


def presentation_word(seed: Optional[int], salt: int | str) -> int:
    base = 0x5F17_B3A3 if seed is None else int(seed)
    salt_word = _stable_text_word(salt) if isinstance(salt, str) else int(salt)
    return _mix_u32(base ^ salt_word ^ 0xC0DE_563A)


def voice_mood_from_statuses(statuses: Mapping[str, int]) -> VoiceMood:
    """Make all six public indexes audible without touching simulation state."""
    physical = float(statuses["physical"]) / 100.0
    emotional = float(statuses["emotional"]) / 100.0
    cognitive = float(statuses["cognitive"]) / 100.0
    social = float(statuses["social"]) / 100.0
    sentient = float(statuses["sentient"]) / 100.0
    spiritual = int(clamp(int(statuses["spiritual"]), -1, 1))
    spiritual_normalized = (spiritual + 1.0) * 0.5
    valence = emotional * 0.50 + social * 0.30 + spiritual_normalized * 0.20
    emotional_intensity = abs(emotional - 0.5) * 2.0
    energy_value = physical * 0.55 + sentient * 0.25 + emotional_intensity * 0.20
    tone = "positive" if valence >= 0.64 else ("negative" if valence <= 0.39 else "neutral")
    energy = 2 if energy_value >= 0.68 else (0 if energy_value <= 0.38 else 1)
    clarity = 2 if cognitive >= 0.68 else (0 if cognitive <= 0.38 else 1)
    return VoiceMood(tone, energy, clarity, spiritual, valence, energy_value, cognitive)


def voice_caption(profile_index: int, cue_key: str, mood: VoiceMood) -> str:
    profile_index = int(clamp(profile_index, 0, len(VOICE_PROFILES) - 1))
    if cue_key == "victory":
        return VICTORY_VOICE_CAPTIONS[profile_index]
    if cue_key == "elimination":
        return "SIGNAL.EXE POWERING DOWN"
    if cue_key.startswith("idle_"):
        return IDLE_VOICE_CAPTIONS.get(cue_key[5:], "IDLE BLEEP DETECTED")
    return VOICE_MOOD_CAPTIONS[profile_index][mood.tone]


def voice_nominal_pitch(profile_index: int, cue_key: str, mood: VoiceMood) -> float:
    profile = VOICE_PROFILES[profile_index]
    tone_shift = 3.5 if mood.tone == "positive" else (-4.0 if mood.tone == "negative" else 0.0)
    energy_shift = (mood.energy - 1) * 2.5
    contour = VOICE_CUE_CONTOURS.get(cue_key, VOICE_CUE_CONTOURS["talk"])
    valence_shift = (mood.valence - 0.5) * 3.0
    return profile.base_hz * (2.0 ** ((tone_shift + energy_shift + valence_shift + contour[0]) / 12.0))


def synthesize_voice_pcm(
    profile_index: int,
    cue_key: str,
    mood: VoiceMood,
    variant: int = 0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> PCMVoice:
    """Generate compact stereo PCM with quantized sample-and-hold chip timbre."""
    if not 0 <= profile_index < len(VOICE_PROFILES):
        raise IndexError("Voice profile index must be between zero and three")
    if sample_rate < 8_000:
        raise ValueError("Voice sample rate is too low")
    upsample_factor = (
        AUDIO_SYNTH_DIVISOR
        if sample_rate % AUDIO_SYNTH_DIVISOR == 0 and sample_rate >= AUDIO_SAMPLE_RATE
        else 1
    )
    synth_rate = sample_rate // upsample_factor
    profile = VOICE_PROFILES[profile_index]
    contour = VOICE_CUE_CONTOURS.get(cue_key, VOICE_CUE_CONTOURS["talk"])
    note_seconds = VOICE_NOTE_SECONDS.get(cue_key, 0.070)
    tempo_scale = (1.17, 1.0, 0.86)[mood.energy]
    tempo_scale *= 1.08 - mood.energy_value * 0.16
    note_seconds *= tempo_scale
    gap_seconds = 0.012 * tempo_scale
    pitch_shift = (
        (3.5 if mood.tone == "positive" else (-4.0 if mood.tone == "negative" else 0.0))
        + (mood.energy - 1) * 2.5
        + (mood.valence - 0.5) * 3.0
        + ((int(variant) % 3) - 1) * 0.35
    )
    wobble_semitones = 0.14 + (1.0 - mood.clarity_value) * 1.22
    hold_samples = (4, 3, 2)[mood.clarity]
    sparkle = 0.05 + max(0.0, mood.energy_value - 0.45) * 0.14
    buzz = (0.08 if mood.tone == "negative" else 0.025) + (2 - mood.clarity) * 0.025
    if mood.spiritual < 0:
        buzz += 0.055
    left_gain = 1.0 - max(0.0, profile.pan) * 0.58
    right_gain = 1.0 + min(0.0, profile.pan) * 0.58
    pcm = array("h")
    mouth_windows: List[Tuple[float, float]] = []
    cursor_seconds = 0.0
    noise_word = presentation_word(
        variant ^ (profile_index << 10),
        f"{cue_key}:{mood.cache_key}",
    ) or 1

    for note_index, semitone in enumerate(contour):
        duration = note_seconds * (1.12 if cue_key == "elimination" else 1.0)
        sample_count = max(1, int(duration * synth_rate))
        mouth_windows.append((cursor_seconds, cursor_seconds + duration))
        phase = 0.0
        held_value = 0.0
        for sample_index in range(sample_count):
            progress = sample_index / max(1, sample_count - 1)
            vibrato = math.sin(progress * math.tau * (2.2 + profile_index * 0.37)) * wobble_semitones
            bend = 0.0
            if cue_key == "elimination":
                bend = -progress * 9.0
            elif cue_key == "victory":
                bend = progress * 1.8
            frequency = profile.base_hz * 2.0 ** ((semitone + pitch_shift + vibrato + bend) / 12.0)
            if mood.spiritual < 0:
                frequency *= 0.992 + math.sin(progress * math.tau * 3.0) * 0.006
            phase = (phase + frequency / synth_rate) % 1.0
            if sample_index % hold_samples == 0:
                if profile.waveform == "triangle":
                    oscillator = 1.0 - 4.0 * abs(phase - 0.5)
                    pulse = 1.0 if phase < profile.duty else -1.0
                    oscillator = oscillator * 0.78 + pulse * 0.22
                elif profile.waveform == "saw":
                    oscillator = phase * 2.0 - 1.0
                    oscillator = oscillator * 0.70 + (1.0 if phase < profile.duty else -1.0) * 0.30
                elif profile.waveform == "hollow":
                    pulse = 1.0 if phase < profile.duty else -1.0
                    overtone_phase = (phase * 1.5) % 1.0
                    oscillator = pulse * 0.72 + (1.0 if overtone_phase < 0.34 else -1.0) * 0.28
                else:
                    oscillator = 1.0 if phase < profile.duty else -1.0
                noise_word ^= (noise_word << 13) & 0xFFFFFFFF
                noise_word ^= noise_word >> 17
                noise_word ^= (noise_word << 5) & 0xFFFFFFFF
                noise = ((noise_word & 0xFFFF) / 32767.5) - 1.0
                fight_noise = 0.20 * (1.0 - progress) if cue_key == "fight" and note_index == 0 else 0.0
                overtone = 1.0 if ((phase * (2.0 + mood.energy * 0.5)) % 1.0) < 0.30 else -1.0
                held_value = oscillator * (1.0 - buzz - fight_noise) + noise * (buzz + fight_noise)
                held_value += overtone * sparkle
                held_value = round(float(clamp(held_value, -1.0, 1.0)) * 15.0) / 15.0
            attack = float(clamp(progress / 0.10, 0.0, 1.0))
            release = float(clamp((1.0 - progress) / 0.18, 0.0, 1.0))
            envelope = attack * release
            amplitude = held_value * envelope * 0.31
            left_sample = int(clamp(amplitude * left_gain, -1.0, 1.0) * 32767)
            right_sample = int(clamp(amplitude * right_gain, -1.0, 1.0) * 32767)
            pcm.extend((left_sample, right_sample) * upsample_factor)
        cursor_seconds += duration
        silence_samples = int(gap_seconds * synth_rate)
        pcm.extend([0] * (silence_samples * AUDIO_CHANNELS * upsample_factor))
        cursor_seconds += gap_seconds

    if sys.byteorder != "little":
        pcm.byteswap()
    return PCMVoice(pcm.tobytes(), cursor_seconds, tuple(mouth_windows))


class AudioDirector:
    """Failure-tolerant serialized voice queue driven by immutable round records."""

    def __init__(
        self,
        seed: Optional[int],
        available: bool,
        muted: bool = False,
        init_error: str = "",
    ) -> None:
        self.seed = seed
        self.available = bool(available and pygame is not None and pygame.mixer.get_init())
        self.muted = bool(muted)
        self.init_error = init_error
        self.cache: "OrderedDict[Tuple[Any, ...], Tuple[Any, PCMVoice]]" = OrderedDict()
        self.pending: "deque[QueuedVoice]" = deque()
        self.active: Optional[ActiveVoice] = None
        self.voice_channel = None
        self.fx_channel = None
        if self.available:
            try:
                pygame.mixer.set_num_channels(max(4, pygame.mixer.get_num_channels()))
                self.voice_channel = pygame.mixer.Channel(0)
                self.fx_channel = pygame.mixer.Channel(1)
            except pygame.error as error:
                self.available = False
                self.init_error = str(error)
        self.reset(seed)

    def reset(self, seed: Optional[int], now: Optional[float] = None) -> None:
        self.seed = seed
        if self.voice_channel is not None:
            self.voice_channel.stop()
        if self.fx_channel is not None:
            self.fx_channel.stop()
        self.pending.clear()
        self.active = None
        self.seen_rounds = 0
        self.seen_event_tokens: set[Tuple[int, int]] = set()
        self.victory_token: Optional[Tuple[int, int]] = None
        self.next_action_time = 0.0
        self.started_at = time.monotonic() if now is None else float(now)
        self.last_idle_slot: List[Optional[int]] = [None] * len(VOICE_PROFILES)
        self.idle_due: List[Optional[Tuple[int, float, str]]] = [None] * len(VOICE_PROFILES)
        self.last_idle_global = -999.0
        self.last_idle_per_npc = [-999.0] * len(VOICE_PROFILES)

    def cleanup(self) -> None:
        if self.voice_channel is not None:
            self.voice_channel.stop()
        if self.fx_channel is not None:
            self.fx_channel.stop()
        self.pending.clear()
        self.active = None
        self.cache.clear()

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self.voice_channel is not None:
            self.voice_channel.stop()
        if self.fx_channel is not None:
            self.fx_channel.stop()
        self.pending.clear()
        self.active = None
        self.next_action_time = time.monotonic()
        self.idle_due = [None] * len(VOICE_PROFILES)
        return self.muted

    @property
    def status_label(self) -> str:
        if self.muted:
            return "8-BIT VOICES: MUTED  [M]"
        if not self.available:
            return "8-BIT VOICES: SILENT FALLBACK"
        return "8-BIT VOICES: LIVE  [M TO MUTE]"

    def _remove_idle_queue(self) -> None:
        self.pending = deque(cue for cue in self.pending if cue.priority > 0)
        if self.active is not None and self.active.priority == 0:
            if self.voice_channel is not None:
                self.voice_channel.fadeout(55)
            self.active = None

    def _retime_pending(self, now: float, spacing: float = 0.18) -> None:
        base = float(now)
        if self.active is not None:
            base = max(base, self.active.start_time + self.active.duration + 0.04)
        self.pending = deque(
            replace(cue, earliest_time=base + index * spacing)
            for index, cue in enumerate(self.pending)
        )
        self.next_action_time = base + len(self.pending) * spacing

    def _enqueue(self, cue: QueuedVoice, now: Optional[float] = None) -> bool:
        dropped = False
        if len(self.pending) >= AUDIO_QUEUE_LIMIT:
            idle_cue = next((item for item in self.pending if item.priority == 0), None)
            if idle_cue is not None:
                self.pending.remove(idle_cue)
                dropped = True
            elif cue.priority >= 2:
                self.pending.popleft()
                dropped = True
            else:
                return False
        self.pending.append(cue)
        if dropped and now is not None:
            self._retime_pending(now)
        return True

    def _queue_action(
        self,
        state: Any,
        event: Any,
        event_index: int,
        now: float,
    ) -> None:
        if event.actor_index > 0:
            speaker = event.actor_index - 1
        elif event.target_index is not None and event.target_index > 0:
            speaker = event.target_index - 1
        elif event.action == "investigate":
            speaker = int(clamp(state.last_reaction_index, 0, 3))
        else:
            return
        participant_index = speaker + 1
        cue_key = (
            "elimination"
            if participant_index in event.eliminated_indices
            else event.action
        )
        if len(event.voice_statuses) == len(STATUS_KEYS):
            mood_statuses = dict(zip(STATUS_KEYS, event.voice_statuses))
        else:
            # Compatibility fallback for records created by an older build.
            mood_statuses = state.npcs[speaker].statuses
        mood = voice_mood_from_statuses(mood_statuses)
        variant = presentation_word(
            self.seed,
            event.round_number * 131 + event_index * 17 + speaker * 997,
        ) & 0xFFFF
        earliest = max(float(now), self.next_action_time)
        admitted = self._enqueue(
            QueuedVoice(
                speaker,
                speaker,
                cue_key,
                mood,
                earliest,
                2,
                variant,
                voice_caption(speaker, cue_key, mood),
            ),
            now,
        )
        if admitted:
            self.next_action_time = max(
                self.next_action_time,
                self.pending[-1].earliest_time + 0.20,
            )

    def _ingest_rounds(self, state: Any, now: float) -> None:
        if len(state.round_history) < self.seen_rounds:
            self.seen_rounds = 0
            self.seen_event_tokens.clear()
        if self.seen_rounds >= len(state.round_history):
            return
        self._remove_idle_queue()
        for record_index in range(self.seen_rounds, len(state.round_history)):
            record = state.round_history[record_index]
            for event_index, event in enumerate(record.events):
                token = (event.round_number, event.actor_index)
                if token in self.seen_event_tokens:
                    continue
                self._queue_action(state, event, event_index, now)
                self.seen_event_tokens.add(token)
        self.seen_rounds = len(state.round_history)

    def _ingest_live_events(self, state: Any, now: float) -> None:
        if state.phase != "resolving_npcs" or not state.last_round_events:
            return
        new_events = [
            (event_index, event)
            for event_index, event in enumerate(state.last_round_events)
            if (event.round_number, event.actor_index) not in self.seen_event_tokens
        ]
        if not new_events:
            return
        self._remove_idle_queue()
        for event_index, event in new_events:
            self._queue_action(state, event, event_index, now)
            self.seen_event_tokens.add((event.round_number, event.actor_index))

    def _ingest_victory(self, state: Any, now: float) -> None:
        winner = state.last_survivor
        if not state.is_ending or winner is None:
            return
        participant_index = state.participants.index(winner)
        token = (state.turn, participant_index)
        if token == self.victory_token:
            return
        self.victory_token = token
        if participant_index > 0:
            speaker = participant_index - 1
            profile_index = speaker
            mood = voice_mood_from_statuses(winner.statuses)
        else:
            speaker = -1
            profile_index = presentation_word(self.seed, state.player.name) % len(VOICE_PROFILES)
            celebratory_statuses = dict(state.player.statuses)
            celebratory_statuses["emotional"] = max(celebratory_statuses["emotional"], 82)
            celebratory_statuses["social"] = max(celebratory_statuses["social"], 72)
            mood = voice_mood_from_statuses(celebratory_statuses)
        # Keep only the terminal round's latest voices. A victory should follow
        # its immediate causes, never wait behind minutes of stale autonomous
        # chatter retained by an intentionally adversarial burst.
        if len(self.pending) > 5:
            self.pending = deque(tuple(self.pending)[-5:])
            self._retime_pending(now, 0.18)
        earliest = max(now + 0.18, self.next_action_time + 0.16)
        admitted = self._enqueue(
            QueuedVoice(
                speaker,
                profile_index,
                "victory",
                mood,
                earliest,
                3,
                presentation_word(self.seed, f"victory:{state.turn}:{participant_index}"),
                voice_caption(profile_index, "victory", mood),
            ),
            now,
        )
        if admitted:
            self.next_action_time = self.pending[-1].earliest_time + 0.40

    def _schedule_idle(self, state: Any, now: float, allow_idle: bool) -> None:
        if not allow_idle or now - self.started_at < 4.0:
            return
        elapsed = max(0.0, now - state.idle_epoch)
        for npc_index, npc in enumerate(state.npcs):
            frame = idle_emote_frame(self.seed, npc_index, elapsed)
            previous = self.last_idle_slot[npc_index]
            if previous is None:
                self.last_idle_slot[npc_index] = frame.slot_index
            elif previous != frame.slot_index:
                self.last_idle_slot[npc_index] = frame.slot_index
                eligibility = presentation_word(
                    self.seed,
                    npc_index * 100_003 + frame.slot_index * 7_919,
                )
                if eligibility % 3 == 0:
                    delay = IDLE_PERSONALITIES[npc_index].slot_seconds * 0.24
                    self.idle_due[npc_index] = (
                        frame.slot_index,
                        now + delay,
                        frame.emote_key,
                    )
            due = self.idle_due[npc_index]
            if due is None or now < due[1]:
                continue
            self.idle_due[npc_index] = None
            if due[0] != frame.slot_index:
                continue
            if not npc_idle_enabled(state.phase, state.player.alive, npc.alive, state.is_ending):
                continue
            if now - self.last_idle_global < 5.5 or now - self.last_idle_per_npc[npc_index] < 16.0:
                continue
            cue_key = f"idle_{due[2]}"
            mood = voice_mood_from_statuses(npc.statuses)
            admitted = self._enqueue(
                QueuedVoice(
                    npc_index,
                    npc_index,
                    cue_key,
                    mood,
                    now,
                    0,
                    presentation_word(self.seed, f"idle:{npc_index}:{frame.slot_index}"),
                    voice_caption(npc_index, cue_key, mood),
                ),
                now,
            )
            if admitted:
                self.last_idle_global = now
                self.last_idle_per_npc[npc_index] = now
            break

    def _rendered_sound(self, cue: QueuedVoice) -> Optional[Tuple[Any, PCMVoice]]:
        key = (cue.profile_index, cue.cue_key, cue.mood.cache_key, cue.variant % 3)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached
        pcm = synthesize_voice_pcm(
            cue.profile_index,
            cue.cue_key,
            cue.mood,
            cue.variant % 3,
        )
        try:
            sound = pygame.mixer.Sound(buffer=pcm.buffer)
            sound.set_volume(0.62 if cue.priority < 3 else 0.76)
        except (pygame.error, ValueError) as error:
            self.available = False
            self.init_error = str(error)
            return None
        rendered = (sound, pcm)
        self.cache[key] = rendered
        while len(self.cache) > AUDIO_CACHE_LIMIT:
            self.cache.popitem(last=False)
        return rendered

    def _start_next(self, now: float) -> None:
        if not self.pending or self.voice_channel is None:
            return
        cue = self.pending[0]
        if now < cue.earliest_time or self.voice_channel.get_busy():
            return
        self.pending.popleft()
        rendered = self._rendered_sound(cue)
        if rendered is None:
            return
        sound, pcm = rendered
        profile = VOICE_PROFILES[cue.profile_index]
        left = 1.0 - max(0.0, profile.pan) * 0.55
        right = 1.0 + min(0.0, profile.pan) * 0.55
        self.voice_channel.set_volume(left, right)
        self.voice_channel.play(sound)
        self.active = ActiveVoice(
            cue.speaker_index,
            cue.cue_key,
            now,
            pcm.duration,
            pcm.mouth_windows,
            cue.caption,
            cue.priority,
        )

    def update(self, state: Any, now: float, allow_idle: bool = True) -> None:
        self._ingest_live_events(state, now)
        self._ingest_rounds(state, now)
        self._ingest_victory(state, now)
        if self.muted or not self.available:
            self.pending.clear()
            self.active = None
            return
        if self.active is not None:
            finished = now >= self.active.start_time + self.active.duration + 0.04
            if finished or (self.voice_channel is not None and not self.voice_channel.get_busy()):
                self.active = None
        self._schedule_idle(state, now, allow_idle and not self.pending and self.active is None)
        if self.active is None:
            self._start_next(now)

    def voice_envelope(self, npc_index: int, now: float) -> float:
        active = self.active
        if active is None or active.speaker_index != npc_index:
            return 0.0
        local_time = now - active.start_time
        for start, end in active.mouth_windows:
            if start <= local_time <= end:
                progress = (local_time - start) / max(1e-6, end - start)
                return math.sin(progress * math.pi) ** 0.58
        return 0.0

    def caption_for_npc(self, npc_index: int, now: float) -> str:
        active = self.active
        if active is None or active.speaker_index != npc_index:
            return ""
        if now <= active.start_time + active.duration + 0.18:
            return active.caption
        return ""

    def stage_caption(self, now: float) -> str:
        active = self.active
        if active is None or active.cue_key != "victory":
            return ""
        return active.caption if now <= active.start_time + active.duration + 0.35 else ""


# ---------------------------------------------------------------------------
# Presentation-only winner choreography and particle finale selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VictoryFrame:
    celebration_key: str
    celebration_label: str
    meme_caption: str
    gesture_key: str
    next_gesture_key: str
    gesture_blend: float
    bounce: float
    body_x: float
    body_roll: float
    head_yaw: float
    head_nod: float
    pulse: float


NPC_VICTORY_GESTURES: Tuple[Tuple[str, ...], ...] = (
    ("viral_point", "affirmation", "seesaw_67", "pixel_wave"),
    ("heart_hands", "retro_chacha", "open_palms", "affirmation"),
    ("play_them_off", "temple_tap", "affirmation", "pixel_wave"),
    ("prayer_pose", "open_palms", "retro_chacha", "heart_hands"),
)
NPC_VICTORY_LABELS: Tuple[str, ...] = (
    "NO-EXIT VICTORY SHUFFLE",
    "CONSENSUS CROWN CHA-CHA",
    "FINAL-FEATURE AIR KEYBOARD",
    "COSMIC VIBE ASCENSION",
)
NPC_VICTORY_CAPTIONS: Tuple[str, ...] = (
    "MEDICALLY SPEAKING: STILL HERE",
    "THE ROOM VOTED. THE ROOM WAS ME.",
    "404 OPPONENTS FOUND: ZERO",
    "THE SIGNAL SAID: ABSOLUTELY.",
)
PLAYER_VICTORY_EFFECTS: Tuple[str, ...] = (
    "FIFTH-WAVE SUPERNOVA",
    "GOLDEN GLITCH FOUNTAIN",
    "CYAN VICTORY VORTEX",
    "PIXEL-CROWN OVERDRIVE",
)


def victory_frame(seed: Optional[int], npc_index: int, elapsed: float) -> VictoryFrame:
    if not 0 <= npc_index < len(NPC_VICTORY_GESTURES):
        raise IndexError("NPC victory index must be between zero and three")
    elapsed = max(0.0, float(elapsed))
    beat_seconds = 1.02 + npc_index * 0.055
    scheduled = elapsed + (presentation_word(seed, npc_index * 0x2D31) & 0xFF) / 255.0 * 0.18
    beat_index = int(scheduled // beat_seconds)
    progress = (scheduled / beat_seconds) - beat_index
    gestures = NPC_VICTORY_GESTURES[npc_index]
    offset = presentation_word(seed, f"victory-loop:{npc_index}") % len(gestures)
    gesture_index = (beat_index + offset) % len(gestures)
    transition = float(clamp((progress - 0.72) / 0.28, 0.0, 1.0))
    transition = transition * transition * (3.0 - 2.0 * transition)
    phase = scheduled * math.tau / beat_seconds
    hop = abs(math.sin(progress * math.pi))
    return VictoryFrame(
        f"npc_{npc_index}_celebration",
        NPC_VICTORY_LABELS[npc_index],
        NPC_VICTORY_CAPTIONS[npc_index],
        gestures[gesture_index],
        gestures[(gesture_index + 1) % len(gestures)],
        transition,
        hop * (0.090 + npc_index * 0.010),
        math.sin(phase * 0.50 + npc_index) * (0.060 + npc_index * 0.008),
        math.sin(phase + npc_index * 0.7) * (3.6 + npc_index * 0.45),
        math.sin(phase * 0.50 + npc_index * 0.9) * 5.0,
        math.sin(phase * 1.5) * 2.1 + hop * 1.4,
        hop * 0.045,
    )


def player_victory_effect_index(
    seed: Optional[int],
    player_name: str,
    cycle: int = 0,
) -> int:
    word = presentation_word(seed, player_name)
    return _mix_u32(word ^ (int(cycle) * 0x9E37_79B9)) % len(PLAYER_VICTORY_EFFECTS)

INK = (13, 17, 20)
PANEL = (24, 29, 32)
PANEL_ALT = (31, 38, 42)
LINE = (74, 84, 88)
WHITE = (235, 239, 235)
MUTED = (157, 168, 168)
GOLD = (230, 181, 72)
CYAN = (74, 190, 201)
GREEN = (92, 190, 124)
RED = (218, 91, 82)
CORAL = (220, 118, 93)
MAGENTA = (208, 104, 177)


def clamp(value: int | float, low: int | float, high: int | float) -> int | float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SpiritualPortraitStyle:
    """Mutually exclusive visual language for the ternary spirituality index."""

    horns: bool
    flames: bool
    halo: bool
    dim_window: bool
    light_scale: float
    halo_layers: int


@dataclass(frozen=True)
class PortraitLightProfile:
    ambient: Tuple[float, float, float, float]
    diffuse: Tuple[float, float, float, float]
    secondary_diffuse: Tuple[float, float, float, float]
    dim_overlay_alpha: float


HORN_BODY_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (3, 4, 6),
    (12, 13, 16),
    (24, 26, 29),
)
HORN_BASE_Y = 0.52
HORN_CENTERLINE: Tuple[Tuple[float, float, float], ...] = (
    (0.000, 0.000, 0.170),
    (0.055, 0.180, 0.135),
    (0.095, 0.385, 0.095),
    (0.070, 0.605, 0.050),
    (-0.010, 0.800, 0.000),
)
HORN_FLAME_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (242, 45, 7),
    (255, 116, 8),
    (255, 199, 36),
    (255, 247, 190),
)
# Inner core to outer bloom: radius grows while opacity falls.
HALO_GLOW_LAYERS: Tuple[Tuple[float, float, float], ...] = (
    (0.720, 0.014, 0.96),
    (0.748, 0.026, 0.48),
    (0.785, 0.048, 0.23),
    (0.835, 0.078, 0.10),
)


def spiritual_portrait_style(value: int) -> SpiritualPortraitStyle:
    if value not in (-1, 0, 1):
        raise ValueError("Spiritual portrait state must be -1, 0, or +1")
    if value < 0:
        return SpiritualPortraitStyle(True, True, False, False, 0.84, 0)
    if value > 0:
        return SpiritualPortraitStyle(False, False, True, False, 1.10, len(HALO_GLOW_LAYERS))
    return SpiritualPortraitStyle(False, False, False, True, 0.44, 0)


def portrait_light_profile(
    portrait_index: int,
    spirituality: int,
    pulse: float = 1.0,
) -> PortraitLightProfile:
    """Return deterministic fixed-function lighting for one portrait viewport."""
    if not 0 <= int(portrait_index) < 4:
        raise IndexError("Portrait light index must be between zero and three")
    style = spiritual_portrait_style(spirituality)
    if portrait_index == 2:
        base_ambient = (0.34, 0.30, 0.14)
        base_diffuse = (1.00, 0.84, 0.38)
    else:
        base_ambient = (0.28, 0.28, 0.28)
        base_diffuse = (0.88, 0.88, 0.84)

    if style.dim_window:
        ambient = tuple(component * style.light_scale for component in base_ambient)
        diffuse = tuple(component * style.light_scale for component in base_diffuse)
        secondary = (0.0, 0.0, 0.0)
        overlay = 0.24
    elif style.halo:
        ambient = tuple(
            min(1.0, base_ambient[channel] * 0.96 + (0.03, 0.10, 0.12)[channel])
            for channel in range(3)
        )
        diffuse = tuple(
            min(1.0, base_diffuse[channel] * 0.78 + (0.55, 0.96, 1.00)[channel] * 0.28)
            for channel in range(3)
        )
        glow = float(clamp(pulse, 0.65, 1.25))
        secondary = (0.12 * glow, 0.44 * glow, 0.58 * glow)
        overlay = 0.0
    else:
        ambient = tuple(
            min(1.0, base_ambient[channel] * style.light_scale + (0.07, 0.015, 0.004)[channel])
            for channel in range(3)
        )
        diffuse = tuple(component * 0.90 for component in base_diffuse)
        fire = float(clamp(pulse, 0.65, 1.25))
        secondary = (0.72 * fire, 0.16 * fire, 0.025 * fire)
        overlay = 0.0
    return PortraitLightProfile(
        tuple(ambient) + (1.0,),
        tuple(diffuse) + (1.0,),
        tuple(secondary) + (1.0,),
        overlay,
    )


def horn_tip_position(side: float) -> Tuple[float, float, float]:
    outward, height, _radius = HORN_CENTERLINE[-1]
    return (float(side) * (0.48 + outward), HORN_BASE_Y + height, 0.115)


def horn_flame_parameters(now: float, phase: float, side: float) -> Tuple[float, float, float]:
    """Return bounded, deterministic flame height, width, and lateral sway."""
    wave = math.sin(float(now) * 8.7 + float(phase) + float(side) * 1.31)
    cross_wave = math.sin(float(now) * 13.1 + float(phase) * 0.7 - float(side))
    height = 0.31 + wave * 0.035 + cross_wave * 0.018
    width = 0.135 + cross_wave * 0.014
    sway = wave * 0.045 + cross_wave * 0.018
    return (height, width, sway)


def normalize_player_name(value: str) -> str:
    """Validate and normalize the short, display-safe player name."""
    if not isinstance(value, str):
        raise TypeError("Player name must be text")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("Enter at least one letter or number.")
    if len(normalized) > MAX_PLAYER_NAME_LENGTH:
        raise ValueError(f"Use {MAX_PLAYER_NAME_LENGTH} characters or fewer.")
    if any(character not in PLAYER_NAME_ALLOWED_CHARS for character in normalized):
        raise ValueError("Use letters, numbers, spaces, apostrophes, or hyphens only.")
    if not any(character.isascii() and character.isalnum() for character in normalized):
        raise ValueError("Enter at least one letter or number.")
    return normalized


def possessive_name(name: str) -> str:
    return f"{name}'" if name.lower().endswith("s") else f"{name}'s"


def mix_color(a: Sequence[int], b: Sequence[int], amount: float) -> Tuple[int, int, int]:
    amount = float(clamp(amount, 0.0, 1.0))
    return tuple(int(a[i] + (b[i] - a[i]) * amount) for i in range(3))


@dataclass(frozen=True)
class ItemSpec:
    key: str
    label: str
    short_label: str
    give_amount: int
    use_amount: int
    color: Tuple[int, int, int]


ITEM_SPECS = (
    ItemSpec("dollars", "Dollars", "Dollars", 5, 5, (112, 196, 123)),
    ItemSpec("water_liters", "Water (liters)", "Water", 1, 1, (78, 171, 218)),
    ItemSpec("food_pounds", "Food (pounds)", "Food", 1, 1, (223, 168, 78)),
    ItemSpec("caffeine_pills", "Caffeine pills", "Caffeine", 1, 1, (205, 124, 77)),
    ItemSpec(
        "acetaminophen_pills",
        "Acetaminophen pills",
        "Acetaminophen",
        1,
        1,
        (205, 116, 146),
    ),
)
ITEM_BY_KEY = {item.key: item for item in ITEM_SPECS}
ITEM_KEYS = tuple(item.key for item in ITEM_SPECS)

# Five dollars and one unit of every supply were the original prototype's default
# exchange sizes. Keeping that relationship gives custom quantities a stable
# barter value without exposing an NPC's private stock.
TRADE_UNIT_VALUES = {
    spec.key: 5.0 / spec.give_amount
    for spec in ITEM_SPECS
}

# Every index contributes monotonically to trade willingness. Spiritual is a
# signed ternary index (-1, 0, or 1); the other five are percentages, and the
# weights sum to one.
TRADE_STATUS_WEIGHTS = {
    "physical": 0.12,
    "emotional": 0.18,
    "cognitive": 0.20,
    "social": 0.22,
    "sentient": 0.18,
    "spiritual": 0.10,
}

# Every item has an explicit relationship to every index. A zero means the
# item does not directly change that index in the abstract model.
ITEM_STATUS_EFFECTS: Dict[str, Dict[str, int]] = {
    "dollars": {
        "physical": 0,
        "emotional": 1,
        "cognitive": 1,
        "social": 2,
        "sentient": 1,
        "spiritual": 0,
    },
    "water_liters": {
        "physical": 5,
        "emotional": 1,
        "cognitive": 2,
        "social": 0,
        "sentient": 1,
        "spiritual": 0,
    },
    "food_pounds": {
        "physical": 6,
        "emotional": 2,
        "cognitive": 1,
        "social": 0,
        "sentient": 1,
        "spiritual": 0,
    },
    "caffeine_pills": {
        "physical": -1,
        "emotional": -1,
        "cognitive": 7,
        "social": -1,
        "sentient": 0,
        "spiritual": 0,
    },
    "acetaminophen_pills": {
        "physical": 5,
        "emotional": 1,
        "cognitive": 0,
        "social": 0,
        "sentient": 0,
        "spiritual": 0,
    },
}

# NPCs reason in usable, diminishing-return supply bands instead of treating a
# million dollars and a million liters of water as interchangeable raw numbers.
# The values are private reserve targets in inventory units; only stock-safe
# prose derived from them is ever exposed to the player.
NPC_ITEM_RESERVES: Dict[str, float] = {
    "dollars": 20.0,
    "water_liters": 3.0,
    "food_pounds": 3.0,
    "caffeine_pills": 2.0,
    "acetaminophen_pills": 2.0,
}
NPC_STATUS_IMPORTANCE: Dict[str, float] = {
    "physical": 1.65,
    "emotional": 1.00,
    "cognitive": 1.25,
    "social": 0.95,
    "sentient": 1.05,
    "spiritual": 0.75,
}
REST_STATUS_EFFECTS: Dict[str, int] = {
    "physical": 8,
    "emotional": 4,
    "cognitive": 3,
    "social": 1,
    "sentient": 1,
    "spiritual": 0,
}

# Small disposition nudges preserve the cast's identities without overruling
# immediate needs. Indexes, target state, supplies, and survival pressure still
# provide the overwhelming majority of every utility score.
NPC_ROLE_ACTION_BIASES: Tuple[Mapping[str, float], ...] = (
    {"listen": 0.16, "give": 0.38, "use": 0.30, "rest": 0.15, "trade": 0.10},
    {"listen": 0.35, "compliment": 0.28, "flirt": 0.16, "give": 0.20, "trade": 0.25, "reflect": 0.12},
    {"talk": 0.28, "steal": 0.12, "trade": 0.30, "reflect": 0.22, "use": 0.18, "fight": 0.08},
    {"listen": 0.20, "flirt": 0.12, "steal": 0.10, "trade": 0.15, "reflect": 0.34, "compliment": 0.12, "rest": 0.10},
)
NPC_ROLE_ITEM_BIASES: Tuple[Mapping[str, float], ...] = (
    {"water_liters": 0.10, "acetaminophen_pills": 0.25},
    {"water_liters": 0.12, "food_pounds": 0.12},
    {"dollars": 0.12, "caffeine_pills": 0.25},
    {"water_liters": 0.15, "food_pounds": 0.12},
)


@dataclass(frozen=True)
class Profile:
    name: str
    role: str
    accent: Tuple[int, int, int]
    skin: Tuple[int, int, int]
    hair: Tuple[int, int, int]
    talk_lines: Tuple[str, ...]
    listen_lines: Tuple[str, ...]


PLAYER_PROFILE = Profile(
    DEFAULT_PLAYER_NAME,
    "Courier",
    (91, 167, 197),
    (192, 145, 111),
    (62, 48, 39),
    (),
    (),
)

NPC_PROFILES = (
    Profile(
        "Mara Voss",
        "Field medic",
        (205, 91, 82),
        (102, 70, 55),
        (30, 24, 23),
        (
            "This room is measuring us. Care may be the only variable it cannot fake.",
            "Numbers matter. So does noticing who has gone quiet.",
            "I can treat pain. I cannot manufacture trust.",
        ),
        (
            "Mara names the fatigue she has hidden and lets the silence settle.",
            "Mara admits that keeping everyone alive has made her afraid to choose.",
        ),
    ),
    Profile(
        "Imani Reed",
        "Mediator",
        (71, 170, 131),
        (129, 82, 61),
        (42, 29, 25),
        (
            "A group survives twice: first in body, then in the stories it tells itself.",
            "Ask what each choice costs the person standing farthest from power.",
            "Agreement is useful. Honest disagreement is often more useful.",
        ),
        (
            "Imani describes the tension in the group without assigning blame.",
            "Imani speaks slowly about home, then asks what home means now.",
        ),
    ),
    Profile(
        "Elias Vale",
        "Systems engineer",
        (218, 164, 68),
        (195, 157, 121),
        (65, 51, 37),
        (
            "Every broken system is still a system. Find the rule it is following.",
            "The fifth pulse changes after every interaction. That is not accidental.",
            "Give me a constraint and I will give you three imperfect tests for it.",
        ),
        (
            "Elias stops solving the problem long enough to describe why it frightens him.",
            "Elias confesses that certainty has become a shelter he cannot maintain.",
        ),
    ),
    Profile(
        "Noor Aster",
        "Pattern reader",
        (139, 112, 201),
        (151, 106, 78),
        (34, 27, 31),
        (
            "A room without a door may still have a direction hidden inside it.",
            "The Fifth Signal may be a machine, or it may be what we become together.",
            "I trust instincts after they have survived a hard question.",
        ),
        (
            "Noor describes a recurring dream of five lights answering in a sealed chamber.",
            "Noor listens in return, making the conversation feel briefly unhurried.",
        ),
    ),
)


@dataclass
class Participant:
    profile: Profile
    statuses: Dict[str, int]
    inventory: Dict[str, int]
    last_caffeine_turn: int = -999
    last_acetaminophen_turn: int = -999
    eliminated: bool = False
    elimination_turn: Optional[int] = None
    gesture_key: str = "idle"
    gesture_turn: int = 0
    gesture_changed_at: float = field(default_factory=time.monotonic)
    display_name: Optional[str] = None

    @property
    def name(self) -> str:
        return self.display_name or self.profile.name

    @property
    def alive(self) -> bool:
        return not self.eliminated

    def adjust(self, key: str, amount: int) -> None:
        if self.eliminated:
            return
        if key == "spiritual":
            self.statuses[key] = int(clamp(self.statuses[key] + amount, -1, 1))
            return
        low = 0 if key == "physical" else 1
        self.statuses[key] = int(clamp(self.statuses[key] + amount, low, 100))

    def set_spiritual(self, value: int) -> None:
        if self.eliminated:
            return
        self.statuses["spiritual"] = int(clamp(int(value), -1, 1))

    def change_item(self, key: str, amount: int) -> None:
        if self.eliminated:
            return
        self.inventory[key] = int(clamp(self.inventory[key] + amount, 0, INVENTORY_CAP))


def random_participant(profile: Profile, rng: random.Random) -> Participant:
    # A fresh room scenario never opens on an already critical participant. Physical
    # Health can later reach zero through exposure, choices, or repeated use.
    statuses = {key: rng.randint(20, 100) for key in RANGED_STATUS_KEYS}
    statuses["spiritual"] = rng.randint(-1, 1)
    inventory = {key: rng.randint(0, 100) for key in ITEM_KEYS}
    return Participant(profile, statuses, inventory)


@dataclass
class TradeDraft:
    target_index: int
    offered_item: str
    offered_quantity: int
    requested_item: str
    requested_quantity: int


@dataclass(frozen=True)
class ExchangePlan:
    offered_item: str
    offered_quantity: int
    requested_item: str
    requested_quantity: int


@dataclass(frozen=True)
class NPCDecision:
    """One scored, presentation-safe plan for an NPC's single round slot."""

    action: str
    target_index: int
    item_key: Optional[str]
    utility: float
    reasoning: str
    exchange: Optional[ExchangePlan] = None


@dataclass(frozen=True)
class RoundEvent:
    round_number: int
    actor_index: int
    action: str
    target_index: Optional[int]
    summary: str
    success: bool = True
    eliminated_indices: Tuple[int, ...] = ()
    gesture_key: Optional[str] = None
    voice_statuses: Tuple[int, ...] = ()
    item_key: Optional[str] = None
    reasoning: str = ""
    decision_score: Optional[float] = None
    exchange: Optional[ExchangePlan] = None
    impact: str = ""


@dataclass(frozen=True)
class RoundRecord:
    """An immutable, compact archive entry for one fully resolved round."""

    round_number: int
    events: Tuple[RoundEvent, ...]
    notices: Tuple[str, ...] = ()

    @property
    def player_led(self) -> bool:
        return bool(self.events and self.events[0].actor_index == 0)


def face_expression_parameters(
    participant: Participant,
    now: float,
    phase: float,
) -> Dict[str, float]:
    """Convert all six public indexes into bounded portrait expression controls."""
    if participant.eliminated:
        return {
            "eye_open": 1.55,
            "mouth_curve": -0.16,
            "mouth_half_width": 0.38,
            "pupil_radius": 0.0,
            "pupil_dx": 0.0,
            "pupil_dy": 0.0,
            "brow_tilt": 0.12,
            "inner_brow": 0.0,
            "fatigue": 1.0,
            "sentient_glint": 0.0,
            "spiritual_glow": 0.0,
            "negative_spirituality": 1.0 if participant.statuses["spiritual"] == -1 else 0.0,
            "eliminated": 1.0,
        }

    physical = participant.statuses["physical"] / 100.0
    emotional = participant.statuses["emotional"] / 100.0
    cognitive = participant.statuses["cognitive"] / 100.0
    social = participant.statuses["social"] / 100.0
    sentient = participant.statuses["sentient"] / 100.0
    spiritual = float(participant.statuses["spiritual"])

    focus_jitter = (1.0 - cognitive) * 0.045
    return {
        # Physical: tired faces develop heavy lids and under-eye stress marks.
        "eye_open": 0.48 + physical * 0.52,
        "fatigue": 1.0 - physical,
        # Emotional: the mouth and outer brow move from frown to smile.
        "mouth_curve": (emotional * 2.0 - 1.0) * 0.13,
        # Social: eye contact centers and the mouth opens as connection rises.
        "mouth_half_width": 0.30 + social * 0.10,
        "pupil_dx": (1.0 - social) * 0.09 * math.sin(phase + 0.8)
        + focus_jitter * math.sin(now * 2.1 + phase),
        # Cognitive: low focus produces an asymmetric vertical eye drift and tension.
        "pupil_dy": focus_jitter * math.sin(now * 1.7 + phase * 1.9),
        "brow_tilt": (0.5 - emotional) * 0.10 + (1.0 - cognitive) * 0.035,
        # Sentient: awareness enlarges the pupils, catchlights, and sympathetic inner brow.
        "pupil_radius": 0.055 + sentient * 0.040,
        "inner_brow": (sentient - 0.5) * 0.055,
        "sentient_glint": sentient,
        # Spiritual: cyan eye/forehead light complements the existing halo.
        "spiritual_glow": max(0.0, spiritual),
        "negative_spirituality": 1.0 if spiritual < 0.0 else 0.0,
        "eliminated": 0.0,
    }


@dataclass(frozen=True)
class StoryChoice:
    label: str
    outcome: str
    next_scene: str
    player_effects: Mapping[str, int] = field(default_factory=dict)
    target_effects: Mapping[str, int] = field(default_factory=dict)
    party_effects: Mapping[str, int] = field(default_factory=dict)
    costs: Mapping[str, int] = field(default_factory=dict)
    gains: Mapping[str, int] = field(default_factory=dict)
    player_spiritual: Optional[int] = None
    target_spiritual: Optional[int] = None
    special: str = ""


@dataclass(frozen=True)
class Scene:
    title: str
    prompt: str
    choices: Tuple[StoryChoice, StoryChoice, StoryChoice]


def build_scenes() -> Dict[str, Scene]:
    return {
        "sealed_room": Scene(
            "Stage 1: The Sealed Room",
            (
                "Five people wake in a windowless room with seamless walls, one metal table, "
                "four dark wall nodes, and no known escape. A soft pulse repeats in groups of five."
            ),
            (
                StoryChoice(
                    "Inspect the walls for a seam",
                    (
                        "You and {npc} test the walls. Four taps return immediately; a fifth "
                        "answer arrives from somewhere inside the room."
                    ),
                    "resonance",
                    {"cognitive": 4, "sentient": 2},
                    {"cognitive": 2, "social": 1},
                ),
                StoryChoice(
                    "Compare your final memories",
                    (
                        "You ask {npc} to begin. Each person remembers a different tone just "
                        "before waking, but nobody remembers entering the room."
                    ),
                    "testimony",
                    {"emotional": 1, "social": 3, "sentient": 2},
                    {"emotional": 2, "social": 3},
                ),
                StoryChoice(
                    "Examine the metal table",
                    (
                        "The table has no drawer, yet its underside hums. You recover one sealed "
                        "water pouch and find five contact marks beneath the rim."
                    ),
                    "mechanism",
                    {"physical": -1, "cognitive": 3},
                    {"cognitive": 2},
                    gains={"water_liters": 1},
                ),
            ),
        ),
        "resonance": Scene(
            "Stage 2A: Echoes in Concrete",
            (
                "The walls return four ordinary echoes and one delayed pulse that changes when "
                "someone speaks, threatens, comforts, or falls silent. The room is listening."
            ),
            (
                StoryChoice(
                    "Tap the four wall nodes in sequence",
                    (
                        "Each node adopts one companion's rhythm. The table hum answers as a "
                        "fifth channel, but its pattern still will not repeat."
                    ),
                    "mechanism",
                    {"cognitive": 4, "sentient": 2},
                    {"cognitive": 3},
                ),
                StoryChoice(
                    "Listen without making a sound",
                    (
                        "In the silence, {npc} notices the delayed pulse copying the emotional "
                        "shape of the group's last exchange rather than its sound."
                    ),
                    "testimony",
                    {"emotional": 2, "cognitive": 3, "sentient": 4},
                    {"cognitive": 3, "sentient": 3},
                    special="echo",
                ),
                StoryChoice(
                    "Strike the wall to force a response",
                    (
                        "The wall does not break. A red waveform records the blow, and the fifth "
                        "pulse returns sharper while {npc} watches the room learn aggression."
                    ),
                    "mechanism",
                    {"physical": -4, "emotional": -2, "cognitive": 2},
                    {"emotional": -2, "sentient": 2},
                ),
            ),
        ),
        "testimony": Scene(
            "Stage 2B: Five Incomplete Accounts",
            (
                "Every remembered tone matches one person, yet the fifth tone changes whenever "
                "the group interacts. Suspicion rises because no one can prove who put them here."
            ),
            (
                StoryChoice(
                    "Invite each person to speak honestly",
                    (
                        "The accounts overlap at one detail: a voice promised that the last "
                        "response, not the first command, would reveal the Fifth Signal."
                    ),
                    "pattern",
                    {"emotional": 3, "social": 5, "sentient": 4},
                    {"emotional": 3, "social": 4},
                    party_effects={"social": 1, "sentient": 1},
                ),
                StoryChoice(
                    "Accuse {npc} of hiding the truth",
                    (
                        "The accusation produces the room's strongest waveform yet. It reveals "
                        "a concealed console, but trust in the room fractures."
                    ),
                    "mechanism",
                    {"emotional": -3, "social": -5, "cognitive": 2},
                    {"emotional": -4, "social": -4},
                    party_effects={"emotional": -1, "social": -1},
                ),
                StoryChoice(
                    "Recreate the five remembered tones",
                    (
                        "Four tones settle into fixed channels. The fifth refuses to exist until "
                        "one person acts and the other four answer in their own ways."
                    ),
                    "pattern",
                    {"cognitive": 5, "sentient": 4},
                    {"cognitive": 3, "sentient": 2},
                    special="echo",
                ),
            ),
        ),
        "mechanism": Scene(
            "Stage 3: The Table Console",
            (
                "A console unfolds beneath the table: four fixed channels, one blank channel, "
                "and a diagram of five figures taking turns. No symbol resembles a door control."
            ),
            (
                StoryChoice(
                    "Bridge the contacts with five dollars in coins",
                    (
                        "The coins complete the circuit. The console displays five empty turn "
                        "slots and waits for a complete sequence of human responses."
                    ),
                    "pattern",
                    {"cognitive": 4, "sentient": 2},
                    {"cognitive": 2},
                    costs={"dollars": 5},
                ),
                StoryChoice(
                    "Feed one liter of water through the coolant port",
                    (
                        "The coolant stabilizes the sensor and exposes a recording: FIRST ACTS. FOUR ANSWER. "
                        "FIFTH EMERGES. The room still offers no obvious exit."
                    ),
                    "pattern",
                    {"physical": -1, "cognitive": 5},
                    {"cognitive": 3},
                    costs={"water_liters": 1},
                ),
                StoryChoice(
                    "Study the symbols without touching them",
                    (
                        "You and {npc} identify the symbols as interaction types: care, inquiry, "
                        "praise, hostility, reflection, and force."
                    ),
                    "pattern",
                    {"cognitive": 6, "sentient": 3},
                    {"cognitive": 4, "social": 2},
                ),
            ),
        ),
        "pattern": Scene(
            "Stage 4: A Complete Round",
            (
                "The console reveals it has recorded every round: {player} first, then one self-chosen "
                "response from each living companion. Missing people leave visibly silent channels."
            ),
            (
                StoryChoice(
                    "Propose a chain of sincere compliments",
                    (
                        "You propose care as the test. Whatever replies actually follow, their "
                        "contrast creates a stable fifth waveform the console can compare."
                    ),
                    "discovery",
                    {"emotional": 4, "social": 6, "sentient": 3},
                    {"emotional": 3, "social": 4},
                    party_effects={"emotional": 2, "social": 2},
                ),
                StoryChoice(
                    "Announce a conflict test and measure the spikes",
                    (
                        "The room records every reply to the hostile setup. The waveform becomes "
                        "jagged, proving that it measures relationship rather than goodness."
                    ),
                    "discovery",
                    {"physical": -3, "emotional": -4, "social": -5, "cognitive": 4},
                    {"emotional": -3, "social": -3},
                    party_effects={"emotional": -2, "social": -2},
                ),
                StoryChoice(
                    "Assign one tone to each person",
                    (
                        "You give structure without dictating the replies. The actual choices "
                        "lock into one pattern, with silent gaps wherever a person is missing."
                    ),
                    "discovery",
                    {"cognitive": 6, "social": 3, "sentient": 5},
                    {"cognitive": 4, "sentient": 3},
                    special="signal_sync",
                ),
            ),
        ),
        "discovery": Scene(
            "Stage 5: The Fifth Signal",
            (
                "The Fifth Signal is identified as the pattern created when one person acts "
                "and the others answer freely. The console labels it REFERENCE A, then opens "
                "a second, fainter sequence beneath it."
            ),
            (
                StoryChoice(
                    "Answer the room together",
                    (
                        "Every survivor answers freely. The console accepts the complete response, "
                        "rotates its channels, and leaves one new interval blank."
                    ),
                    "signal_refrain",
                    {"emotional": 5, "social": 7, "sentient": 6},
                    party_effects={"emotional": 3, "social": 4, "sentient": 3},
                    player_spiritual=1,
                ),
                StoryChoice(
                    "Force a violent overload",
                    (
                        "You drive the console past its limit. White noise becomes a physical "
                        "shock; when the static clears, the waveform is still running one place "
                        "to the left of where it began."
                    ),
                    "signal_afterimage",
                    {"physical": -8, "emotional": -5, "cognitive": 2, "social": -6},
                    party_effects={"physical": -5, "emotional": -3, "social": -3},
                    player_spiritual=-1,
                    special="violent_override",
                ),
                StoryChoice(
                    "Refuse the room's final demand",
                    (
                        "You refuse to turn the group into a passcode. The console records the "
                        "refusal as another answer and pairs every living channel with a dim echo."
                    ),
                    "signal_counterpoint",
                    {"emotional": 2, "cognitive": 4, "social": 2, "sentient": 6},
                    party_effects={"sentient": 2},
                    player_spiritual=1,
                ),
            ),
        ),
        "signal_refrain": Scene(
            "Stage 6: The Refrain",
            (
                "The accepted waveform returns one beat later. Four wall nodes replay fragments "
                "while the console leaves a new interval blank."
            ),
            (
                StoryChoice(
                    "Compare the blank interval with the last round",
                    (
                        "The interval fills with the emotional contour of the latest exchange. "
                        "The console marks MATCH, rotates the channels, and produces an afterimage."
                    ),
                    "signal_afterimage",
                    {"cognitive": 4, "sentient": 3},
                    {"cognitive": 2, "sentient": 1},
                    special="echo",
                ),
                StoryChoice(
                    "Let {npc} choose the channel order",
                    (
                        "{npc}'s order locks into place. A thin trace leaves the display and "
                        "begins moving between the table and the four wall nodes."
                    ),
                    "signal_afterimage",
                    {"emotional": 2, "social": 4, "sentient": 2},
                    {"cognitive": 3, "social": 3, "sentient": 2},
                    party_effects={"social": 1},
                ),
                StoryChoice(
                    "Strike the console through the refrain",
                    (
                        "The console throws the blow back as a perfect red waveform. When the "
                        "ringing fades, the blank interval has moved rather than disappeared."
                    ),
                    "signal_counterpoint",
                    {"physical": -6, "emotional": -4, "cognitive": 2, "social": -3},
                    {"emotional": -2, "sentient": 2},
                    party_effects={"emotional": -1},
                    player_spiritual=-1,
                    special="violent_override",
                ),
            ),
        ),
        "signal_afterimage": Scene(
            "Stage 7: The Moving Trace",
            (
                "A narrow trace travels from console to wall node and back. Whenever someone "
                "approaches, it relocates and presents a different timing key."
            ),
            (
                StoryChoice(
                    "Follow the trace without touching it",
                    (
                        "The trace circles the room and returns to the table. Its path resolves "
                        "into a five-part checksum beside the original waveform."
                    ),
                    "signal_counterpoint",
                    {"cognitive": 5, "sentient": 4},
                    {"cognitive": 2, "sentient": 2},
                ),
                StoryChoice(
                    "Interrupt its prediction with silence",
                    (
                        "The silence registers as a valid response. The trace duplicates itself, "
                        "and both copies begin replaying the accepted refrain at different speeds."
                    ),
                    "signal_refrain",
                    {"emotional": 3, "cognitive": 3, "sentient": 4},
                    {"emotional": 2, "sentient": 3},
                    special="echo",
                ),
                StoryChoice(
                    "Pull the console contacts apart",
                    (
                        "The contacts snap loose with a painful spark, but the wall nodes carry "
                        "the pulse without them and label the next sample COUNTERPOINT."
                    ),
                    "signal_counterpoint",
                    {"physical": -4, "emotional": -2, "cognitive": 3},
                    {"emotional": -2, "cognitive": 2},
                ),
            ),
        ),
        "signal_counterpoint": Scene(
            "Stage 8: The Counterpoint",
            (
                "Each living channel now carries a dim partner waveform. The display calls one "
                "ACTION and one ANSWER, then swaps the labels after every completed round."
            ),
            (
                StoryChoice(
                    "Match the shadow channels to remembered voices",
                    (
                        "The console accepts every match, compares them with the latest exchange, "
                        "and requests a fresh sample of the refrain."
                    ),
                    "signal_refrain",
                    {"cognitive": 6, "social": 2, "sentient": 4},
                    {"cognitive": 3, "sentient": 2},
                ),
                StoryChoice(
                    "Ask {npc} which waveform feels true",
                    (
                        "{npc}'s choice becomes the reference. The remaining traces reorganize "
                        "around it, and a moving afterimage marks their differences."
                    ),
                    "signal_afterimage",
                    {"emotional": 3, "social": 4, "sentient": 3},
                    {"emotional": 3, "social": 4, "sentient": 3},
                    party_effects={"social": 1},
                ),
                StoryChoice(
                    "Supply contradictory answers",
                    (
                        "Red and white harmonics braid together. The console marks the conflict "
                        "VALID, clears its labels, and begins the original question again."
                    ),
                    "signal_refrain",
                    {"emotional": -3, "cognitive": 4, "social": -2, "sentient": 3},
                    {"emotional": -2, "sentient": 2},
                    party_effects={"social": -1},
                ),
            ),
        ),
    }


SCENES = build_scenes()


class GameState:
    ACTIONS = (
        "talk",
        "listen",
        "compliment",
        "flirt",
        "antagonize",
        "fight",
        "give",
        "steal",
        "trade",
        "reflect",
        "use",
        "rest",
        "self_eliminate",
    )
    NPC_ACTIONS = (
        "talk",
        "listen",
        "compliment",
        "flirt",
        "antagonize",
        "fight",
        "give",
        "steal",
        "trade",
        "reflect",
        "use",
        "rest",
    )
    TARGET_ACTIONS = {
        "talk",
        "listen",
        "compliment",
        "flirt",
        "antagonize",
        "fight",
        "give",
        "steal",
        "trade",
        "reflect",
    }
    SELF_ACTIONS = {"use", "rest", "self_eliminate"}

    def __init__(
        self,
        seed: Optional[int] = None,
        player_name: str = DEFAULT_PLAYER_NAME,
        paced_rounds: bool = False,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.participants = [random_participant(PLAYER_PROFILE, self.rng)]
        self.participants.extend(random_participant(profile, self.rng) for profile in NPC_PROFILES)
        self.player.display_name = normalize_player_name(player_name)
        self.selected_npc_index = 0
        self.selected_item = "water_liters"
        self.trade_draft: Optional[TradeDraft] = None
        self.scene_id = "sealed_room"
        self.chapter = 1
        self.turn = 0
        self.phase = "awaiting_player"
        self.paced_rounds = bool(paced_rounds)
        self.current_round_player_led = True
        self.acted_this_round: set[int] = set()
        self.last_round_events: List[RoundEvent] = []
        self.round_notices: List[str] = []
        self.round_history: List[RoundRecord] = []
        self.fifth_signal_found = False
        self.last_fight_indices: Tuple[int, ...] = ()
        self.last_fight_time = -999.0
        self.last_exposure_strain = 0
        self.message = (
            f"There is no visible door and nobody remembers entering. Choose {possessive_name(self.player.name)} first "
            "interaction; each living companion will then choose one response of their own."
        )
        self.history: List[str] = []
        self.ending_title = ""
        self.ending_text = ""
        self.last_reaction_index = 0
        self.last_reaction_time = time.monotonic()
        # Render-only idle clocks are anchored independently from self.rng.
        # They may advance for hours without perturbing one simulation choice.
        self.idle_epoch = time.monotonic()
        self.validate()

    @property
    def player(self) -> Participant:
        return self.participants[0]

    @property
    def npcs(self) -> List[Participant]:
        return self.participants[1:]

    @property
    def living_npcs(self) -> List[Participant]:
        return [npc for npc in self.npcs if npc.alive]

    @property
    def living_participants(self) -> List[Participant]:
        return [participant for participant in self.participants if participant.alive]

    @property
    def last_survivor(self) -> Optional[Participant]:
        living = self.living_participants
        return living[0] if len(living) == 1 else None

    @property
    def selected_npc(self) -> Participant:
        return self.npcs[self.selected_npc_index]

    @property
    def story_target_name(self) -> str:
        target = self.selected_npc
        return target.name if target.alive else f"the memory of {target.name}"

    @property
    def current_scene(self) -> Optional[Scene]:
        return SCENES.get(self.scene_id)

    @property
    def is_ending(self) -> bool:
        return self.phase == "ending"

    @property
    def is_game_over(self) -> bool:
        return self.is_ending

    @property
    def round_number(self) -> int:
        return self.turn

    def select_npc(self, index: int) -> None:
        if self.is_ending or self.player.eliminated:
            return
        if self.phase != "awaiting_player":
            self.message = "The current one-second action sequence must finish first."
            return
        index = int(clamp(index, 0, len(self.npcs) - 1))
        npc = self.npcs[index]
        if npc.eliminated:
            self.message = f"{npc.name} has been eliminated and can no longer interact or trade."
            return
        self.selected_npc_index = index
        self.trade_draft = None
        self.message = f"You turn your attention to {npc.name}, the {npc.profile.role}."

    def cycle_item(self) -> None:
        if self.is_ending or self.player.eliminated:
            return
        if self.phase != "awaiting_player":
            self.message = "Inventory selection unlocks after the current action sequence."
            return
        index = ITEM_KEYS.index(self.selected_item)
        self.selected_item = ITEM_KEYS[(index + 1) % len(ITEM_KEYS)]

    def begin_trade(self) -> bool:
        if self.is_ending or self.player.eliminated:
            return False
        if self.phase != "awaiting_player":
            self.message = "Wait until every current action has resolved."
            return False
        target = self.selected_npc
        if target.eliminated:
            self.message = f"{target.name} has been eliminated and cannot trade."
            return False
        offered = ITEM_BY_KEY[self.selected_item]
        offered_index = ITEM_KEYS.index(offered.key)
        requested = ITEM_BY_KEY[ITEM_KEYS[(offered_index + 1) % len(ITEM_KEYS)]]
        self.trade_draft = TradeDraft(
            target_index=self.selected_npc_index,
            offered_item=offered.key,
            offered_quantity=offered.give_amount,
            requested_item=requested.key,
            requested_quantity=requested.give_amount,
        )
        self.message = (
            f"Build a private-stock trade proposal for {target.name}. "
            "Choose both items and type exact quantities."
        )
        return True

    def cancel_trade(self) -> None:
        if self.trade_draft is not None:
            self.trade_draft = None
            self.message = "Trade proposal cancelled. No time passed and no items changed hands."

    def set_trade_item(self, side: str, item_key: str) -> None:
        if self.trade_draft is None:
            raise RuntimeError("No trade proposal is open")
        if item_key not in ITEM_KEYS:
            raise KeyError(f"Unknown trade item: {item_key}")
        if side == "offer":
            self.trade_draft.offered_item = item_key
        elif side == "request":
            self.trade_draft.requested_item = item_key
        else:
            raise ValueError("Trade side must be 'offer' or 'request'")

    def set_trade_quantity(self, side: str, quantity: int) -> None:
        if self.trade_draft is None:
            raise RuntimeError("No trade proposal is open")
        if type(quantity) is not int:
            raise TypeError("Trade quantities must be whole numbers")
        if not 1 <= quantity <= INVENTORY_CAP:
            raise ValueError(f"Trade quantities must be between 1 and {INVENTORY_CAP:,}")
        if side == "offer":
            self.trade_draft.offered_quantity = quantity
        elif side == "request":
            self.trade_draft.requested_quantity = quantity
        else:
            raise ValueError("Trade side must be 'offer' or 'request'")

    def adjust_trade_quantity(self, side: str, amount: int) -> None:
        if self.trade_draft is None:
            raise RuntimeError("No trade proposal is open")
        current = (
            self.trade_draft.offered_quantity
            if side == "offer"
            else self.trade_draft.requested_quantity
        )
        next_quantity = int(clamp(current + int(amount), 1, INVENTORY_CAP))
        self.set_trade_quantity(side, next_quantity)

    def _react(self) -> None:
        self.last_reaction_index = self.selected_npc_index
        self.last_reaction_time = time.monotonic()

    @staticmethod
    def _trade_status_score(participant: Participant) -> float:
        score = 0.0
        for key, weight in TRADE_STATUS_WEIGHTS.items():
            value = float(participant.statuses[key])
            normalized = value if key == "spiritual" else value / 100.0
            score += normalized * weight
        return score

    def _trade_willingness_between(
        self,
        proposer: Participant,
        target: Participant,
        exchange: ExchangePlan,
    ) -> float:
        if proposer.eliminated or target.eliminated:
            return 0.0
        if exchange.offered_item not in ITEM_KEYS or exchange.requested_item not in ITEM_KEYS:
            raise KeyError("Trade exchange contains an unknown item")
        if exchange.offered_quantity <= 0 or exchange.requested_quantity <= 0:
            raise ValueError("Trade exchange quantities must be positive")
        proposer_score = self._trade_status_score(proposer)
        target_score = self._trade_status_score(target)
        status_readiness = proposer_score * 0.40 + target_score * 0.60
        offer_value = TRADE_UNIT_VALUES[exchange.offered_item] * exchange.offered_quantity
        request_value = TRADE_UNIT_VALUES[exchange.requested_item] * exchange.requested_quantity
        fairness_ratio = offer_value / max(1.0, request_value)
        fairness_adjustment = 0.20 * float(clamp(fairness_ratio - 1.0, -1.0, 1.0))
        return float(clamp(0.10 + status_readiness * 0.80 + fairness_adjustment, 0.05, 0.95))

    @staticmethod
    def _flirt_readiness(participant: Participant) -> float:
        physical = participant.statuses["physical"] / 100.0
        emotional = participant.statuses["emotional"] / 100.0
        cognitive = participant.statuses["cognitive"] / 100.0
        social = participant.statuses["social"] / 100.0
        sentient = participant.statuses["sentient"] / 100.0
        spiritual = (participant.statuses["spiritual"] + 1.0) / 2.0
        return (
            0.08 * physical
            + 0.24 * emotional
            + 0.14 * cognitive
            + 0.24 * social
            + 0.20 * sentient
            + 0.10 * spiritual
        )

    def flirt_chance(self, actor: Participant, target: Participant) -> float:
        return float(
            clamp(
                0.08
                + 0.40 * self._flirt_readiness(actor)
                + 0.40 * self._flirt_readiness(target),
                0.08,
                0.88,
            )
        )

    @staticmethod
    def _steal_stealth(participant: Participant) -> float:
        physical = participant.statuses["physical"] / 100.0
        emotional = participant.statuses["emotional"] / 100.0
        cognitive = participant.statuses["cognitive"] / 100.0
        social = participant.statuses["social"] / 100.0
        sentient = participant.statuses["sentient"] / 100.0
        spiritual = (participant.statuses["spiritual"] + 1.0) / 2.0
        return (
            0.22 * physical
            + 0.10 * emotional
            + 0.30 * cognitive
            + 0.10 * social
            + 0.18 * sentient
            + 0.10 * (1.0 - spiritual)
        )

    @staticmethod
    def _steal_guard(participant: Participant) -> float:
        physical = participant.statuses["physical"] / 100.0
        emotional = participant.statuses["emotional"] / 100.0
        cognitive = participant.statuses["cognitive"] / 100.0
        social = participant.statuses["social"] / 100.0
        sentient = participant.statuses["sentient"] / 100.0
        spiritual = (participant.statuses["spiritual"] + 1.0) / 2.0
        return (
            0.12 * physical
            + 0.10 * emotional
            + 0.30 * cognitive
            + 0.14 * social
            + 0.24 * sentient
            + 0.10 * spiritual
        )

    def steal_chance(self, actor: Participant, target: Participant) -> float:
        return float(
            clamp(
                0.50 + 0.62 * (self._steal_stealth(actor) - self._steal_guard(target)),
                0.08,
                0.90,
            )
        )

    def trade_willingness(self, draft: Optional[TradeDraft] = None) -> float:
        draft = draft or self.trade_draft
        if draft is None:
            raise RuntimeError("No trade proposal is open")
        if type(draft.target_index) is not int or not 0 <= draft.target_index < len(self.npcs):
            raise ValueError("Trade target index is invalid")
        target = self.npcs[draft.target_index]
        if target.eliminated or self.player.eliminated:
            return 0.0

        return self._trade_willingness_between(
            self.player,
            target,
            ExchangePlan(
                draft.offered_item,
                draft.offered_quantity,
                draft.requested_item,
                draft.requested_quantity,
            ),
        )

    def trade_local_error(self, draft: Optional[TradeDraft] = None) -> str:
        """Return a proposal error knowable without exposing private NPC stock."""
        draft = draft or self.trade_draft
        if draft is None:
            return "Open the trade builder before confirming a proposal."
        if type(draft.target_index) is not int or not 0 <= draft.target_index < len(self.npcs):
            return "That trade target is no longer available."
        target = self.npcs[draft.target_index]
        if target.eliminated:
            return f"{target.name} has been eliminated and cannot trade."
        if draft.offered_item not in ITEM_KEYS or draft.requested_item not in ITEM_KEYS:
            return "The proposal contains an unknown inventory item."
        quantities = (draft.offered_quantity, draft.requested_quantity)
        if any(type(quantity) is not int for quantity in quantities):
            return "Trade quantities must be whole numbers."
        if any(not 1 <= quantity <= INVENTORY_CAP for quantity in quantities):
            return f"Trade quantities must be between 1 and {INVENTORY_CAP:,}."
        if (
            draft.offered_item == draft.requested_item
            and draft.offered_quantity == draft.requested_quantity
        ):
            return "That proposal returns the exact same items; change one of the quantities."
        if self.player.inventory[draft.offered_item] < draft.offered_quantity:
            offered = ITEM_BY_KEY[draft.offered_item]
            return f"You do not have {draft.offered_quantity:,} {offered.short_label.lower()} to offer."

        player_after = dict(self.player.inventory)
        player_after[draft.offered_item] -= draft.offered_quantity
        player_after[draft.requested_item] += draft.requested_quantity
        if any(not 0 <= amount <= INVENTORY_CAP for amount in player_after.values()):
            return "Your inventory has no capacity for the requested return quantity."
        return ""

    def _resolve_eliminations(self) -> List[Participant]:
        living_before = self.living_participants
        doomed = [
            participant
            for participant in living_before
            if participant.statuses["physical"] <= 0
        ]

        # A fight or room-strain tick can reduce all remaining bodies to zero at
        # once. The room always retains exactly one signal: the strongest
        # remaining mind wins a stable, index-tied survival check at 1 PHYS.
        if doomed and len(doomed) == len(living_before):
            survivor = max(
                doomed,
                key=lambda participant: (
                    participant.statuses["cognitive"]
                    + participant.statuses["sentient"]
                    + participant.statuses["emotional"]
                    + participant.statuses["social"]
                    + participant.statuses["spiritual"] * 10,
                    participant.statuses["cognitive"],
                    -self._participant_index(participant),
                ),
            )
            survivor.statuses["physical"] = 1
            doomed.remove(survivor)

        newly_eliminated: List[Participant] = []
        for participant in doomed:
            participant.statuses["physical"] = 0
            participant.eliminated = True
            participant.elimination_turn = self.turn
            newly_eliminated.append(participant)

        if self.selected_npc.eliminated:
            for index, npc in enumerate(self.npcs):
                if npc.alive:
                    self.selected_npc_index = index
                    break
        if self.trade_draft is not None and self.npcs[self.trade_draft.target_index].eliminated:
            self.trade_draft = None
        if self.player.eliminated:
            self.trade_draft = None

        if len(self.living_participants) == 1:
            self._finish_last_survivor()
        elif self.player.eliminated:
            self.phase = "autonomous"
            self.message = (
                f"{self.player.name} has been eliminated. The surviving companions continue their "
                "interactions without player control."
            )
        return newly_eliminated

    @staticmethod
    def _elimination_notice(participants: Iterable[Participant]) -> str:
        names = [participant.name for participant in participants if participant.profile is not PLAYER_PROFILE]
        if not names:
            return ""
        if len(names) == 1:
            subject = names[0]
        else:
            subject = ", ".join(names[:-1]) + f" and {names[-1]}"
        verb = "has" if len(names) == 1 else "have"
        return f" {subject} {verb} been eliminated after Physical Health reached zero."

    def _apply_effects(self, participant: Participant, effects: Mapping[str, int]) -> None:
        for key, amount in effects.items():
            if key not in STATUS_KEYS:
                raise KeyError(f"Unknown status effect: {key}")
            participant.adjust(key, int(amount))

    def _change_snapshot(self) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
        return tuple(
            (
                tuple(participant.statuses[key] for key in STATUS_KEYS),
                tuple(participant.inventory[key] for key in ITEM_KEYS),
            )
            for participant in self.participants
        )

    def _describe_changes(
        self,
        before: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...],
    ) -> str:
        """Describe observable deltas without exposing anyone's remaining stock."""
        participant_changes: List[str] = []
        for index, participant in enumerate(self.participants):
            old_statuses, old_inventory = before[index]
            details: List[str] = []
            for status_index, key in enumerate(STATUS_KEYS):
                old_value = old_statuses[status_index]
                new_value = participant.statuses[key]
                if old_value == new_value:
                    continue
                if key == "spiritual":
                    details.append(f"SP {old_value}->{new_value}")
                else:
                    details.append(f"{STATUS_SHORT[key]} {new_value - old_value:+d}")
            for item_index, key in enumerate(ITEM_KEYS):
                delta = participant.inventory[key] - old_inventory[item_index]
                if delta:
                    details.append(f"{ITEM_BY_KEY[key].short_label} {delta:+d}")
            if participant.eliminated and old_statuses[0] > 0:
                details.append("ELIMINATED")
            if details:
                participant_changes.append(f"{participant.name}: {', '.join(details)}")
        return " | ".join(participant_changes) or "No measurable status or inventory change."

    def _advance_time(self) -> None:
        self.turn += 1

    def _apply_end_of_round_exposure(self) -> List[Participant]:
        strain = 1 if self.turn % 4 == 0 else 0
        if self.fifth_signal_found and self.turn % 2 == 0:
            strain += 1
        if self.player.eliminated:
            strain += 2
        self.last_exposure_strain = strain
        if strain:
            for participant in self.participants:
                if participant.eliminated:
                    continue
                participant.adjust("physical", -strain)
                if participant.statuses["emotional"] < 25:
                    participant.adjust("cognitive", -1)
        return self._resolve_eliminations()

    def _can_pay(self, costs: Mapping[str, int]) -> bool:
        return all(self.player.inventory[key] >= amount for key, amount in costs.items())

    def _cost_description(self, costs: Mapping[str, int]) -> str:
        parts = []
        for key, amount in costs.items():
            parts.append(f"{amount} {ITEM_BY_KEY[key].short_label.lower()}")
        return ", ".join(parts)

    def _project_item_effects(
        self,
        participant: Participant,
        item_key: str,
    ) -> Tuple[Dict[str, int], str]:
        """Forecast one item's exact status effect without mutating simulation state."""
        if item_key not in ITEM_BY_KEY:
            raise KeyError(f"Unknown item effect: {item_key}")
        effects = dict(ITEM_STATUS_EFFECTS[item_key])
        note = ""
        if item_key == "caffeine_pills":
            repeated = self.turn - participant.last_caffeine_turn <= 3
            if repeated:
                effects["physical"] -= 4
                effects["emotional"] -= 3
                effects["cognitive"] -= 2
                effects["social"] -= 1
                note = " Repeated caffeine use brings jitters and a sharper physical cost."
        elif item_key == "acetaminophen_pills":
            repeated = self.turn - participant.last_acetaminophen_turn <= 4
            if repeated:
                effects = {
                    "physical": -8,
                    "emotional": -2,
                    "cognitive": -3,
                    "social": 0,
                    "sentient": -1,
                    "spiritual": 0,
                }
                note = " Repeated medicine use causes a serious health penalty in this abstract model."
        return effects, note

    @staticmethod
    def _realized_effects(
        participant: Participant,
        effects: Mapping[str, int],
    ) -> Dict[str, int]:
        """Return clamped deltas exactly as Participant.adjust would realize them."""
        realized: Dict[str, int] = {}
        for key in STATUS_KEYS:
            current = participant.statuses[key]
            amount = int(effects.get(key, 0))
            if key == "spiritual":
                future = int(clamp(current + amount, -1, 1))
            else:
                low = 0 if key == "physical" else 1
                future = int(clamp(current + amount, low, 100))
            realized[key] = future - current
        return realized

    def _effect_value(
        self,
        participant: Participant,
        effects: Mapping[str, int],
    ) -> float:
        """Score a projected effect against all six of the recipient's needs."""
        realized = self._realized_effects(participant, effects)
        value = 0.0
        for key in STATUS_KEYS:
            current = participant.statuses[key]
            delta = realized[key]
            importance = NPC_STATUS_IMPORTANCE[key]
            if key == "spiritual":
                normalized = (current + 1.0) / 2.0
                scale = 0.80
            else:
                normalized = current / 100.0
                scale = 0.10
            if delta >= 0:
                need_multiplier = 0.35 + 1.65 * (1.0 - normalized)
            else:
                need_multiplier = 1.00 + 1.75 * (1.0 - normalized)
            value += delta * scale * importance * need_multiplier
        return value

    @staticmethod
    def _item_security(participant: Participant, item_key: str) -> float:
        amount = max(0.0, float(participant.inventory[item_key]))
        reserve = NPC_ITEM_RESERVES[item_key]
        return amount / (amount + reserve) if amount else 0.0

    def _resource_security(self, participant: Participant) -> float:
        # Every one of the five private quantities contributes to every plan.
        return sum(self._item_security(participant, key) for key in ITEM_KEYS) / len(ITEM_KEYS)

    def _apply_item_effect(self, participant: Participant, item_key: str) -> str:
        effects, note = self._project_item_effects(participant, item_key)
        if item_key == "caffeine_pills":
            participant.last_caffeine_turn = self.turn
        elif item_key == "acetaminophen_pills":
            participant.last_acetaminophen_turn = self.turn
        self._apply_effects(participant, effects)
        return note

    def _participant_index(self, participant: Participant) -> int:
        for index, candidate in enumerate(self.participants):
            if candidate is participant:
                return index
        raise ValueError("Participant is not part of this room")

    def _begin_round(self) -> bool:
        if self.phase != "awaiting_player" or self.is_ending:
            return False
        self.phase = "resolving_npcs"
        self.acted_this_round.clear()
        self.last_round_events = []
        self.round_notices = []
        self.current_round_player_led = True
        self._advance_time()
        return True

    def _record_event(
        self,
        actor_index: int,
        action: str,
        target_index: Optional[int],
        summary: str,
        success: bool,
        eliminated: Iterable[Participant] = (),
        gesture_key: Optional[str] = None,
        item_key: Optional[str] = None,
        reasoning: str = "",
        decision_score: Optional[float] = None,
        exchange: Optional[ExchangePlan] = None,
        impact: str = "",
    ) -> RoundEvent:
        if actor_index in self.acted_this_round:
            raise AssertionError("A participant attempted to act twice in one round")
        eliminated_indices = tuple(self._participant_index(person) for person in eliminated)
        voice_participant_index = (
            actor_index
            if actor_index > 0
            else (target_index if target_index is not None and target_index > 0 else None)
        )
        voice_statuses = (
            tuple(
                self.participants[voice_participant_index].statuses[key]
                for key in STATUS_KEYS
            )
            if voice_participant_index is not None
            else ()
        )
        event = RoundEvent(
            round_number=self.turn,
            actor_index=actor_index,
            action=action,
            target_index=target_index,
            summary=summary,
            success=success,
            eliminated_indices=eliminated_indices,
            gesture_key=gesture_key,
            voice_statuses=voice_statuses,
            item_key=item_key,
            reasoning=reasoning,
            decision_score=decision_score,
            exchange=exchange,
            impact=impact,
        )
        self.acted_this_round.add(actor_index)
        self.last_round_events.append(event)
        return event

    def _archive_current_round(self) -> None:
        """Freeze the completed round once, including terminal partial rounds."""
        if not self.last_round_events:
            raise AssertionError("A completed round must contain at least one action")
        if self.round_history and self.round_history[-1].round_number >= self.turn:
            raise AssertionError("A round was archived twice or out of order")
        self.round_history.append(
            RoundRecord(
                round_number=self.turn,
                events=tuple(self.last_round_events),
                notices=tuple(self.round_notices),
            )
        )

    @staticmethod
    def _combat_score(participant: Participant) -> float:
        return (
            participant.statuses["physical"] * 0.30
            + participant.statuses["emotional"] * 0.15
            + participant.statuses["cognitive"] * 0.20
            + participant.statuses["social"] * 0.10
            + participant.statuses["sentient"] * 0.20
            + participant.statuses["spiritual"] * 5.0
        )

    def _resolve_fight(self, attacker: Participant, defender: Participant) -> Tuple[str, bool]:
        attack_score = self._combat_score(attacker)
        guard_score = self._combat_score(defender)
        hit_chance = float(clamp(0.50 + (attack_score - guard_score) / 180.0, 0.15, 0.90))
        hit = self.rng.random() <= hit_chance
        if hit:
            defender_damage = self.rng.randint(8, 18) + int(clamp((attack_score - guard_score) / 25.0, 0, 7))
            attacker_damage = self.rng.randint(1, 5)
        else:
            attacker_damage = self.rng.randint(4, 10)
            defender_damage = self.rng.randint(1, 3)

        attacker.adjust("physical", -attacker_damage)
        defender.adjust("physical", -defender_damage)
        self._apply_effects(attacker, {"emotional": -3, "social": -5, "sentient": -1})
        self._apply_effects(defender, {"emotional": -3, "social": -5, "sentient": -1})
        attacker.adjust("spiritual", -1)
        defender.adjust("spiritual", -1)
        if hit:
            attacker.adjust("cognitive", 1)
            defender.adjust("cognitive", -1)
        else:
            attacker.adjust("cognitive", -1)

        self.last_fight_indices = (
            self._participant_index(attacker),
            self._participant_index(defender),
        )
        self.last_fight_time = time.monotonic()
        if hit:
            return (
                f"fights {defender.name}: them -{defender_damage}, self -{attacker_damage} PHYS",
                True,
            )
        return (
            f"fights {defender.name}: repelled, self -{attacker_damage}, them -{defender_damage} PHYS",
            False,
        )

    def _resolve_flirt(
        self,
        actor: Participant,
        target: Participant,
        roll: Optional[float] = None,
    ) -> Tuple[str, bool]:
        if roll is not None and not 0.0 <= float(roll) <= 1.0:
            raise ValueError("A supplied flirt roll must be between zero and one")
        chance = self.flirt_chance(actor, target)
        roll_value = self.rng.random() if roll is None else float(roll)
        if roll_value <= chance:
            self._apply_effects(
                actor,
                {"emotional": 3, "cognitive": 1, "social": 3, "sentient": 2},
            )
            self._apply_effects(
                target,
                {"emotional": 4, "cognitive": 1, "social": 4, "sentient": 2},
            )
            actor.adjust("spiritual", 1)
            target.adjust("spiritual", 1)
            return (
                f"offers {target.name} a playful, low-pressure flirt; {target.name} warmly reciprocates",
                True,
            )
        self._apply_effects(actor, {"emotional": -2, "cognitive": 1, "social": -1, "sentient": 1})
        self._apply_effects(target, {"social": 1, "sentient": 1})
        return (
            f"flirts with {target.name}; {target.name} gently declines and {actor.name} accepts the boundary",
            False,
        )

    def _resolve_steal(
        self,
        actor: Participant,
        target: Participant,
        item_key: str,
        roll: Optional[float] = None,
    ) -> Tuple[str, bool]:
        if item_key not in ITEM_BY_KEY:
            raise KeyError(f"Unknown stolen item: {item_key}")
        if roll is not None and not 0.0 <= float(roll) <= 1.0:
            raise ValueError("A supplied steal roll must be between zero and one")
        spec = ITEM_BY_KEY[item_key]
        quantity = spec.give_amount
        if target.inventory[item_key] < quantity:
            self._apply_effects(actor, {"emotional": -2, "cognitive": -1, "social": -3})
            actor.adjust("spiritual", -1)
            return (
                f"searches {target.name}'s accessible supplies for {spec.short_label.lower()}, but finds no exposed unit",
                False,
            )
        if actor.inventory[item_key] > INVENTORY_CAP - quantity:
            return f"cannot conceal more {spec.short_label.lower()}; carrying capacity is full", False
        chance = self.steal_chance(actor, target)
        roll_value = self.rng.random() if roll is None else float(roll)
        if roll_value <= chance:
            target.change_item(item_key, -quantity)
            actor.change_item(item_key, quantity)
            self._apply_effects(
                actor,
                {"emotional": 1, "cognitive": 2, "social": -5, "sentient": -2},
            )
            self._apply_effects(
                target,
                {"emotional": -4, "cognitive": -1, "social": -6, "sentient": -2},
            )
            actor.adjust("spiritual", -1)
            target.adjust("spiritual", -1)
            return (
                f"steals {quantity} {spec.short_label.lower()} from {target.name} without being stopped",
                True,
            )
        self._apply_effects(
            actor,
            {"emotional": -3, "cognitive": -2, "social": -5, "sentient": -1},
        )
        self._apply_effects(
            target,
            {"emotional": -2, "cognitive": 1, "social": -4, "sentient": 1},
        )
        actor.adjust("spiritual", -1)
        return f"tries to steal {spec.short_label.lower()} from {target.name}, but is caught", False

    def _resolve_npc_trade(
        self,
        actor: Participant,
        target: Participant,
        exchange: ExchangePlan,
        roll: Optional[float] = None,
    ) -> Tuple[str, bool]:
        if roll is not None and not 0.0 <= float(roll) <= 1.0:
            raise ValueError("A supplied trade roll must be between zero and one")
        offered = ITEM_BY_KEY[exchange.offered_item]
        requested = ITEM_BY_KEY[exchange.requested_item]
        if actor.inventory[exchange.offered_item] < exchange.offered_quantity:
            return "withdraws a trade after reassessing private stock", False
        if target.inventory[exchange.requested_item] < exchange.requested_quantity:
            return f"proposes a trade to {target.name}, but no accessible requested bundle is available", False
        actor_after = dict(actor.inventory)
        target_after = dict(target.inventory)
        actor_after[exchange.offered_item] -= exchange.offered_quantity
        target_after[exchange.offered_item] += exchange.offered_quantity
        target_after[exchange.requested_item] -= exchange.requested_quantity
        actor_after[exchange.requested_item] += exchange.requested_quantity
        if any(not 0 <= amount <= INVENTORY_CAP for amount in actor_after.values()):
            return f"withdraws the proposal because {actor.name} cannot carry the requested bundle", False
        if any(not 0 <= amount <= INVENTORY_CAP for amount in target_after.values()):
            return f"withdraws the proposal because {target.name} cannot carry the offered bundle", False
        chance = self._trade_willingness_between(actor, target, exchange)
        roll_value = self.rng.random() if roll is None else float(roll)
        if roll_value > chance:
            actor.adjust("emotional", -1)
            actor.adjust("social", -2)
            target.adjust("social", -1)
            return f"offers {target.name} a private-stock trade; the proposal is declined", False
        actor.inventory = actor_after
        target.inventory = target_after
        self._apply_effects(actor, {"cognitive": 2, "social": 3, "sentient": 1})
        self._apply_effects(target, {"emotional": 1, "social": 2})
        return (
            f"trades {exchange.offered_quantity} {offered.short_label.lower()} with {target.name} "
            f"for {exchange.requested_quantity} {requested.short_label.lower()}",
            True,
        )

    def _resolve_actor_action(
        self,
        actor: Participant,
        target: Participant,
        action: str,
        item_key: Optional[str] = None,
        exchange: Optional[ExchangePlan] = None,
    ) -> Tuple[str, bool]:
        if actor.eliminated or target.eliminated:
            return "cannot act because one participant has been eliminated", False

        if action == "talk":
            self._apply_effects(actor, {"emotional": 1, "cognitive": 1, "social": 2, "sentient": 1})
            self._apply_effects(target, {"emotional": 1, "cognitive": 1, "social": 2})
            return f"talks with {target.name}", True

        if action == "listen":
            self._apply_effects(actor, {"emotional": 1, "cognitive": 2, "social": 1, "sentient": 2})
            self._apply_effects(target, {"emotional": 4, "social": 3, "sentient": 2})
            if actor.statuses["sentient"] >= 72 and target.statuses["sentient"] >= 72:
                actor.set_spiritual(1)
            return f"listens closely to {target.name}", True

        if action == "compliment":
            self._apply_effects(actor, {"emotional": 1, "social": 2, "sentient": 1})
            self._apply_effects(target, {"emotional": 6, "social": 4, "sentient": 2})
            actor.adjust("spiritual", 1)
            target.adjust("spiritual", 1)
            return f"compliments {target.name}, raising trust", True

        if action == "flirt":
            return self._resolve_flirt(actor, target)

        if action == "antagonize":
            self._apply_effects(actor, {"emotional": -1, "social": -3, "sentient": -1})
            self._apply_effects(
                target,
                {"emotional": -6, "cognitive": -2, "social": -5, "sentient": -1},
            )
            actor.adjust("spiritual", -1)
            target.adjust("spiritual", -1)
            return f"antagonizes {target.name}, deepening hostility", True

        if action == "reflect":
            self._apply_effects(actor, {"emotional": 3, "cognitive": 1, "social": 2, "sentient": 5})
            self._apply_effects(target, {"emotional": 3, "cognitive": 1, "social": 3, "sentient": 5})
            if actor.statuses["sentient"] + target.statuses["sentient"] >= 130:
                actor.set_spiritual(1)
                target.set_spiritual(1)
            return f"reflects with {target.name} on what the room is measuring", True

        if action == "fight":
            return self._resolve_fight(actor, target)

        if action == "give":
            key = item_key or self.selected_item
            spec = ITEM_BY_KEY[key]
            if actor.inventory[key] < spec.give_amount:
                actor.adjust("emotional", -1)
                return f"cannot give {spec.give_amount} {spec.short_label.lower()} to {target.name}", False
            if target.inventory[key] > INVENTORY_CAP - spec.give_amount:
                return f"finds that {target.name} cannot carry more {spec.short_label.lower()}", False
            actor.change_item(key, -spec.give_amount)
            target.change_item(key, spec.give_amount)
            self._apply_effects(actor, {"emotional": 1, "social": 3, "sentient": 2})
            target.adjust("social", 2)
            # Giving transfers the supply; it does not duplicate its use effect.
            # The recipient may consume it in a later one-action slot, which
            # keeps item quantities meaningful and prevents infinite healing by
            # shuttling the same ration back and forth.
            return f"gives {target.name} {spec.give_amount} {spec.short_label.lower()}", True

        if action == "steal":
            key = item_key or self.selected_item
            return self._resolve_steal(actor, target, key)

        if action == "trade":
            if exchange is None:
                raise ValueError("NPC trade actions require a frozen exchange plan")
            return self._resolve_npc_trade(actor, target, exchange)

        if action == "use":
            key = item_key or self.selected_item
            spec = ITEM_BY_KEY[key]
            if actor.inventory[key] < spec.use_amount:
                actor.adjust("emotional", -1)
                return f"cannot use {spec.short_label.lower()}; inventory is too low", False
            actor.change_item(key, -spec.use_amount)
            note = self._apply_item_effect(actor, key)
            if key == "dollars":
                description = "uses five dollars as conductive strips on the table"
            elif key == "water_liters":
                description = "drinks one measured liter of water"
            elif key == "food_pounds":
                description = "eats one pound of food"
            elif key == "caffeine_pills":
                description = "uses one caffeine pill"
            else:
                description = "uses one acetaminophen pill"
            if note:
                description += "; repeated use causes a penalty"
            return description, True

        if action == "rest":
            ready = actor.inventory["water_liters"] >= 1 and actor.inventory["food_pounds"] >= 1
            if ready:
                actor.change_item("water_liters", -1)
                actor.change_item("food_pounds", -1)
                self._apply_effects(actor, REST_STATUS_EFFECTS)
                return "rests against the wall using one water and one food", True
            self._apply_effects(actor, {"physical": -2, "emotional": -2, "cognitive": -1})
            return "tries to rest without both water and food", False

        if action == "self_eliminate":
            if actor is not self.player:
                raise ValueError("Self-elimination is a player-only action")
            actor.statuses["physical"] = 0
            return "chooses self-elimination, reducing their PHYS to zero", True

        raise ValueError(f"Unsupported resolved action: {action}")

    def _execute_actor_slot(
        self,
        actor_index: int,
        action: str,
        target_index: int,
        item_key: Optional[str] = None,
        reasoning: str = "",
        decision_score: Optional[float] = None,
        exchange: Optional[ExchangePlan] = None,
    ) -> RoundEvent:
        actor = self.participants[actor_index]
        target = self.participants[target_index]
        if actor_index in self.acted_this_round:
            raise AssertionError("Actor already used this round's interaction")
        if actor.eliminated:
            raise ValueError("Eliminated participant cannot act")
        if target.eliminated:
            raise ValueError("Eliminated participant cannot be targeted")
        if action == "self_eliminate" and actor_index != 0:
            raise ValueError("Self-elimination is a player-only action")
        if action in self.SELF_ACTIONS and actor_index != target_index:
            raise ValueError("Self-care actions must target the actor")
        if action not in self.SELF_ACTIONS and actor_index == target_index:
            raise ValueError("A targeted interaction cannot target its actor")
        effective_item_key = item_key
        if action in ("give", "use", "steal"):
            effective_item_key = item_key or self.selected_item
            if effective_item_key not in ITEM_KEYS:
                raise KeyError(f"Unknown action item: {effective_item_key}")
        elif item_key is not None:
            raise ValueError(f"Action {action!r} cannot carry an item choice")
        if action == "trade":
            if exchange is None:
                raise ValueError("Trade actions require an exchange plan")
        elif exchange is not None:
            raise ValueError(f"Action {action!r} cannot carry an exchange plan")
        if decision_score is not None and not math.isfinite(decision_score):
            raise ValueError("NPC decision utility must be finite")
        before_changes = self._change_snapshot()

        # NPCs choose a physical aside for every action.  Player actions remain
        # gesture-free so the game never invents body language for the player.
        gesture_key: Optional[str] = None
        if actor_index > 0:
            gesture_key = self._choose_npc_gesture(actor, action, effective_item_key)
            actor.gesture_key = gesture_key
            actor.gesture_turn = self.turn
            actor.gesture_changed_at = time.monotonic()

        summary, success = self._resolve_actor_action(
            actor,
            target,
            action,
            effective_item_key,
            exchange,
        )
        newly_eliminated = self._resolve_eliminations()
        if newly_eliminated:
            names = ", ".join(person.name for person in newly_eliminated)
            summary += f"; {names} eliminated"
        impact = self._describe_changes(before_changes)
        return self._record_event(
            actor_index=actor_index,
            action=action,
            target_index=target_index,
            summary=summary,
            success=success,
            eliminated=newly_eliminated,
            gesture_key=gesture_key,
            item_key=effective_item_key,
            reasoning=reasoning,
            decision_score=decision_score,
            exchange=exchange,
            impact=impact,
        )

    def _npc_survival_pressure(self) -> float:
        living = len(self.living_participants)
        pressure = (5 - living) * 0.35 + min(8.0, max(0, self.turn - 8) * 0.10)
        if self.player.eliminated:
            pressure += 1.75 + min(8.0, self.turn * 0.12) + max(0, 4 - living) * 0.30
        if living <= 2:
            pressure += 1.75
        return pressure

    def _npc_action_weights(self, npc: Participant) -> Dict[str, float]:
        """Expose all twelve six-index action priors used by the deeper planner."""
        physical = npc.statuses["physical"] / 100.0
        emotional = npc.statuses["emotional"] / 100.0
        cognitive = npc.statuses["cognitive"] / 100.0
        social = npc.statuses["social"] / 100.0
        sentient = npc.statuses["sentient"] / 100.0
        spiritual_raw = float(npc.statuses["spiritual"])
        light = max(0.0, spiritual_raw)
        shadow = max(0.0, -spiritual_raw)
        neutral = 1.0 - abs(spiritual_raw)
        fatigue = 1.0 - physical
        distress = 1.0 - emotional
        confusion = 1.0 - cognitive
        isolation = 1.0 - social
        detachment = 1.0 - sentient
        security = self._resource_security(npc)
        scarcity = 1.0 - security
        survival_pressure = self._npc_survival_pressure()
        weights = {
            "talk": 0.75 + 0.40 * physical + 0.35 * emotional + 0.70 * cognitive
            + 0.35 * social + 0.35 * sentient + 0.15 * neutral + 0.25 * security,
            "listen": 0.65 + 0.15 * physical + 0.25 * emotional + 0.35 * cognitive
            + 0.55 * social + 1.00 * sentient + 0.25 * light + 0.25 * security,
            "compliment": 0.45 + 0.15 * physical + 0.80 * emotional + 0.20 * cognitive
            + 0.90 * social + 0.75 * sentient + 0.60 * light + 0.25 * security,
            "flirt": 0.55 + 0.10 * physical + 0.78 * emotional + 0.32 * cognitive
            + 0.90 * social + 0.68 * sentient + 0.55 * light + 0.12 * neutral
            + 0.18 * security - 0.28 * shadow,
            "antagonize": 0.18 + 0.30 * physical + 1.10 * distress + 0.30 * confusion
            + 0.90 * isolation + 0.60 * detachment + 0.90 * shadow + 0.45 * scarcity,
            "reflect": 0.40 + 0.10 * physical + 0.35 * emotional + 0.75 * cognitive
            + 0.25 * social + 1.00 * sentient + 0.80 * light + 0.15 * neutral
            + 0.25 * security,
            "fight": (
                0.08
                + (0.60 * physical + 0.30 * cognitive)
                * (0.70 + distress + isolation + shadow + survival_pressure)
                + 0.25 * confusion
                + 0.20 * detachment
                - 0.50 * sentient
                - 0.25 * light
                + 0.45 * scarcity
                + 0.24 * survival_pressure
            ),
            "give": 0.28 + 0.15 * physical + 0.55 * emotional + 0.25 * cognitive
            + 0.70 * social + 0.90 * sentient + 0.60 * light + 0.20 * security,
            "steal": 0.12 + 0.38 * physical + 0.16 * distress + 0.52 * cognitive
            + 0.50 * isolation + 0.22 * sentient + 0.72 * shadow + 0.72 * scarcity
            + 0.16 * survival_pressure - 0.20 * light,
            "trade": 0.22 + 0.12 * physical + 0.38 * emotional + 0.82 * cognitive
            + 0.68 * social + 0.46 * sentient + 0.22 * light + 0.12 * neutral
            + 0.38 * security,
            "use": 0.20 + 0.85 * fatigue + 0.30 * distress + 0.25 * confusion
            + 0.12 * isolation + 0.12 * detachment + 0.12 * shadow + 0.20 * scarcity,
            "rest": 0.25 + 1.20 * fatigue + 0.45 * distress + 0.35 * confusion
            + 0.10 * isolation + 0.10 * detachment + 0.12 * neutral + 0.18 * security,
        }
        return {action: max(0.03, utility) for action, utility in weights.items()}

    def _npc_gesture_weights(self, npc: Participant, action: str) -> Dict[str, float]:
        """Return expressive pose weights driven by action and all six indexes."""
        if action not in self.NPC_ACTIONS:
            raise ValueError(f"NPC gesture requested for unsupported action: {action}")

        # Percentage indexes use a 0..1 intensity. Spirituality is ternary, so
        # -1, 0, +1 become 0, .5, 1 while retaining a neutral-specific term.
        physical = npc.statuses["physical"] / 100.0
        emotional = npc.statuses["emotional"] / 100.0
        cognitive = npc.statuses["cognitive"] / 100.0
        social = npc.statuses["social"] / 100.0
        sentient = npc.statuses["sentient"] / 100.0
        spiritual_raw = npc.statuses["spiritual"]
        spiritual = (spiritual_raw + 1.0) / 2.0
        spiritual_neutral = 1.0 - abs(float(spiritual_raw))

        weights = {key: 0.16 for key in NPC_GESTURE_KEYS}

        # Physical: vigor favors large, committed poses; fatigue folds inward.
        weights["sprint_pose"] += 2.10 * physical
        weights["fist_clench"] += 1.05 * physical
        weights["self_hug"] += 1.85 * (1.0 - physical)
        weights["guarded_cross"] += 0.75 * (1.0 - physical)

        # Emotional: buoyancy opens affectionate/dance choices; distress reads
        # as a guarded reaction or a theatrical turn away.
        weights["heart_hands"] += 2.00 * emotional
        weights["retro_chacha"] += 1.05 * emotional
        weights["dramatic_turn"] += 1.55 * (1.0 - emotional)
        weights["self_hug"] += 0.85 * (1.0 - emotional)

        # Cognitive: clarity favors precise/head-led gestures, while confusion
        # admits shrug-like rhythm and playful uncertainty.
        weights["temple_tap"] += 2.05 * cognitive
        weights["play_them_off"] += 1.10 * cognitive
        weights["funky_ehh"] += 1.35 * (1.0 - cognitive)
        weights["seesaw_67"] += 0.80 * (1.0 - cognitive)

        # Social: connection opens arms toward others; isolation closes them.
        weights["open_palms"] += 1.95 * social
        weights["pixel_wave"] += 1.00 * social
        weights["guarded_cross"] += 1.45 * (1.0 - social)
        weights["dramatic_turn"] += 0.65 * (1.0 - social)

        # Sentient: heightened awareness becomes inward/contemplative; a low
        # reading favors kinetic deflection and the intentionally awkward groove.
        weights["prayer_pose"] += 2.15 * sentient
        weights["affirmation"] += 0.85 * sentient
        weights["sprint_pose"] += 0.90 * (1.0 - sentient)
        weights["funky_ehh"] += 0.85 * (1.0 - sentient)

        # Spiritual: +1 favors alignment, -1 favors combative/dismissive poses,
        # and exactly zero adds a deliberately unresolved back-and-forth beat.
        weights["affirmation"] += 1.95 * spiritual
        weights["prayer_pose"] += 1.15 * spiritual
        weights["fist_clench"] += 1.35 * (1.0 - spiritual)
        weights["dramatic_turn"] += 0.80 * (1.0 - spiritual)
        weights["seesaw_67"] += 1.15 * spiritual_neutral
        weights["play_them_off"] += 0.65 * spiritual_neutral

        # The Full Final Release emotes combine all six public indexes into readable
        # non-verbal decisions. Positive terms keep every weight valid for
        # random.choices while still allowing action context to lead.
        emotional_extreme = abs(emotional - 0.5) * 2.0
        weights["spitting"] += (
            0.55 * physical
            + 1.55 * (1.0 - emotional)
            + 0.50 * (1.0 - cognitive)
            + 1.85 * (1.0 - social)
            + 0.85 * (1.0 - sentient)
            + 1.35 * (1.0 - spiritual)
        )
        weights["yawning"] += (
            2.20 * (1.0 - physical)
            + 0.45 * (1.0 - emotional)
            + 1.35 * (1.0 - cognitive)
            + 0.30 * (1.0 - social)
            + 0.90 * (1.0 - sentient)
            + 0.30 * spiritual_neutral
        )
        weights["chomping"] += (
            1.10 * physical
            + 0.35 * (1.0 - emotional)
            + 0.55 * (1.0 - cognitive)
            + 0.20 * (1.0 - social)
            + 0.45 * (1.0 - sentient)
            + 0.35 * (1.0 - spiritual)
        )
        weights["bowing"] += (
            0.45 * physical
            + 1.10 * emotional
            + 0.70 * cognitive
            + 1.85 * social
            + 1.45 * sentient
            + 1.50 * spiritual
            + 0.35 * spiritual_neutral
        )
        weights["head_banging"] += (
            1.75 * physical
            + 0.75 * emotional_extreme
            + 1.10 * (1.0 - cognitive)
            + 0.55 * (1.0 - social)
            + 0.70 * (1.0 - sentient)
            + 0.95 * (1.0 - spiritual)
        )

        # Context is intentionally strong but not exclusive.  The ordered bias
        # makes the first pose the action's clearest read, with room for status-
        # driven alternatives and occasional personality-rich surprises.
        for rank, key in enumerate(ACTION_GESTURE_CONTEXT[action]):
            weights[key] += 3.40 - rank * 0.45
        return weights

    def _choose_npc_gesture(
        self,
        npc: Participant,
        action: str,
        item_key: Optional[str] = None,
    ) -> str:
        weights = self._npc_gesture_weights(npc, action)
        if action == "rest":
            weights["yawning"] += 7.0
        elif action == "give":
            weights["bowing"] += 4.5
        elif action == "use" and item_key is not None:
            item_gesture = NPC_ITEM_GESTURE_CONTEXT[item_key]
            weights[item_gesture] += 8.0
        return self.rng.choices(
            list(NPC_GESTURE_KEYS),
            weights=[weights[key] for key in NPC_GESTURE_KEYS],
            k=1,
        )[0]

    def _last_event_for_actor(self, actor_index: int) -> Optional[RoundEvent]:
        for record in reversed(self.round_history):
            for event in record.events:
                if event.actor_index == actor_index:
                    return event
        return None

    def _npc_target_modifier(
        self,
        npc: Participant,
        action: str,
        target: Participant,
    ) -> float:
        """Score whom an action serves or threatens using visible target state."""
        physical = target.statuses["physical"] / 100.0
        emotional = target.statuses["emotional"] / 100.0
        cognitive = target.statuses["cognitive"] / 100.0
        social = target.statuses["social"] / 100.0
        sentient = target.statuses["sentient"] / 100.0
        shadow = max(0.0, -float(target.statuses["spiritual"]))
        distress = 1.0 - emotional
        confusion = 1.0 - cognitive
        isolation = 1.0 - social
        fading = 1.0 - sentient
        player_salience = 0.12 if target is self.player else 0.0
        if action == "talk":
            return 0.20 * distress + 0.30 * confusion + 0.20 * isolation + player_salience
        if action == "listen":
            return 0.85 * distress + 0.35 * confusion + 0.65 * isolation + 0.25 * fading
        if action == "compliment":
            return 0.75 * distress + 0.85 * isolation + 0.22 * fading + 0.28 * shadow
        if action == "flirt":
            receptivity = (
                0.08 * physical
                + 0.25 * emotional
                + 0.12 * cognitive
                + 0.26 * social
                + 0.19 * sentient
                + 0.10 * max(0.0, float(target.statuses["spiritual"]))
            )
            return 0.60 * receptivity + 0.10 * distress + player_salience
        if action == "antagonize":
            threat = 0.45 * physical + 0.35 * cognitive + 0.20 * social
            return 0.42 * threat + 0.25 * shadow
        if action == "reflect":
            return 0.30 * distress + 0.55 * confusion + 0.25 * isolation + 0.55 * fading + player_salience
        if action == "fight":
            threat = 0.48 * physical + 0.34 * cognitive + 0.18 * social
            vulnerability = 1.0 - physical
            clarity = npc.statuses["cognitive"] / 100.0
            terminal_focus = 0.70 * vulnerability if self.player.eliminated else 0.20 * vulnerability
            return 0.72 * (clarity * threat + (1.0 - clarity) * vulnerability) + terminal_focus
        if action == "steal":
            return (
                0.18 * (1.0 - physical)
                + 0.10 * distress
                + 0.27 * confusion
                + 0.18 * isolation
                + 0.22 * fading
                + 0.05 * shadow
            )
        if action == "trade":
            readiness = (
                0.08 * physical
                + 0.18 * emotional
                + 0.28 * cognitive
                + 0.24 * social
                + 0.16 * sentient
                + 0.06 * max(0.0, float(target.statuses["spiritual"]))
            )
            return 0.35 * readiness + player_salience
        raise ValueError(f"No NPC target model for action: {action}")

    @staticmethod
    def _visible_need_phrase(participant: Participant) -> str:
        phrases = {
            "physical": "visible physical strain",
            "emotional": "emotional strain",
            "cognitive": "cognitive overload",
            "social": "social isolation",
            "sentient": "fading awareness",
        }
        weakest = min(RANGED_STATUS_KEYS, key=lambda key: participant.statuses[key])
        if participant.statuses["spiritual"] < 0 and participant.statuses[weakest] >= 55:
            return "a hostile spiritual turn"
        return phrases[weakest]

    def _npc_decision_reasoning(
        self,
        npc: Participant,
        action: str,
        target: Participant,
        item_key: Optional[str],
        exchange: Optional[ExchangePlan] = None,
    ) -> str:
        """Return concise rationale that never reveals exact private stock."""
        physical = npc.statuses["physical"] / 100.0
        emotional = npc.statuses["emotional"] / 100.0
        cognitive = npc.statuses["cognitive"] / 100.0
        social = npc.statuses["social"] / 100.0
        sentient = npc.statuses["sentient"] / 100.0
        spiritual = float(npc.statuses["spiritual"])
        light = max(0.0, spiritual)
        shadow = max(0.0, -spiritual)
        neutral = 1.0 - abs(spiritual)

        def dominant(*signals: Tuple[str, float]) -> str:
            return max(signals, key=lambda signal: signal[1])[0]

        if action == "use" and item_key is not None:
            effects, _ = self._project_item_effects(npc, item_key)
            realized = self._realized_effects(npc, effects)
            helped = max(STATUS_KEYS, key=lambda key: realized[key] * NPC_STATUS_IMPORTANCE[key])
            need = {
                "physical": "physical fatigue",
                "emotional": "emotional strain",
                "cognitive": "cognitive overload",
                "social": "social isolation",
                "sentient": "fading awareness",
                "spiritual": "spiritual imbalance",
            }[helped]
            return f"{need.capitalize()} makes private {ITEM_BY_KEY[item_key].short_label.lower()} the best self-care supply."
        if action == "give" and item_key is not None:
            return (
                f"{possessive_name(target.name)} {self._visible_need_phrase(target)} and an available private "
                f"{ITEM_BY_KEY[item_key].short_label.lower()} reserve favor sharing."
            )
        if action == "steal" and item_key is not None:
            return (
                f"Resource pressure, cognitive timing, and {possessive_name(target.name)} visible "
                f"distraction make exposed {ITEM_BY_KEY[item_key].short_label.lower()} tempting."
            )
        if action == "trade" and exchange is not None:
            return (
                f"Cognitive planning and complementary needs favor offering "
                f"{ITEM_BY_KEY[exchange.offered_item].short_label.lower()} for "
                f"{ITEM_BY_KEY[exchange.requested_item].short_label.lower()} with {target.name}."
            )
        if action == "rest":
            pressure = dominant(
                ("physical fatigue", 1.20 * (1.0 - physical)),
                ("emotional strain", 0.45 * (1.0 - emotional)),
                ("cognitive load", 0.35 * (1.0 - cognitive)),
                ("social exhaustion", 0.10 * (1.0 - social)),
                ("fading awareness", 0.10 * (1.0 - sentient)),
                ("spiritual uncertainty", 0.12 * neutral),
            )
            return f"{pressure.capitalize()} outweighs the cost of private food and water."
        drivers = {
            "talk": dominant(
                ("physical stability", 0.40 * physical),
                ("emotional balance", 0.35 * emotional),
                ("cognitive clarity", 0.70 * cognitive),
                ("social connection", 0.35 * social),
                ("awareness", 0.35 * sentient),
                ("spiritual uncertainty", 0.15 * neutral),
            ),
            "listen": dominant(
                ("physical stability", 0.15 * physical),
                ("emotional balance", 0.25 * emotional),
                ("cognitive clarity", 0.35 * cognitive),
                ("social connection", 0.55 * social),
                ("awareness", 1.00 * sentient),
                ("spiritual alignment", 0.25 * light),
            ),
            "compliment": dominant(
                ("physical stability", 0.15 * physical),
                ("emotional balance", 0.80 * emotional),
                ("cognitive clarity", 0.20 * cognitive),
                ("social connection", 0.90 * social),
                ("awareness", 0.75 * sentient),
                ("spiritual alignment", 0.60 * light),
            ),
            "flirt": dominant(
                ("physical confidence", 0.10 * physical),
                ("emotional openness", 0.78 * emotional),
                ("cognitive confidence", 0.32 * cognitive),
                ("social confidence", 0.90 * social),
                ("awareness", 0.68 * sentient),
                ("spiritual alignment", 0.55 * light),
            ),
            "antagonize": dominant(
                ("physical momentum", 0.30 * physical),
                ("emotional strain", 1.10 * (1.0 - emotional)),
                ("cognitive uncertainty", 0.30 * (1.0 - cognitive)),
                ("social isolation", 0.90 * (1.0 - social)),
                ("detachment", 0.60 * (1.0 - sentient)),
                ("spiritual shadow", 0.90 * shadow),
            ),
            "reflect": dominant(
                ("physical stability", 0.10 * physical),
                ("emotional balance", 0.35 * emotional),
                ("cognitive clarity", 0.75 * cognitive),
                ("social connection", 0.25 * social),
                ("awareness", 1.00 * sentient),
                ("spiritual alignment", 0.80 * light),
                ("spiritual uncertainty", 0.15 * neutral),
            ),
            "fight": dominant(
                ("physical capacity", 0.60 * physical),
                ("cognitive threat analysis", 0.30 * cognitive),
                ("emotional strain", 1.0 - emotional),
                ("social isolation", 1.0 - social),
                ("spiritual shadow", shadow),
                ("survival pressure", 0.24 * self._npc_survival_pressure()),
            ),
        }
        driver = drivers[action]
        reasons = {
            "talk": f"{driver.capitalize()} and the survival calculus favor talking with {target.name}.",
            "listen": f"{driver.capitalize()} focuses attention on {target.name}'s {self._visible_need_phrase(target)}.",
            "compliment": f"{driver.capitalize()} and the survival calculus favor reassuring {target.name}.",
            "flirt": f"{driver.capitalize()} makes a playful, low-pressure overture toward {target.name} feel worth risking.",
            "antagonize": f"{driver.capitalize()} and room pressure turn suspicion toward {target.name}.",
            "reflect": f"{driver.capitalize()} and the survival calculus favor reflection with {target.name}.",
            "fight": f"{driver.capitalize()} marks {target.name} as the chief physical threat.",
        }
        return reasons[action]

    def _decision_repetition_penalty(
        self,
        actor_index: int,
        action: str,
        target_index: int,
        item_key: Optional[str],
        exchange: Optional[ExchangePlan] = None,
    ) -> float:
        previous = self._last_event_for_actor(actor_index)
        if previous is None:
            return 0.0
        penalty = 0.0
        if previous.action == action:
            penalty += 0.44
        if previous.target_index == target_index:
            penalty += 0.10
        if item_key is not None and previous.item_key == item_key:
            penalty += 0.16
        if exchange is not None and previous.exchange == exchange:
            penalty += 0.16
        return penalty

    def _npc_decision_candidates(self, npc: Participant) -> Tuple[NPCDecision, ...]:
        """Enumerate one-turn plans over action, target, and private item use."""
        if npc.eliminated:
            return ()
        actor_index = self._participant_index(npc)
        profile_index = actor_index - 1
        action_biases = NPC_ROLE_ACTION_BIASES[profile_index]
        item_biases = NPC_ROLE_ITEM_BIASES[profile_index]
        priors = self._npc_action_weights(npc)
        candidates: List[NPCDecision] = []
        other_indices = [
            index
            for index, participant in enumerate(self.participants)
            if participant.alive and participant is not npc
        ]

        for action in ("talk", "listen", "compliment", "flirt", "antagonize", "reflect", "fight"):
            for target_index in other_indices:
                target = self.participants[target_index]
                utility = (
                    priors[action]
                    + action_biases.get(action, 0.0)
                    + self._npc_target_modifier(npc, action, target)
                    + (0.55 * self.flirt_chance(npc, target) if action == "flirt" else 0.0)
                    - self._decision_repetition_penalty(actor_index, action, target_index, None)
                )
                candidates.append(
                    NPCDecision(
                        action,
                        target_index,
                        None,
                        utility,
                        self._npc_decision_reasoning(npc, action, target, None),
                    )
                )

        for item_key in ITEM_KEYS:
            spec = ITEM_BY_KEY[item_key]
            if npc.inventory[item_key] < spec.use_amount:
                continue
            effects, _ = self._project_item_effects(npc, item_key)
            realized = self._realized_effects(npc, effects)
            if npc.statuses["physical"] + realized["physical"] <= 0:
                continue
            benefit = self._effect_value(npc, effects)
            # Full-status characters conserve supplies; recent unsafe doses are
            # rejected because their exact forecast has non-positive utility.
            if benefit <= 0.035:
                continue
            scarcity = 1.0 - self._item_security(npc, item_key)
            utility = (
                priors["use"]
                + action_biases.get("use", 0.0)
                + item_biases.get(item_key, 0.0)
                + 1.20 * benefit
                - 0.62 * scarcity
                - self._decision_repetition_penalty(actor_index, "use", actor_index, item_key)
            )
            candidates.append(
                NPCDecision(
                    "use",
                    actor_index,
                    item_key,
                    utility,
                    self._npc_decision_reasoning(npc, "use", npc, item_key),
                )
            )

        if npc.inventory["water_liters"] >= 1 and npc.inventory["food_pounds"] >= 1:
            rest_benefit = self._effect_value(npc, REST_STATUS_EFFECTS)
            if rest_benefit > 0.035:
                provision_scarcity = (
                    2.0
                    - self._item_security(npc, "water_liters")
                    - self._item_security(npc, "food_pounds")
                ) / 2.0
                utility = (
                    priors["rest"]
                    + action_biases.get("rest", 0.0)
                    + 0.82 * rest_benefit
                    - 0.72 * provision_scarcity
                    - self._decision_repetition_penalty(actor_index, "rest", actor_index, None)
                )
                candidates.append(
                    NPCDecision(
                        "rest",
                        actor_index,
                        None,
                        utility,
                        self._npc_decision_reasoning(npc, "rest", npc, None),
                    )
                )

        for target_index in other_indices:
            target = self.participants[target_index]
            for item_key in ITEM_KEYS:
                spec = ITEM_BY_KEY[item_key]
                if npc.inventory[item_key] < spec.give_amount:
                    continue
                if target.inventory[item_key] > INVENTORY_CAP - spec.give_amount:
                    continue
                target_effects, _ = self._project_item_effects(target, item_key)
                target_realized = self._realized_effects(target, target_effects)
                if target.statuses["physical"] + target_realized["physical"] <= 0:
                    continue
                target_benefit = self._effect_value(target, target_effects)
                if target_benefit <= 0.035:
                    continue
                own_effects, _ = self._project_item_effects(npc, item_key)
                own_opportunity = max(0.0, self._effect_value(npc, own_effects))
                security = self._item_security(npc, item_key)
                utility = (
                    priors["give"]
                    + action_biases.get("give", 0.0)
                    + item_biases.get(item_key, 0.0)
                    + 0.92 * target_benefit
                    - 0.78 * own_opportunity
                    + 0.35 * security
                    - 0.70 * (1.0 - security)
                    - self._decision_repetition_penalty(actor_index, "give", target_index, item_key)
                )
                candidates.append(
                    NPCDecision(
                        "give",
                        target_index,
                        item_key,
                        utility,
                        self._npc_decision_reasoning(npc, "give", target, item_key),
                    )
                )

        for target_index in other_indices:
            target = self.participants[target_index]
            for item_key in ITEM_KEYS:
                spec = ITEM_BY_KEY[item_key]
                if target.inventory[item_key] < spec.give_amount:
                    continue
                if npc.inventory[item_key] > INVENTORY_CAP - spec.give_amount:
                    continue
                own_effects, _ = self._project_item_effects(npc, item_key)
                target_effects, _ = self._project_item_effects(target, item_key)
                own_need = max(0.0, self._effect_value(npc, own_effects))
                own_need += 0.45 * (1.0 - self._item_security(npc, item_key))
                victim_cost = max(0.0, self._effect_value(target, target_effects))
                prosocial_brake = (
                    npc.statuses["emotional"] / 100.0
                    + npc.statuses["social"] / 100.0
                    + npc.statuses["sentient"] / 100.0
                    + max(0.0, float(npc.statuses["spiritual"]))
                ) / 4.0
                utility = (
                    priors["steal"]
                    + action_biases.get("steal", 0.0)
                    + item_biases.get(item_key, 0.0)
                    + self._npc_target_modifier(npc, "steal", target)
                    + 0.90 * own_need
                    + 0.55 * self.steal_chance(npc, target)
                    - 0.25 * victim_cost * prosocial_brake
                    - self._decision_repetition_penalty(
                        actor_index,
                        "steal",
                        target_index,
                        item_key,
                    )
                )
                candidates.append(
                    NPCDecision(
                        "steal",
                        target_index,
                        item_key,
                        utility,
                        self._npc_decision_reasoning(npc, "steal", target, item_key),
                    )
                )

        for target_index in other_indices:
            target = self.participants[target_index]
            for offered_key in ITEM_KEYS:
                offered_spec = ITEM_BY_KEY[offered_key]
                offered_quantity = offered_spec.give_amount
                if npc.inventory[offered_key] < offered_quantity:
                    continue
                for requested_key in ITEM_KEYS:
                    if requested_key == offered_key:
                        continue
                    requested_spec = ITEM_BY_KEY[requested_key]
                    requested_quantity = requested_spec.give_amount
                    if target.inventory[requested_key] < requested_quantity:
                        continue
                    exchange = ExchangePlan(
                        offered_key,
                        offered_quantity,
                        requested_key,
                        requested_quantity,
                    )
                    actor_after = dict(npc.inventory)
                    target_after = dict(target.inventory)
                    actor_after[offered_key] -= offered_quantity
                    target_after[offered_key] += offered_quantity
                    target_after[requested_key] -= requested_quantity
                    actor_after[requested_key] += requested_quantity
                    if any(not 0 <= amount <= INVENTORY_CAP for amount in actor_after.values()):
                        continue
                    if any(not 0 <= amount <= INVENTORY_CAP for amount in target_after.values()):
                        continue

                    requested_effects, _ = self._project_item_effects(npc, requested_key)
                    offered_effects, _ = self._project_item_effects(npc, offered_key)
                    target_receives, _ = self._project_item_effects(target, offered_key)
                    target_surrenders, _ = self._project_item_effects(target, requested_key)
                    actor_requested_need = max(0.0, self._effect_value(npc, requested_effects))
                    actor_requested_need += 0.45 * (1.0 - self._item_security(npc, requested_key))
                    actor_offered_need = max(0.0, self._effect_value(npc, offered_effects))
                    target_offered_need = max(0.0, self._effect_value(target, target_receives))
                    target_requested_need = max(0.0, self._effect_value(target, target_surrenders))
                    acceptance = self._trade_willingness_between(npc, target, exchange)
                    utility = (
                        priors["trade"]
                        + action_biases.get("trade", 0.0)
                        + self._npc_target_modifier(npc, "trade", target)
                        + 0.90 * actor_requested_need
                        - 0.55 * actor_offered_need
                        + 0.30 * (target_offered_need - target_requested_need)
                        + 0.45 * acceptance
                        - self._decision_repetition_penalty(
                            actor_index,
                            "trade",
                            target_index,
                            None,
                            exchange,
                        )
                    )
                    candidates.append(
                        NPCDecision(
                            "trade",
                            target_index,
                            None,
                            utility,
                            self._npc_decision_reasoning(
                                npc,
                                "trade",
                                target,
                                None,
                                exchange,
                            ),
                            exchange,
                        )
                    )
        return tuple(candidates)

    def _choose_npc_decision(self, npc: Participant) -> NPCDecision:
        candidates = self._npc_decision_candidates(npc)
        if not candidates:
            raise RuntimeError("NPC has no legal one-interaction plan")
        best_utility = max(candidate.utility for candidate in candidates)
        finalists = [
            candidate
            for candidate in candidates
            if math.isclose(candidate.utility, best_utility, rel_tol=0.0, abs_tol=1e-9)
        ]
        return self.rng.choice(finalists)

    def _choose_npc_action(self, npc: Participant) -> str:
        """Compatibility view for callers that need only the chosen verb."""
        return self._choose_npc_decision(npc).action

    def _choose_npc_target(self, npc: Participant, action: str) -> Participant:
        if action in ("use", "rest"):
            return npc
        candidates = [person for person in self.participants if person.alive and person is not npc]
        if not candidates:
            raise RuntimeError("NPC has no living interaction target")
        if action in ("compliment", "flirt", "listen"):
            weights = [max(1.0, 202.0 - person.statuses["emotional"] - person.statuses["social"]) for person in candidates]
        elif action in ("antagonize", "fight", "steal"):
            weights = [max(1.0, person.statuses["physical"] + person.statuses["cognitive"]) for person in candidates]
        else:
            weights = [1.6 if person is self.player else 1.0 for person in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _run_npc_phase(self) -> None:
        for actor_index in range(1, len(self.participants)):
            if self.is_ending:
                break
            npc = self.participants[actor_index]
            if npc.eliminated:
                continue
            decision = self._choose_npc_decision(npc)
            self._execute_actor_slot(
                actor_index,
                decision.action,
                decision.target_index,
                decision.item_key,
                decision.reasoning,
                decision.utility,
                decision.exchange,
            )

    def _next_npc_actor_index(self) -> Optional[int]:
        for actor_index in range(1, len(self.participants)):
            npc = self.participants[actor_index]
            if npc.alive and actor_index not in self.acted_this_round:
                return actor_index
        return None

    def _advance_autonomous_story(self) -> None:
        story_route = {
            "sealed_room": "resonance",
            "resonance": "mechanism",
            "testimony": "pattern",
            "mechanism": "pattern",
            "pattern": "discovery",
            "discovery": "signal_refrain",
            "signal_refrain": "signal_afterimage",
            "signal_afterimage": "signal_counterpoint",
            "signal_counterpoint": "signal_refrain",
        }
        self.scene_id = story_route[self.scene_id]
        if self.scene_id in {
            "discovery",
            "signal_refrain",
            "signal_afterimage",
            "signal_counterpoint",
        }:
            self.fifth_signal_found = True

    def _finalize_current_round(self) -> None:
        """Apply room strain and archive one fully paced or immediate round."""
        exposure_eliminations: List[Participant] = []
        if not self.is_ending:
            exposure_eliminations = self._apply_end_of_round_exposure()
            if self.last_exposure_strain:
                self.round_notices.append(
                    f"ROOM STRAIN: the room applied {self.last_exposure_strain} PHYS strain "
                    "to every living participant"
                    + ("; low emotional state also reduced COG." if any(
                        participant.alive and participant.statuses["emotional"] < 25
                        for participant in self.participants
                    ) else ".")
                )
            for participant in exposure_eliminations:
                self.round_notices.append(
                    f"ROOM STRAIN: {participant.name} reached 0 PHYS and was eliminated."
                )
        if not self.is_ending:
            if self.current_round_player_led:
                companion_actions = max(0, len(self.last_round_events) - 1)
                if self.player.alive:
                    self.phase = "awaiting_player"
                    self.message = (
                        f"Round {self.turn} complete: {self.player.name} acted first and {companion_actions} "
                        "living companion actions followed at one-second intervals."
                        + self._elimination_notice(exposure_eliminations)
                    )
                else:
                    self.phase = "autonomous"
                    self.message = (
                        f"Round {self.turn} complete. {self.player.name} is eliminated; {companion_actions} "
                        "survivor actions followed and the room continues."
                        + self._elimination_notice(exposure_eliminations)
                    )
            else:
                self.phase = "autonomous"
                self._advance_autonomous_story()
                self.message = (
                    f"Autonomous round {self.turn}: {len(self.last_round_events)} surviving NPC "
                    "interactions resolved at one-second intervals."
                    + self._elimination_notice(exposure_eliminations)
                )
        self._archive_current_round()

    def _complete_round(self) -> None:
        if self.paced_rounds and not self.is_ending:
            self.phase = "resolving_npcs"
            next_actor = self._next_npc_actor_index()
            if next_actor is not None:
                self.message = (
                    f"Round {self.turn}, moment {len(self.last_round_events)}: "
                    f"{self.participants[self.last_round_events[-1].actor_index].name} acted. "
                    f"{self.participants[next_actor].name} considers the next move."
                )
                return
        if not self.is_ending:
            self._run_npc_phase()
        self._finalize_current_round()

    def advance_resolution_moment(self) -> bool:
        """Resolve exactly one pending NPC action in a paced round."""
        if self.is_ending or self.phase != "resolving_npcs":
            return False
        actor_index = self._next_npc_actor_index()
        if actor_index is None:
            self._finalize_current_round()
            self.validate()
            return True
        npc = self.participants[actor_index]
        decision = self._choose_npc_decision(npc)
        self._execute_actor_slot(
            actor_index,
            decision.action,
            decision.target_index,
            decision.item_key,
            decision.reasoning,
            decision.utility,
            decision.exchange,
        )
        next_actor = self._next_npc_actor_index()
        if self.is_ending or next_actor is None:
            self._finalize_current_round()
        else:
            # Elimination handling may temporarily select autonomous mode when
            # the player dies; the current round still owes later NPCs a slot.
            self.phase = "resolving_npcs"
            self.message = (
                f"Round {self.turn}, moment {len(self.last_round_events)}: {npc.name} acted. "
                f"{self.participants[next_actor].name} considers the next move."
            )
        self.validate()
        return True

    def advance_autonomous_round(self) -> bool:
        """Begin or resolve one NPC-only round after the player's elimination."""
        if self.player.alive or self.is_ending or self.phase != "autonomous":
            return False
        if len(self.living_participants) <= 1:
            self._finish_last_survivor()
            return False

        self.phase = "resolving_npcs"
        self.acted_this_round.clear()
        self.last_round_events = []
        self.round_notices = []
        self.current_round_player_led = False
        self._advance_time()
        if self.paced_rounds:
            self.advance_resolution_moment()
        else:
            self._run_npc_phase()
            self._finalize_current_round()
        self.validate()
        return True

    def interact(self, action: str) -> None:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown interaction: {action}")
        if self.is_ending or self.player.eliminated:
            return

        if action == "trade":
            if self.trade_draft is None:
                self.begin_trade()
            else:
                self.confirm_trade()
            self.validate()
            return

        target = self.player if action in self.SELF_ACTIONS else self.selected_npc
        if action in self.TARGET_ACTIONS and target.eliminated:
            self.message = f"{target.name} has been eliminated and cannot take part in that action."
            return
        if not self._begin_round():
            self.message = "A round is already resolving."
            return
        self._react()
        self._execute_actor_slot(
            0,
            action,
            self._participant_index(target),
            self.selected_item if action in ("give", "use", "steal") else None,
        )
        self._complete_round()
        self.validate()

    def confirm_trade(self, roll: Optional[float] = None) -> bool:
        draft = self.trade_draft
        if draft is None:
            self.message = "Open the trade builder before confirming a proposal."
            return False
        local_error = self.trade_local_error(draft)
        if local_error:
            malformed = (
                type(draft.target_index) is not int
                or not 0 <= draft.target_index < len(self.npcs)
                or draft.offered_item not in ITEM_KEYS
                or draft.requested_item not in ITEM_KEYS
                or type(draft.offered_quantity) is not int
                or type(draft.requested_quantity) is not int
                or not 1 <= draft.offered_quantity <= INVENTORY_CAP
                or not 1 <= draft.requested_quantity <= INVENTORY_CAP
            )
            target_eliminated = (
                type(draft.target_index) is int
                and 0 <= draft.target_index < len(self.npcs)
                and self.npcs[draft.target_index].eliminated
            )
            if malformed or target_eliminated:
                self.trade_draft = None
            self.message = local_error
            return False

        target = self.npcs[draft.target_index]
        if roll is not None and not 0.0 <= float(roll) <= 1.0:
            raise ValueError("A supplied trade roll must be between zero and one")

        chance = self.trade_willingness(draft)
        if not self._begin_round():
            self.message = "A round is already resolving."
            return False
        before_changes = self._change_snapshot()
        roll_value = self.rng.random() if roll is None else float(roll)
        self.trade_draft = None
        success = False
        if target.inventory[draft.requested_item] < draft.requested_quantity:
            requested = ITEM_BY_KEY[draft.requested_item]
            summary = (
                f"proposes a trade to {target.name}, but private stock cannot fill "
                f"the {requested.short_label.lower()} request"
            )
        else:
            player_after = dict(self.player.inventory)
            target_after = dict(target.inventory)
            player_after[draft.offered_item] -= draft.offered_quantity
            target_after[draft.offered_item] += draft.offered_quantity
            target_after[draft.requested_item] -= draft.requested_quantity
            player_after[draft.requested_item] += draft.requested_quantity
            if any(not 0 <= amount <= INVENTORY_CAP for amount in target_after.values()):
                summary = f"proposes a trade, but {target.name} cannot carry the offer"
            elif roll_value > chance:
                self.player.adjust("emotional", -1)
                self.player.adjust("social", -2)
                target.adjust("social", -1)
                summary = f"has a {chance:.0%} trade proposal declined by {target.name}"
            else:
                self.player.inventory = player_after
                target.inventory = target_after
                self._apply_effects(self.player, {"cognitive": 2, "social": 3, "sentient": 1})
                self._apply_effects(target, {"emotional": 1, "social": 2})
                offered = ITEM_BY_KEY[draft.offered_item]
                requested = ITEM_BY_KEY[draft.requested_item]
                summary = (
                    f"trades {draft.offered_quantity:,} {offered.short_label.lower()} with "
                    f"{target.name} for {draft.requested_quantity:,} {requested.short_label.lower()}"
                )
                success = True

        self._record_event(
            actor_index=0,
            action="trade",
            target_index=draft.target_index + 1,
            summary=summary,
            success=success,
            exchange=ExchangePlan(
                draft.offered_item,
                draft.offered_quantity,
                draft.requested_item,
                draft.requested_quantity,
            ),
            impact=self._describe_changes(before_changes),
        )
        self._complete_round()
        self.validate()
        return success

    def choose_story(self, choice_index: int) -> None:
        if self.is_ending or self.player.eliminated:
            return
        scene = self.current_scene
        if scene is None or not 0 <= choice_index < len(scene.choices):
            raise IndexError("Story choice is outside the current scene")

        choice = scene.choices[choice_index]
        if not self._begin_round():
            self.message = "A round is already resolving."
            return
        before_changes = self._change_snapshot()
        target = self.selected_npc
        self._react()
        can_pay = self._can_pay(choice.costs)

        if not can_pay:
            missing = self._cost_description(choice.costs)
            self._apply_effects(
                self.player,
                {"physical": -1, "emotional": -3, "cognitive": 1, "social": -2},
            )
            outcome = (
                f"You cannot provide {missing}. The room rejects the attempted test, so this "
                "part of the mechanism remains unsolved."
            )
        else:
            for key, amount in choice.costs.items():
                self.player.change_item(key, -amount)
            for key, amount in choice.gains.items():
                self.player.change_item(key, amount)
            self._apply_effects(self.player, choice.player_effects)
            self._apply_effects(target, choice.target_effects)
            for participant in self.participants:
                self._apply_effects(participant, choice.party_effects)
            if choice.player_spiritual is not None:
                self.player.set_spiritual(choice.player_spiritual)
            if choice.target_spiritual is not None:
                target.set_spiritual(choice.target_spiritual)
            target_name = target.name if target.alive else f"the memory of {target.name}"
            target_role = target.profile.role if target.alive else "absent companion"
            outcome = choice.outcome.format(
                npc=target_name,
                role=target_role,
                player=self.player.name,
            )
            if target.alive:
                outcome = self._resolve_special(choice.special, outcome, target)

        target_label = target.name if target.alive else f"the memory of {target.name}"
        summary_label = choice.label.format(npc=target_label, player=self.player.name)
        self.history.append(f"{scene.title}: {summary_label}")
        if can_pay:
            self.chapter += 1
        choice_eliminations = self._resolve_eliminations()
        self._record_event(
            actor_index=0,
            action="investigate",
            target_index=self._participant_index(target),
            summary=f"investigates: {summary_label}",
            success=can_pay,
            eliminated=choice_eliminations,
            impact=self._describe_changes(before_changes),
        )
        self.message = outcome + self._elimination_notice(choice_eliminations)
        self._complete_round()
        if self.is_game_over:
            self.validate()
            return

        next_scene = choice.next_scene if can_pay else self.scene_id
        if next_scene not in SCENES:
            raise ValueError(f"Story route is not a room scene: {next_scene}")
        self.scene_id = next_scene
        if next_scene == "discovery":
            self.fifth_signal_found = True
        self.message = outcome + self._elimination_notice(choice_eliminations)
        self.validate()

    def _resolve_special(self, special: str, outcome: str, target: Participant) -> str:
        if special == "echo":
            score = (
                self.player.statuses["cognitive"]
                + self.player.statuses["sentient"]
                + target.statuses["cognitive"]
                + target.statuses["sentient"]
            ) / 4.0
            if score >= 58:
                self._apply_effects(self.player, {"cognitive": 2, "sentient": 2})
                target.adjust("sentient", 2)
                return outcome + " Together you isolate the fifth waveform from the room's hum."
            self._apply_effects(self.player, {"emotional": -1, "cognitive": 1})
            return outcome + " The pattern remains uncertain, but the failed test narrows it."

        if special == "signal_sync":
            self.fifth_signal_found = True
            if self.player.statuses["sentient"] + target.statuses["sentient"] >= 120:
                self.player.set_spiritual(1)
                target.set_spiritual(1)
                return outcome + " The cyan fifth channel mirrors both faces at once."
            return outcome + " The fifth channel stabilizes without revealing why it chose white."

        if special == "violent_override" and self.player.statuses["cognitive"] < 50:
            self.player.adjust("physical", -4)
            return outcome + f" The feedback catches {self.player.name} before the pattern can be controlled."
        return outcome

    def _finish_last_survivor(self) -> None:
        winner = self.last_survivor
        if winner is None:
            return
        self.trade_draft = None
        self.ending_title = f"{winner.name}: Last Survivor"
        if winner is self.player:
            self.ending_text = (
                f"Only {self.player.name} remains alive. Four channels go dark, the wall nodes settle into "
                "one pulse, and the Fifth Signal display begins another unreadable sequence."
            )
        else:
            self.ending_text = (
                f"Only {winner.name} remains alive. The wall nodes settle into one pulse, and "
                "the Fifth Signal display begins another unreadable sequence."
            )
        self.phase = "ending"
        self.scene_id = "ending"
        self.message = self.ending_text

    def public_status_snapshot(self) -> Dict[str, Any]:
        return {
            "room": {
                "sealed": True,
                "scene": self.scene_id,
                "round": self.turn,
                "phase": self.phase,
                "fifth_signal_found": self.fifth_signal_found,
                "living_count": len(self.living_participants),
                "winner": self.last_survivor.name if self.is_ending and self.last_survivor else None,
            },
            "player": {
                "name": self.player.name,
                "statuses": dict(self.player.statuses),
                "inventory": dict(self.player.inventory),
                "eliminated": self.player.eliminated,
                "gesture_key": None,
                "gesture_turn": None,
            },
            "npcs": [
                {
                    "name": npc.name,
                    "role": npc.profile.role,
                    "statuses": dict(npc.statuses),
                    "eliminated": npc.eliminated,
                    "gesture_key": npc.gesture_key,
                    "gesture_label": GESTURE_LABELS[npc.gesture_key],
                    "gesture_turn": npc.gesture_turn,
                }
                for npc in self.npcs
            ],
            "last_round": [
                {
                    "actor_index": event.actor_index,
                    "action": event.action,
                    "target_index": event.target_index,
                    "success": event.success,
                    "eliminated_indices": event.eliminated_indices,
                    "gesture_key": event.gesture_key,
                    "item_key": event.item_key,
                    "exchange": (
                        {
                            "offered_item": event.exchange.offered_item,
                            "offered_quantity": event.exchange.offered_quantity,
                            "requested_item": event.exchange.requested_item,
                            "requested_quantity": event.exchange.requested_quantity,
                        }
                        if event.exchange is not None
                        else None
                    ),
                    "reasoning": event.reasoning,
                    "impact": event.impact,
                    "voice_statuses": (
                        dict(zip(STATUS_KEYS, event.voice_statuses))
                        if event.voice_statuses
                        else None
                    ),
                }
                for event in self.last_round_events
            ],
            "round_notices": list(self.round_notices),
        }

    def validate(self) -> None:
        def validate_voice_snapshot(event: RoundEvent) -> None:
            expected = event.actor_index > 0 or (
                event.target_index is not None and event.target_index > 0
            )
            if not expected:
                assert event.voice_statuses == ()
                return
            assert len(event.voice_statuses) == len(STATUS_KEYS)
            for key, value in zip(STATUS_KEYS, event.voice_statuses):
                if key == "spiritual":
                    assert value in (-1, 0, 1)
                else:
                    low = 0 if key == "physical" else 1
                    assert isinstance(value, int) and low <= value <= 100

        def validate_decision_metadata(event: RoundEvent) -> None:
            if event.item_key is not None:
                assert event.item_key in ITEM_KEYS
                assert event.action in ("give", "use", "steal")
            if event.action in ("give", "use", "steal"):
                assert event.item_key in ITEM_KEYS
            if event.exchange is not None:
                assert event.action == "trade"
                assert event.exchange.offered_item in ITEM_KEYS
                assert event.exchange.requested_item in ITEM_KEYS
                assert 1 <= event.exchange.offered_quantity <= INVENTORY_CAP
                assert 1 <= event.exchange.requested_quantity <= INVENTORY_CAP
            if event.action == "trade":
                assert event.exchange is not None
            assert isinstance(event.impact, str) and event.impact.strip()
            if event.actor_index > 0:
                assert isinstance(event.reasoning, str) and event.reasoning.strip()
                assert isinstance(event.decision_score, float)
                assert math.isfinite(event.decision_score)
                assert event.target_index is not None
                if event.action in ("use", "rest"):
                    assert event.target_index == event.actor_index
                else:
                    assert event.target_index != event.actor_index
            else:
                assert event.reasoning == ""
                assert event.decision_score is None

        assert len(self.participants) == 5
        assert len(self.npcs) == 4
        assert self.player.display_name == normalize_player_name(self.player.name)
        assert all(npc.display_name is None for npc in self.npcs)
        assert 0 <= self.selected_npc_index < 4
        assert self.selected_item in ITEM_KEYS
        assert self.phase in ("awaiting_player", "resolving_npcs", "autonomous", "ending")
        assert isinstance(self.paced_rounds, bool)
        assert isinstance(self.current_round_player_led, bool)
        assert isinstance(self.idle_epoch, float) and math.isfinite(self.idle_epoch)
        if self.is_ending:
            assert self.phase == "ending"
            assert self.scene_id == "ending"
            assert len(self.living_participants) == 1
            assert self.last_survivor is not None
        else:
            assert self.scene_id in SCENES
            assert len(self.living_participants) >= 2
        if self.player.eliminated and not self.is_ending:
            assert self.phase in ("resolving_npcs", "autonomous")
        actors = [event.actor_index for event in self.last_round_events]
        assert len(actors) == len(set(actors))
        assert actors == sorted(actors)
        assert self.acted_this_round == set(actors)
        assert all(isinstance(notice, str) and notice for notice in self.round_notices)
        if actors and actors[0] != 0:
            assert self.player.eliminated
            assert all(actor > 0 for actor in actors)
        elif actors:
            assert actors[0] == 0
        for event in self.last_round_events:
            assert event.round_number == self.turn
            assert 0 <= event.actor_index < len(self.participants)
            assert event.action in self.ACTIONS or event.action == "investigate"
            if event.actor_index > 0:
                assert event.action in self.NPC_ACTIONS
                assert event.gesture_key in NPC_GESTURE_KEYS
            else:
                assert event.gesture_key is None
            if event.target_index is not None:
                assert 0 <= event.target_index < len(self.participants)
            assert all(0 <= index < len(self.participants) for index in event.eliminated_indices)
            validate_voice_snapshot(event)
            validate_decision_metadata(event)
        archived_round_numbers = [record.round_number for record in self.round_history]
        assert archived_round_numbers == sorted(set(archived_round_numbers))
        assert all(1 <= round_number <= self.turn for round_number in archived_round_numbers)
        for record in self.round_history:
            assert isinstance(record, RoundRecord)
            assert isinstance(record.events, tuple) and record.events
            assert isinstance(record.notices, tuple)
            record_actors = [event.actor_index for event in record.events]
            assert record_actors == sorted(set(record_actors))
            assert all(event.round_number == record.round_number for event in record.events)
            assert all(
                event.gesture_key in NPC_GESTURE_KEYS
                if event.actor_index > 0
                else event.gesture_key is None
                for event in record.events
            )
            for event in record.events:
                validate_voice_snapshot(event)
                validate_decision_metadata(event)
            assert all(isinstance(notice, str) and notice for notice in record.notices)
        if self.last_round_events and self.phase != "resolving_npcs":
            assert self.round_history
            assert self.round_history[-1].round_number == self.turn
            assert self.round_history[-1].events == tuple(self.last_round_events)
            assert self.round_history[-1].notices == tuple(self.round_notices)
        if self.trade_draft is not None:
            assert self.phase == "awaiting_player"
            assert self.player.alive
            assert type(self.trade_draft.target_index) is int
            assert 0 <= self.trade_draft.target_index < len(self.npcs)
            assert self.trade_draft.offered_item in ITEM_KEYS
            assert self.trade_draft.requested_item in ITEM_KEYS
            assert type(self.trade_draft.offered_quantity) is int
            assert type(self.trade_draft.requested_quantity) is int
            assert 1 <= self.trade_draft.offered_quantity <= INVENTORY_CAP
            assert 1 <= self.trade_draft.requested_quantity <= INVENTORY_CAP
            assert self.npcs[self.trade_draft.target_index].alive
        for participant_index, participant in enumerate(self.participants):
            assert set(participant.statuses) == set(STATUS_KEYS)
            assert set(participant.inventory) == set(ITEM_KEYS)
            for key in RANGED_STATUS_KEYS:
                value = participant.statuses[key]
                low = 0 if key == "physical" else 1
                assert isinstance(value, int) and low <= value <= 100
            assert participant.statuses["spiritual"] in (-1, 0, 1)
            assert participant.eliminated == (participant.statuses["physical"] == 0)
            if participant.eliminated:
                assert participant.elimination_turn is not None
            assert participant.gesture_key in GESTURE_LABELS
            assert type(participant.gesture_turn) is int
            assert 0 <= participant.gesture_turn <= self.turn
            assert isinstance(participant.gesture_changed_at, float)
            assert math.isfinite(participant.gesture_changed_at)
            if participant_index == 0:
                assert participant.gesture_key == "idle"
                assert participant.gesture_turn == 0
            elif participant.gesture_turn == 0:
                assert participant.gesture_key == "idle"
            else:
                assert participant.gesture_key in NPC_GESTURE_KEYS
            for value in participant.inventory.values():
                assert isinstance(value, int) and 0 <= value <= INVENTORY_CAP


@dataclass
class HitRegion:
    rect: Tuple[float, float, float, float]
    kind: str
    value: Any

    def contains(self, position: Tuple[int, int]) -> bool:
        x, y, width, height = self.rect
        return x <= position[0] <= x + width and y <= position[1] <= y + height


def load_graphics() -> None:
    global pygame, GL, GLU
    try:
        import pygame as pygame_module
    except ImportError as exc:
        raise SystemExit("This game requires pygame. Install it with: pip install pygame") from exc
    try:
        from OpenGL import GL as gl_module
        from OpenGL import GLU as glu_module
    except ImportError as exc:
        raise SystemExit(
            "This game requires PyOpenGL. Install it with: pip install PyOpenGL PyOpenGL_accelerate"
        ) from exc
    pygame = pygame_module
    GL = gl_module
    GLU = glu_module


def gl_color(color: Sequence[int], alpha: float = 1.0) -> None:
    GL.glColor4f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha)


def draw_rect(x: float, y: float, width: float, height: float, color: Sequence[int], alpha: float = 1.0) -> None:
    gl_color(color, alpha)
    GL.glBegin(GL.GL_QUADS)
    GL.glVertex2f(x, y)
    GL.glVertex2f(x + width, y)
    GL.glVertex2f(x + width, y + height)
    GL.glVertex2f(x, y + height)
    GL.glEnd()


def draw_outline(
    x: float,
    y: float,
    width: float,
    height: float,
    color: Sequence[int],
    line_width: float = 1.0,
) -> None:
    GL.glLineWidth(line_width)
    gl_color(color)
    GL.glBegin(GL.GL_LINE_LOOP)
    GL.glVertex2f(x, y)
    GL.glVertex2f(x + width, y)
    GL.glVertex2f(x + width, y + height)
    GL.glVertex2f(x, y + height)
    GL.glEnd()
    GL.glLineWidth(1.0)


def _begin_bound_texture(texture_id: int) -> bool:
    """Bind an optional RGBA material texture using fixed-function modulation."""
    if not texture_id:
        return False
    GL.glEnable(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexEnvi(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
    return True


def _end_bound_texture(was_bound: bool) -> None:
    if was_bound:
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)


def draw_box(
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    color: Sequence[int],
    texture_id: int = 0,
) -> None:
    x0, x1 = center_x - size_x / 2.0, center_x + size_x / 2.0
    y0, y1 = center_y - size_y / 2.0, center_y + size_y / 2.0
    z0, z1 = center_z - size_z / 2.0, center_z + size_z / 2.0
    textured = _begin_bound_texture(texture_id)
    gl_color(color)
    GL.glBegin(GL.GL_QUADS)
    for normal, vertices in (
        ((0, 1, 0), ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))),
        ((0, -1, 0), ((x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0))),
        ((0, 0, 1), ((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1))),
        ((0, 0, -1), ((x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0))),
        ((1, 0, 0), ((x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (x1, y0, z0))),
        ((-1, 0, 0), ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))),
    ):
        GL.glNormal3f(*normal)
        for texcoord, vertex in zip(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), vertices):
            GL.glTexCoord2f(*texcoord)
            GL.glVertex3f(*vertex)
    GL.glEnd()
    _end_bound_texture(textured)


_SPHERE_MESHES: Dict[Tuple[int, int], int] = {}
_CYLINDER_MESHES: Dict[int, int] = {}


def _sphere_mesh(rings: int, segments: int) -> int:
    """Return a cached unit sphere display list for the legacy GL portrait pass."""
    key = (rings, segments)
    cached = _SPHERE_MESHES.get(key)
    if cached is not None:
        return cached
    mesh = int(GL.glGenLists(1))
    if mesh <= 0:
        raise RuntimeError("OpenGL could not allocate the portrait sphere mesh")
    GL.glNewList(mesh, GL.GL_COMPILE)
    for ring in range(rings):
        theta0 = math.pi * ring / rings
        theta1 = math.pi * (ring + 1) / rings
        y0 = math.cos(theta0)
        y1 = math.cos(theta1)
        radial0 = math.sin(theta0)
        radial1 = math.sin(theta1)
        GL.glBegin(GL.GL_QUAD_STRIP)
        for segment in range(segments + 1):
            angle = math.tau * segment / segments
            u = segment / segments
            x0 = math.cos(angle) * radial0
            z0 = math.sin(angle) * radial0
            x1 = math.cos(angle) * radial1
            z1 = math.sin(angle) * radial1
            GL.glTexCoord2f(u, ring / rings)
            GL.glNormal3f(x0, y0, z0)
            GL.glVertex3f(x0, y0, z0)
            GL.glTexCoord2f(u, (ring + 1) / rings)
            GL.glNormal3f(x1, y1, z1)
            GL.glVertex3f(x1, y1, z1)
        GL.glEnd()
    GL.glEndList()
    _SPHERE_MESHES[key] = mesh
    return mesh


def draw_sphere(
    radius: float,
    color: Sequence[int],
    rings: int = 14,
    segments: int = 28,
    texture_id: int = 0,
) -> None:
    """Draw a scaled cached mesh; high detail stays cheap across four portraits."""
    textured = _begin_bound_texture(texture_id)
    gl_color(color)
    GL.glPushMatrix()
    GL.glScalef(radius, radius, radius)
    GL.glCallList(_sphere_mesh(max(4, rings), max(8, segments)))
    GL.glPopMatrix()
    _end_bound_texture(textured)


def _cylinder_mesh(segments: int) -> int:
    cached = _CYLINDER_MESHES.get(segments)
    if cached is not None:
        return cached
    mesh = int(GL.glGenLists(1))
    if mesh <= 0:
        raise RuntimeError("OpenGL could not allocate the portrait cylinder mesh")
    GL.glNewList(mesh, GL.GL_COMPILE)
    GL.glBegin(GL.GL_QUAD_STRIP)
    for index in range(segments + 1):
        angle = math.tau * index / segments
        x = math.cos(angle)
        z = math.sin(angle)
        GL.glTexCoord2f(index / segments, 0.0)
        GL.glNormal3f(x, 0.0, z)
        GL.glVertex3f(x, -0.5, z)
        GL.glTexCoord2f(index / segments, 1.0)
        GL.glVertex3f(x, 0.5, z)
    GL.glEnd()
    for y, normal in ((-0.5, -1.0), (0.5, 1.0)):
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glNormal3f(0.0, normal, 0.0)
        GL.glTexCoord2f(0.5, 0.5)
        GL.glVertex3f(0.0, y, 0.0)
        for index in range(segments + 1):
            angle = math.tau * index / segments
            if normal < 0.0:
                angle = -angle
            GL.glTexCoord2f(0.5 + math.cos(angle) * 0.5, 0.5 + math.sin(angle) * 0.5)
            GL.glVertex3f(math.cos(angle), y, math.sin(angle))
        GL.glEnd()
    GL.glEndList()
    _CYLINDER_MESHES[segments] = mesh
    return mesh


def draw_cylinder(
    radius: float,
    height: float,
    color: Sequence[int],
    segments: int = 24,
    texture_id: int = 0,
) -> None:
    textured = _begin_bound_texture(texture_id)
    gl_color(color)
    GL.glPushMatrix()
    GL.glScalef(radius, height, radius)
    GL.glCallList(_cylinder_mesh(max(8, segments)))
    GL.glPopMatrix()
    _end_bound_texture(textured)


def draw_cylinder_between(
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
    color: Sequence[int],
    segments: int = 24,
    texture_id: int = 0,
) -> None:
    """Draw one cached cylinder along an arbitrary 3-D segment."""
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    dz = float(end[2] - start[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-6:
        return
    midpoint = (
        (float(start[0]) + float(end[0])) * 0.5,
        (float(start[1]) + float(end[1])) * 0.5,
        (float(start[2]) + float(end[2])) * 0.5,
    )
    # Cached cylinders point along +Y.  Y cross direction gives the rotation
    # axis that carries the mesh onto the requested segment.
    angle = math.degrees(math.acos(float(clamp(dy / length, -1.0, 1.0))))
    axis_x, axis_z = dz, -dx
    GL.glPushMatrix()
    GL.glTranslatef(*midpoint)
    if abs(axis_x) + abs(axis_z) > 1e-6:
        GL.glRotatef(angle, axis_x, 0.0, axis_z)
    elif dy < 0.0:
        GL.glRotatef(180.0, 1.0, 0.0, 0.0)
    draw_cylinder(radius, length, color, segments, texture_id)
    GL.glPopMatrix()


class GLTextureBank:
    """Small, failure-tolerant bank for the shipped CC0 material derivatives."""

    REPEATING_KEYS = frozenset(("skin", "hair", "cloth"))

    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self.ids: Dict[str, int] = {}
        self.errors: Dict[str, str] = {}
        for key, filename in TEXTURE_ASSET_FILES.items():
            self.ids[key] = self._load(key, asset_dir / filename)

    def _load(self, key: str, path: Path) -> int:
        if not path.is_file():
            self.errors[key] = "missing"
            return 0
        texture = 0
        try:
            surface = pygame.image.load(str(path)).convert_alpha()
            width, height = surface.get_size()
            data = pygame.image.tostring(surface, "RGBA", True)
            texture = int(GL.glGenTextures(1))
            if texture <= 0:
                raise RuntimeError("glGenTextures returned zero")
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            wrap = GL.GL_REPEAT if key in self.REPEATING_KEYS else GL.GL_CLAMP_TO_EDGE
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA,
                width,
                height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                data,
            )
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            return texture
        except Exception as exc:  # graceful flat-material fallback
            if texture:
                GL.glDeleteTextures([texture])
            self.errors[key] = str(exc)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            return 0

    def get(self, key: str) -> int:
        return self.ids.get(key, 0)

    @property
    def loaded_count(self) -> int:
        return sum(bool(texture) for texture in self.ids.values())

    def cleanup(self) -> None:
        textures = [texture for texture in self.ids.values() if texture]
        if textures:
            GL.glDeleteTextures(textures)
        self.ids.clear()


def draw_gradient_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    top: Sequence[int],
    bottom: Sequence[int],
    alpha: float = 1.0,
) -> None:
    GL.glBegin(GL.GL_QUADS)
    gl_color(top, alpha)
    GL.glVertex2f(x, y)
    GL.glVertex2f(x + width, y)
    gl_color(bottom, alpha)
    GL.glVertex2f(x + width, y + height)
    GL.glVertex2f(x, y + height)
    GL.glEnd()


def draw_textured_quad_2d(
    texture_id: int,
    x: float,
    y: float,
    width: float,
    height: float,
    color: Sequence[int] = WHITE,
    alpha: float = 1.0,
    uv: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> None:
    if not texture_id:
        return
    textured = _begin_bound_texture(texture_id)
    gl_color(color, alpha)
    u0, v0, u1, v1 = uv
    GL.glBegin(GL.GL_QUADS)
    GL.glTexCoord2f(u0, v1)
    GL.glVertex2f(x, y)
    GL.glTexCoord2f(u1, v1)
    GL.glVertex2f(x + width, y)
    GL.glTexCoord2f(u1, v0)
    GL.glVertex2f(x + width, y + height)
    GL.glTexCoord2f(u0, v0)
    GL.glVertex2f(x, y + height)
    GL.glEnd()
    _end_bound_texture(textured)


def draw_textured_nine_slice(
    texture_id: int,
    rect: Tuple[float, float, float, float],
    color: Sequence[int],
    alpha: float,
    border: float = 14.0,
    fill_center: bool = False,
) -> None:
    """Stretch a CC0 sci-fi frame without stretching its corner hardware."""
    if not texture_id:
        return
    x, y, width, height = rect
    bx = min(border, width / 3.0)
    by = min(border, height / 3.0)
    xs = (x, x + bx, x + width - bx, x + width)
    ys = (y, y + by, y + height - by, y + height)
    us = (0.0, 0.18, 0.82, 1.0)
    vs = (0.0, 0.18, 0.82, 1.0)
    textured = _begin_bound_texture(texture_id)
    gl_color(color, alpha)
    GL.glBegin(GL.GL_QUADS)
    for row in range(3):
        for column in range(3):
            if not fill_center and row == 1 and column == 1:
                continue
            x0, x1 = xs[column], xs[column + 1]
            y0, y1 = ys[row], ys[row + 1]
            u0, u1 = us[column], us[column + 1]
            v0, v1 = vs[row], vs[row + 1]
            GL.glTexCoord2f(u0, v1)
            GL.glVertex2f(x0, y0)
            GL.glTexCoord2f(u1, v1)
            GL.glVertex2f(x1, y0)
            GL.glTexCoord2f(u1, v0)
            GL.glVertex2f(x1, y1)
            GL.glTexCoord2f(u0, v0)
            GL.glVertex2f(x0, y1)
    GL.glEnd()
    _end_bound_texture(textured)


def draw_textured_sprite_3d(
    texture_id: int,
    x: float,
    y: float,
    z: float,
    width: float,
    height: float,
    color: Sequence[int],
    alpha: float,
    rotation: float = 0.0,
) -> None:
    if not texture_id or alpha <= 0.0:
        return
    GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glDisable(GL.GL_LIGHTING)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDepthMask(GL.GL_FALSE)
    textured = _begin_bound_texture(texture_id)
    GL.glPushMatrix()
    GL.glTranslatef(x, y, z)
    GL.glRotatef(rotation, 0.0, 0.0, 1.0)
    gl_color(color, alpha)
    GL.glBegin(GL.GL_QUADS)
    GL.glTexCoord2f(0.0, 0.0)
    GL.glVertex3f(-width / 2.0, -height / 2.0, 0.0)
    GL.glTexCoord2f(1.0, 0.0)
    GL.glVertex3f(width / 2.0, -height / 2.0, 0.0)
    GL.glTexCoord2f(1.0, 1.0)
    GL.glVertex3f(width / 2.0, height / 2.0, 0.0)
    GL.glTexCoord2f(0.0, 1.0)
    GL.glVertex3f(-width / 2.0, height / 2.0, 0.0)
    GL.glEnd()
    GL.glPopMatrix()
    _end_bound_texture(textured)
    GL.glPopAttrib()


def draw_textured_sprite_batch_3d(
    texture_id: int,
    sprites: Sequence[
        Tuple[float, float, float, float, float, Sequence[int], float, float]
    ],
) -> None:
    """Draw many camera-facing sprites with one texture bind and state change."""
    if not texture_id or not sprites:
        return
    GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glDisable(GL.GL_LIGHTING)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDepthMask(GL.GL_FALSE)
    textured = _begin_bound_texture(texture_id)
    GL.glBegin(GL.GL_QUADS)
    for x, y, z, width, height, color, alpha, rotation in sprites:
        if alpha <= 0.0:
            continue
        angle = math.radians(rotation)
        cosine, sine = math.cos(angle), math.sin(angle)
        half_width, half_height = width / 2.0, height / 2.0
        corners = (
            (-half_width, -half_height, 0.0, 0.0),
            (half_width, -half_height, 1.0, 0.0),
            (half_width, half_height, 1.0, 1.0),
            (-half_width, half_height, 0.0, 1.0),
        )
        gl_color(color, alpha)
        for local_x, local_y, u, v in corners:
            rotated_x = local_x * cosine - local_y * sine
            rotated_y = local_x * sine + local_y * cosine
            GL.glTexCoord2f(u, v)
            GL.glVertex3f(x + rotated_x, y + rotated_y, z)
    GL.glEnd()
    _end_bound_texture(textured)
    GL.glPopAttrib()


class GLText:
    def __init__(self, asset_dir: Path = ASSET_DIR) -> None:
        pygame.font.init()
        future_font = asset_dir / "KenneyFuture.ttf"
        future_narrow_font = asset_dir / "KenneyFutureNarrow.ttf"

        def interface_font(size: int, narrow: bool = False) -> Any:
            selected = future_narrow_font if narrow and future_narrow_font.is_file() else future_font
            if selected.is_file():
                return pygame.font.Font(str(selected), size)
            return pygame.font.SysFont("dejavusansmono", size)

        self.fonts = {
            "tiny": interface_font(13, True),
            "small": interface_font(16, True),
            "normal": interface_font(19, True),
            "heading": interface_font(24),
            "title": interface_font(36),
            "portrait": pygame.font.SysFont("dejavusans", 22, bold=True),
        }
        self.cache: OrderedDict[Tuple[str, Tuple[int, int, int], str], Tuple[int, int, int]] = OrderedDict()
        self.cache_limit = 700

    def _texture(self, text: str, color: Tuple[int, int, int], size: str) -> Tuple[int, int, int]:
        key = (text, color, size)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        surface = self.fonts[size].render(text, True, color)
        # OpenGL treats v=0 as the texture's lower edge. Flip Pygame's
        # top-down rows during upload, then map v=1 to the quad's top edge.
        data = pygame.image.tostring(surface, "RGBA", True)
        width, height = surface.get_size()
        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            data,
        )
        self.cache[key] = (texture, width, height)
        if len(self.cache) > self.cache_limit:
            _old_key, (old_texture, _width, _height) = self.cache.popitem(last=False)
            GL.glDeleteTextures([old_texture])
        return texture, width, height

    def measure(self, text: str, size: str = "normal") -> Tuple[int, int]:
        return self.fonts[size].size(text)

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int] = WHITE,
        size: str = "normal",
        center: bool = False,
    ) -> None:
        texture, width, height = self._texture(str(text), color, size)
        if center:
            x -= width / 2.0
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 1.0)
        GL.glVertex2f(x, y)
        GL.glTexCoord2f(1.0, 1.0)
        GL.glVertex2f(x + width, y)
        GL.glTexCoord2f(1.0, 0.0)
        GL.glVertex2f(x + width, y + height)
        GL.glTexCoord2f(0.0, 0.0)
        GL.glVertex2f(x, y + height)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)

    def wrap(self, text: str, max_width: int, size: str = "normal") -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self.measure(candidate, size)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def wrapped(
        self,
        text: str,
        x: float,
        y: float,
        max_width: int,
        color: Tuple[int, int, int] = WHITE,
        size: str = "normal",
        max_lines: Optional[int] = None,
        line_height: Optional[int] = None,
    ) -> float:
        lines = self.wrap(text, max_width, size)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            while lines and self.measure(lines[-1] + "...", size)[0] > max_width:
                words = lines[-1].split()
                if len(words) <= 1:
                    break
                lines[-1] = " ".join(words[:-1])
            if lines:
                lines[-1] += "..."
        step = line_height or self.measure("Ag", size)[1] + 3
        for line in lines:
            self.draw(line, x, y, color, size)
            y += step
        return y

    def cleanup(self) -> None:
        if self.cache:
            GL.glDeleteTextures([value[0] for value in self.cache.values()])
            self.cache.clear()


def initialize_optional_audio(enabled: bool) -> Tuple[bool, str]:
    """Open the dedicated chip-voice mixer, or return a silent fallback."""
    if not enabled:
        return False, "disabled"
    try:
        desired = (AUDIO_SAMPLE_RATE, -16, AUDIO_CHANNELS)
        pygame.mixer.pre_init(AUDIO_SAMPLE_RATE, -16, AUDIO_CHANNELS, 512)
        initialized = pygame.mixer.get_init()
        if initialized is not None and initialized != desired:
            # Sound(buffer=...) is interpreted in the current mixer format.
            # Reopen a mismatched embedding so chip timing, signedness, and
            # stereo panning cannot silently play at half/double speed.
            pygame.mixer.quit()
            initialized = None
        if initialized is None:
            pygame.mixer.init(AUDIO_SAMPLE_RATE, -16, AUDIO_CHANNELS, 512)
        initialized = pygame.mixer.get_init()
        if initialized != desired:
            if initialized is not None:
                pygame.mixer.quit()
            return False, f"unsupported mixer format: {initialized!r}"
        return True, ""
    except pygame.error as error:
        return False, str(error)


class GameApp:
    def __init__(
        self,
        seed: Optional[int] = None,
        player_name: Optional[str] = None,
        audio_enabled: bool = True,
        start_muted: bool = False,
    ) -> None:
        audio_available, audio_error = initialize_optional_audio(audio_enabled)
        pygame.display.init()
        pygame.font.init()
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        flags = pygame.DOUBLEBUF | pygame.OPENGL
        try:
            pygame.display.set_mode((WIDTH, HEIGHT), flags, vsync=1)
        except (TypeError, pygame.error):
            pygame.display.set_mode((WIDTH, HEIGHT), flags)
        pygame.display.set_caption(TITLE)
        self._init_gl()
        self.materials = GLTextureBank(ASSET_DIR)
        self.clock = pygame.time.Clock()
        self.seed = seed
        initial_name = DEFAULT_PLAYER_NAME if player_name is None else player_name
        self.player_name = normalize_player_name(initial_name)
        self.name_entry_active = player_name is None
        self.name_entry_text = self.player_name
        self.name_entry_error = ""
        self.name_replace_on_type = True
        self.state = GameState(seed, self.player_name, paced_rounds=True)
        self.audio = AudioDirector(seed, audio_available, start_muted, audio_error)
        self.text = GLText(ASSET_DIR)
        self.regions: List[HitRegion] = []
        self.trade_edit_side = "offer"
        self.trade_quantity_text = {"offer": "1", "request": "1"}
        self.trade_replace_on_type = True
        # None follows the newest entry. An integer pins a historical round even
        # while autonomous survivor rounds continue to arrive.
        self.activity_history_index: Optional[int] = None
        self.next_participant_action_time: Optional[float] = None
        # The final NPC-to-player boundary receives the same full beat as every
        # other participant handoff. The round is already archived while this
        # presentation-side lock counts down, so no simulation mutation waits.
        self.player_turn_cooldown = False
        self.running = True
        self.presentation_seed = (
            presentation_word(seed, PRESENTATION_SALT)
            if seed is not None
            else random.SystemRandom().randrange(0x1_0000_0000)
        )
        self.victory_cycle = 0
        self.victory_started_at: Optional[float] = None
        self.victory_signature: Optional[Tuple[int, int]] = None
        self._init_portrait_effects(seed)
        self._init_victory_effects()

    def _init_gl(self) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glShadeModel(GL.GL_SMOOTH)

    def restart(self) -> None:
        self.state = GameState(self.seed, self.player_name, paced_rounds=True)
        self.audio.reset(self.seed)
        self.trade_edit_side = "offer"
        self.trade_quantity_text = {"offer": "1", "request": "1"}
        self.trade_replace_on_type = True
        self.activity_history_index = None
        self.next_participant_action_time = None
        self.player_turn_cooldown = False
        self.victory_cycle += 1
        self.victory_started_at = None
        self.victory_signature = None
        self._init_victory_effects()

    def _start_with_player_name(self, value: str) -> bool:
        try:
            accepted_name = normalize_player_name(value)
        except (TypeError, ValueError) as error:
            self.name_entry_error = str(error)
            return False
        self.player_name = accepted_name
        self.name_entry_text = accepted_name
        self.name_entry_error = ""
        self.name_replace_on_type = True
        self.name_entry_active = False
        self.restart()
        return True

    def _cancel_player_name_entry(self) -> None:
        self._start_with_player_name(DEFAULT_PLAYER_NAME)

    def _set_name_entry_text(self, value: str) -> None:
        filtered = "".join(
            character for character in value if character in PLAYER_NAME_ALLOWED_CHARS
        )[:MAX_PLAYER_NAME_LENGTH]
        self.name_entry_text = filtered
        self.name_entry_error = ""

    def _handle_name_key(self, event: Any) -> None:
        key = event.key
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._start_with_player_name(self.name_entry_text)
            return
        if key == pygame.K_ESCAPE:
            self._cancel_player_name_entry()
            return
        if key == pygame.K_F5:
            self.name_entry_text = DEFAULT_PLAYER_NAME
            self.name_entry_error = ""
            self.name_replace_on_type = True
            return
        if key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            current = "" if self.name_replace_on_type else self.name_entry_text[:-1]
            self.name_replace_on_type = False
            self._set_name_entry_text(current)
            return
        character = getattr(event, "unicode", "")
        if len(character) == 1 and character in PLAYER_NAME_ALLOWED_CHARS:
            current = "" if self.name_replace_on_type else self.name_entry_text
            self.name_replace_on_type = False
            self._set_name_entry_text(current + character)

    def _has_live_activity(self) -> bool:
        return bool(
            self.state.phase == "resolving_npcs"
            and self.state.last_round_events
            and (
                not self.state.round_history
                or self.state.round_history[-1].round_number < self.state.turn
            )
        )

    def _activity_record_count(self) -> int:
        return len(self.state.round_history) + int(self._has_live_activity())

    def _selected_activity_record(self) -> Optional[RoundRecord]:
        if self.activity_history_index is None:
            if self._has_live_activity():
                return RoundRecord(
                    self.state.turn,
                    tuple(self.state.last_round_events),
                    tuple(self.state.round_notices),
                )
            if not self.state.round_history:
                return None
            return self.state.round_history[-1]
        if not self.state.round_history:
            return None
        self.activity_history_index = int(
            clamp(self.activity_history_index, 0, len(self.state.round_history) - 1)
        )
        return self.state.round_history[self.activity_history_index]

    def _activity_record_index(self) -> Optional[int]:
        total = self._activity_record_count()
        if total == 0:
            return None
        if self.activity_history_index is None:
            return total - 1
        return int(clamp(self.activity_history_index, 0, len(self.state.round_history) - 1))

    def _scroll_activity(self, direction: int) -> None:
        """Move -1 toward older rounds and +1 toward newer rounds."""
        current = self._activity_record_index()
        if current is None or direction == 0:
            return
        total = self._activity_record_count()
        next_index = int(clamp(current + direction, 0, total - 1))
        self.activity_history_index = (
            None if next_index == total - 1 else next_index
        )

    def _player_controls_locked(self) -> bool:
        """Keep player mutations out of every one-second handoff window."""
        return self.state.phase == "resolving_npcs" or self.player_turn_cooldown

    def _update_action_pacing(self, now: float) -> None:
        """Advance at most one participant action after a visible one-second beat."""
        if self.player_turn_cooldown:
            if self.next_participant_action_time is None:
                self.next_participant_action_time = now + ACTION_MOMENT_SECONDS
                return
            if now + 1e-9 < self.next_participant_action_time:
                return
            self.player_turn_cooldown = False
            self.next_participant_action_time = None
            return
        waiting = self.state.phase == "resolving_npcs" or (
            self.state.player.eliminated
            and not self.state.is_ending
            and self.state.phase == "autonomous"
        )
        if not waiting:
            self.next_participant_action_time = None
            return
        if self.next_participant_action_time is None:
            self.next_participant_action_time = now + ACTION_MOMENT_SECONDS
            return
        if now + 1e-9 < self.next_participant_action_time:
            return
        if self.state.phase == "resolving_npcs":
            self.state.advance_resolution_moment()
        else:
            # A paced autonomous round begins with its first NPC action at this
            # due moment; later NPCs receive their own one-second beats.
            self.state.advance_autonomous_round()
        waiting_after = self.state.phase == "resolving_npcs" or (
            self.state.player.eliminated
            and not self.state.is_ending
            and self.state.phase == "autonomous"
        )
        if waiting_after:
            self.next_participant_action_time = now + ACTION_MOMENT_SECONDS
        elif self.state.phase == "awaiting_player" and not self.state.is_ending:
            # NPC 4 -> player is a real handoff too. Archive immediately, then
            # hold controls for one visible beat before accepting a new round.
            self.player_turn_cooldown = True
            self.next_participant_action_time = now + ACTION_MOMENT_SECONDS
        else:
            self.next_participant_action_time = None

    def run(self, capture_path: Optional[str] = None) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)
                elif event.type == pygame.MOUSEWHEEL:
                    if event.y:
                        self._scroll_activity(-1 if event.y > 0 else 1)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._handle_click(event.pos)
            now = time.monotonic()
            self._update_action_pacing(now)
            self.audio.update(
                self.state,
                now,
                allow_idle=not self.name_entry_active and capture_path is None,
            )
            self.render(now)
            if capture_path is not None:
                # Some Mesa software-renderer paths finish a large batch of
                # newly created text textures over several frames (notably the
                # complete ending archive). Cycle both buffers while warming so
                # --capture records the same complete UI seen during play.
                for warm_frame in range(1, 4):
                    pygame.display.flip()
                    self.render(now + warm_frame / FPS)
                GL.glFinish()
                self._capture_frame(capture_path)
                self.running = False
            pygame.display.flip()
            self.clock.tick(FPS)
        self.audio.cleanup()
        self.text.cleanup()
        self.materials.cleanup()
        pygame.quit()

    def _capture_frame(self, path: str) -> None:
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        previous_read_buffer = int(GL.glGetIntegerv(GL.GL_READ_BUFFER))
        # render() composes into the double-buffered back surface. Selecting it
        # explicitly avoids stale or partial front-buffer screenshots.
        GL.glReadBuffer(GL.GL_BACK)
        pixels = GL.glReadPixels(0, 0, WIDTH, HEIGHT, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        GL.glReadBuffer(previous_read_buffer)
        surface = pygame.image.fromstring(bytes(pixels), (WIDTH, HEIGHT), "RGBA", True)
        pygame.image.save(surface, path)

    def _open_trade(self) -> None:
        if not self.state.begin_trade():
            return
        draft = self.state.trade_draft
        self.trade_edit_side = "offer"
        self.trade_quantity_text = {
            "offer": str(draft.offered_quantity),
            "request": str(draft.requested_quantity),
        }
        self.trade_replace_on_type = True

    def _focus_trade_quantity(self, side: str) -> None:
        self.trade_edit_side = side
        self.trade_replace_on_type = True

    def _set_trade_quantity_from_text(self, side: str, text: str) -> None:
        if self.state.trade_draft is None:
            return
        digits = "".join(character for character in text if character in "0123456789")[:7]
        if not digits:
            self.trade_quantity_text[side] = ""
            self.state.set_trade_quantity(side, 1)
            return
        value = int(digits)
        value = int(clamp(value, 1, INVENTORY_CAP))
        self.trade_quantity_text[side] = str(value)
        self.state.set_trade_quantity(side, value)

    def _adjust_trade_quantity(self, side: str, amount: int) -> None:
        if self.state.trade_draft is None:
            return
        self.state.adjust_trade_quantity(side, amount)
        draft = self.state.trade_draft
        value = draft.offered_quantity if side == "offer" else draft.requested_quantity
        self.trade_quantity_text[side] = str(value)
        self._focus_trade_quantity(side)

    def _confirm_trade(self) -> None:
        if self.state.trade_draft is None:
            return
        if not self.trade_quantity_text["offer"] or not self.trade_quantity_text["request"]:
            self.state.message = "Enter a whole-number quantity on both sides of the proposal."
            return
        self.state.confirm_trade()
        self.trade_replace_on_type = True

    def _handle_trade_key(self, event: Any) -> None:
        key = event.key
        if key == pygame.K_ESCAPE:
            self.state.cancel_trade()
            return
        if key == pygame.K_F5:
            self.restart()
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._confirm_trade()
            return
        if key == pygame.K_TAB:
            side = "request" if self.trade_edit_side == "offer" else "offer"
            self._focus_trade_quantity(side)
            return
        if key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            current = self.trade_quantity_text[self.trade_edit_side]
            if self.trade_replace_on_type:
                current = ""
            else:
                current = current[:-1]
            self.trade_replace_on_type = False
            self._set_trade_quantity_from_text(self.trade_edit_side, current)
            return
        if key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self._adjust_trade_quantity(self.trade_edit_side, 1)
            return
        if key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
            self._adjust_trade_quantity(self.trade_edit_side, -1)
            return
        character = getattr(event, "unicode", "")
        if len(character) == 1 and character in "0123456789":
            current = "" if self.trade_replace_on_type else self.trade_quantity_text[self.trade_edit_side]
            self.trade_replace_on_type = False
            self._set_trade_quantity_from_text(self.trade_edit_side, current + character)

    def _handle_key(self, event: Any) -> None:
        key = event.key
        if key == pygame.K_m:
            self.audio.toggle_mute()
            return
        if self.name_entry_active:
            self._handle_name_key(event)
            return
        if key == pygame.K_PAGEUP:
            self._scroll_activity(-1)
            return
        if key == pygame.K_PAGEDOWN:
            self._scroll_activity(1)
            return
        if self.state.trade_draft is not None:
            self._handle_trade_key(event)
            return
        if self.state.player.eliminated and not self.state.is_ending:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_F5:
                self.restart()
            return
        if self.state.is_game_over:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_F5 or key in (pygame.K_RETURN, pygame.K_SPACE):
                self.restart()
            return
        if self.state.is_ending:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_F5 or key in (pygame.K_RETURN, pygame.K_SPACE):
                self.restart()
            return
        if self._player_controls_locked():
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_F5:
                self.restart()
            return
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_F5:
            self.restart()
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self.state.select_npc(key - pygame.K_1)
        elif key in (pygame.K_F1, pygame.K_F2, pygame.K_F3) and not self.state.is_ending:
            self.state.choose_story((pygame.K_F1, pygame.K_F2, pygame.K_F3).index(key))
        elif key == pygame.K_TAB:
            self.state.cycle_item()
        elif key == pygame.K_t:
            self.state.interact("talk")
        elif key == pygame.K_l:
            self.state.interact("listen")
        elif key == pygame.K_c:
            self.state.interact("compliment")
        elif key == pygame.K_v:
            self.state.interact("flirt")
        elif key == pygame.K_a:
            self.state.interact("antagonize")
        elif key == pygame.K_f:
            self.state.interact("fight")
        elif key == pygame.K_g:
            self.state.interact("give")
        elif key == pygame.K_s:
            self.state.interact("steal")
        elif key == pygame.K_b:
            self._open_trade()
        elif key == pygame.K_r:
            self.state.interact("reflect")
        elif key == pygame.K_u:
            self.state.interact("use")
        elif key == pygame.K_x:
            self.state.interact("rest")

    def _handle_click(self, position: Tuple[int, int]) -> None:
        if self.name_entry_active:
            for region in reversed(self.regions):
                if not region.contains(position):
                    continue
                if region.kind == "name_confirm":
                    self._start_with_player_name(self.name_entry_text)
                elif region.kind == "name_cancel":
                    self._cancel_player_name_entry()
                elif region.kind == "name_field":
                    self.name_replace_on_type = False
                return
            return
        for region in reversed(self.regions):
            if region.contains(position) and region.kind == "activity_scroll":
                self._scroll_activity(region.value)
                return
        if self.state.player.eliminated and not self.state.is_ending:
            return
        if self.state.is_game_over:
            for region in reversed(self.regions):
                if region.contains(position) and region.kind == "restart":
                    self.restart()
                    return
            return
        if self.state.is_ending:
            for region in reversed(self.regions):
                if region.contains(position) and region.kind == "restart":
                    self.restart()
                    return
            return
        if self._player_controls_locked():
            return
        if self.state.trade_draft is not None:
            for region in reversed(self.regions):
                if not region.contains(position):
                    continue
                if region.kind == "trade_item":
                    side, item_key = region.value
                    self.state.set_trade_item(side, item_key)
                elif region.kind == "trade_focus":
                    self._focus_trade_quantity(region.value)
                elif region.kind == "trade_adjust":
                    side, amount = region.value
                    self._adjust_trade_quantity(side, amount)
                elif region.kind == "trade_confirm":
                    self._confirm_trade()
                elif region.kind == "trade_cancel":
                    self.state.cancel_trade()
                return
            return
        if position[1] < TOP_HEIGHT:
            self.state.select_npc(min(3, int(position[0] / (WIDTH / 4))))
            return
        for region in reversed(self.regions):
            if not region.contains(position):
                continue
            if region.kind == "item":
                self.state.selected_item = region.value
            elif region.kind == "action":
                if region.value == "trade":
                    self._open_trade()
                else:
                    self.state.interact(region.value)
            elif region.kind == "story":
                self.state.choose_story(region.value)
            elif region.kind == "restart":
                self.restart()
            return

    def render(self, now: float) -> None:
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glViewport(0, 0, WIDTH, HEIGHT)
        GL.glClearColor(INK[0] / 255.0, INK[1] / 255.0, INK[2] / 255.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.state.is_game_over:
            self._begin_2d()
            self.regions = []
            self._draw_game_over_overlay(now)
        else:
            self._draw_portraits(now)
            self._begin_2d()
            self.regions = []
            self._draw_lower_interface()
            # Draw labels and the release/countdown strip over the HUD base;
            # the strip itself now lives below the avatar-effect viewports.
            self._draw_portrait_overlays(now)
            if self.state.trade_draft is not None:
                self.regions = []
                self._draw_trade_builder()
            if self.name_entry_active:
                self.regions = []
                self._draw_name_entry_overlay()
        self._draw_watermark()

    def _draw_watermark(self) -> None:
        credit_width, credit_height = self.text.measure(CREDIT_WATERMARK, "small")
        if self.state.trade_draft is not None:
            x = WIDTH - credit_width - 28.0
            y = TOP_HEIGHT + 15.0
        else:
            x = (WIDTH - credit_width) / 2.0
            y = HEIGHT - credit_height - 3.0
        watermark_rect = (x - 8, y - 2, credit_width + 16, credit_height + 4)
        draw_gradient_rect(*watermark_rect, (9, 18, 22), (3, 7, 10), 0.94)
        draw_textured_nine_slice(
            self.materials.get("ui_panel"),
            watermark_rect,
            CYAN,
            0.11,
            5.0,
        )
        draw_outline(*watermark_rect, CYAN, 1.0)
        self.text.draw(CREDIT_WATERMARK, x, y, mix_color(CYAN, WHITE, 0.35), "small")
        audio_label = self.audio.status_label
        audio_width, audio_height = self.text.measure(audio_label, "tiny")
        audio_rect = (8.0, HEIGHT - audio_height - 6.0, audio_width + 14.0, audio_height + 4.0)
        draw_rect(*audio_rect, (3, 7, 10), 0.90)
        draw_outline(*audio_rect, GOLD if self.audio.available and not self.audio.muted else LINE, 1.0)
        self.text.draw(
            audio_label,
            audio_rect[0] + 7,
            audio_rect[1] + 2,
            GOLD if self.audio.available and not self.audio.muted else MUTED,
            "tiny",
        )

    def _fit_text(self, value: str, size: str, max_width: float) -> str:
        """Ellipsize a single display line to a measured pixel boundary."""
        if self.text.measure(value, size)[0] <= max_width:
            return value
        suffix = "..."
        candidate = value.rstrip()
        while candidate and self.text.measure(candidate + suffix, size)[0] > max_width:
            candidate = candidate[:-1].rstrip()
        return (candidate + suffix) if candidate else suffix

    def _fitting_font(
        self,
        value: str,
        sizes: Sequence[str],
        max_width: float,
    ) -> str:
        for size in sizes:
            if self.text.measure(value, size)[0] <= max_width:
                return size
        return sizes[-1]

    def _draw_portraits(self, now: float) -> None:
        portrait_width = WIDTH // 4
        viewport_y = HEIGHT - TOP_HEIGHT
        GL.glEnable(GL.GL_SCISSOR_TEST)
        for index, npc in enumerate(self.state.npcs):
            viewport_x = index * portrait_width
            GL.glViewport(viewport_x, viewport_y, portrait_width, TOP_HEIGHT)
            GL.glScissor(viewport_x, viewport_y, portrait_width, TOP_HEIGHT)
            accent = npc.profile.accent
            background = mix_color((16, 20, 22), accent, 0.16)
            style = spiritual_portrait_style(npc.statuses["spiritual"])
            if style.dim_window:
                background = tuple(int(channel * 0.48) for channel in background)
            GL.glClearColor(
                background[0] / 255.0,
                background[1] / 255.0,
                background[2] / 255.0,
                1.0,
            )
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GLU.gluPerspective(34.0, portrait_width / TOP_HEIGHT, 0.1, 100.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GL.glTranslatef(0.0, -0.05, -7.3)

            self._draw_portrait_background_effect(index, now)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            light_profile = self._configure_portrait_lights(index, npc.statuses["spiritual"], now)
            self._draw_portrait_model(npc, index, now)
            if light_profile.dim_overlay_alpha > 0.0:
                self._draw_portrait_dim_veil(light_profile.dim_overlay_alpha)
            GL.glDisable(GL.GL_LIGHT1)
        GL.glDisable(GL.GL_SCISSOR_TEST)

    def _configure_portrait_lights(
        self,
        portrait_index: int,
        spirituality: int,
        now: float,
    ) -> PortraitLightProfile:
        """Apply one isolated spirituality-aware fixed-function light rig."""
        pulse = 0.96 + math.sin(now * 3.1 + portrait_index * 1.37) * 0.10
        profile = portrait_light_profile(portrait_index, spirituality, pulse)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (-2.5, 4.0, 5.0, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, profile.ambient)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, profile.diffuse)
        GL.glLightfv(
            GL.GL_LIGHT0,
            GL.GL_SPECULAR,
            tuple(component * 0.24 for component in profile.diffuse[:3]) + (1.0,),
        )
        GL.glDisable(GL.GL_LIGHT1)
        if any(component > 0.001 for component in profile.secondary_diffuse[:3]):
            GL.glEnable(GL.GL_LIGHT1)
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, (0.0, 2.35, 2.8, 1.0))
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, profile.secondary_diffuse)
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_SPECULAR, profile.secondary_diffuse)
            GL.glLightf(GL.GL_LIGHT1, GL.GL_CONSTANT_ATTENUATION, 1.0)
            GL.glLightf(GL.GL_LIGHT1, GL.GL_LINEAR_ATTENUATION, 0.16)
            GL.glLightf(GL.GL_LIGHT1, GL.GL_QUADRATIC_ATTENUATION, 0.035)
        return profile

    def _draw_portrait_dim_veil(self, alpha: float) -> None:
        """Darken only the currently scissored 3-D avatar window."""
        GL.glPushAttrib(
            GL.GL_ENABLE_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_CURRENT_BIT
        )
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        gl_color((2, 3, 5), float(clamp(alpha, 0.0, 0.75)))
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-1.0, -1.0, 0.0)
        GL.glVertex3f(1.0, -1.0, 0.0)
        GL.glVertex3f(1.0, 1.0, 0.0)
        GL.glVertex3f(-1.0, 1.0, 0.0)
        GL.glEnd()
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopAttrib()

    def _init_portrait_effects(self, seed: Optional[int]) -> None:
        """Precompute bounded effect fields without touching the simulation RNG."""
        effect_seed = 0x5F17A1 if seed is None else (int(seed) ^ 0x5F17A1)
        rng = random.Random(effect_seed)

        firework_particles = []
        for burst in range(7):
            center_x = rng.uniform(-1.34, 1.34)
            center_y = rng.uniform(-0.35, 1.72)
            burst_offset = (burst / 7.0 + rng.uniform(-0.025, 0.025)) % 1.0
            color_index = burst % 7
            ray_count = 17
            angle_offset = rng.uniform(0.0, math.tau)
            for ray in range(ray_count):
                angle = angle_offset + math.tau * ray / ray_count + rng.uniform(-0.055, 0.055)
                firework_particles.append(
                    (
                        center_x,
                        center_y,
                        burst_offset,
                        angle,
                        rng.uniform(0.72, 1.22),
                        color_index,
                        rng.uniform(0.75, 1.20),
                    )
                )
        self._firework_particles = tuple(firework_particles)

        fog_anchors = (
            (-1.34, 1.42), (1.34, 1.42), (-1.38, 0.42), (1.38, 0.42),
            (-1.36, -0.72), (1.36, -0.72), (-0.92, 1.78), (0.92, 1.78),
            (-1.10, -1.68), (1.10, -1.68), (-0.58, 1.66), (0.58, 1.66),
            (-1.46, -1.38), (1.46, -1.38), (0.0, 1.86),
        )
        fog_blobs = []
        for blob in range(15):
            anchor_x, anchor_y = fog_anchors[blob]
            fog_blobs.append(
                (
                    anchor_x + rng.uniform(-0.10, 0.10),
                    anchor_y + rng.uniform(-0.10, 0.10),
                    rng.uniform(0.54, 1.02),
                    rng.uniform(0.10, 0.34),
                    rng.uniform(0.07, 0.25),
                    rng.uniform(0.0, math.tau),
                    rng.uniform(0.16, 0.34),
                    blob % 3,
                )
            )
        self._fog_blobs = tuple(fog_blobs)

        rain_drops = []
        for _ in range(54):
            rain_drops.append(
                (
                    rng.uniform(-1.72, 1.72),
                    rng.uniform(0.0, 4.8),
                    rng.uniform(0.22, 0.48),
                    rng.uniform(1.28, 2.35),
                    rng.uniform(0.035, 0.105),
                )
            )
        self._rain_drops = tuple(rain_drops)

        lightning_segments = []
        nodes = []
        bolt_x = -1.18
        for node in range(14):
            bolt_y = 2.18 - node * (4.30 / 13.0)
            nodes.append((bolt_x + rng.uniform(-0.18, 0.18), bolt_y))
            bolt_x += rng.uniform(-0.09, 0.13)
        for node in range(len(nodes) - 1):
            lightning_segments.append((nodes[node][0], nodes[node][1], nodes[node + 1][0], nodes[node + 1][1], 1.0))
            if node in (3, 6, 9):
                direction = 1.0 if node != 6 else -1.0
                branch_x = nodes[node][0]
                branch_y = nodes[node][1]
                for branch in range(3):
                    next_x = branch_x + direction * rng.uniform(0.19, 0.34)
                    next_y = branch_y - rng.uniform(0.16, 0.31)
                    lightning_segments.append((branch_x, branch_y, next_x, next_y, 0.62))
                    branch_x, branch_y = next_x, next_y
        self._lightning_segments = tuple(lightning_segments)

        raster_cells = []
        columns, rows = 18, 14
        for row in range(rows):
            for column in range(columns):
                raster_cells.append(
                    (
                        -1.68 + column * (3.36 / columns),
                        -2.18 + row * (4.36 / rows),
                        rng.uniform(0.055, 0.125),
                        rng.uniform(0.0, math.tau),
                        rng.uniform(0.72, 1.82),
                        rng.randrange(5),
                        rng.uniform(0.32, 0.82),
                    )
                )
        self._raster_cells = tuple(raster_cells)

    def _init_victory_effects(self) -> None:
        """Precompute a bounded finale field without consuming GameState RNG."""
        self.player_victory_style = player_victory_effect_index(
            self.presentation_seed,
            self.player_name,
            self.victory_cycle,
        )
        word = _mix_u32(
            self.presentation_seed
            ^ _stable_text_word(self.player_name)
            ^ (self.victory_cycle * 0x9E37_79B9)
        )
        self.victory_visual_word = word
        rng = random.Random(word)
        particles = []
        for particle_index in range(112):
            particles.append(
                (
                    rng.uniform(0.0, math.tau),
                    rng.random(),
                    rng.uniform(0.55, 1.55),
                    rng.uniform(0.45, 1.90),
                    rng.uniform(-1.0, 1.0),
                    rng.uniform(0.025, 0.085),
                    particle_index % 7,
                )
            )
        self._victory_particles = tuple(particles)

    def _victory_elapsed(self, now: float) -> float:
        winner = self.state.last_survivor
        if winner is None:
            return 0.0
        participant_index = self.state.participants.index(winner)
        signature = (self.state.turn, participant_index)
        if signature != self.victory_signature or self.victory_started_at is None:
            self.victory_signature = signature
            self.victory_started_at = float(now)
        return max(0.0, float(now) - self.victory_started_at)

    def _draw_portrait_background_effect(self, index: int, now: float) -> None:
        """Render one isolated effect field before its opaque portrait model."""
        attribs = (
            GL.GL_ENABLE_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_LINE_BIT
            | GL.GL_POINT_BIT
            | GL.GL_CURRENT_BIT
        )
        GL.glPushAttrib(attribs)
        GL.glPushMatrix()
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        if index == 0:
            self._draw_firework_field(now)
        elif index == 1:
            self._draw_luminous_fog(now)
        elif index == 2:
            self._draw_black_rain_and_lightning(now)
        else:
            self._draw_raster_field(now)
        GL.glPopMatrix()
        GL.glPopAttrib()

    def _draw_firework_field(self, now: float) -> None:
        palette = (
            (255, 84, 77),
            (255, 190, 55),
            (89, 220, 255),
            (176, 102, 255),
            (86, 241, 151),
            (255, 103, 205),
            (244, 244, 226),
        )
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glLineWidth(1.55)
        GL.glBegin(GL.GL_LINES)
        for center_x, center_y, offset, angle, speed, color_index, shimmer in self._firework_particles:
            progress = (now * 0.245 + offset) % 1.0
            fade = (1.0 - progress) ** 1.55
            distance = speed * progress * 1.58
            trail_progress = max(0.0, progress - 0.065)
            trail_distance = speed * trail_progress * 1.58
            color = palette[color_index]
            gl_color(color, fade * 0.18)
            GL.glVertex3f(
                center_x + math.cos(angle) * trail_distance,
                center_y + math.sin(angle) * trail_distance - trail_progress * trail_progress * 0.72,
                -1.72,
            )
            gl_color(color, min(1.0, fade * shimmer * 0.92))
            GL.glVertex3f(
                center_x + math.cos(angle) * distance,
                center_y + math.sin(angle) * distance - progress * progress * 0.72,
                -1.72,
            )
        GL.glEnd()
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(3.2)
        GL.glBegin(GL.GL_POINTS)
        for center_x, center_y, offset, angle, speed, color_index, shimmer in self._firework_particles:
            progress = (now * 0.245 + offset) % 1.0
            fade = (1.0 - progress) ** 1.45
            distance = speed * progress * 1.58
            gl_color(palette[color_index], min(1.0, fade * shimmer))
            GL.glVertex3f(
                center_x + math.cos(angle) * distance,
                center_y + math.sin(angle) * distance - progress * progress * 0.72,
                -1.71,
            )
        GL.glEnd()
        # Sparse CC0 star sprites add a soft high-energy core without replacing
        # the crisp procedural trails.
        star_sprites = []
        for particle in self._firework_particles[::7]:
            center_x, center_y, offset, angle, speed, color_index, shimmer = particle
            progress = (now * 0.245 + offset) % 1.0
            fade = (1.0 - progress) ** 1.35
            distance = speed * progress * 1.58
            star_sprites.append(
                (
                    center_x + math.cos(angle) * distance,
                    center_y + math.sin(angle) * distance - progress * progress * 0.72,
                    -1.69,
                    0.18 + fade * 0.16,
                    0.18 + fade * 0.16,
                    palette[color_index],
                    min(0.78, fade * shimmer),
                    math.degrees(angle) + now * 18.0,
                )
            )
        draw_textured_sprite_batch_3d(self.materials.get("particle_star"), star_sprites)

    def _draw_luminous_fog(self, now: float) -> None:
        palette = (
            (70, 224, 235),
            (174, 88, 245),
            (74, 213, 146),
        )
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glBegin(GL.GL_TRIANGLES)
        for base_x, base_y, radius, drift_x, drift_y, phase, speed, color_index in self._fog_blobs:
            center_x = base_x + math.sin(now * speed + phase) * drift_x
            center_y = base_y + math.cos(now * speed * 0.73 + phase) * drift_y
            breathing_radius = radius * (0.88 + math.sin(now * speed * 1.7 + phase) * 0.12)
            color = palette[color_index]
            center_alpha = 0.155 + 0.045 * math.sin(now * speed + phase)
            for segment in range(18):
                angle_a = math.tau * segment / 18.0
                angle_b = math.tau * (segment + 1) / 18.0
                gl_color(color, center_alpha)
                GL.glVertex3f(center_x, center_y, -1.88)
                gl_color(color, 0.006)
                GL.glVertex3f(
                    center_x + math.cos(angle_a) * breathing_radius,
                    center_y + math.sin(angle_a) * breathing_radius * 0.72,
                    -1.88,
                )
                GL.glVertex3f(
                    center_x + math.cos(angle_b) * breathing_radius,
                    center_y + math.sin(angle_b) * breathing_radius * 0.72,
                    -1.88,
                )
        GL.glEnd()
        smoke_sprites = []
        for base_x, base_y, radius, drift_x, drift_y, phase, speed, color_index in self._fog_blobs:
            center_x = base_x + math.sin(now * speed + phase) * drift_x
            center_y = base_y + math.cos(now * speed * 0.73 + phase) * drift_y
            scale = radius * (1.14 + math.sin(now * speed * 1.2 + phase) * 0.08)
            smoke_sprites.append(
                (
                    center_x,
                    center_y,
                    -1.84,
                    scale * 1.55,
                    scale,
                    palette[color_index],
                    0.08,
                    math.degrees(phase) + now * speed * 7.0,
                )
            )
        draw_textured_sprite_batch_3d(self.materials.get("particle_smoke"), smoke_sprites)

    def _draw_black_rain_and_lightning(self, now: float) -> None:
        flash_phase = (now * 0.31) % 1.0
        flash = max(0.0, 1.0 - abs(flash_phase - 0.055) / 0.055)
        secondary = max(0.0, 1.0 - abs(flash_phase - 0.135) / 0.026) * 0.58
        intensity = max(flash, secondary)

        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glBegin(GL.GL_TRIANGLES)
        light_sprites = []
        for glow in range(3):
            center_x = -1.05 + glow * 0.98
            center_y = 0.72 - glow * 0.54
            radius = 1.16 - glow * 0.13
            alpha = 0.072 + intensity * (0.095 - glow * 0.014)
            for segment in range(20):
                angle_a = math.tau * segment / 20.0
                angle_b = math.tau * (segment + 1) / 20.0
                gl_color((255, 211, 56), alpha)
                GL.glVertex3f(center_x, center_y, -1.92)
                gl_color((255, 180, 30), 0.0)
                GL.glVertex3f(center_x + math.cos(angle_a) * radius, center_y + math.sin(angle_a) * radius, -1.92)
                GL.glVertex3f(center_x + math.cos(angle_b) * radius, center_y + math.sin(angle_b) * radius, -1.92)
        GL.glEnd()

        for glow in range(3):
            center_x = -1.05 + glow * 0.98
            center_y = 0.72 - glow * 0.54
            radius = 1.16 - glow * 0.13
            light_sprites.append(
                (
                    center_x,
                    center_y,
                    -1.89,
                    radius * 1.62,
                    radius * 1.62,
                    (255, 220, 74),
                    0.08 + intensity * 0.18,
                    now * (8.0 + glow * 2.0),
                )
            )
        draw_textured_sprite_batch_3d(self.materials.get("particle_light"), light_sprites)

        if intensity > 0.01:
            GL.glLineWidth(6.0)
            GL.glBegin(GL.GL_LINES)
            for x1, y1, x2, y2, branch_alpha in self._lightning_segments:
                gl_color((155, 104, 8), intensity * branch_alpha * 0.48)
                GL.glVertex3f(x1, y1, -1.76)
                GL.glVertex3f(x2, y2, -1.76)
            GL.glEnd()
            GL.glLineWidth(2.0)
            GL.glBegin(GL.GL_LINES)
            for x1, y1, x2, y2, branch_alpha in self._lightning_segments:
                gl_color((255, 235, 82), intensity * branch_alpha)
                GL.glVertex3f(x1, y1, -1.75)
                GL.glVertex3f(x2, y2, -1.75)
            GL.glEnd()

        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glLineWidth(2.35)
        GL.glBegin(GL.GL_LINES)
        for base_x, offset, length, speed, slant in self._rain_drops:
            rain_y = 2.32 - ((now * speed + offset) % 4.78)
            wind = math.sin(now * 0.37 + offset) * 0.035
            gl_color((0, 0, 0), 0.91)
            GL.glVertex3f(base_x + wind, rain_y, -1.66)
            GL.glVertex3f(base_x + wind + slant, rain_y - length, -1.66)
        GL.glEnd()

    def _draw_raster_field(self, now: float) -> None:
        palette = (
            (52, 221, 209),
            (235, 67, 136),
            (247, 190, 48),
            (105, 112, 246),
            (225, 238, 232),
        )
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glBegin(GL.GL_QUADS)
        for base_x, base_y, size, phase, speed, color_index, activity in self._raster_cells:
            signal = math.sin(now * speed + phase) * 0.5 + 0.5
            if signal < activity:
                continue
            jitter_x = math.sin(now * 4.1 + phase) * 0.018
            jitter_y = math.cos(now * 3.3 + phase) * 0.012
            half_size = size * (0.66 + signal * 0.44)
            alpha = (signal - activity) / max(0.05, 1.0 - activity) * 0.62
            gl_color(palette[color_index], alpha)
            GL.glVertex3f(base_x + jitter_x - half_size, base_y + jitter_y - half_size, -1.82)
            GL.glVertex3f(base_x + jitter_x + half_size, base_y + jitter_y - half_size, -1.82)
            GL.glVertex3f(base_x + jitter_x + half_size, base_y + jitter_y + half_size, -1.82)
            GL.glVertex3f(base_x + jitter_x - half_size, base_y + jitter_y + half_size, -1.82)

        scan_y = 2.16 - ((now * 0.78) % 4.34)
        gl_color((80, 238, 224), 0.12)
        GL.glVertex3f(-1.72, scan_y - 0.035, -1.80)
        GL.glVertex3f(1.72, scan_y - 0.035, -1.80)
        gl_color((180, 255, 242), 0.26)
        GL.glVertex3f(1.72, scan_y + 0.035, -1.80)
        GL.glVertex3f(-1.72, scan_y + 0.035, -1.80)
        GL.glEnd()
        magic_sprites = []
        for cell_index, cell in enumerate(self._raster_cells[::39]):
            base_x, base_y, size, phase, speed, color_index, activity = cell
            signal = math.sin(now * speed + phase) * 0.5 + 0.5
            if signal < activity * 0.72:
                continue
            sprite_size = 0.25 + size * 1.6
            magic_sprites.append(
                (
                    base_x + math.sin(now * 2.0 + phase) * 0.05,
                    base_y + math.cos(now * 1.7 + phase) * 0.05,
                    -1.77,
                    sprite_size,
                    sprite_size,
                    palette[color_index],
                    0.12 + signal * 0.20,
                    now * (12.0 + cell_index) + math.degrees(phase),
                )
            )
        draw_textured_sprite_batch_3d(self.materials.get("particle_magic"), magic_sprites)

    def _draw_victory_particle_field(
        self,
        elapsed: float,
        style_index: int,
        accent: Sequence[int],
    ) -> None:
        """Render one of four bounded, generated OpenGL finale fields."""
        palette = (
            tuple(accent),
            GOLD,
            CYAN,
            GREEN,
            (255, 95, 154),
            (170, 104, 255),
            WHITE,
        )
        positions = []
        for angle, phase, speed, radial, drift, size, color_index in self._victory_particles:
            if style_index == 0:  # Fifth-Wave Supernova
                progress = (elapsed * 0.38 * speed + phase) % 1.0
                theta = angle + elapsed * 0.22
                radius = progress * radial * 1.46
                previous_radius = max(0.0, radius - 0.18)
                x = math.cos(theta) * radius
                y = math.sin(theta) * radius
                previous = (math.cos(theta) * previous_radius, math.sin(theta) * previous_radius)
                alpha = (1.0 - progress) ** 1.35
            elif style_index == 1:  # Golden Glitch Fountain
                progress = (elapsed * 0.48 * speed + phase) % 1.0
                previous_progress = max(0.0, progress - 0.045)
                x = drift * (0.28 + progress * 1.18) + math.sin(angle) * 0.22
                y = -1.58 + progress * (3.40 + radial * 0.42) - progress * progress * 2.10
                previous = (
                    drift * (0.28 + previous_progress * 1.18) + math.sin(angle) * 0.22,
                    -1.58
                    + previous_progress * (3.40 + radial * 0.42)
                    - previous_progress * previous_progress * 2.10,
                )
                alpha = math.sin(progress * math.pi) ** 0.72
            elif style_index == 2:  # Cyan Victory Vortex
                progress = (elapsed * 0.32 * speed + phase) % 1.0
                theta = angle + elapsed * (0.82 + speed * 0.42) + progress * math.tau * 1.7
                radius = 0.18 + progress * radial
                x = math.cos(theta) * radius
                y = math.sin(theta) * radius * 0.76 + drift * 0.18
                previous_theta = theta - 0.18
                previous = (
                    math.cos(previous_theta) * max(0.12, radius - 0.05),
                    math.sin(previous_theta) * max(0.12, radius - 0.05) * 0.76 + drift * 0.18,
                )
                alpha = 0.25 + (1.0 - progress) * 0.72
            else:  # Pixel-Crown Overdrive
                progress = (elapsed * 0.42 * speed + phase) % 1.0
                x = drift * 1.52 + math.sin(angle + elapsed * 1.8) * 0.13
                y = -1.70 + progress * 3.55
                previous = (x, y - 0.16)
                alpha = math.sin(progress * math.pi) ** 0.66
            positions.append((x, y, previous[0], previous[1], size, color_index, alpha))

        attribs = (
            GL.GL_ENABLE_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_LINE_BIT
            | GL.GL_POINT_BIT
            | GL.GL_CURRENT_BIT
        )
        GL.glPushAttrib(attribs)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glLineWidth(1.8 if style_index != 3 else 2.5)
        GL.glBegin(GL.GL_LINES)
        for x, y, previous_x, previous_y, _size, color_index, alpha in positions:
            gl_color(palette[color_index], alpha * 0.20)
            GL.glVertex3f(previous_x, previous_y, -1.72)
            gl_color(palette[color_index], min(1.0, alpha * 0.96))
            GL.glVertex3f(x, y, -1.70)
        GL.glEnd()
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(4.0 if style_index != 3 else 5.5)
        GL.glBegin(GL.GL_POINTS)
        for x, y, _previous_x, _previous_y, _size, color_index, alpha in positions:
            gl_color(palette[color_index], min(1.0, alpha * 1.16))
            GL.glVertex3f(x, y, -1.66)
        GL.glEnd()
        # Four expanding fifth-channel rings keep the finale legible even when
        # a seeded fountain or vortex happens to cross its quietest instant.
        GL.glLineWidth(1.4)
        for ring_index in range(4):
            ring_progress = (elapsed * (0.24 + style_index * 0.025) + ring_index / 4.0) % 1.0
            ring_radius = 0.22 + ring_progress * 1.46
            gl_color(palette[(ring_index + style_index) % len(palette)], (1.0 - ring_progress) * 0.40)
            GL.glBegin(GL.GL_LINE_LOOP)
            for point in range(36):
                ring_angle = math.tau * point / 36.0
                GL.glVertex3f(
                    math.cos(ring_angle) * ring_radius,
                    math.sin(ring_angle) * ring_radius * (0.74 + style_index * 0.04),
                    -1.64,
                )
            GL.glEnd()
        sprite_texture = (
            "particle_star",
            "particle_light",
            "particle_smoke",
            "particle_magic",
        )[style_index]
        sprites = []
        for sprite_index, position in enumerate(positions[::5]):
            x, y, _previous_x, _previous_y, size, color_index, alpha = position
            sprite_size = 0.16 + size * 2.2
            sprites.append(
                (
                    x,
                    y,
                    -1.62,
                    sprite_size,
                    sprite_size,
                    palette[color_index],
                    min(0.86, alpha * 0.78),
                    elapsed * (28.0 + sprite_index % 7) + sprite_index * 19.0,
                )
            )
        draw_textured_sprite_batch_3d(self.materials.get(sprite_texture), sprites)
        GL.glPopAttrib()

    def _draw_victory_crown(
        self,
        elapsed: float,
        accent: Sequence[int],
        bounce: float,
    ) -> None:
        GL.glPushAttrib(
            GL.GL_ENABLE_BIT
            | GL.GL_CURRENT_BIT
            | GL.GL_LINE_BIT
            | GL.GL_DEPTH_BUFFER_BIT
        )
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(3.0)
        GL.glPushMatrix()
        GL.glTranslatef(0.0, 1.56 + bounce * 0.45, 0.82)
        GL.glRotatef(math.sin(elapsed * 0.9) * 14.0, 1.0, 0.0, 0.0)
        GL.glRotatef(elapsed * 38.0, 0.0, 0.0, 1.0)
        gl_color(mix_color(accent, GOLD, 0.52), 0.94)
        GL.glBegin(GL.GL_LINE_LOOP)
        for point in range(32):
            angle = math.tau * point / 32.0
            GL.glVertex3f(math.cos(angle) * 0.54, math.sin(angle) * 0.12, 0.0)
        GL.glEnd()
        GL.glBegin(GL.GL_LINE_STRIP)
        for point in range(9):
            x = -0.48 + point * 0.12
            peak = 0.24 if point % 2 else 0.04
            GL.glVertex3f(x, 0.08 + peak, 0.02)
        GL.glEnd()
        GL.glPopMatrix()
        GL.glPopAttrib()

    def _draw_victory_stage(
        self,
        now: float,
        rect: Tuple[float, float, float, float],
    ) -> None:
        winner = self.state.last_survivor
        if winner is None:
            return
        elapsed = self._victory_elapsed(now)
        x, y, width, height = rect
        viewport = (
            int(x),
            int(HEIGHT - y - height),
            max(1, int(width)),
            max(1, int(height)),
        )
        attribs = (
            GL.GL_ENABLE_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_LIGHTING_BIT
            | GL.GL_SCISSOR_BIT
            | GL.GL_VIEWPORT_BIT
            | GL.GL_CURRENT_BIT
        )
        GL.glPushAttrib(attribs)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glViewport(*viewport)
        GL.glScissor(*viewport)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(33.0, width / max(1.0, height), 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, -0.12, -7.25)
        victory_dim_alpha = 0.0

        if winner is self.state.player:
            accent = mix_color(CYAN, GOLD, 0.34)
            GL.glPushMatrix()
            GL.glScalef(1.22, 1.18, 1.0)
            self._draw_victory_particle_field(
                elapsed,
                self.player_victory_style,
                accent,
            )
            GL.glPopMatrix()
            # A generated fifth-channel core anchors the otherwise avatar-free
            # player finale in true 3-D space.
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (-2.0, 3.5, 4.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.22, 0.26, 0.28, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (0.75, 0.96, 1.0, 1.0))
            GL.glPushMatrix()
            GL.glRotatef(elapsed * 42.0, 0.3, 1.0, 0.2)
            pulse = 0.76 + math.sin(elapsed * 3.4) * 0.08
            draw_sphere(pulse, accent, 18, 36, self.materials.get("particle_magic"))
            GL.glPopMatrix()
        else:
            npc_index = self.state.npcs.index(winner)
            frame = victory_frame(self.seed, npc_index, elapsed)
            style = (self.player_victory_style + npc_index + 1) % len(PLAYER_VICTORY_EFFECTS)
            self._draw_victory_particle_field(elapsed, style, winner.profile.accent)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            light_profile = self._configure_portrait_lights(
                npc_index,
                winner.statuses["spiritual"],
                now,
            )
            victory_dim_alpha = light_profile.dim_overlay_alpha
            self._draw_portrait_model(winner, npc_index, now, frame)
            self._draw_victory_crown(elapsed, winner.profile.accent, frame.bounce)
            GL.glDisable(GL.GL_LIGHT1)

        if victory_dim_alpha > 0.0:
            self._draw_portrait_dim_veil(victory_dim_alpha)

        GL.glPopAttrib()
        self._begin_2d()

    def _draw_portrait_model(
        self,
        npc: Participant,
        index: int,
        now: float,
        victory: Optional[VictoryFrame] = None,
    ) -> None:
        phase = index * 1.37
        skin_texture = self.materials.get("skin")
        cloth_texture = self.materials.get("cloth")
        health = npc.statuses["physical"] / 100.0
        emotional = npc.statuses["emotional"] / 100.0
        cognitive = npc.statuses["cognitive"] / 100.0
        sentient = npc.statuses["sentient"] / 100.0
        idle_enabled = False if victory is not None else npc_idle_enabled(
            self.state.phase,
            self.state.player.alive,
            npc.alive,
            self.state.is_ending,
        )
        idle_frame = idle_emote_frame(
            self.state.seed,
            index,
            max(0.0, now - self.state.idle_epoch),
        )
        idle_motion = idle_motion_parameters(idle_frame, index)
        gesture_key = (
            victory.gesture_key
            if victory is not None
            else ("idle" if npc.eliminated else str(getattr(npc, "gesture_key", "idle")))
        )
        gesture_age = 0.0 if victory is not None else max(0.0, now - npc.gesture_changed_at)
        emote_motion = action_emote_motion(gesture_key, gesture_age, index)
        action_weight = 1.0 if victory is not None else action_gesture_override_weight(
            npc.gesture_turn,
            gesture_age,
            idle_enabled,
        )
        emote_weight = 0.0 if victory is not None or npc.eliminated else action_weight
        idle_weight = (1.0 - action_weight) if idle_enabled else 0.0
        breath_wave = math.sin(now * (0.78 + (1.0 - health) * 0.38) + phase)
        bob = 0.0 if npc.eliminated else (
            breath_wave * (0.020 + health * 0.020)
            + idle_motion["body_y"] * idle_weight
        )
        body_x = 0.0 if npc.eliminated else idle_motion["body_x"] * idle_weight
        body_roll = 0.0 if npc.eliminated else (
            idle_motion["body_roll"] * idle_weight
            + emote_motion["body_roll"] * emote_weight
        )
        body_pitch = 0.0 if npc.eliminated else emote_motion["body_pitch"] * emote_weight
        bob += emote_motion["body_y"] * emote_weight
        yaw = 0.0 if npc.eliminated else (
            math.sin(now * 0.55 + phase) * (2.0 + cognitive * 3.0)
            + idle_motion["head_yaw"] * idle_weight
        )
        head_tilt = 0.0 if npc.eliminated else (
            math.sin(now * 0.42 + phase * 1.7) * (1.2 + emotional * 1.8)
            + idle_motion["head_tilt"] * idle_weight
        )
        head_nod = 0.0 if npc.eliminated else (
            math.sin(now * 0.73 + phase * 0.6) * (0.7 + sentient * 1.3)
            + idle_motion["head_nod"] * idle_weight
            + emote_motion["head_nod"] * emote_weight
        )
        head_tilt += emote_motion["head_tilt"] * emote_weight
        local_head_yaw = emote_motion["head_yaw"] * emote_weight
        if victory is not None and npc.alive:
            bob += victory.bounce
            body_x += victory.body_x
            body_roll += victory.body_roll
            yaw += victory.head_yaw
            head_nod += victory.head_nod
        reaction_age = now - self.state.last_reaction_time
        pulse = 0.0
        if (
            npc.alive
            and index == self.state.last_reaction_index
            and 0.0 <= reaction_age < 0.7
        ):
            pulse = (1.0 - reaction_age / 0.7) * 0.08
        if victory is not None:
            pulse += victory.pulse
        fight_age = now - self.state.last_fight_time
        in_recent_fight = (
            index + 1 in self.state.last_fight_indices
            and 0.0 <= fight_age < 0.8
        )
        fight_shake = 0.0
        if in_recent_fight:
            fight_shake = math.sin(now * 42.0 + index) * (1.0 - fight_age / 0.8) * 0.08

        GL.glPushMatrix()
        GL.glTranslatef(fight_shake + body_x, bob, 0.0)
        GL.glRotatef(body_roll, 0.0, 0.0, 1.0)
        GL.glRotatef(body_pitch, 1.0, 0.0, 0.0)
        GL.glRotatef(yaw, 0.0, 1.0, 0.0)
        GL.glScalef(1.0 + pulse, 1.0 + pulse, 1.0 + pulse)

        jacket_fade = 0.72 if npc.eliminated else (1.0 - health) * 0.28
        jacket = mix_color(npc.profile.accent, (55, 58, 56), jacket_fade)
        if in_recent_fight:
            jacket = mix_color(jacket, RED, 0.42)
        chest_breath = 1.0 if npc.eliminated else 1.0 + breath_wave * 0.012
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.08, 0.09, 0.10, 1.0))
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 14.0)
        GL.glPushMatrix()
        GL.glScalef(chest_breath, 1.0, chest_breath)
        draw_box(0.0, -1.42, 0.0, 2.35, 1.65, 0.95, jacket, cloth_texture)
        for side in (-1.0, 1.0):
            GL.glPushMatrix()
            GL.glTranslatef(side * 1.04, -1.18, -0.02)
            GL.glScalef(1.18, 0.76, 0.92)
            draw_sphere(0.46, jacket, 12, 24, cloth_texture)
            GL.glPopMatrix()
        self._draw_outfit_details(npc.profile, jacket, index)
        GL.glPopMatrix()

        self._draw_articulated_arms(
            npc,
            index,
            now,
            phase,
            jacket,
            idle_frame,
            idle_weight,
            action_weight,
            gesture_age,
            victory,
        )

        GL.glPushMatrix()
        GL.glTranslatef(0.0, -0.52, 0.0)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.08, 0.06, 0.05, 1.0))
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 24.0)
        draw_cylinder(0.31, 0.55, npc.profile.skin, 24, skin_texture)
        GL.glPopMatrix()

        GL.glPushMatrix()
        GL.glTranslatef(0.0, 0.40 + (0.0 if npc.eliminated else breath_wave * 0.008), 0.0)
        GL.glRotatef(head_tilt, 0.0, 0.0, 1.0)
        GL.glRotatef(head_nod, 1.0, 0.0, 0.0)
        GL.glRotatef(local_head_yaw, 0.0, 1.0, 0.0)
        draw_sphere(0.92, npc.profile.skin, 20, 40, skin_texture)
        self._draw_ears(npc.profile)
        self._draw_hair(npc.profile, index)
        if npc.statuses["spiritual"] == -1:
            self._draw_horns(now, phase)
        self._draw_face(
            npc,
            index,
            now,
            phase,
            idle_motion["gaze_x"] * idle_weight,
            1.0 if victory is not None else 0.0,
            emote_motion,
            emote_weight,
        )
        self._draw_action_emote_particles(
            gesture_key,
            gesture_age,
            index,
            npc.profile.accent,
            emote_motion,
            emote_weight,
        )
        if npc.alive and npc.statuses["spiritual"] == 1:
            self._draw_halo(now, phase)
        GL.glPopMatrix()
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 0.0)
        GL.glPopMatrix()

    @staticmethod
    def _blend_point(
        start: Sequence[float],
        end: Sequence[float],
        amount: float,
    ) -> Tuple[float, float, float]:
        return (
            float(start[0] + (end[0] - start[0]) * amount),
            float(start[1] + (end[1] - start[1]) * amount),
            float(start[2] + (end[2] - start[2]) * amount),
        )

    def _gesture_arm_target(
        self,
        gesture_key: str,
        side: float,
        now: float,
        phase: float,
        gesture_age: Optional[float] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, str]:
        """Return an elbow, palm, finger heading, and hand articulation."""
        idle = (
            (side * 1.27, -1.43, 0.23),
            (side * 1.20, -1.86, 0.43),
            180.0 + side * 4.0,
            "relaxed",
        )
        float_wave = math.sin(now * 1.65 + phase)
        quick_wave = math.sin(now * 5.1 + phase * 1.4)

        if gesture_key == "idle":
            elbow, palm, angle, style = idle
            return (
                (elbow[0] + side * float_wave * 0.015, elbow[1], elbow[2]),
                (palm[0] + side * float_wave * 0.020, palm[1] + float_wave * 0.018, palm[2]),
                angle + quick_wave * 1.6,
                style,
            )
        if gesture_key == "open_palms":
            return (
                (side * 1.18, -1.03 + float_wave * 0.025, 0.39),
                (side * 0.76, -0.38 + float_wave * 0.035, 0.82),
                -side * (8.0 + quick_wave * 2.5),
                "open",
            )
        if gesture_key == "heart_hands":
            heartbeat = max(0.0, math.sin(now * 3.4 + phase)) * 0.025
            return (
                (side * 0.99, -1.04, 0.42),
                (side * (0.285 - heartbeat), -0.54 + heartbeat, 0.92),
                -side * 50.0,
                "heart",
            )
        if gesture_key == "temple_tap":
            if side > 0.0:
                return (
                    (1.16, -0.57, 0.46),
                    (0.73 + quick_wave * 0.018, 0.29, 0.91),
                    -8.0 + quick_wave * 2.0,
                    "point",
                )
            return ((-1.15, -1.28, 0.30), (-0.35, -0.79, 0.73), 30.0, "relaxed")
        if gesture_key == "guarded_cross":
            return (
                (side * 1.04, -0.94, 0.45),
                (-side * 0.52, -0.72 + side * 0.12, 0.88 + side * 0.025),
                -side * 92.0,
                "fist",
            )
        if gesture_key == "fist_clench":
            tension = max(0.0, quick_wave) * 0.035
            return (
                (side * 1.13, -0.97, 0.42),
                (side * (0.75 - tension), -0.41 + tension, 0.91),
                -side * 4.0,
                "fist",
            )
        if gesture_key == "prayer_pose":
            return (
                (side * 0.93, -0.92, 0.42),
                (side * 0.145, -0.43 + float_wave * 0.012, 0.96),
                0.0,
                "prayer",
            )
        if gesture_key == "self_hug":
            return (
                (side * 0.98, -0.91, 0.48),
                (-side * 0.73, -0.69 + side * 0.10, 0.90),
                -side * 96.0,
                "grip",
            )
        if gesture_key == "viral_point":
            if side < 0.0:
                return ((-1.06, -0.96, 0.40), (-0.68, -0.31, 0.88), -90.0, "point")
            return ((0.97, -0.89, 0.42), (0.09, -0.50, 0.92), -90.0, "point")
        if gesture_key == "seesaw_67":
            seesaw = math.sin(now * 3.1 + phase) * 0.16
            # Keep the lowered palm above the portrait subtitle while the two
            # hands continue their opposing up/down rhythm.
            base_y = 0.02 if side < 0.0 else -0.88
            return (
                (side * 1.12, -0.94 - side * seesaw * 0.30, 0.40),
                (side * 0.82, base_y - side * seesaw, 0.88),
                0.0 if side < 0.0 else 90.0,
                "open",
            )
        if gesture_key == "funky_ehh":
            return (
                (side * 1.22, -0.76 + float_wave * 0.035, 0.39),
                (side * 1.10, -0.06 + quick_wave * 0.055, 0.84),
                side * (20.0 + quick_wave * 12.0),
                "open",
            )
        if gesture_key == "sprint_pose":
            pump = math.sin(now * 4.8 + phase) * 0.10
            if side < 0.0:
                return ((-0.94, -0.73 + pump, 0.45), (-0.58, -1.25 + pump, 0.87), 180.0, "fist")
            return ((0.98, -1.16 - pump, 0.42), (0.54, -0.55 - pump, 0.90), 0.0, "fist")
        if gesture_key == "affirmation":
            if side > 0.0:
                return ((1.05, -0.91, 0.43), (0.72, -0.30 + float_wave * 0.020, 0.91), 0.0, "thumbs_up")
            return ((-1.02, -1.07, 0.40), (-0.25, -0.79, 0.84), 34.0, "relaxed")
        if gesture_key == "dramatic_turn":
            sweep = math.sin(now * 1.8 + phase) * 0.06
            if side < 0.0:
                return ((-1.08, -0.82, 0.42), (-0.98 + sweep, 0.05, 0.86), -25.0, "open")
            return ((0.97, -1.00, 0.43), (0.20, -0.68, 0.88), -20.0, "relaxed")
        if gesture_key == "play_them_off":
            flourish = math.sin(now * 2.7 + phase + side) * 0.07
            return (
                (side * 1.12, -0.75 + flourish, 0.42),
                (side * 1.13, -0.07 - flourish, 0.88),
                side * (38.0 + quick_wave * 5.0),
                "open",
            )
        if gesture_key == "retro_chacha":
            dance = math.sin(now * 3.35 + phase) * 0.10
            if side < 0.0:
                return ((-1.03, -0.78 + dance, 0.44), (-0.66 + dance, 0.23, 0.90), -18.0, "open")
            return ((1.08, -0.91 - dance, 0.41), (0.72 - dance, -0.31, 0.89), 106.0, "open")
        if gesture_key == "pixel_wave":
            if side > 0.0:
                return (
                    (1.11, -0.54, 0.44),
                    (0.84, 0.37 + float_wave * 0.025, 0.90),
                    quick_wave * 18.0,
                    "open",
                )
            return ((-1.11, -1.25, 0.31), (-0.36, -0.86, 0.78), 30.0, "relaxed")
        if gesture_key == "spitting":
            if side > 0.0:
                return ((1.08, -0.72, 0.44), (0.55, -0.12, 0.96), -45.0, "relaxed")
            return ((-1.06, -1.05, 0.41), (-0.45, -0.72, 0.88), 45.0, "fist")
        if gesture_key == "yawning":
            if side > 0.0:
                return ((1.06, -0.64, 0.44), (0.38, -0.03, 0.99), -55.0, "open")
            return ((-1.18, -1.34, 0.28), (-1.02, -1.72, 0.50), 170.0, "relaxed")
        if gesture_key == "chomping":
            beat = math.sin((now if gesture_age is None else gesture_age) * 18.0)
            if side > 0.0:
                return (
                    (1.07, -0.70 + beat * 0.025, 0.44),
                    (0.31, -0.12 + beat * 0.018, 0.99),
                    -60.0 + beat * 3.0,
                    "grip",
                )
            return ((-1.02, -1.00, 0.41), (-0.42, -0.72, 0.88), 55.0, "grip")
        if gesture_key == "bowing":
            return (
                (side * 0.96, -1.00, 0.43),
                (side * 0.16, -0.55, 0.96),
                0.0,
                "prayer",
            )
        if gesture_key == "head_banging":
            beat = math.sin((now if gesture_age is None else gesture_age) * 18.0)
            return (
                (side * 1.05, -0.74 + beat * 0.05, 0.44),
                (side * 0.66, -0.22 + beat * 0.07, 0.92),
                -side * 8.0,
                "fist",
            )
        return idle

    def _idle_arm_target(
        self,
        frame: IdleEmoteFrame,
        side: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, str]:
        """Return one restrained, personality-specific idle hand performance."""
        wave = math.sin(frame.motion_phase)
        detail = math.sin(frame.motion_phase * 2.15 + side * 0.8)
        energy = IDLE_PERSONALITY_BY_KEY[frame.personality_key].hand_energy
        emote_key = frame.emote_key

        if emote_key == "room_scan":
            if side > 0.0:
                return (
                    (1.15, -1.26, 0.30),
                    (0.93 + detail * 0.025, -1.47, 0.62),
                    172.0 + detail * 3.0,
                    "relaxed",
                )
            return ((-1.24, -1.42, 0.24), (-1.17, -1.83, 0.44), 178.0, "relaxed")
        if emote_key == "cuff_check":
            if side < 0.0:
                return ((-1.04, -1.25, 0.37), (0.48, -1.13, 0.84), -72.0, "grip")
            return ((1.08, -1.36, 0.31), (0.72, -1.37, 0.74), 155.0, "relaxed")
        if emote_key == "knuckle_roll":
            return (
                (side * 1.10, -1.25, 0.34),
                (side * (0.58 + wave * 0.035), -1.11 + side * detail * 0.022, 0.82),
                side * (16.0 + detail * 5.0),
                "fist",
            )
        if emote_key == "heart_breath":
            heartbeat = max(0.0, math.sin(frame.motion_phase * 2.0)) * 0.025
            return (
                (side * 0.99, -1.09, 0.41),
                (side * (0.32 - heartbeat), -0.68 + heartbeat, 0.91),
                -side * 48.0,
                "heart",
            )
        if emote_key == "reassure_palm":
            if side > 0.0:
                return (
                    (1.10, -1.08, 0.42),
                    (0.73, -0.61 + wave * 0.028, 0.88),
                    -8.0 + detail * 2.5,
                    "open",
                )
            return ((-1.19, -1.40, 0.25), (-1.06, -1.77, 0.46), 177.0, "relaxed")
        if emote_key == "soft_wave":
            if side > 0.0:
                return (
                    (1.14, -0.91, 0.40),
                    (0.91, -0.29 + wave * 0.025, 0.88),
                    detail * 10.0 * energy,
                    "open",
                )
            return ((-1.22, -1.41, 0.24), (-1.13, -1.82, 0.44), 177.0, "relaxed")
        if emote_key == "chin_think":
            if side > 0.0:
                return (
                    (1.09, -0.77, 0.44),
                    (0.42, -0.05 + detail * 0.012, 0.92),
                    -55.0,
                    "relaxed",
                )
            return ((-1.05, -1.03, 0.40), (0.31, -0.91, 0.81), -82.0, "grip")
        if emote_key == "air_type":
            tap = detail * 0.055 * energy
            return (
                (side * 1.05, -1.05, 0.43),
                (side * 0.54, -0.80 + tap, 0.91),
                side * (76.0 + detail * 7.0),
                "open",
            )
        if emote_key == "temple_count":
            if side > 0.0:
                return (
                    (1.15, -0.55, 0.45),
                    (0.74 + detail * 0.014, 0.27, 0.91),
                    -8.0 + detail * 2.0,
                    "point",
                )
            return ((-1.18, -1.35, 0.28), (-0.96, -1.66, 0.55), 168.0, "relaxed")
        if emote_key == "signal_trace":
            if side > 0.0:
                return (
                    (1.12, -0.77, 0.44),
                    (0.80 + wave * 0.10, -0.05 + detail * 0.08, 0.91),
                    -24.0 + detail * 8.0,
                    "point",
                )
            return ((-1.17, -1.34, 0.27), (-0.93, -1.65, 0.56), 164.0, "relaxed")
        if emote_key == "prayer_breath":
            return (
                (side * 0.96, -1.00, 0.43),
                (side * 0.16, -0.51 + wave * 0.018, 0.95),
                0.0,
                "prayer",
            )
        if emote_key == "palm_orbit":
            orbit_y = math.sin(frame.motion_phase + (0.0 if side < 0.0 else math.pi)) * 0.11
            return (
                (side * 1.10, -0.98 - orbit_y * 0.22, 0.42),
                (side * 0.71, -0.42 + orbit_y, 0.91),
                side * (8.0 + detail * 4.0),
                "open",
            )
        return self._gesture_arm_target("idle", side, 0.0, 0.0)

    def _draw_articulated_arms(
        self,
        npc: Participant,
        index: int,
        now: float,
        phase: float,
        jacket: Tuple[int, int, int],
        idle_frame: IdleEmoteFrame,
        idle_weight: float,
        action_weight: float,
        gesture_age: float,
        victory: Optional[VictoryFrame] = None,
    ) -> None:
        """Render two animated sleeve/forearm/hand chains in front of the torso."""
        gesture_key = (
            victory.gesture_key
            if victory is not None
            else ("idle" if npc.eliminated else str(getattr(npc, "gesture_key", "idle")))
        )
        pose_now = 0.0 if npc.eliminated else now
        pose_phase = 0.0 if npc.eliminated else phase
        for side in (-1.0, 1.0):
            idle_elbow, idle_palm, idle_angle, _idle_style = self._gesture_arm_target(
                "idle", side, pose_now, pose_phase
            )
            emote_elbow, emote_palm, emote_angle, emote_style = self._idle_arm_target(
                idle_frame,
                side,
            )
            emote_weight = float(clamp(idle_frame.blend * idle_weight, 0.0, 1.0))
            ambient_elbow = self._blend_point(idle_elbow, emote_elbow, emote_weight)
            ambient_palm = self._blend_point(idle_palm, emote_palm, emote_weight)
            ambient_angle = idle_angle + (emote_angle - idle_angle) * emote_weight
            ambient_style = emote_style if emote_weight > 0.48 else "relaxed"
            action_elbow, action_palm, action_angle, action_style = self._gesture_arm_target(
                gesture_key,
                side,
                pose_now,
                pose_phase,
                None if victory is not None else gesture_age,
            )
            if victory is not None:
                next_elbow, next_palm, next_angle, next_style = self._gesture_arm_target(
                    victory.next_gesture_key,
                    side,
                    pose_now,
                    pose_phase,
                )
                action_elbow = self._blend_point(
                    action_elbow,
                    next_elbow,
                    victory.gesture_blend,
                )
                action_palm = self._blend_point(
                    action_palm,
                    next_palm,
                    victory.gesture_blend,
                )
                action_angle += (next_angle - action_angle) * victory.gesture_blend
                if victory.gesture_blend > 0.52:
                    action_style = next_style
                action_weight = 1.0
            action_weight = float(clamp(action_weight, 0.0, 1.0))
            elbow = self._blend_point(ambient_elbow, action_elbow, action_weight)
            palm = self._blend_point(ambient_palm, action_palm, action_weight)
            hand_angle = ambient_angle + (action_angle - ambient_angle) * action_weight
            hand_style = action_style if action_weight > 0.48 else ambient_style
            shoulder = (side * 1.04, -1.09, 0.24)
            self._draw_articulated_arm(
                shoulder,
                elbow,
                palm,
                hand_angle,
                hand_style,
                npc.profile.skin,
                jacket,
                side,
                now,
                phase,
            )

    def _draw_articulated_arm(
        self,
        shoulder: Sequence[float],
        elbow: Sequence[float],
        palm: Sequence[float],
        hand_angle: float,
        hand_style: str,
        skin: Tuple[int, int, int],
        jacket: Tuple[int, int, int],
        side: float,
        now: float,
        phase: float,
    ) -> None:
        sleeve_shadow = mix_color(jacket, INK, 0.30)
        cuff = mix_color(jacket, WHITE, 0.46)
        cloth_texture = self.materials.get("cloth")
        textured = _begin_bound_texture(cloth_texture)
        draw_cylinder_between(shoulder, elbow, 0.185, jacket, 24)
        draw_cylinder_between(elbow, palm, 0.145, jacket, 24)
        for joint, radius in ((shoulder, 0.205), (elbow, 0.175)):
            GL.glPushMatrix()
            GL.glTranslatef(*joint)
            draw_sphere(radius, jacket, 12, 24)
            GL.glPopMatrix()

        cuff_start = self._blend_point(elbow, palm, 0.78)
        cuff_end = self._blend_point(elbow, palm, 0.95)
        draw_cylinder_between(cuff_start, cuff_end, 0.158, cuff, 22)
        # Raised front-facing piping makes each sleeve readable without adding
        # uncached or per-frame tessellation.
        seam_start = (shoulder[0], shoulder[1], shoulder[2] + 0.175)
        seam_end = (elbow[0], elbow[1], elbow[2] + 0.145)
        draw_cylinder_between(seam_start, seam_end, 0.014, sleeve_shadow, 12)
        _end_bound_texture(textured)
        self._draw_hand(palm, hand_angle, hand_style, skin, side, now, phase)

    def _draw_hand(
        self,
        center: Sequence[float],
        hand_angle: float,
        style: str,
        skin: Tuple[int, int, int],
        side: float,
        now: float,
        phase: float,
    ) -> None:
        """Draw an animated palm, thumb, and two phalanges for every finger."""
        nail = mix_color(skin, WHITE, 0.24)
        joint_shadow = mix_color(skin, (96, 48, 45), 0.18)
        skin_texture = self.materials.get("skin")
        textured = _begin_bound_texture(skin_texture)
        GL.glPushMatrix()
        GL.glTranslatef(*center)
        GL.glRotatef(-hand_angle, 0.0, 0.0, 1.0)
        GL.glPushMatrix()
        GL.glScalef(0.205, 0.245, 0.115)
        draw_sphere(1.0, skin, 14, 28)
        GL.glPopMatrix()

        finger_xs = (-0.132, -0.044, 0.044, 0.132)
        lengths = (0.245, 0.335, 0.315, 0.235)
        if style == "prayer":
            finger_xs = tuple(value * 0.74 for value in finger_xs)
        hand_pulse = math.sin(now * 4.6 + phase + side) * 0.012
        for finger_index, (finger_x, finger_length) in enumerate(zip(finger_xs, lengths)):
            base = (finger_x, 0.145, 0.015)
            is_pointing = style == "point" and finger_index == (1 if side > 0.0 else 2)
            curled = style in ("fist", "thumbs_up") or (style == "point" and not is_pointing)
            half_curled = style in ("relaxed", "grip", "heart")
            if curled:
                knuckle = (finger_x, 0.205, 0.078)
                tip = (finger_x * 0.90, 0.105, 0.155)
            else:
                extra = 0.055 if is_pointing else 0.0
                bend = 0.0
                if half_curled:
                    bend = 0.060 if style != "heart" else 0.095
                if style == "open":
                    bend += math.sin(now * 2.9 + phase + finger_index * 0.7) * 0.010
                knuckle = (
                    finger_x * (1.08 if style == "open" else 1.0),
                    0.145 + finger_length * 0.53,
                    0.018 + bend * 0.35,
                )
                tip = (
                    finger_x * (1.15 if style == "open" else 1.0),
                    0.145 + finger_length + extra - bend,
                    0.020 + bend + (hand_pulse if style == "heart" else 0.0),
                )
            draw_cylinder_between(base, knuckle, 0.034, skin, 16)
            draw_cylinder_between(knuckle, tip, 0.030, skin, 16)
            GL.glPushMatrix()
            GL.glTranslatef(*knuckle)
            draw_sphere(0.041, joint_shadow, 8, 16)
            GL.glPopMatrix()
            if not curled:
                GL.glPushMatrix()
                GL.glTranslatef(tip[0], tip[1], tip[2] + 0.018)
                GL.glScalef(0.72, 1.0, 0.36)
                draw_sphere(0.036, nail, 8, 16)
                GL.glPopMatrix()

        thumb_root = (-side * 0.155, -0.015, 0.025)
        if style == "thumbs_up":
            thumb_joint = (-side * 0.105, 0.18, 0.055)
            thumb_tip = (-side * 0.075, 0.405, 0.060)
        elif style == "prayer":
            thumb_joint = (-side * 0.105, 0.05, 0.105)
            thumb_tip = (-side * 0.035, 0.16, 0.125)
        else:
            spread = 0.33 if style in ("open", "heart") else 0.24
            thumb_joint = (-side * 0.245, 0.045, 0.050)
            thumb_tip = (-side * spread, 0.145, 0.060 if style != "fist" else 0.135)
        draw_cylinder_between(thumb_root, thumb_joint, 0.042, skin, 16)
        draw_cylinder_between(thumb_joint, thumb_tip, 0.036, skin, 16)
        GL.glPushMatrix()
        GL.glTranslatef(*thumb_joint)
        draw_sphere(0.048, joint_shadow, 8, 16)
        GL.glPopMatrix()
        if style in ("open", "heart", "thumbs_up"):
            GL.glPushMatrix()
            GL.glTranslatef(thumb_tip[0], thumb_tip[1], thumb_tip[2] + 0.018)
            GL.glScalef(0.75, 1.0, 0.38)
            draw_sphere(0.041, nail, 8, 16)
            GL.glPopMatrix()
        GL.glPopMatrix()
        _end_bound_texture(textured)

    def _draw_outfit_details(
        self,
        profile: Profile,
        jacket: Tuple[int, int, int],
        index: int,
    ) -> None:
        """Layer collars, lapels, a shirt front, and a small individual emblem."""
        shirt = mix_color(jacket, WHITE, 0.52)
        shadow = mix_color(jacket, INK, 0.30)
        cloth_texture = self.materials.get("cloth")
        textured = _begin_bound_texture(cloth_texture)
        draw_box(0.0, -0.93, 0.50, 0.42, 0.78, 0.10, shirt)
        for side in (-1.0, 1.0):
            GL.glPushMatrix()
            GL.glTranslatef(side * 0.31, -1.02, 0.55)
            GL.glRotatef(side * 25.0, 0.0, 0.0, 1.0)
            draw_box(0.0, 0.0, 0.0, 0.25, 0.92, 0.09, shadow)
            GL.glPopMatrix()
            GL.glPushMatrix()
            GL.glTranslatef(side * 0.18, -0.62, 0.54)
            GL.glRotatef(side * 36.0, 0.0, 0.0, 1.0)
            draw_box(0.0, 0.0, 0.0, 0.20, 0.38, 0.08, shirt)
            GL.glPopMatrix()
        _end_bound_texture(textured)
        button_color = mix_color(profile.accent, GOLD, 0.38)
        for row in range(3):
            GL.glPushMatrix()
            GL.glTranslatef(0.0, -1.00 - row * 0.24, 0.585)
            GL.glRotatef(90.0, 1.0, 0.0, 0.0)
            draw_cylinder(0.045, 0.025, button_color, 14)
            GL.glPopMatrix()
        emblem_x = -0.72 if index % 2 == 0 else 0.72
        GL.glPushMatrix()
        GL.glTranslatef(emblem_x, -1.13, 0.59)
        GL.glScalef(1.0, 0.72, 0.45)
        draw_sphere(0.105, mix_color(profile.accent, WHITE, 0.28), 8, 16)
        GL.glPopMatrix()

    def _draw_ears(self, profile: Profile) -> None:
        ear_shadow = mix_color(profile.skin, (105, 55, 52), 0.20)
        skin_texture = self.materials.get("skin")
        textured = _begin_bound_texture(skin_texture)
        for side in (-1.0, 1.0):
            GL.glPushMatrix()
            GL.glTranslatef(side * 0.88, 0.02, 0.02)
            GL.glScalef(0.38, 0.72, 0.32)
            draw_sphere(0.31, profile.skin, 12, 24)
            GL.glPopMatrix()
            GL.glPushMatrix()
            GL.glTranslatef(side * 0.905, 0.015, 0.105)
            GL.glScalef(0.28, 0.58, 0.22)
            draw_sphere(0.19, ear_shadow, 10, 20)
            GL.glPopMatrix()
        _end_bound_texture(textured)

    def _draw_horns(self, now: float, phase: float) -> None:
        """Draw two swept black horns with true apexes and live tip flames."""
        segments = 18
        GL.glPushAttrib(
            GL.GL_ENABLE_BIT
            | GL.GL_LIGHTING_BIT
            | GL.GL_CURRENT_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
        )
        GL.glEnable(GL.GL_LIGHTING)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.52, 0.54, 0.58, 1.0))
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 82.0)

        for side in (-1.0, 1.0):
            centers = [
                (side * (0.48 + outward), HORN_BASE_Y + height, 0.08, radius)
                for outward, height, radius in HORN_CENTERLINE
            ]

            # A closed black base disappears naturally beneath the hair while
            # preventing a hollow silhouette during wide idle head turns.
            base_x, base_y, base_z, base_radius = centers[0]
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glNormal3f(0.0, -1.0, 0.0)
            gl_color(HORN_BODY_COLORS[0])
            GL.glVertex3f(base_x, base_y, base_z)
            for segment in range(segments + 1):
                angle = -math.tau * segment / segments
                gl_color(HORN_BODY_COLORS[1])
                GL.glVertex3f(
                    base_x + math.cos(angle) * base_radius,
                    base_y,
                    base_z + math.sin(angle) * base_radius * 0.72,
                )
            GL.glEnd()

            # Three swept frustums carry a graphite highlight but remain within
            # a genuinely black palette (all body channels stay below 30).
            for ring_index in range(len(centers) - 2):
                x0, y0, z0, radius0 = centers[ring_index]
                x1, y1, z1, radius1 = centers[ring_index + 1]
                GL.glBegin(GL.GL_QUAD_STRIP)
                for segment in range(segments + 1):
                    angle = math.tau * segment / segments
                    cosine = math.cos(angle)
                    sine = math.sin(angle)
                    highlight = max(0.0, sine)
                    body_color = (
                        HORN_BODY_COLORS[2]
                        if highlight > 0.68
                        else HORN_BODY_COLORS[1]
                        if highlight > 0.12
                        else HORN_BODY_COLORS[0]
                    )
                    gl_color(body_color)
                    GL.glNormal3f(cosine, 0.18, sine)
                    GL.glVertex3f(
                        x0 + cosine * radius0,
                        y0,
                        z0 + sine * radius0 * 0.72,
                    )
                    GL.glVertex3f(
                        x1 + cosine * radius1,
                        y1,
                        z1 + sine * radius1 * 0.72,
                    )
                GL.glEnd()

            # The last ring terminates in one mathematical apex instead of a
            # rounded sphere, keeping the horns visibly sharp at 1080p.
            ring_x, ring_y, ring_z, ring_radius = centers[-2]
            tip_x, tip_y, tip_z, _tip_radius = centers[-1]
            GL.glBegin(GL.GL_TRIANGLES)
            for segment in range(segments):
                angle0 = math.tau * segment / segments
                angle1 = math.tau * (segment + 1) / segments
                for angle in (angle0, angle1):
                    gl_color(HORN_BODY_COLORS[1 if math.sin(angle) > 0.0 else 0])
                    GL.glNormal3f(math.cos(angle), 0.32, math.sin(angle))
                    GL.glVertex3f(
                        ring_x + math.cos(angle) * ring_radius,
                        ring_y,
                        ring_z + math.sin(angle) * ring_radius * 0.72,
                    )
                gl_color(HORN_BODY_COLORS[2])
                GL.glNormal3f(side * 0.08, 0.78, 0.24)
                GL.glVertex3f(tip_x, tip_y, tip_z)
            GL.glEnd()
            self._draw_horn_flame(horn_tip_position(side), now, phase, side)
        GL.glPopAttrib()

    def _draw_horn_flame(
        self,
        tip: Tuple[float, float, float],
        now: float,
        phase: float,
        side: float,
    ) -> None:
        """Layer additive red, orange, yellow, and white flame tongues."""
        height, width, sway = horn_flame_parameters(now, phase, side)
        GL.glPushAttrib(
            GL.GL_ENABLE_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_CURRENT_BIT
            | GL.GL_POINT_BIT
        )
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        base_x, base_y, base_z = tip
        tongue_specs = (
            (1.00, 1.00, 0.00, 0, 0.68),
            (0.72, 0.72, -0.08 * side, 1, 0.82),
            (0.48, 0.48, 0.07 * side, 2, 0.92),
            (0.28, 0.33, 0.00, 3, 0.98),
        )
        GL.glBegin(GL.GL_TRIANGLES)
        for height_scale, width_scale, offset, color_index, alpha in tongue_specs:
            flame_color = HORN_FLAME_PALETTE[color_index]
            local_width = width * width_scale
            apex_x = base_x + sway * height_scale + offset
            apex_y = base_y + height * height_scale
            z = base_z + 0.055 + color_index * 0.004
            gl_color(flame_color, alpha * 0.84)
            GL.glVertex3f(base_x - local_width, base_y - 0.015, z)
            gl_color(flame_color, alpha)
            GL.glVertex3f(base_x + local_width, base_y - 0.015, z)
            gl_color(flame_color, 0.05)
            GL.glVertex3f(apex_x, apex_y, z)
        GL.glEnd()

        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(3.8)
        GL.glBegin(GL.GL_POINTS)
        for ember in range(5):
            rise = (0.55 + ember * 0.16 + now * (0.72 + ember * 0.05)) % 1.0
            drift = math.sin(now * 5.2 + phase + ember * 1.71 + side) * width * (0.35 + rise)
            gl_color(HORN_FLAME_PALETTE[ember % 3], (1.0 - rise) * 0.72)
            GL.glVertex3f(
                base_x + sway * 0.45 + drift,
                base_y + height * (0.72 + rise * 0.72),
                base_z + 0.07,
            )
        GL.glEnd()
        GL.glPopAttrib()

    def _draw_hair(self, profile: Profile, index: int) -> None:
        hair = profile.hair
        hair_texture = self.materials.get("hair")
        cloth_texture = self.materials.get("cloth")
        # The fiber map already carries anisotropic-looking strand highlights;
        # suppress a second broad fixed-function specular bloom.
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 0.0)
        textured = _begin_bound_texture(cloth_texture if index == 3 else hair_texture)
        if index == 0:
            GL.glPushMatrix()
            GL.glTranslatef(0.0, 0.52, -0.02)
            GL.glScalef(1.02, 0.58, 0.98)
            draw_sphere(0.92, hair, 14, 28)
            GL.glPopMatrix()
        elif index == 1:
            for angle in (-2.6, -2.05, -1.5, -0.95, -0.4, 0.15):
                GL.glPushMatrix()
                GL.glTranslatef(math.cos(angle) * 0.73, 0.47 + math.sin(angle) * 0.18, -0.02)
                draw_sphere(0.30, hair, 11, 22)
                GL.glPopMatrix()
            for x in (-0.55, 0.0, 0.55):
                GL.glPushMatrix()
                GL.glTranslatef(x, 0.98 - abs(x) * 0.16, -0.02)
                draw_sphere(0.34, hair, 11, 22)
                GL.glPopMatrix()
        elif index == 2:
            GL.glPushMatrix()
            GL.glTranslatef(-0.18, 0.68, -0.03)
            GL.glScalef(1.06, 0.45, 1.0)
            draw_sphere(0.90, hair, 14, 28)
            GL.glPopMatrix()
            draw_box(0.70, 0.34, -0.05, 0.24, 1.08, 0.66, hair)
        else:
            scarf = mix_color(profile.accent, (235, 235, 225), 0.20)
            GL.glPushMatrix()
            GL.glTranslatef(0.0, 0.52, -0.04)
            GL.glScalef(1.04, 0.60, 1.0)
            draw_sphere(0.93, scarf, 14, 28)
            GL.glPopMatrix()
            draw_box(-0.72, -0.08, -0.08, 0.30, 1.20, 0.56, scarf)
        _end_bound_texture(textured)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.08, 0.06, 0.05, 1.0))
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 24.0)

    def _draw_face(
        self,
        npc: Participant,
        index: int,
        now: float,
        phase: float,
        idle_gaze_x: float = 0.0,
        victory_energy: float = 0.0,
        gesture_motion: Optional[Mapping[str, float]] = None,
        gesture_weight: float = 0.0,
    ) -> None:
        expression = face_expression_parameters(npc, now, phase)
        gesture_motion = gesture_motion or {}
        gesture_weight = float(clamp(gesture_weight, 0.0, 1.0))
        if npc.alive and gesture_weight > 0.0:
            expression = dict(expression)
            width_scale = 1.0 + (
                float(gesture_motion.get("mouth_width_scale", 1.0)) - 1.0
            ) * gesture_weight
            expression["mouth_half_width"] = float(
                clamp(expression["mouth_half_width"] * width_scale, 0.16, 0.44)
            )
        if npc.alive and victory_energy > 0.0:
            expression = dict(expression)
            expression["mouth_curve"] = max(
                expression["mouth_curve"],
                0.075 + math.sin(now * 3.4 + phase) * 0.018,
            )
            expression["eye_open"] = max(expression["eye_open"], 0.76)
        physical = npc.statuses["physical"] / 100.0
        emotional = npc.statuses["emotional"] / 100.0
        cognitive = npc.statuses["cognitive"] / 100.0
        social = npc.statuses["social"] / 100.0
        # Each face owns a different blink clock. A raised-sine envelope avoids the
        # single-frame snap of the earlier portraits and occasionally produces a
        # natural double blink when physical health is low.
        blink_period = 3.15 + physical * 1.75 + index * 0.17
        blink_phase = (now + phase * 0.37) % blink_period
        blink_close = 0.0
        blink_windows = ((0.0, 0.19),)
        if physical < 0.58:
            blink_windows += ((0.31, 0.14),)
        for blink_start, blink_duration in blink_windows:
            if blink_start <= blink_phase <= blink_start + blink_duration:
                progress = (blink_phase - blink_start) / blink_duration
                blink_close = max(blink_close, math.sin(progress * math.pi) ** 2)
        if npc.eliminated:
            eye_scale = expression["eye_open"]
            eye_radius = 0.215
        else:
            eye_scale = max(0.055, expression["eye_open"] * (1.0 - blink_close * 0.94))
            eye_multiplier = 1.0 + (
                float(gesture_motion.get("eye_scale", 1.0)) - 1.0
            ) * gesture_weight
            eye_scale = float(clamp(eye_scale * eye_multiplier, 0.055, 1.18))
            eye_radius = 0.17
        saccade_tick = math.floor((now + phase * 1.91) * (0.62 + (1.0 - cognitive) * 0.48))
        saccade_x = math.sin(saccade_tick * 12.9898 + phase * 3.1) * (0.016 + (1.0 - cognitive) * 0.035)
        saccade_y = math.sin(saccade_tick * 7.233 + phase * 1.7) * (0.009 + (1.0 - cognitive) * 0.022)
        micro_x = math.sin(now * 8.2 + phase) * 0.006 * (1.0 - cognitive)
        gaze_x = float(
            clamp(
                expression["pupil_dx"]
                + saccade_x
                + micro_x
                + idle_gaze_x
                + float(gesture_motion.get("gaze_x", 0.0)) * gesture_weight,
                -0.12,
                0.12,
            )
        )
        gaze_y = float(
            clamp(
                expression["pupil_dy"]
                + saccade_y
                + float(gesture_motion.get("gaze_y", 0.0)) * gesture_weight,
                -0.09,
                0.07,
            )
        )
        eye_y = 0.18
        for eye_x in (-0.34, 0.34):
            GL.glPushMatrix()
            GL.glTranslatef(eye_x, eye_y, 0.81)
            GL.glScalef(1.0, eye_scale, 0.55)
            draw_sphere(
                eye_radius,
                (250, 250, 246) if npc.eliminated else (236, 235, 224),
                12,
                24,
            )
            GL.glPopMatrix()
            if npc.alive:
                # Keep the colored response in the iris and give every living
                # portrait an unmistakably black central pupil.  The previous
                # cyan mix colored the entire pupil at positive spirituality.
                GL.glPushMatrix()
                GL.glTranslatef(
                    eye_x + gaze_x,
                    eye_y + gaze_y,
                    0.91,
                )
                GL.glScalef(1.0, eye_scale, 0.60)
                iris_base = mix_color(npc.profile.accent, (32, 55, 54), 0.58)
                iris_color = mix_color(
                    iris_base,
                    CYAN,
                    expression["spiritual_glow"] * 0.52,
                )
                draw_sphere(expression["pupil_radius"], iris_color, 10, 20)
                GL.glPopMatrix()

                # A cropped CC0 iris map supplies radial fibers while the
                # profile color and spirituality glow retain character state.
                iris_diameter = expression["pupil_radius"] * 2.12
                draw_textured_sprite_3d(
                    self.materials.get("iris"),
                    eye_x + gaze_x,
                    eye_y + gaze_y,
                    0.974,
                    iris_diameter,
                    iris_diameter * eye_scale,
                    iris_color,
                    0.88,
                )

                black_pupil_radius = expression["pupil_radius"] * 0.47
                GL.glPushMatrix()
                GL.glTranslatef(
                    eye_x + gaze_x,
                    eye_y + gaze_y,
                    0.948,
                )
                GL.glScalef(1.0, eye_scale, 0.64)
                draw_sphere(black_pupil_radius, (2, 3, 3), 10, 20)
                GL.glPopMatrix()
                if expression["sentient_glint"] > 0.24:
                    GL.glPushMatrix()
                    GL.glTranslatef(
                        eye_x + gaze_x - black_pupil_radius * 0.28,
                        eye_y + gaze_y + black_pupil_radius * 0.42,
                        0.980,
                    )
                    glint_color = mix_color(
                        WHITE,
                        CYAN,
                        expression["spiritual_glow"] * 0.75,
                    )
                    draw_sphere(0.009 + expression["sentient_glint"] * 0.009, glint_color, 7, 14)
                    GL.glPopMatrix()

        GL.glPushMatrix()
        GL.glTranslatef(0.0, -0.06, 0.88)
        GL.glRotatef(90.0, 1.0, 0.0, 0.0)
        nose_shadow = mix_color(npc.profile.skin, (80, 50, 42), 0.18)
        draw_cylinder(0.055, 0.25, nose_shadow, 18)
        GL.glPopMatrix()
        GL.glPushMatrix()
        GL.glTranslatef(0.0, -0.105, 1.01)
        GL.glScalef(1.20, 0.72, 0.86)
        draw_sphere(0.075, mix_color(npc.profile.skin, WHITE, 0.035), 11, 22)
        GL.glPopMatrix()

        reaction_age = now - self.state.last_reaction_time
        reaction_envelope = 0.0
        if npc.alive and index == self.state.last_reaction_index and 0.0 <= reaction_age < 1.55:
            reaction_envelope = 1.0 - reaction_age / 1.55
        speech_wave = max(0.0, math.sin(now * (5.0 + index * 0.27) + phase * 2.2))
        voice_level = self.audio.voice_envelope(index, now)
        jaw_open = 0.0
        if npc.alive:
            jaw_open = 0.006 + social * 0.008
            jaw_open += reaction_envelope * (0.025 + social * 0.065) * speech_wave
            jaw_open += voice_level * (0.035 + social * 0.060)
            jaw_open += victory_energy * max(0.0, math.sin(now * 4.1 + phase)) * 0.018
            jaw_open += float(gesture_motion.get("jaw_open", 0.0)) * gesture_weight
            jaw_open = float(clamp(jaw_open, 0.0, 0.20))
        brow_motion = 0.0 if npc.eliminated else math.sin(now * 1.12 + phase * 0.8) * 0.012
        brow_motion += reaction_envelope * (emotional - 0.5) * 0.075
        brow_motion += voice_level * (emotional - 0.5) * 0.055
        brow_motion += victory_energy * math.sin(now * 2.6 + phase) * 0.018
        brow_motion += float(gesture_motion.get("brow_offset", 0.0)) * gesture_weight

        GL.glDisable(GL.GL_LIGHTING)
        if npc.alive and jaw_open > 0.004:
            mouth_center_y = -0.335 - expression["mouth_curve"] * 0.34
            gl_color(mix_color((58, 19, 27), RED, emotional * 0.14))
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glVertex3f(0.0, mouth_center_y, 0.905)
            for point in range(19):
                angle = math.tau * point / 18.0
                GL.glVertex3f(
                    math.cos(angle) * expression["mouth_half_width"] * 0.86,
                    mouth_center_y + math.sin(angle) * jaw_open,
                    0.907,
                )
            GL.glEnd()
            if jaw_open > 0.025:
                GL.glLineWidth(2.0)
                gl_color((228, 220, 201), min(0.85, jaw_open * 13.0))
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(-expression["mouth_half_width"] * 0.54, mouth_center_y - jaw_open * 0.28, 0.912)
                GL.glVertex3f(expression["mouth_half_width"] * 0.54, mouth_center_y - jaw_open * 0.28, 0.912)
                GL.glEnd()
        GL.glLineWidth(4.0)
        gl_color(RED if npc.eliminated else (92, 42, 43))
        GL.glBegin(GL.GL_LINE_STRIP)
        for point in range(9):
            half_width = expression["mouth_half_width"]
            x = -half_width + point * (half_width * 2.0 / 8.0)
            curve = 1.0 - (x / half_width) ** 2
            y = -0.33 - expression["mouth_curve"] * curve - jaw_open * 0.24
            GL.glVertex3f(x, y, 0.91)
        GL.glEnd()
        if npc.alive and jaw_open > 0.012:
            GL.glLineWidth(2.6)
            gl_color(mix_color((92, 42, 43), RED, 0.22))
            GL.glBegin(GL.GL_LINE_STRIP)
            half_width = expression["mouth_half_width"] * 0.88
            for point in range(9):
                x = -half_width + point * (half_width * 2.0 / 8.0)
                curve = 1.0 - (x / half_width) ** 2
                GL.glVertex3f(
                    x,
                    -0.33 - expression["mouth_curve"] * curve * 0.42 + jaw_open * curve,
                    0.914,
                )
            GL.glEnd()

        GL.glLineWidth(3.0)
        gl_color(RED if npc.eliminated else mix_color(npc.profile.hair, (145, 100, 85), 0.25))
        for side in (-1, 1):
            center = side * 0.34
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(
                center + side * 0.18,
                0.43 + expression["brow_tilt"] + brow_motion,
                0.94,
            )
            GL.glVertex3f(
                center - side * 0.18,
                0.43 + expression["inner_brow"] - expression["brow_tilt"] * 0.30 + brow_motion * 0.55,
                0.94,
            )
            GL.glEnd()

        if npc.eliminated:
            GL.glLineWidth(2.5)
            gl_color((239, 34, 38))
            for eye_x in (-0.34, 0.34):
                for vertical_offset in (-0.045, 0.045):
                    GL.glBegin(GL.GL_LINE_STRIP)
                    for point in range(7):
                        x = eye_x - 0.17 + point * (0.34 / 6.0)
                        y = eye_y + vertical_offset + math.sin(point * 2.2 + eye_x) * 0.025
                        GL.glVertex3f(x, y, 0.955)
                    GL.glEnd()
        else:
            # Animated upper lids, nostrils, cheek creases, and a chin contour add
            # definition without introducing expensive per-frame mesh generation.
            GL.glLineWidth(1.4 + blink_close * 1.7)
            gl_color(mix_color(npc.profile.skin, npc.profile.hair, 0.36), 0.88)
            for eye_x in (-0.34, 0.34):
                GL.glBegin(GL.GL_LINE_STRIP)
                for point in range(9):
                    ratio = point / 8.0
                    x = eye_x - eye_radius + ratio * eye_radius * 2.0
                    arch = math.sin(ratio * math.pi) * eye_radius * eye_scale * 0.82
                    lid_drop = blink_close * eye_radius * 0.26
                    GL.glVertex3f(x, eye_y + arch - lid_drop, 0.958)
                GL.glEnd()

            GL.glPointSize(2.5)
            gl_color(mix_color(nose_shadow, INK, 0.34), 0.78)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex3f(-0.037, -0.135, 1.078)
            GL.glVertex3f(0.037, -0.135, 1.078)
            GL.glEnd()

            cheek_lift = (emotional - 0.5) * 0.026 + reaction_envelope * emotional * 0.018
            GL.glLineWidth(1.25)
            gl_color(mix_color(npc.profile.skin, (126, 64, 61), 0.26), 0.56)
            for side in (-1.0, 1.0):
                GL.glBegin(GL.GL_LINE_STRIP)
                GL.glVertex3f(side * 0.50, -0.13 + cheek_lift, 0.858)
                GL.glVertex3f(side * 0.59, -0.18 + cheek_lift, 0.806)
                GL.glVertex3f(side * 0.63, -0.24 + cheek_lift * 0.65, 0.768)
                GL.glEnd()
            GL.glBegin(GL.GL_LINE_STRIP)
            for point in range(9):
                x = -0.22 + point * 0.055
                y = -0.57 + math.sin(point / 8.0 * math.pi) * 0.028
                GL.glVertex3f(x, y, 0.755)
            GL.glEnd()

            if expression["fatigue"] > 0.12:
                GL.glLineWidth(1.5 + expression["fatigue"] * 1.5)
                gl_color(mix_color(npc.profile.skin, RED, 0.42))
                for eye_x in (-0.34, 0.34):
                    GL.glBegin(GL.GL_LINE_STRIP)
                    GL.glVertex3f(eye_x - 0.13, 0.01, 0.94)
                    GL.glVertex3f(eye_x, -0.015 - expression["fatigue"] * 0.018, 0.95)
                    GL.glVertex3f(eye_x + 0.13, 0.01, 0.94)
                    GL.glEnd()
            if expression["spiritual_glow"] > 0.0:
                GL.glLineWidth(2.0)
                gl_color(CYAN, 0.86)
                GL.glBegin(GL.GL_LINE_LOOP)
                for point in range(12):
                    angle = math.tau * point / 12.0
                    GL.glVertex3f(math.cos(angle) * 0.075, 0.70 + math.sin(angle) * 0.075, 0.94)
                GL.glEnd()
        GL.glPointSize(1.0)
        GL.glLineWidth(1.0)
        GL.glEnable(GL.GL_LIGHTING)

    def _draw_action_emote_particles(
        self,
        gesture_key: str,
        gesture_age: float,
        npc_index: int,
        accent: Tuple[int, int, int],
        motion: Mapping[str, float],
        gesture_weight: float,
    ) -> None:
        """Draw a tiny bounded accent field in head-local portrait space."""
        if gesture_key not in ("spitting", "chomping", "head_banging"):
            return
        gesture_weight = float(clamp(gesture_weight, 0.0, 1.0))
        if gesture_weight <= 0.001:
            return

        spit = float(motion.get("spit_strength", 0.0)) * gesture_weight
        crumbs = float(motion.get("crumb_strength", 0.0)) * gesture_weight
        sparks = float(motion.get("spark_strength", 0.0)) * gesture_weight
        if max(spit, crumbs, sparks) <= 0.001:
            return

        attribs = (
            GL.GL_ENABLE_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_POINT_BIT
            | GL.GL_LINE_BIT
            | GL.GL_CURRENT_BIT
        )
        GL.glPushAttrib(attribs)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_POINT_SMOOTH)

        if spit > 0.001:
            direction = -1.0 if npc_index % 2 else 1.0
            travel = float(clamp((gesture_age - 0.22) / 0.42, 0.0, 1.0))
            GL.glPointSize(4.8)
            GL.glBegin(GL.GL_POINTS)
            for droplet in range(7):
                lane = droplet - 3
                distance = 0.10 + travel * (0.26 + droplet * 0.035)
                jitter = math.sin((droplet + 1) * 12.9898 + npc_index * 1.71) * 0.018
                gl_color(mix_color(CYAN, GREEN, droplet / 8.0), spit * (0.92 - droplet * 0.055))
                GL.glVertex3f(
                    direction * distance + jitter * travel,
                    -0.33 + lane * 0.012 - travel * travel * (0.08 + droplet * 0.006),
                    1.03 + (droplet % 2) * 0.012,
                )
            GL.glEnd()

        if crumbs > 0.001:
            fall = float(clamp(gesture_age / 1.35, 0.0, 1.0))
            GL.glPointSize(4.0)
            GL.glBegin(GL.GL_POINTS)
            for crumb in range(5):
                direction = -1.0 if crumb % 2 else 1.0
                drift = 0.06 + crumb * 0.025
                gl_color(mix_color(GOLD, accent, 0.24), crumbs * (0.88 - crumb * 0.09))
                GL.glVertex3f(
                    direction * drift * (0.65 + fall),
                    -0.34 - fall * (0.05 + crumb * 0.015),
                    1.025 + (crumb % 3) * 0.008,
                )
            GL.glEnd()

        if sparks > 0.001:
            GL.glPointSize(5.2)
            GL.glBegin(GL.GL_POINTS)
            for spark in range(4):
                side = -1.0 if spark % 2 == 0 else 1.0
                height = 0.30 + (spark // 2) * 0.22
                gl_color(mix_color(GOLD, accent, spark / 5.0), sparks * (0.92 - spark * 0.08))
                GL.glVertex3f(side * (0.73 + spark * 0.035), height, 0.98)
            GL.glEnd()
            GL.glLineWidth(1.8)
            GL.glBegin(GL.GL_LINES)
            gl_color(GOLD, sparks * 0.72)
            for side in (-1.0, 1.0):
                x = side * 0.79
                GL.glVertex3f(x - 0.045, 0.53, 0.985)
                GL.glVertex3f(x + 0.045, 0.53, 0.985)
                GL.glVertex3f(x, 0.485, 0.985)
                GL.glVertex3f(x, 0.575, 0.985)
            GL.glEnd()
        GL.glPopAttrib()

    def _draw_halo(self, now: float, phase: float) -> None:
        """Render a depth-aware additive halo core and multi-band bloom."""
        pulse = 1.0 + math.sin(now * 1.8 + phase) * 0.025
        GL.glPushAttrib(
            GL.GL_ENABLE_BIT
            | GL.GL_COLOR_BUFFER_BIT
            | GL.GL_DEPTH_BUFFER_BIT
            | GL.GL_CURRENT_BIT
            | GL.GL_POINT_BIT
        )
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glPushMatrix()
        GL.glTranslatef(0.0, 1.15, 0.0)
        GL.glRotatef(68.0, 1.0, 0.0, 0.0)
        for layer_index, (base_radius, thickness, base_alpha) in enumerate(HALO_GLOW_LAYERS):
            radius = base_radius * pulse
            inner_radius = radius - thickness
            outer_radius = radius + thickness
            color = (
                mix_color(WHITE, GOLD, 0.16)
                if layer_index == 0
                else mix_color(WHITE, CYAN, min(1.0, layer_index / 3.0))
            )
            GL.glBegin(GL.GL_QUAD_STRIP)
            for point in range(65):
                angle = math.tau * point / 64.0
                cosine, sine = math.cos(angle), math.sin(angle)
                gl_color(color, base_alpha * 0.72)
                GL.glVertex3f(cosine * inner_radius, sine * inner_radius, 0.0)
                gl_color(color, base_alpha)
                GL.glVertex3f(cosine * outer_radius, sine * outer_radius, 0.0)
            GL.glEnd()

        # Orbiting motes make the glow perceptible even on bright backgrounds
        # without turning the ring into an opaque disc.
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(3.6)
        GL.glBegin(GL.GL_POINTS)
        for mote in range(12):
            angle = math.tau * mote / 12.0 + now * 0.18
            radius = (0.755 + math.sin(now * 2.3 + mote) * 0.018) * pulse
            gl_color(mix_color(CYAN, WHITE, 0.42), 0.42 + 0.20 * math.sin(now * 2.0 + mote) ** 2)
            GL.glVertex3f(math.cos(angle) * radius, math.sin(angle) * radius, 0.008)
        GL.glEnd()
        GL.glPopMatrix()
        GL.glPopAttrib()

    def _begin_2d(self) -> None:
        GL.glViewport(0, 0, WIDTH, HEIGHT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0.0, WIDTH, HEIGHT, 0.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def _draw_portrait_overlays(self, now: float) -> None:
        portrait_width = WIDTH / 4.0
        for index, npc in enumerate(self.state.npcs):
            x = index * portrait_width
            selected = index == self.state.selected_npc_index
            if npc.eliminated:
                draw_rect(x, 0, portrait_width, TOP_HEIGHT - 58, (35, 4, 7), 0.28)
            draw_rect(x, TOP_HEIGHT - 58, portrait_width, 58, (10, 13, 15), 0.82)
            border = RED if npc.eliminated else (GOLD if selected else mix_color(LINE, npc.profile.accent, 0.35))
            draw_outline(x + 2, 2, portrait_width - 4, TOP_HEIGHT - 4, border, 3.0 if selected else 1.0)
            self.text.draw(npc.name, x + portrait_width / 2, TOP_HEIGHT - 53, WHITE, "portrait", True)
            voice_caption_text = self.audio.caption_for_npc(index, now)
            voice_level = self.audio.voice_envelope(index, now)
            if voice_level > 0.01:
                for bar in range(5):
                    bar_height = 3.0 + voice_level * (5.0 + 8.0 * abs(math.sin(now * 9.0 + bar * 1.3)))
                    draw_rect(x + 12 + bar * 6, TOP_HEIGHT - 39 - bar_height, 3, bar_height, GOLD, 0.92)
            show_idle_label = not voice_caption_text and npc_idle_enabled(
                self.state.phase,
                self.state.player.alive,
                npc.alive,
                self.state.is_ending,
            ) and action_gesture_override_weight(
                npc.gesture_turn,
                max(0.0, now - npc.gesture_changed_at),
                True,
            ) < 0.5
            if show_idle_label:
                idle_frame = idle_emote_frame(
                    self.state.seed,
                    index,
                    max(0.0, now - self.state.idle_epoch),
                )
                gesture_label = IDLE_EMOTE_LABELS[idle_frame.emote_key]
            else:
                gesture_label = GESTURE_LABELS.get(
                    npc.gesture_key,
                    npc.gesture_key.replace("_", " ").title(),
                )
            if npc.eliminated:
                portrait_subtitle = "ELIMINATED"
            elif voice_caption_text:
                portrait_subtitle = f"8-BIT VOICE  ·  {voice_caption_text}"
            else:
                portrait_subtitle = f"{npc.profile.role}  ·  {gesture_label}"
            subtitle_font = "small" if npc.eliminated else "tiny"
            subtitle_limit = int(portrait_width - 18)
            if self.text.measure(portrait_subtitle, subtitle_font)[0] > subtitle_limit:
                shortened = portrait_subtitle
                while shortened and self.text.measure(shortened + "...", subtitle_font)[0] > subtitle_limit:
                    shortened = shortened[:-1]
                portrait_subtitle = shortened.rstrip(" ·") + "..."
            self.text.draw(
                portrait_subtitle,
                x + portrait_width / 2,
                TOP_HEIGHT - 30,
                RED if npc.eliminated else (GOLD if voice_caption_text else (npc.profile.accent if selected else MUTED)),
                subtitle_font,
                True,
            )
            if npc.eliminated:
                GL.glLineWidth(13.0)
                gl_color((105, 9, 14), 0.82)
                GL.glBegin(GL.GL_LINES)
                GL.glVertex2f(x + 30, 25)
                GL.glVertex2f(x + portrait_width - 30, TOP_HEIGHT - 82)
                GL.glVertex2f(x + portrait_width - 30, 25)
                GL.glVertex2f(x + 30, TOP_HEIGHT - 82)
                GL.glEnd()
                GL.glLineWidth(7.0)
                gl_color((247, 42, 47), 0.96)
                GL.glBegin(GL.GL_LINES)
                GL.glVertex2f(x + 30, 25)
                GL.glVertex2f(x + portrait_width - 30, TOP_HEIGHT - 82)
                GL.glVertex2f(x + portrait_width - 30, 25)
                GL.glVertex2f(x + 30, TOP_HEIGHT - 82)
                GL.glEnd()
                GL.glLineWidth(1.0)

        if self.next_participant_action_time is not None and self.player_turn_cooldown:
            remaining = max(0.0, self.next_participant_action_time - now)
            signal_text = f"{RELEASE_LABEL} | ONE-SECOND BEAT | YOUR TURN IN {remaining:.1f}s"
            signal_color = CYAN
        elif self.next_participant_action_time is not None and (
            self.state.phase == "resolving_npcs"
            or (self.state.player.eliminated and self.state.phase == "autonomous")
        ):
            remaining = max(0.0, self.next_participant_action_time - now)
            next_actor_index = self.state._next_npc_actor_index()
            if next_actor_index is None and self.state.phase == "autonomous":
                next_actor_index = next(
                    (
                        index
                        for index in range(1, len(self.state.participants))
                        if self.state.participants[index].alive
                    ),
                    None,
                )
            next_name = (
                self.state.participants[next_actor_index].name.split()[0].upper()
                if next_actor_index is not None
                else "ROOM"
            )
            signal_text = f"{RELEASE_LABEL} | ONE-SECOND BEAT | NEXT {next_name} IN {remaining:.1f}s"
            signal_color = MAGENTA
        else:
            signal_text = (
                f"{RELEASE_LABEL} | FIFTH SIGNAL: DISCOVERED"
                if self.state.fifth_signal_found
                else f"{RELEASE_LABEL} | SEALED ROOM | FIFTH SIGNAL: UNKNOWN"
            )
            signal_color = CYAN if self.state.fifth_signal_found else GOLD
        ribbon_width = min(700.0, WIDTH * 0.36)
        ribbon = ((WIDTH - ribbon_width) / 2.0, TOP_HEIGHT + 2.0, ribbon_width, 20.0)
        draw_gradient_rect(*ribbon, mix_color((8, 12, 14), signal_color, 0.09), (4, 7, 9), 0.94)
        draw_textured_nine_slice(
            self.materials.get("ui_button"),
            ribbon,
            signal_color,
            0.14,
            5.0,
        )
        draw_outline(*ribbon, signal_color, 1.5)
        self.text.draw(signal_text, WIDTH / 2.0, TOP_HEIGHT + 4.0, signal_color, "small", True)

    def _draw_lower_interface(self) -> None:
        draw_gradient_rect(
            0,
            TOP_HEIGHT,
            WIDTH,
            HEIGHT - TOP_HEIGHT,
            mix_color(INK, CYAN, 0.055),
            mix_color(INK, (2, 5, 8), 0.42),
        )
        # Fine fixed-function scanlines and a cyan horizon make the lower HUD
        # read as one luminous instrument surface rather than flat rectangles.
        draw_rect(0, TOP_HEIGHT, WIDTH, 2, CYAN, 0.24)
        for scan_y in range(TOP_HEIGHT + 5, HEIGHT, 7):
            draw_rect(0, scan_y, WIDTH, 1, CYAN, 0.018)
        # Reserve a clean twenty-pixel channel for the release/countdown strip.
        # The panels keep their original bottom edge, so choices do not move.
        panel_y = TOP_HEIGHT + 26
        panel_height = 270
        self._draw_story_panel(18, panel_y, 798, panel_height)
        self._draw_status_panel(830, panel_y, 687, panel_height)
        self._draw_inventory_panel(1531, panel_y, 371, panel_height)
        if self.state.player.eliminated and not self.state.is_ending:
            self._draw_spectator_panel(TOP_HEIGHT + 308)
        else:
            self._draw_story_choices(TOP_HEIGHT + 308)
        if not self.state.is_ending and self.state.player.alive:
            self._draw_action_choices(TOP_HEIGHT + 392)

    def _draw_spectator_panel(self, y: float) -> None:
        margin = 14.0
        height = 145.0
        self._draw_premium_panel(
            (margin, y, WIDTH - margin * 2, height),
            RED,
            0.20,
        )
        self.text.draw(
            f"{self.state.player.name.upper()} ELIMINATED | AUTONOMOUS SURVIVOR ROUND",
            WIDTH / 2.0,
            y + 19,
            RED,
            "portrait",
            True,
        )
        living = ", ".join(participant.name for participant in self.state.living_participants)
        self.text.wrapped(
            f"No player action is available. The remaining participants continue choosing one interaction each. Alive: {living}.",
            margin + 70,
            y + 61,
            int(WIDTH - margin * 2 - 140),
            WHITE,
            "normal",
            2,
            23,
        )
        self.text.draw("The next survivor round resolves automatically. F5 restarts; Esc quits.", WIDTH / 2.0, y + 116, MUTED, "small", True)

    def _draw_premium_panel(
        self,
        rect: Tuple[float, float, float, float],
        accent: Tuple[int, int, int] = CYAN,
        texture_alpha: float = 0.14,
    ) -> None:
        x, y, width, height = rect
        draw_gradient_rect(
            x,
            y,
            width,
            height,
            mix_color(PANEL_ALT, accent, 0.08),
            mix_color(PANEL, INK, 0.30),
            0.98,
        )
        draw_textured_nine_slice(
            self.materials.get("ui_panel"),
            rect,
            mix_color(accent, WHITE, 0.22),
            texture_alpha,
            16.0,
        )
        draw_outline(x, y, width, height, mix_color(LINE, accent, 0.32), 1.0)
        draw_outline(x + 3, y + 3, width - 6, height - 6, mix_color(INK, accent, 0.18), 1.0)
        draw_rect(x + 14, y + 2, max(0.0, width - 28), 1, accent, 0.36)

    def _panel(self, x: float, y: float, width: float, height: float, title: str) -> None:
        self._draw_premium_panel((x, y, width, height), CYAN, 0.11)
        draw_gradient_rect(
            x + 3,
            y + 3,
            width - 6,
            31,
            mix_color(PANEL_ALT, CYAN, 0.12),
            mix_color(PANEL_ALT, INK, 0.18),
            0.94,
        )
        self.text.draw(
            self._fit_text(title, "normal", width - 24),
            x + 12,
            y + 7,
            WHITE,
            "normal",
        )

    def _draw_story_panel(self, x: float, y: float, width: float, height: float) -> None:
        activity_record = self._selected_activity_record()
        activity_index = self._activity_record_index()
        activity_total = self._activity_record_count()
        live_activity = bool(
            self._has_live_activity()
            and self.activity_history_index is None
            and activity_index == activity_total - 1
        )
        title = (
            self.state.ending_title
            if self.state.is_ending
            else f"Round {self.state.turn} | {self.state.current_scene.title}"
        )
        if activity_record is not None:
            available_title_width = int(width - 154)
            if self.text.measure(title, "normal")[0] > available_title_width:
                trimmed = title
                while trimmed and self.text.measure(trimmed + "...", "normal")[0] > available_title_width:
                    trimmed = trimmed[:-1]
                title = trimmed.rstrip() + "..."
        self._panel(x, y, width, height, title)
        if activity_record is not None:
            older_enabled = activity_index is not None and activity_index > 0
            newer_enabled = (
                activity_index is not None
                and activity_index < activity_total - 1
            )
            self._draw_button(
                (x + width - 132, y + 6, 60, 22),
                "OLDER",
                "activity_scroll",
                -1,
                GOLD,
                enabled=older_enabled,
            )
            self._draw_button(
                (x + width - 68, y + 6, 60, 22),
                "NEWER",
                "activity_scroll",
                1,
                CYAN,
                enabled=newer_enabled,
            )

        latest_round_was_investigation = bool(
            self.state.last_round_events
            and self.state.last_round_events[0].action == "investigate"
        )
        if self.state.is_ending:
            prompt = self.state.ending_text
        elif live_activity:
            prompt = self.state.message
        elif latest_round_was_investigation:
            prompt = self.state.message
        else:
            prompt = self.state.current_scene.prompt.format(
                npc=self.state.story_target_name,
                player=self.state.player.name,
            )
        # Activity rows are the room's readable audit trail, so give them the
        # vertical space instead of repeating the scene prompt above them. The
        # normal story view still receives the full prompt and message area.
        if activity_record is not None:
            cursor = y + 43
        else:
            cursor = self.text.wrapped(
                prompt,
                x + 12,
                y + 43,
                int(width - 24),
                MUTED,
                "small",
                6,
                19,
            )
            cursor += 4
        if activity_record is not None:
            sequence_label = (
                "NPC SURVIVOR ROUND -> EACH LIVING NPC ONCE"
                if not activity_record.player_led
                else "PLAYER FIRST -> EACH LIVING NPC ONCE"
            )
            position = (activity_index + 1) if activity_index is not None else 1
            activity_kind = (
                f"LIVE MOMENT {len(activity_record.events)}"
                if live_activity
                else f"ACTIVITY {position}/{activity_total}"
            )
            self.text.draw(
                f"{activity_kind} | "
                f"ROUND {activity_record.round_number} | {sequence_label}",
                x + 12,
                cursor,
                GOLD,
                "tiny",
            )
            cursor += 18
            show_reasoning_rows = len(activity_record.events) <= 4
            for event in activity_record.events[:5]:
                actor = self.state.participants[event.actor_index]
                target = (
                    self.state.participants[event.target_index]
                    if event.target_index is not None
                    else None
                )
                target_text = (
                    f" -> {target.name.split()[0]}"
                    if target is not None and target is not actor
                    else " -> SELF"
                    if target is actor
                    else ""
                )
                supply_text = ""
                if event.item_key:
                    supply_text = f" {ITEM_BY_KEY[event.item_key].short_label.upper()}"
                elif event.exchange:
                    supply_text = (
                        f" {ITEM_BY_KEY[event.exchange.offered_item].short_label.upper()}"
                        f"/{ITEM_BY_KEY[event.exchange.requested_item].short_label.upper()}"
                    )
                result_text = "SUCCESS" if event.success else "DECLINED/FAILED"
                pose_text = (
                    f" | {event.gesture_key.replace('_', ' ').upper()}"
                    if event.gesture_key
                    else ""
                )
                header = (
                    f"{actor.name.split()[0].upper()}{target_text.upper()} | "
                    f"{event.action.upper()}{supply_text} | {result_text}{pose_text}"
                )
                compact_impact = event.impact
                for participant in self.state.participants:
                    compact_impact = compact_impact.replace(
                        f"{participant.name}:",
                        f"{participant.name.split()[0]}:",
                    )
                color = CYAN if event.actor_index == 0 else actor.profile.accent
                if event.action in ("fight", "self_eliminate") or event.eliminated_indices:
                    color = RED
                elif event.action in ("antagonize", "steal"):
                    color = CORAL
                elif event.action == "flirt":
                    color = MAGENTA
                elif event.action == "compliment":
                    color = GREEN
                self.text.draw(
                    self._fit_text(header, "tiny", width - 24),
                    x + 12,
                    cursor,
                    color,
                    "tiny",
                )
                cursor += 13
                self.text.draw(
                    self._fit_text(f"OUTCOME {event.summary}", "tiny", width - 34),
                    x + 22,
                    cursor,
                    WHITE,
                    "tiny",
                )
                cursor += 13
                self.text.draw(
                    self._fit_text(f"IMPACT {compact_impact}", "tiny", width - 34),
                    x + 22,
                    cursor,
                    GOLD,
                    "tiny",
                )
                cursor += 13
                if show_reasoning_rows and event.reasoning:
                    self.text.draw(
                        self._fit_text(f"WHY {event.reasoning}", "tiny", width - 34),
                        x + 22,
                        cursor,
                        MUTED,
                        "tiny",
                    )
                    cursor += 13
            if activity_record.notices and cursor <= y + height - 15:
                notice = " | ".join(activity_record.notices)
                wrapped = self.text.wrap(notice, int(width - 24), "tiny")
                self.text.draw(wrapped[0] if wrapped else notice, x + 12, cursor, RED, "tiny")
            return
        remaining_lines = max(2, int((y + height - 9 - cursor) / 17))
        self.text.wrapped(
            self.state.message,
            x + 12,
            cursor,
            int(width - 24),
            WHITE,
            "small",
            remaining_lines,
            17,
        )

    def _status_color(self, value: int, spiritual: bool = False) -> Tuple[int, int, int]:
        if spiritual:
            if value < 0:
                return RED
            return CYAN if value > 0 else MUTED
        if value < 34:
            return RED
        if value < 67:
            return GOLD
        return GREEN

    def _draw_status_panel(self, x: float, y: float, width: float, height: float) -> None:
        self._panel(x, y, width, height, "Status indexes (PHYS 0 = eliminated)")
        name_width = 145
        cell_width = (width - name_width - 16) / 6.0
        header_y = y + 48
        for column, key in enumerate(STATUS_KEYS):
            self.text.draw(
                STATUS_SHORT[key],
                x + name_width + cell_width * (column + 0.5),
                header_y,
                MUTED,
                "tiny",
                True,
            )
        row_y = y + 79
        for index, participant in enumerate(self.state.participants):
            selected = index == self.state.selected_npc_index + 1
            if participant.eliminated:
                row_color = mix_color(PANEL, (104, 12, 17), 0.46)
            elif index == 0:
                row_color = mix_color(PANEL, PLAYER_PROFILE.accent, 0.14)
            elif selected:
                row_color = mix_color(PANEL, participant.profile.accent, 0.18)
            else:
                row_color = PANEL_ALT if index % 2 == 0 else PANEL
            draw_rect(x + 6, row_y - 3, width - 12, 34, row_color)
            label_color = RED if participant.eliminated else (GOLD if selected else (CYAN if index == 0 else WHITE))
            label = participant.name.split()[0] + ("  X" if participant.eliminated else "")
            label = self._fit_text(label, "small", name_width - 22)
            self.text.draw(label, x + 12, row_y + 5, label_color, "small")
            for column, key in enumerate(STATUS_KEYS):
                value = participant.statuses[key]
                center_x = x + name_width + cell_width * (column + 0.5)
                color = RED if participant.eliminated else self._status_color(value, key == "spiritual")
                self.text.draw(str(value), center_x, row_y + 4, color, "small", True)
                if key != "spiritual":
                    bar_x = center_x - cell_width * 0.34
                    bar_width = cell_width * 0.68
                    draw_rect(bar_x, row_y + 27, bar_width, 3, (48, 54, 56))
                    draw_rect(bar_x, row_y + 27, bar_width * value / 100.0, 3, color)
            row_y += 38

    def _draw_inventory_panel(self, x: float, y: float, width: float, height: float) -> None:
        inventory_owner = possessive_name(self.state.player.name)
        title = (
            f"{inventory_owner} inventory (locked)"
            if self.state.player.eliminated
            else f"{inventory_owner} inventory"
        )
        self._panel(x, y, width, height, title)
        row_y = y + 53
        mouse = pygame.mouse.get_pos()
        for spec in ITEM_SPECS:
            row = (x + 6, row_y, width - 12, 38)
            selected = spec.key == self.state.selected_item
            hovered = HitRegion(row, "", None).contains(mouse)
            fill = mix_color(PANEL_ALT, spec.color, 0.22 if selected else 0.09)
            if hovered:
                fill = mix_color(fill, WHITE, 0.07)
            draw_rect(*row, fill)
            if selected:
                draw_outline(*row, spec.color, 2.0)
            draw_rect(x + 14, row_y + 12, 13, 13, spec.color)
            self.text.draw(spec.label, x + 38, row_y + 8, WHITE if selected else MUTED, "small")
            count = f"{self.state.player.inventory[spec.key]:,}"
            count_width = self.text.measure(count, "small")[0]
            self.text.draw(count, x + width - 14 - count_width, row_y + 8, spec.color, "small")
            self.regions.append(HitRegion(row, "item", spec.key))
            row_y += 42

    def _draw_trade_item_column(
        self,
        side: str,
        x: float,
        y: float,
        width: float,
    ) -> None:
        draft = self.state.trade_draft
        selected_key = draft.offered_item if side == "offer" else draft.requested_item
        mouse = pygame.mouse.get_pos()
        for spec in ITEM_SPECS:
            rect = (x, y, width, 27.0)
            region = HitRegion(rect, "trade_item", (side, spec.key))
            selected = spec.key == selected_key
            hovered = region.contains(mouse)
            fill = mix_color(PANEL_ALT, spec.color, 0.24 if selected else 0.08)
            if hovered:
                fill = mix_color(fill, WHITE, 0.08)
            draw_rect(*rect, fill)
            if selected:
                draw_outline(*rect, spec.color, 2.0)
            draw_rect(x + 9, y + 8, 11, 11, spec.color)
            self.text.draw(spec.label, x + 28, y + 4, WHITE if selected else MUTED, "small")
            if side == "offer":
                stock_text = f"You have {self.state.player.inventory[spec.key]:,}"
            else:
                stock_text = "NPC stock private"
            stock_width = self.text.measure(stock_text, "tiny")[0]
            self.text.draw(stock_text, x + width - stock_width - 9, y + 6, spec.color if selected else MUTED, "tiny")
            self.regions.append(region)
            y += 30.0

    def _draw_trade_quantity_controls(
        self,
        side: str,
        x: float,
        y: float,
        width: float,
    ) -> None:
        accent = CYAN if side == "offer" else CORAL
        button_width = 54.0
        gap = 6.0
        field_width = width - button_width * 4 - gap * 4
        cursor = x
        for label, amount in (("-10", -10), ("-1", -1)):
            self._draw_button(
                (cursor, y, button_width, 36),
                label,
                "trade_adjust",
                (side, amount),
                accent,
            )
            cursor += button_width + gap
        field = (cursor, y, field_width, 36.0)
        active = side == self.trade_edit_side
        draw_rect(*field, mix_color(PANEL_ALT, accent, 0.22 if active else 0.10))
        draw_outline(*field, accent if active else LINE, 2.0 if active else 1.0)
        quantity = self.trade_quantity_text[side]
        display_quantity = quantity + (" |" if active and quantity else "")
        if not display_quantity:
            display_quantity = "type quantity"
        quantity_width = self.text.measure(display_quantity, "normal")[0]
        self.text.draw(
            display_quantity,
            cursor + (field_width - quantity_width) / 2.0,
            y + 7,
            WHITE if quantity else MUTED,
            "normal",
        )
        self.regions.append(HitRegion(field, "trade_focus", side))
        cursor += field_width + gap
        for label, amount in (("+1", 1), ("+10", 10)):
            self._draw_button(
                (cursor, y, button_width, 36),
                label,
                "trade_adjust",
                (side, amount),
                accent,
            )
            cursor += button_width + gap

    def _draw_trade_builder(self) -> None:
        draft = self.state.trade_draft
        if draft is None:
            return
        target = self.state.npcs[draft.target_index]
        draw_gradient_rect(
            0,
            TOP_HEIGHT,
            WIDTH,
            HEIGHT - TOP_HEIGHT,
            mix_color((7, 10, 12), target.profile.accent, 0.08),
            (4, 7, 10),
            0.99,
        )
        self._draw_premium_panel(
            (14, TOP_HEIGHT + 6, WIDTH - 28, HEIGHT - TOP_HEIGHT - 12),
            target.profile.accent,
            0.13,
        )

        self.text.draw(
            f"Private-stock trade proposal with {target.name}",
            28,
            TOP_HEIGHT + 14,
            WHITE,
            "heading",
        )
        self.text.draw(
            f"Select any item and exact quantity. Submitting is {possessive_name(self.state.player.name)} one action; every living NPC then acts once.",
            28,
            TOP_HEIGHT + 43,
            MUTED,
            "small",
        )

        left_x = 28.0
        column_gap = 30.0
        column_width = (WIDTH - left_x * 2.0 - column_gap) / 2.0
        right_x = left_x + column_width + column_gap
        self.text.draw("YOU OFFER", left_x, TOP_HEIGHT + 69, CYAN, "normal")
        self.text.draw("YOU REQUEST FROM NPC", right_x, TOP_HEIGHT + 69, CORAL, "normal")
        self._draw_trade_item_column("offer", left_x, TOP_HEIGHT + 94, column_width)
        self._draw_trade_item_column("request", right_x, TOP_HEIGHT + 94, column_width)

        self.text.draw("Offer quantity", left_x, TOP_HEIGHT + 246, MUTED, "small")
        self.text.draw("Request quantity", right_x, TOP_HEIGHT + 246, MUTED, "small")
        self._draw_trade_quantity_controls("offer", left_x, TOP_HEIGHT + 268, column_width)
        self._draw_trade_quantity_controls("request", right_x, TOP_HEIGHT + 268, column_width)

        if not self.trade_quantity_text["offer"] or not self.trade_quantity_text["request"]:
            local_error = "Enter a whole-number quantity on both sides of the proposal."
        else:
            local_error = self.state.trade_local_error(draft)
        chance = None if local_error else self.state.trade_willingness(draft)
        chance_color = (
            RED
            if chance is None or chance < 0.40
            else (GREEN if chance >= 0.67 else GOLD)
        )
        offered = ITEM_BY_KEY[draft.offered_item]
        requested = ITEM_BY_KEY[draft.requested_item]
        offer_quantity = self.trade_quantity_text["offer"] or "?"
        request_quantity = self.trade_quantity_text["request"] or "?"
        summary = (
            f"{offer_quantity} {offered.short_label.lower()}  ->  "
            f"{request_quantity} {requested.short_label.lower()}"
        )
        self.text.draw(summary, 28, TOP_HEIGHT + 316, WHITE, "normal")
        willingness = "Proposal unavailable" if chance is None else f"Willingness: {chance:.0%}"
        willingness_width = self.text.measure(willingness, "normal")[0]
        self.text.draw(willingness, WIDTH - 28 - willingness_width, TOP_HEIGHT + 316, chance_color, "normal")
        self.text.draw(
            "PHYS + EMO + COG + SOC + SENT + SP for both people affect this chance; item value and quantity affect fairness.",
            28,
            TOP_HEIGHT + 341,
            MUTED,
            "tiny",
        )
        self.text.wrapped(
            local_error or self.state.message,
            28,
            TOP_HEIGHT + 386,
            WIDTH - 650,
            RED if local_error else MUTED,
            "tiny",
            2,
            14,
        )
        self._draw_button(
            (WIDTH - 540, TOP_HEIGHT + 452, 220, 48),
            "Cancel (Esc)",
            "trade_cancel",
            None,
            MUTED,
        )
        self._draw_button(
            (WIDTH - 306, TOP_HEIGHT + 452, 278, 48),
            "Fix proposal" if chance is None else f"Propose trade ({chance:.0%})",
            "trade_confirm",
            None,
            chance_color,
            True,
            chance is not None,
        )

    def _draw_name_entry_overlay(self) -> None:
        draw_rect(0, 0, WIDTH, HEIGHT, (3, 7, 10), 0.88)
        panel_width = 820.0
        panel_height = 470.0
        panel_x = (WIDTH - panel_width) / 2.0
        panel_y = (HEIGHT - panel_height) / 2.0
        panel = (panel_x, panel_y, panel_width, panel_height)
        # Nested restrained glows give the start card depth without obscuring
        # the animated room that remains visible behind it.
        draw_rect(panel[0] - 12, panel[1] - 12, panel[2] + 24, panel[3] + 24, CYAN, 0.06)
        self._draw_premium_panel(panel, CYAN, 0.20)
        draw_outline(*panel, CYAN, 3.0)
        draw_outline(panel[0] + 8, panel[1] + 8, panel[2] - 16, panel[3] - 16, LINE, 1.0)

        self.text.draw("IDENTIFY THE FIFTH CHANNEL", WIDTH / 2.0, panel_y + 38, CYAN, "title", True)
        self.text.draw(
            "Choose the name carried into the sealed room",
            WIDTH / 2.0,
            panel_y + 92,
            WHITE,
            "portrait",
            True,
        )
        self.text.draw(
            f"1-{MAX_PLAYER_NAME_LENGTH} chars  |  letters, numbers, spaces, ' and -",
            WIDTH / 2.0,
            panel_y + 128,
            MUTED,
            "small",
            True,
        )

        field = (panel_x + 70.0, panel_y + 174.0, panel_width - 140.0, 72.0)
        draw_rect(*field, mix_color(PANEL_ALT, CYAN, 0.12), 1.0)
        draw_outline(*field, CYAN, 2.0)
        display_text = self.name_entry_text + " |"
        if not self.name_entry_text:
            display_text = "Type a name... |"
        self.text.draw(
            display_text,
            field[0] + 18,
            field[1] + 18,
            WHITE if self.name_entry_text else MUTED,
            "heading",
        )
        count_text = f"{len(self.name_entry_text)}/{MAX_PLAYER_NAME_LENGTH}"
        count_width = self.text.measure(count_text, "small")[0]
        self.text.draw(
            count_text,
            field[0] + field[2] - count_width - 14,
            field[1] + 41,
            CYAN,
            "small",
        )
        self.regions.append(HitRegion(field, "name_field", None))

        if self.name_entry_error:
            self.text.draw(self.name_entry_error, WIDTH / 2.0, panel_y + 264, RED, "small", True)
        else:
            self.text.draw(
                "Enter confirms. Escape keeps the default name Tyler.",
                WIDTH / 2.0,
                panel_y + 264,
                MUTED,
                "small",
                True,
            )
        self._draw_button(
            (panel_x + 70, panel_y + 314, 326, 64),
            "Use Tyler (Esc)",
            "name_cancel",
            None,
            MUTED,
        )
        self._draw_button(
            (panel_x + 424, panel_y + 314, 326, 64),
            "Enter the Room",
            "name_confirm",
            None,
            CYAN,
            True,
        )
        self.text.draw(
            "The accepted name persists when the room is restarted.",
            WIDTH / 2.0,
            panel_y + 416,
            GOLD,
            "tiny",
            True,
        )

    def _draw_game_over_overlay(self, now: float) -> None:
        draw_gradient_rect(0, 0, WIDTH, HEIGHT, (14, 5, 12), (2, 5, 9), 0.96)
        for scan_y in range(4, HEIGHT, 8):
            draw_rect(0, scan_y, WIDTH, 1, RED, 0.018)
        self.text.draw("FINAL SIGNAL", WIDTH / 2.0, 22, RED, "title", True)
        self.text.draw(self.state.ending_title, WIDTH / 2.0, 63, WHITE, "portrait", True)

        summary_panel = (28.0, 106.0, 500.0, 880.0)
        winner = self.state.last_survivor
        winner_accent = (
            mix_color(CYAN, GOLD, 0.42)
            if winner is self.state.player
            else (winner.profile.accent if winner is not None else RED)
        )
        self._draw_premium_panel(summary_panel, winner_accent, 0.18)
        draw_outline(*summary_panel, winner_accent, 3.0)
        outcome_label = "PLAYER SIGNAL ASCENDANT" if winner is self.state.player else "NPC SIGNAL ASCENDANT"
        self.text.draw(outcome_label, 50, 125, winner_accent, "heading")

        stage_rect = (48.0, 160.0, 460.0, 360.0)
        draw_gradient_rect(*stage_rect, mix_color(INK, winner_accent, 0.15), (3, 6, 10), 1.0)
        self._draw_victory_stage(now, stage_rect)
        draw_textured_nine_slice(
            self.materials.get("ui_panel"),
            stage_rect,
            winner_accent,
            0.30,
            12.0,
        )
        draw_outline(*stage_rect, winner_accent, 2.0)
        stage_caption = self.audio.stage_caption(now)
        if winner is self.state.player and winner is not None:
            name_font = self._fitting_font(
                winner.name.upper(),
                ("title", "heading", "portrait", "normal", "small"),
                stage_rect[2] - 48,
            )
            stage_player_name = self._fit_text(
                winner.name.upper(),
                name_font,
                stage_rect[2] - 48,
            )
            draw_rect(stage_rect[0] + 18, stage_rect[1] + 84, stage_rect[2] - 36, 65, INK, 0.72)
            self.text.draw(
                stage_player_name,
                stage_rect[0] + stage_rect[2] / 2,
                stage_rect[1] + 93,
                WHITE,
                name_font,
                True,
            )
            celebration_label = PLAYER_VICTORY_EFFECTS[self.player_victory_style]
        elif winner is not None:
            npc_index = self.state.npcs.index(winner)
            finale_frame = victory_frame(
                self.seed,
                npc_index,
                self._victory_elapsed(now),
            )
            celebration_label = (
                finale_frame.meme_caption
                if int(self._victory_elapsed(now) / 2.4) % 2
                else finale_frame.celebration_label
            )
        else:
            celebration_label = "FINAL CHANNEL UNKNOWN"
        draw_rect(stage_rect[0] + 7, stage_rect[1] + stage_rect[3] - 48, stage_rect[2] - 14, 41, INK, 0.82)
        if winner is not None:
            lower_name_font = self._fitting_font(
                winner.name.upper(),
                ("portrait", "normal", "small", "tiny"),
                stage_rect[2] - 30,
            )
            self.text.draw(
                self._fit_text(winner.name.upper(), lower_name_font, stage_rect[2] - 30),
                stage_rect[0] + stage_rect[2] / 2,
                stage_rect[1] + stage_rect[3] - 44,
                WHITE,
                lower_name_font,
                True,
            )
        stage_line = self._fit_text(
            stage_caption or celebration_label,
            "tiny",
            stage_rect[2] - 28,
        )
        self.text.draw(
            stage_line,
            stage_rect[0] + stage_rect[2] / 2,
            stage_rect[1] + stage_rect[3] - 21,
            GOLD if stage_caption else winner_accent,
            "tiny",
            True,
        )

        self.text.wrapped(
            self.state.ending_text,
            50,
            550,
            456,
            WHITE,
            "small",
            6,
            22,
        )

        total_actions = sum(len(record.events) for record in self.state.round_history)
        self.text.draw("ROOM RECORD", 50, 700, GOLD, "normal")
        self.text.draw(
            f"R {len(self.state.round_history)}  |  ACTIONS {total_actions}  |  FIFTH SIGNAL "
            f"{'FOUND' if self.state.fifth_signal_found else 'UNRESOLVED'}",
            50,
            730,
            MUTED,
            "tiny",
        )
        self.text.draw("FINAL CHANNELS", 50, 765, GOLD, "normal")
        for participant_index, participant in enumerate(self.state.participants):
            survivor = participant is self.state.last_survivor
            marker = "ALIVE" if survivor else "LOST"
            color = GREEN if survivor else RED
            column = 0 if participant_index < 3 else 1
            row = participant_index if column == 0 else participant_index - 3
            channel_label = self._fit_text(
                f"{marker}: {participant.name}",
                "tiny",
                210,
            )
            self.text.draw(
                channel_label,
                55 + column * 225,
                800 + row * 26,
                color,
                "tiny",
            )

        self._draw_button(
            (50, 900, 456, 58),
            "Restart the Room (Enter)",
            "restart",
            None,
            winner_accent,
            True,
        )
        self.text.draw("Esc quits  |  M toggles 8-bit voices", 278, 963, MUTED, "tiny", True)
        self._draw_end_activity_archive(548.0, 106.0, 1344.0, 880.0)

    def _draw_end_activity_archive(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Render every field from one immutable record with full-history paging."""
        self._draw_premium_panel((x, y, width, height), CYAN, 0.16)
        draw_outline(x, y, width, height, CYAN, 2.0)
        record = self._selected_activity_record()
        record_index = self._activity_record_index()
        total = len(self.state.round_history)

        self.text.draw("COMPLETE ACTIVITY ARCHIVE", x + 18, y + 18, CYAN, "heading")
        if record is None or record_index is None:
            self.text.wrapped(
                "No completed round was recorded before the room reached its final state.",
                x + 28,
                y + 92,
                int(width - 56),
                MUTED,
                "normal",
                4,
                24,
            )
            return

        older_enabled = record_index > 0
        newer_enabled = record_index < total - 1
        self._draw_button(
            (x + width - 214, y + 12, 92, 32),
            "OLDER",
            "activity_scroll",
            -1,
            GOLD,
            enabled=older_enabled,
        )
        self._draw_button(
            (x + width - 112, y + 12, 92, 32),
            "NEWER",
            "activity_scroll",
            1,
            CYAN,
            enabled=newer_enabled,
        )

        sequence_label = (
            "PLAYER FIRST, THEN EACH LIVING NPC"
            if record.player_led
            else "AUTONOMOUS: EACH LIVING NPC ONCE"
        )
        self.text.draw(
            f"ENTRY {record_index + 1}/{total}  |  ROUND {record.round_number}",
            x + 20,
            y + 58,
            GOLD,
            "normal",
        )
        self.text.draw(sequence_label, x + 20, y + 82, MUTED, "tiny")

        cursor = y + 112.0
        content_width = int(width - 54)
        for event in record.events:
            actor = self.state.participants[event.actor_index]
            target = (
                self.state.participants[event.target_index]
                if event.target_index is not None
                else None
            )
            event_color = CYAN if event.actor_index == 0 else actor.profile.accent
            if event.action in ("fight", "self_eliminate") or event.eliminated_indices:
                event_color = RED
            elif event.action in ("antagonize", "steal"):
                event_color = CORAL
            elif event.action == "flirt":
                event_color = MAGENTA
            elif event.action == "compliment":
                event_color = GREEN
            draw_rect(x + 18, cursor + 2, 4, 42, event_color)

            header = f"{actor.name}  ·  {event.action.replace('_', ' ').upper()}"
            if target is not None and target is not actor:
                header += f"  ->  {target.name}"
            if event.gesture_key:
                header += f"  ·  {GESTURE_LABELS[event.gesture_key]}"
            if event.item_key:
                header += f"  ·  {ITEM_BY_KEY[event.item_key].short_label}"
            if event.exchange:
                header += (
                    f"  ·  {event.exchange.offered_quantity} "
                    f"{ITEM_BY_KEY[event.exchange.offered_item].short_label} FOR "
                    f"{event.exchange.requested_quantity} "
                    f"{ITEM_BY_KEY[event.exchange.requested_item].short_label}"
                )
            header += "  ·  SUCCESS" if event.success else "  ·  DECLINED/FAILED"
            self.text.draw(header, x + 30, cursor, event_color, "small")
            cursor += 20
            summary_lines = self.text.wrap(event.summary, content_width, "tiny")
            for line in summary_lines:
                self.text.draw(line, x + 30, cursor, WHITE, "tiny")
                cursor += 15
            impact_lines = self.text.wrap(
                f"IMPACT: {event.impact}",
                content_width,
                "tiny",
            )
            for line in impact_lines:
                self.text.draw(line, x + 30, cursor, GOLD, "tiny")
                cursor += 15
            if event.reasoning:
                reasoning_lines = self.text.wrap(
                    f"AI REASONING: {event.reasoning}",
                    content_width,
                    "tiny",
                )
                for line in reasoning_lines:
                    self.text.draw(line, x + 30, cursor, MUTED, "tiny")
                    cursor += 15
            if event.eliminated_indices:
                eliminated_names = ", ".join(
                    self.state.participants[index].name
                    for index in event.eliminated_indices
                )
                self.text.draw(
                    f"ELIMINATED: {eliminated_names}",
                    x + 30,
                    cursor,
                    RED,
                    "tiny",
                )
                cursor += 15
            cursor += 8

        if record.notices:
            self.text.draw("ROUND NOTICES", x + 20, cursor, RED, "small")
            cursor += 20
            for notice in record.notices:
                for line in self.text.wrap(notice, content_width, "tiny"):
                    self.text.draw(line, x + 30, cursor, RED, "tiny")
                    cursor += 15

        footer_y = y + height - 28
        draw_rect(x + 12, footer_y - 5, width - 24, 25, PANEL_ALT, 0.92)
        self.text.draw(
            "Mouse wheel or PageUp/PageDown reviews the frozen archive  ·  no entries can be changed",
            x + width / 2.0,
            footer_y,
            MUTED,
            "tiny",
            True,
        )

    def _draw_button(
        self,
        rect: Tuple[float, float, float, float],
        label: str,
        kind: str,
        value: Any,
        accent: Tuple[int, int, int],
        prominent: bool = False,
        enabled: bool = True,
    ) -> None:
        mouse = pygame.mouse.get_pos()
        region = HitRegion(rect, kind, value)
        hovered = enabled and region.contains(mouse)
        amount = 0.25 if prominent else 0.13
        fill = (
            mix_color(PANEL_ALT, accent, amount + (0.10 if hovered else 0.0))
            if enabled
            else mix_color(PANEL, MUTED, 0.08)
        )
        pulse = (math.sin(time.monotonic() * 2.4) * 0.5 + 0.5) if prominent else 0.0
        draw_gradient_rect(
            rect[0],
            rect[1],
            rect[2],
            rect[3],
            mix_color(fill, WHITE, 0.08 + pulse * 0.04),
            mix_color(fill, INK, 0.28),
            1.0,
        )
        if prominent or hovered:
            draw_textured_nine_slice(
                self.materials.get("ui_button"),
                rect,
                accent if enabled else MUTED,
                (0.26 if hovered else 0.17) + pulse * 0.05,
                min(12.0, rect[3] * 0.28),
            )
        draw_rect(rect[0] + 8, rect[1] + 2, max(0.0, rect[2] - 16), 1, WHITE, 0.12)
        draw_rect(rect[0] + 8, rect[1] + rect[3] - 3, max(0.0, rect[2] - 16), 1, INK, 0.48)
        outline = accent if enabled and (hovered or prominent) else LINE
        draw_outline(*rect, outline, 2.0 if hovered else 1.0)
        button_font = "small"
        lines = self.text.wrap(label, int(rect[2] - 20), button_font)
        if len(lines) > 1 and rect[3] <= 38:
            button_font = "tiny"
            lines = self.text.wrap(label, int(rect[2] - 20), button_font)
        lines = lines[:2]
        if lines and len(self.text.wrap(label, int(rect[2] - 20), button_font)) > 2:
            lines[-1] = self._fit_text(lines[-1] + "...", button_font, rect[2] - 20)
        line_height = 15 if button_font == "tiny" else 17
        start_y = rect[1] + (rect[3] - len(lines) * line_height) / 2.0
        for offset, line in enumerate(lines):
            self.text.draw(
                line,
                rect[0] + rect[2] / 2.0,
                start_y + offset * line_height,
                WHITE if enabled else MUTED,
                button_font,
                True,
            )
        if enabled:
            self.regions.append(region)

    def _draw_story_choices(self, y: float) -> None:
        margin = 14.0
        gap = 10.0
        height = 68.0
        if self.state.is_ending:
            self._draw_button(
                (margin, y, WIDTH - margin * 2, height),
                "Restart the Room",
                "restart",
                None,
                GOLD,
                True,
            )
            return
        scene = self.state.current_scene
        width = (WIDTH - margin * 2 - gap * 2) / 3.0
        for index, choice in enumerate(scene.choices):
            label = "ROUND ACTION: " + choice.label.format(
                npc=self.state.story_target_name,
                player=self.state.player.name,
            )
            self._draw_button(
                (margin + index * (width + gap), y, width, height),
                label,
                "story",
                index,
                self.state.selected_npc.profile.accent,
                True,
                self.state.phase == "awaiting_player" and not self.player_turn_cooldown,
            )

    def _draw_action_choices(self, y: float) -> None:
        margin = 14.0
        gap = 8.0
        row_gap = 6.0
        height = 38.0
        columns = 7
        width = (WIDTH - margin * 2 - gap * (columns - 1)) / columns
        item = ITEM_BY_KEY[self.state.selected_item].short_label
        buttons = (
            ("Talk", "talk", self.state.selected_npc.profile.accent),
            ("Listen", "listen", CYAN),
            ("Compliment (+)", "compliment", GREEN),
            ("Flirt", "flirt", MAGENTA),
            ("Antagonize (-)", "antagonize", CORAL),
            ("Physical Fight", "fight", RED),
            (f"Give {item}", "give", ITEM_BY_KEY[self.state.selected_item].color),
            (f"Steal {item}", "steal", CORAL),
            ("Set up trade", "trade", GOLD),
            ("Reflect", "reflect", self.state.selected_npc.profile.accent),
            (f"Use {item}", "use", ITEM_BY_KEY[self.state.selected_item].color),
            ("Rest", "rest", MUTED),
            (
                f"SELF ELIMINATE - {self.state.player.name.upper()} ONLY",
                "self_eliminate",
                RED,
            ),
        )
        for index, (label, action, accent) in enumerate(buttons):
            enabled = (
                self.state.phase == "awaiting_player"
                and not self.player_turn_cooldown
                and (
                    bool(self.state.living_npcs) or action in self.state.SELF_ACTIONS
                )
            )
            row = index // columns
            column = index % columns
            button_width = width * 2 + gap if action == "self_eliminate" else width
            self._draw_button(
                (
                    margin + column * (width + gap),
                    y + row * (height + row_gap),
                    button_width,
                    height,
                ),
                label,
                "action",
                action,
                accent,
                prominent=action == "self_eliminate",
                enabled=enabled,
            )


def run_checks() -> None:
    def set_statuses(participant: Participant, ranged: int, spiritual: int) -> None:
        for key in RANGED_STATUS_KEYS:
            participant.statuses[key] = ranged
        participant.statuses["spiritual"] = spiritual

    def force_npc_decisions(
        state: GameState,
        action: str,
        target_picker: Any,
        item_key: Optional[str] = None,
    ) -> None:
        def choose(npc: Participant) -> NPCDecision:
            target = target_picker(npc)
            return NPCDecision(
                action,
                state._participant_index(target),
                item_key,
                999.0,
                "Forced deterministic validation decision.",
            )

        state._choose_npc_decision = choose  # type: ignore[method-assign]

    class SilentAudioProbe:
        def reset(self, seed: Optional[int], now: Optional[float] = None) -> None:
            self.seed = seed

    assert RELEASE_LABEL == "FULL FINAL RELEASE"
    assert PRESENTATION_SALT == "full-final-release-presentation"
    assert TITLE == "The Fifth Signal - Full Final Release"
    assert (WIDTH, HEIGHT, TOP_HEIGHT) == (1920, 1080, 540)
    assert ACTION_MOMENT_SECONDS == 1.0
    assert set(GameState.NPC_ACTIONS) == set(GameState.ACTIONS) - {"self_eliminate"}
    assert {"flirt", "steal", "trade"} <= set(GameState.NPC_ACTIONS)
    assert "self_eliminate" not in GameState.NPC_ACTIONS
    assert CREDIT_WATERMARK == "Created by OpenAI ChatGPT Codex 5.6 Sol Ultra"

    negative_style = spiritual_portrait_style(-1)
    neutral_style = spiritual_portrait_style(0)
    positive_style = spiritual_portrait_style(1)
    assert negative_style.horns and negative_style.flames
    assert not negative_style.halo and not negative_style.dim_window
    assert neutral_style.dim_window and 0.0 < neutral_style.light_scale < 1.0
    assert not (neutral_style.horns or neutral_style.flames or neutral_style.halo)
    assert positive_style.halo and positive_style.halo_layers == len(HALO_GLOW_LAYERS) >= 3
    assert not (positive_style.horns or positive_style.flames or positive_style.dim_window)
    try:
        spiritual_portrait_style(2)
    except ValueError:
        pass
    else:
        raise AssertionError("An invalid spirituality render state was accepted")

    for portrait_index in range(4):
        dark_profile = portrait_light_profile(portrait_index, 0)
        horn_profile = portrait_light_profile(portrait_index, -1)
        halo_profile = portrait_light_profile(portrait_index, 1)
        assert 0.0 < dark_profile.dim_overlay_alpha < 0.75
        assert horn_profile.dim_overlay_alpha == halo_profile.dim_overlay_alpha == 0.0
        assert all(
            dark_profile.ambient[channel] < horn_profile.ambient[channel]
            and dark_profile.ambient[channel] < halo_profile.ambient[channel]
            and dark_profile.diffuse[channel] < horn_profile.diffuse[channel]
            and dark_profile.diffuse[channel] < halo_profile.diffuse[channel]
            for channel in range(3)
        )
        assert all(
            profile.ambient[3] == profile.diffuse[3] == profile.secondary_diffuse[3] == 1.0
            for profile in (dark_profile, horn_profile, halo_profile)
        )

    assert max(channel for color in HORN_BODY_COLORS for channel in color) <= 29
    horn_heights = [ring[1] for ring in HORN_CENTERLINE]
    horn_radii = [ring[2] for ring in HORN_CENTERLINE]
    assert horn_heights == sorted(horn_heights)
    assert all(first > second for first, second in zip(horn_radii, horn_radii[1:]))
    assert horn_radii[-1] == 0.0
    assert horn_tip_position(-1.0)[0] == -horn_tip_position(1.0)[0]
    assert horn_tip_position(1.0)[1] > 1.3
    flame_samples = [
        horn_flame_parameters(sample_time, 0.73, side)
        for sample_time in (0.0, 0.17, 0.41, 1.03)
        for side in (-1.0, 1.0)
    ]
    assert len(set(flame_samples)) > 4
    assert all(0.24 <= height <= 0.38 for height, _width, _sway in flame_samples)
    assert all(0.10 <= width <= 0.17 for _height, width, _sway in flame_samples)
    assert all(abs(sway) <= 0.07 for _height, _width, sway in flame_samples)
    assert HORN_FLAME_PALETTE[0][0] > HORN_FLAME_PALETTE[0][1]
    assert HORN_FLAME_PALETTE[2][0] > 240 and HORN_FLAME_PALETTE[2][1] > 170
    halo_radii = [layer[0] for layer in HALO_GLOW_LAYERS]
    halo_alphas = [layer[2] for layer in HALO_GLOW_LAYERS]
    assert halo_radii == sorted(halo_radii)
    assert all(first > second for first, second in zip(halo_alphas, halo_alphas[1:]))
    assert all(layer[1] > 0.0 for layer in HALO_GLOW_LAYERS)
    assert set(TEXTURE_ASSET_FILES) == {
        "skin", "hair", "cloth", "iris", "particle_star", "particle_smoke",
        "particle_light", "particle_magic", "ui_panel", "ui_button",
    }
    assert all((ASSET_DIR / filename).is_file() for filename in TEXTURE_ASSET_FILES.values())
    assert (ASSET_DIR / "KenneyFuture.ttf").is_file()
    assert (ASSET_DIR / "KenneyFutureNarrow.ttf").is_file()
    assert (ASSET_DIR / "ASSET_LICENSES.md").is_file()
    assert normalize_player_name("  Nova   Stone  ") == "Nova Stone"
    assert normalize_player_name("Echo-7") == "Echo-7"
    assert possessive_name("Nova") == "Nova's"
    assert possessive_name("Iris") == "Iris'"
    for invalid_name in ("", "- ' -", "A" * (MAX_PLAYER_NAME_LENGTH + 1), "<Nova>", "Renée"):
        try:
            normalize_player_name(invalid_name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe player name was accepted: {invalid_name!r}")
    try:
        normalize_player_name(None)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Non-text player name was accepted")

    custom_state = GameState(77, "Nova Stone")
    assert custom_state.player.name == "Nova Stone"
    assert custom_state.player.profile is PLAYER_PROFILE
    assert custom_state.public_status_snapshot()["player"]["name"] == "Nova Stone"
    assert "Nova Stone's first interaction" in custom_state.message
    assert "Nova Stone first" in SCENES["pattern"].prompt.format(
        player=custom_state.player.name,
        npc=custom_state.story_target_name,
    )
    for npc in custom_state.npcs:
        npc.statuses["physical"] = 0
    custom_state._resolve_eliminations()
    assert custom_state.is_ending and custom_state.last_survivor is custom_state.player
    assert custom_state.ending_title.startswith("Nova Stone:")
    assert "Only Nova Stone remains alive" in custom_state.ending_text

    name_app = GameApp.__new__(GameApp)
    name_app.seed = 77
    name_app.player_name = DEFAULT_PLAYER_NAME
    name_app.name_entry_active = True
    name_app.name_entry_text = DEFAULT_PLAYER_NAME
    name_app.name_entry_error = ""
    name_app.name_replace_on_type = True
    name_app.trade_edit_side = "offer"
    name_app.trade_quantity_text = {"offer": "1", "request": "1"}
    name_app.trade_replace_on_type = True
    name_app.activity_history_index = None
    name_app.next_participant_action_time = None
    name_app.player_turn_cooldown = False
    name_app.audio = SilentAudioProbe()
    name_app.presentation_seed = presentation_word(77, "check-name-app")
    name_app.victory_cycle = 0
    name_app.victory_started_at = None
    name_app.victory_signature = None
    assert name_app._start_with_player_name("  Echo   Seven  ")
    assert not name_app.name_entry_active
    assert name_app.player_name == name_app.state.player.name == "Echo Seven"
    name_app.restart()
    assert name_app.state.player.name == "Echo Seven"
    assert not name_app._start_with_player_name("<unsafe>")
    assert name_app.player_name == "Echo Seven" and name_app.name_entry_error
    name_app._set_name_entry_text("A<very> long_name that is filtered")
    assert len(name_app.name_entry_text) <= MAX_PLAYER_NAME_LENGTH
    assert all(character in PLAYER_NAME_ALLOWED_CHARS for character in name_app.name_entry_text)
    name_app._cancel_player_name_entry()
    assert name_app.player_name == name_app.state.player.name == DEFAULT_PLAYER_NAME

    # Interactive rounds mutate one participant at a time. Fabricated monotonic
    # timestamps prove that 0.999 seconds does nothing and 1.000 second admits
    # exactly one new living actor without sleeping in the test suite.
    pace_app = GameApp.__new__(GameApp)
    pace_app.state = GameState(78, "Pace Probe", paced_rounds=True)
    pace_app.activity_history_index = None
    pace_app.next_participant_action_time = None
    pace_app.player_turn_cooldown = False
    for pace_participant in pace_app.state.participants:
        set_statuses(pace_participant, 100, 1)
        for item_key in ITEM_KEYS:
            pace_participant.inventory[item_key] = 100
    pace_app.state.interact("talk")
    assert pace_app.state.phase == "resolving_npcs"
    assert [event.actor_index for event in pace_app.state.last_round_events] == [0]
    assert not pace_app.state.round_history
    live_record = pace_app._selected_activity_record()
    assert live_record is not None and [event.actor_index for event in live_record.events] == [0]
    pace_app._update_action_pacing(100.0)
    assert pace_app.next_participant_action_time == 101.0
    pace_app._update_action_pacing(100.999)
    assert [event.actor_index for event in pace_app.state.last_round_events] == [0]
    pace_app._update_action_pacing(101.0)
    assert [event.actor_index for event in pace_app.state.last_round_events] == [0, 1]
    pace_app._update_action_pacing(101.999)
    assert [event.actor_index for event in pace_app.state.last_round_events] == [0, 1]
    for due_time, expected_actor in ((102.0, 2), (103.0, 3), (104.0, 4)):
        pace_app._update_action_pacing(due_time)
        assert pace_app.state.last_round_events[-1].actor_index == expected_actor
    assert pace_app.state.phase == "awaiting_player"
    assert len(pace_app.state.round_history) == 1
    assert pace_app.player_turn_cooldown
    assert pace_app.next_participant_action_time == 105.0
    assert pace_app._player_controls_locked()
    pace_app._update_action_pacing(104.999)
    assert pace_app.player_turn_cooldown
    pace_app._update_action_pacing(105.0)
    assert not pace_app.player_turn_cooldown
    assert pace_app.next_participant_action_time is None
    assert not pace_app._player_controls_locked()
    assert pace_app._activity_record_count() == 1
    pace_app.state.validate()

    paced_elimination = GameState(79, "Pace Exit", paced_rounds=True)
    for pace_participant in paced_elimination.participants:
        set_statuses(pace_participant, 100, 1)
    force_npc_decisions(
        paced_elimination,
        "compliment",
        lambda npc: next(
            participant
            for participant in paced_elimination.living_participants
            if participant is not npc
        ),
    )
    paced_elimination.interact("self_eliminate")
    assert paced_elimination.player.eliminated
    assert paced_elimination.phase == "resolving_npcs"
    for expected_actor in (1, 2, 3, 4):
        assert paced_elimination.advance_resolution_moment()
        assert paced_elimination.last_round_events[-1].actor_index == expected_actor
    assert paced_elimination.phase == "autonomous"
    assert [event.actor_index for event in paced_elimination.round_history[-1].events] == [0, 1, 2, 3, 4]

    effect_probe = GameApp.__new__(GameApp)
    repeat_effect_probe = GameApp.__new__(GameApp)
    effect_probe._init_portrait_effects(42)
    repeat_effect_probe._init_portrait_effects(42)
    for probe in (effect_probe, repeat_effect_probe):
        probe.presentation_seed = presentation_word(42, PRESENTATION_SALT)
        probe.player_name = "Nova Stone"
        probe.victory_cycle = 0
        probe._init_victory_effects()
    assert len(effect_probe._firework_particles) == 119
    assert len(effect_probe._fog_blobs) == 15
    assert len(effect_probe._rain_drops) == 54
    assert len(effect_probe._raster_cells) == 18 * 14
    assert effect_probe._lightning_segments
    assert effect_probe._firework_particles == repeat_effect_probe._firework_particles
    assert effect_probe._fog_blobs == repeat_effect_probe._fog_blobs
    assert effect_probe._rain_drops == repeat_effect_probe._rain_drops
    assert effect_probe._raster_cells == repeat_effect_probe._raster_cells
    assert len(effect_probe._victory_particles) == 112
    assert effect_probe._victory_particles == repeat_effect_probe._victory_particles
    assert effect_probe.player_victory_style == repeat_effect_probe.player_victory_style

    generated_spiritual_values: set[int] = set()
    for seed in range(40):
        state = GameState(seed)
        state.validate()
        for participant in state.participants:
            for key in RANGED_STATUS_KEYS:
                assert 20 <= participant.statuses[key] <= 100
            assert participant.statuses["spiritual"] in (-1, 0, 1)
            generated_spiritual_values.add(participant.statuses["spiritual"])
            for count in participant.inventory.values():
                assert 0 <= count <= 100

        snapshot = state.public_status_snapshot()
        assert "inventory" in snapshot["player"]
        assert "eliminated" in snapshot["player"]
        assert len(snapshot["npcs"]) == 4
        assert all("inventory" not in npc_view for npc_view in snapshot["npcs"])
        assert all("eliminated" in npc_view for npc_view in snapshot["npcs"])
        snapshot["player"]["statuses"]["physical"] = -99
        snapshot["player"]["inventory"]["dollars"] = -99
        snapshot["npcs"][0]["statuses"]["physical"] = -99
        assert state.player.statuses["physical"] >= 20
        assert state.player.inventory["dollars"] >= 0
        assert state.npcs[0].statuses["physical"] >= 20

    assert generated_spiritual_values == {-1, 0, 1}

    assert set(ITEM_STATUS_EFFECTS) == set(ITEM_KEYS)
    assert all(set(effects) == set(STATUS_KEYS) for effects in ITEM_STATUS_EFFECTS.values())
    assert set(NPC_ITEM_RESERVES) == set(ITEM_KEYS)
    assert set(NPC_STATUS_IMPORTANCE) == set(STATUS_KEYS)
    assert set(REST_STATUS_EFFECTS) == set(STATUS_KEYS)
    assert len(NPC_ROLE_ACTION_BIASES) == len(NPC_PROFILES) == 4
    assert len(NPC_ROLE_ITEM_BIASES) == len(NPC_PROFILES)
    assert all(value > 0.0 for value in NPC_ITEM_RESERVES.values())
    assert all(value > 0.0 for value in NPC_STATUS_IMPORTANCE.values())
    assert set(TRADE_STATUS_WEIGHTS) == set(STATUS_KEYS)
    assert all(weight > 0.0 for weight in TRADE_STATUS_WEIGHTS.values())
    assert math.isclose(sum(TRADE_STATUS_WEIGHTS.values()), 1.0)
    assert "".join(character for character in "1²①2" if character in "0123456789") == "12"

    # Flirting is reciprocal and boundary-aware; every index on both people
    # raises readiness monotonically, while a decline remains a valid outcome.
    social_probe = GameState(730)
    flirt_actor = social_probe.npcs[0]
    flirt_target = social_probe.player
    set_statuses(flirt_actor, 1, -1)
    set_statuses(flirt_target, 1, -1)
    low_flirt_chance = social_probe.flirt_chance(flirt_actor, flirt_target)
    for participant in (flirt_actor, flirt_target):
        for key in STATUS_KEYS:
            old_value = participant.statuses[key]
            participant.statuses[key] = 1 if key == "spiritual" else 100
            assert social_probe.flirt_chance(flirt_actor, flirt_target) > low_flirt_chance
            participant.statuses[key] = old_value
    set_statuses(flirt_actor, 50, 0)
    set_statuses(flirt_target, 50, 0)
    success_summary, flirted = social_probe._resolve_flirt(flirt_actor, flirt_target, roll=0.0)
    assert flirted and "reciprocates" in success_summary
    decline_probe = GameState(731)
    set_statuses(decline_probe.npcs[0], 50, 0)
    set_statuses(decline_probe.player, 50, 0)
    decline_summary, flirted = decline_probe._resolve_flirt(
        decline_probe.npcs[0],
        decline_probe.player,
        roll=1.0,
    )
    assert not flirted and "accepts the boundary" in decline_summary

    # Steal chance uses all six indexes on thief and guard. Every item transfers
    # atomically on success and never reveals the target's remaining balance.
    steal_probe = GameState(732)
    thief = steal_probe.npcs[0]
    guard = steal_probe.player
    set_statuses(thief, 1, 1)
    set_statuses(guard, 1, -1)
    base_steal_chance = steal_probe.steal_chance(thief, guard)
    for key in RANGED_STATUS_KEYS:
        old_value = thief.statuses[key]
        thief.statuses[key] = 100
        assert steal_probe.steal_chance(thief, guard) > base_steal_chance
        thief.statuses[key] = old_value
    thief.set_spiritual(-1)
    assert steal_probe.steal_chance(thief, guard) > base_steal_chance
    thief.set_spiritual(1)
    for key in RANGED_STATUS_KEYS:
        old_value = guard.statuses[key]
        guard.statuses[key] = 100
        assert steal_probe.steal_chance(thief, guard) < base_steal_chance
        guard.statuses[key] = old_value
    guard.set_spiritual(1)
    assert steal_probe.steal_chance(thief, guard) < base_steal_chance

    for stolen_item in ITEM_KEYS:
        item_probe = GameState(733)
        thief = item_probe.npcs[0]
        guard = item_probe.player
        thief.inventory[stolen_item] = 0
        guard.inventory[stolen_item] = 100
        quantity = ITEM_BY_KEY[stolen_item].give_amount
        total_before = thief.inventory[stolen_item] + guard.inventory[stolen_item]
        summary, stolen = item_probe._resolve_steal(
            thief,
            guard,
            stolen_item,
            roll=0.0,
        )
        assert stolen and f"steals {quantity}" in summary
        assert thief.inventory[stolen_item] == quantity
        assert guard.inventory[stolen_item] == 100 - quantity
        assert thief.inventory[stolen_item] + guard.inventory[stolen_item] == total_before

        failed_probe = GameState(734)
        failed_thief = failed_probe.npcs[0]
        failed_guard = failed_probe.player
        failed_thief.inventory[stolen_item] = 0
        failed_guard.inventory[stolen_item] = 100
        inventory_before = [dict(person.inventory) for person in failed_probe.participants]
        summary, stolen = failed_probe._resolve_steal(
            failed_thief,
            failed_guard,
            stolen_item,
            roll=1.0,
        )
        assert not stolen and "caught" in summary
        assert [person.inventory for person in failed_probe.participants] == inventory_before

    shortage_probe = GameState(735)
    shortage_probe.player.inventory["water_liters"] = 0
    inventory_before = [dict(person.inventory) for person in shortage_probe.participants]
    shortage_summary, stolen = shortage_probe._resolve_steal(
        shortage_probe.npcs[0],
        shortage_probe.player,
        "water_liters",
        roll=0.0,
    )
    assert not stolen and "finds no exposed unit" in shortage_summary
    assert [person.inventory for person in shortage_probe.participants] == inventory_before
    assert "0" not in shortage_summary

    # NPC trade uses every distinct standard bundle pairing, is atomic, and
    # resolves inside the proposer's one action without consuming the target's.
    exchange_probe = GameState(736)
    for participant in exchange_probe.participants:
        set_statuses(participant, 50, 0)
        for item_key in ITEM_KEYS:
            participant.inventory[item_key] = 100
    exchange_actor = exchange_probe.npcs[0]
    trade_candidates = [
        decision
        for decision in exchange_probe._npc_decision_candidates(exchange_actor)
        if decision.action == "trade" and decision.target_index == 0
    ]
    assert len(trade_candidates) == len(ITEM_KEYS) * (len(ITEM_KEYS) - 1) == 20
    exchange = ExchangePlan("water_liters", 1, "food_pounds", 1)
    totals_before = {
        key: exchange_actor.inventory[key] + exchange_probe.player.inventory[key]
        for key in ITEM_KEYS
    }
    summary, traded = exchange_probe._resolve_npc_trade(
        exchange_actor,
        exchange_probe.player,
        exchange,
        roll=0.0,
    )
    assert traded and "trades" in summary
    assert all(
        exchange_actor.inventory[key] + exchange_probe.player.inventory[key] == totals_before[key]
        for key in ITEM_KEYS
    )
    declined_inventories = [dict(person.inventory) for person in exchange_probe.participants]
    _summary, traded = exchange_probe._resolve_npc_trade(
        exchange_actor,
        exchange_probe.player,
        exchange,
        roll=1.0,
    )
    assert not traded
    assert [person.inventory for person in exchange_probe.participants] == declined_inventories

    # The Full Final Release plans one complete action-target-item combination. Every status
    # index and every private inventory quantity independently moves utility.
    planner = GameState(741)
    planner_npc = planner.npcs[0]
    set_statuses(planner_npc, 50, 0)
    for item_key in ITEM_KEYS:
        planner_npc.inventory[item_key] = int(NPC_ITEM_RESERVES[item_key])
    status_channels = {
        "physical": "talk",
        "emotional": "compliment",
        "cognitive": "talk",
        "social": "compliment",
        "sentient": "listen",
    }
    for status_key, action in status_channels.items():
        planner_npc.statuses[status_key] = 10
        low_utility = planner._npc_action_weights(planner_npc)[action]
        planner_npc.statuses[status_key] = 90
        high_utility = planner._npc_action_weights(planner_npc)[action]
        assert high_utility > low_utility
        planner_npc.statuses[status_key] = 50
    planner_npc.set_spiritual(-1)
    shadow_reflection = planner._npc_action_weights(planner_npc)["reflect"]
    planner_npc.set_spiritual(1)
    light_reflection = planner._npc_action_weights(planner_npc)["reflect"]
    assert light_reflection > shadow_reflection

    set_statuses(planner_npc, 10, 0)
    planner_npc.statuses["cognitive"] = 100
    assert "Cognitive clarity" in planner._npc_decision_reasoning(
        planner_npc,
        "talk",
        planner.player,
        None,
    )
    set_statuses(planner_npc, 10, 0)
    planner_npc.statuses["sentient"] = 100
    assert "Awareness" in planner._npc_decision_reasoning(
        planner_npc,
        "listen",
        planner.player,
        None,
    )
    set_statuses(planner_npc, 100, 0)
    planner_npc.statuses["physical"] = 1
    assert "Physical fatigue" in planner._npc_decision_reasoning(
        planner_npc,
        "rest",
        planner_npc,
        None,
    )
    set_statuses(planner_npc, 50, 0)

    for varied_item in ITEM_KEYS:
        for item_key in ITEM_KEYS:
            planner_npc.inventory[item_key] = 0
        empty_weights = planner._npc_action_weights(planner_npc)
        assert planner._item_security(planner_npc, varied_item) == 0.0
        planner_npc.inventory[varied_item] = ITEM_BY_KEY[varied_item].use_amount
        one_charge_security = planner._item_security(planner_npc, varied_item)
        planner_npc.inventory[varied_item] = INVENTORY_CAP
        abundant_security = planner._item_security(planner_npc, varied_item)
        abundant_weights = planner._npc_action_weights(planner_npc)
        assert 0.0 < one_charge_security < abundant_security < 1.0
        assert abundant_weights["talk"] > empty_weights["talk"]
        assert abundant_weights["antagonize"] < empty_weights["antagonize"]

    # Every item's authored effects can become a legal self-care plan; larger
    # reserves help without linearly overwhelming the six status indexes.
    for varied_item in ITEM_KEYS:
        item_planner = GameState(742)
        item_npc = item_planner.npcs[0]
        set_statuses(item_npc, 30, 0)
        for item_key in ITEM_KEYS:
            item_npc.inventory[item_key] = 0
        spec = ITEM_BY_KEY[varied_item]
        item_npc.inventory[varied_item] = spec.use_amount
        one_charge = [
            decision
            for decision in item_planner._npc_decision_candidates(item_npc)
            if decision.action == "use" and decision.item_key == varied_item
        ]
        assert len(one_charge) == 1
        item_npc.inventory[varied_item] = 100
        abundant = [
            decision
            for decision in item_planner._npc_decision_candidates(item_npc)
            if decision.action == "use" and decision.item_key == varied_item
        ]
        assert len(abundant) == 1
        assert abundant[0].utility > one_charge[0].utility
        projected, _ = item_planner._project_item_effects(item_npc, varied_item)
        assert set(projected) == set(STATUS_KEYS)
        assert item_planner._effect_value(item_npc, projected) > 0.0

    unsafe_planner = GameState(743)
    unsafe_npc = unsafe_planner.npcs[0]
    set_statuses(unsafe_npc, 30, 0)
    unsafe_planner.turn = 10
    unsafe_npc.last_caffeine_turn = 9
    unsafe_npc.last_acetaminophen_turn = 9
    unsafe_npc.inventory["caffeine_pills"] = 100
    unsafe_npc.inventory["acetaminophen_pills"] = 100
    for unsafe_item in ("caffeine_pills", "acetaminophen_pills"):
        projected, warning = unsafe_planner._project_item_effects(unsafe_npc, unsafe_item)
        assert warning and unsafe_planner._effect_value(unsafe_npc, projected) < 0.0
        assert not any(
            decision.action == "use" and decision.item_key == unsafe_item
            for decision in unsafe_planner._npc_decision_candidates(unsafe_npc)
        )

    fatal_self_care = GameState(7431)
    fatal_npc = fatal_self_care.npcs[0]
    set_statuses(fatal_npc, 100, 0)
    fatal_npc.statuses["physical"] = 1
    fatal_npc.statuses["cognitive"] = 1
    fatal_npc.inventory["caffeine_pills"] = 100
    assert not any(
        decision.action == "use" and decision.item_key == "caffeine_pills"
        for decision in fatal_self_care._npc_decision_candidates(fatal_npc)
    )

    full_planner = GameState(744)
    for full_participant in full_planner.participants:
        set_statuses(full_participant, 100, 1)
        for item_key in ITEM_KEYS:
            full_participant.inventory[item_key] = 100
    full_candidates = full_planner._npc_decision_candidates(full_planner.npcs[0])
    assert not any(decision.action in ("use", "rest", "give") for decision in full_candidates)

    # Carefully isolated states prove that self-use, provisioned rest, and
    # resource sharing can each win rather than merely appearing as dead code.
    use_planner = GameState(745)
    for person in use_planner.participants:
        set_statuses(person, 100, 1)
        for item_key in ITEM_KEYS:
            person.inventory[item_key] = 0
    set_statuses(use_planner.npcs[0], 15, 0)
    use_planner.npcs[0].inventory["water_liters"] = 10
    use_decision = use_planner._choose_npc_decision(use_planner.npcs[0])
    assert use_decision.action == "use" and use_decision.item_key == "water_liters"
    assert use_decision.target_index == 1

    rest_planner = GameState(746)
    for person in rest_planner.participants:
        set_statuses(person, 100, 1)
        for item_key in ITEM_KEYS:
            person.inventory[item_key] = 0
    set_statuses(rest_planner.npcs[0], 15, 0)
    rest_planner.npcs[0].inventory["water_liters"] = 10
    rest_planner.npcs[0].inventory["food_pounds"] = 10
    rest_decision = rest_planner._choose_npc_decision(rest_planner.npcs[0])
    assert rest_decision.action == "rest" and rest_decision.target_index == 1

    give_planner = GameState(747)
    for person in give_planner.participants:
        set_statuses(person, 100, 1)
        for item_key in ITEM_KEYS:
            person.inventory[item_key] = 0
    set_statuses(give_planner.player, 10, -1)
    give_planner.npcs[0].inventory["food_pounds"] = 100
    give_decision = give_planner._choose_npc_decision(give_planner.npcs[0])
    assert give_decision.action == "give" and give_decision.item_key == "food_pounds"
    assert give_decision.target_index == 0

    first_planner = GameState(748)
    second_planner = GameState(748)
    first_decisions = tuple(first_planner._choose_npc_decision(npc) for npc in first_planner.npcs)
    second_decisions = tuple(second_planner._choose_npc_decision(npc) for npc in second_planner.npcs)
    assert first_decisions == second_decisions
    assert all(
        decision.action in GameState.NPC_ACTIONS
        and decision.reasoning
        and math.isfinite(decision.utility)
        for decision in first_decisions
    )
    hidden_stock_adjectives = ("scarce", "balanced", "secure", "well-protected")
    assert all(
        not any(word in decision.reasoning.lower() for word in hidden_stock_adjectives)
        for decision in first_decisions
    )

    # Spirituality is a signed, bounded status with distinct face, decision,
    # trade, combat, and status-panel behavior at -1, 0, and +1.
    state = GameState(80)
    participant = state.player
    participant.set_spiritual(-99)
    assert participant.statuses["spiritual"] == -1
    participant.adjust("spiritual", -1)
    assert participant.statuses["spiritual"] == -1
    participant.adjust("spiritual", 1)
    assert participant.statuses["spiritual"] == 0
    participant.adjust("spiritual", 99)
    assert participant.statuses["spiritual"] == 1

    npc = state.selected_npc
    set_statuses(npc, 50, 0)
    neutral_face = face_expression_parameters(npc, 2.75, 0.83)
    npc.set_spiritual(-1)
    negative_face = face_expression_parameters(npc, 2.75, 0.83)
    npc.set_spiritual(1)
    positive_face = face_expression_parameters(npc, 2.75, 0.83)
    assert neutral_face["negative_spirituality"] == neutral_face["spiritual_glow"] == 0.0
    assert negative_face["negative_spirituality"] == 1.0
    assert negative_face["spiritual_glow"] == 0.0
    assert positive_face["negative_spirituality"] == 0.0
    assert positive_face["spiritual_glow"] == 1.0

    set_statuses(npc, 50, 0)
    neutral_weights = state._npc_action_weights(npc)
    npc.set_spiritual(-1)
    negative_weights = state._npc_action_weights(npc)
    npc.set_spiritual(1)
    positive_weights = state._npc_action_weights(npc)
    assert all(
        weight > 0.0
        for weights in (negative_weights, neutral_weights, positive_weights)
        for weight in weights.values()
    )
    assert negative_weights["antagonize"] > neutral_weights["antagonize"]
    assert negative_weights["fight"] > neutral_weights["fight"]
    assert positive_weights["reflect"] > neutral_weights["reflect"]

    # The gesture vocabulary is render-stable, every NPC action has several
    # contextual readings, and all six indexes move pose weights monotonically.
    assert set(GESTURE_LABELS) == {"idle", *NPC_GESTURE_KEYS}
    assert len(GESTURE_LABELS) == 22
    assert len(ACTION_EMOTE_KEYS) == len(set(ACTION_EMOTE_KEYS)) == 5
    assert set(ACTION_EMOTE_KEYS) <= set(NPC_GESTURE_KEYS)
    assert set(ACTION_GESTURE_CONTEXT) == set(GameState.NPC_ACTIONS)
    assert set(NPC_ITEM_GESTURE_CONTEXT) == set(ITEM_KEYS)
    assert all(key in NPC_GESTURE_KEYS for key in NPC_ITEM_GESTURE_CONTEXT.values())
    assert NPC_ITEM_GESTURE_CONTEXT["food_pounds"] == "chomping"
    assert NPC_ITEM_GESTURE_CONTEXT["caffeine_pills"] == "head_banging"
    assert NPC_ITEM_GESTURE_CONTEXT["water_liters"] != "chomping"
    assert all(
        len(context) == len(set(context)) == 4
        and all(key in NPC_GESTURE_KEYS for key in context)
        for context in ACTION_GESTURE_CONTEXT.values()
    )

    # The five Full Final Release emotes are pure presentation functions with bounded
    # one-shot motion; sampling them cannot perturb later NPC reasoning.
    emote_rng_before = state.rng.getstate()
    for emote_index, emote_key in enumerate(ACTION_EMOTE_KEYS):
        sampled_motion = [
            action_emote_motion(emote_key, age, emote_index % 4)
            for age in (0.0, 0.2, 0.5, 0.9, 1.55, 2.5)
        ]
        assert all(
            math.isfinite(value)
            for motion in sampled_motion
            for value in motion.values()
        )
        assert all(-0.05 <= motion["body_y"] <= 0.05 for motion in sampled_motion)
        assert all(-14.0 <= motion["body_pitch"] <= 14.0 for motion in sampled_motion)
        assert all(-4.0 <= motion["body_roll"] <= 4.0 for motion in sampled_motion)
        assert all(-9.0 <= motion["head_yaw"] <= 9.0 for motion in sampled_motion)
        assert all(-3.0 <= motion["head_tilt"] <= 3.0 for motion in sampled_motion)
        assert all(-13.0 <= motion["head_nod"] <= 13.0 for motion in sampled_motion)
        assert all(0.50 <= motion["eye_scale"] <= 1.0 for motion in sampled_motion)
        assert all(0.0 <= motion["jaw_open"] <= 0.14 for motion in sampled_motion)
        assert all(0.50 <= motion["mouth_width_scale"] <= 1.0 for motion in sampled_motion)
        assert action_emote_motion(emote_key, 2.5, emote_index % 4)["jaw_open"] == 0.0
    assert state.rng.getstate() == emote_rng_before

    # Four disjoint idle repertoires run from a render-only, seed-stable clock.
    # Sampling them must never advance the simulation's random generator.
    assert len(IDLE_PERSONALITIES) == 4
    assert len({profile.key for profile in IDLE_PERSONALITIES}) == 4
    assert len({profile.label for profile in IDLE_PERSONALITIES}) == 4
    idle_emote_sets = [set(profile.emotes) for profile in IDLE_PERSONALITIES]
    assert all(len(emotes) == 3 for emotes in idle_emote_sets)
    assert all(
        idle_emote_sets[left].isdisjoint(idle_emote_sets[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )
    assert set(IDLE_EMOTE_LABELS) == set().union(*idle_emote_sets)
    idle_probe = GameState(811)
    rng_before_idle_sampling = idle_probe.rng.getstate()
    sampled_idle_sequences: List[Tuple[str, ...]] = []
    for npc_index, personality in enumerate(IDLE_PERSONALITIES):
        first = idle_emote_frame(811, npc_index, 19.75)
        repeat = idle_emote_frame(811, npc_index, 19.75)
        assert first == repeat
        assert first.personality_key == personality.key
        assert first.personality_label == personality.label
        assert first.emote_key in personality.emotes
        assert first.next_emote_key in personality.emotes
        assert 0.0 <= first.slot_progress < 1.0
        assert 0.0 <= first.blend <= 1.0
        samples = tuple(
            idle_emote_frame(811, npc_index, moment).emote_key
            for moment in range(0, 100, 2)
        )
        assert set(samples) == set(personality.emotes)
        sampled_idle_sequences.append(samples)
        motion = idle_motion_parameters(first, npc_index)
        assert set(motion) == {
            "body_x",
            "body_y",
            "body_roll",
            "head_yaw",
            "head_tilt",
            "head_nod",
            "gaze_x",
            "hand_energy",
        }
        assert all(math.isfinite(value) for value in motion.values())
    assert len(set(sampled_idle_sequences)) == 4
    assert idle_probe.rng.getstate() == rng_before_idle_sampling
    assert idle_emote_frame(811, 0, 19.75) != idle_emote_frame(812, 0, 19.75)

    # Idles own only the live player-decision window. Action gestures hold, then
    # ease back into the persistent schedule; elimination and non-player phases
    # leave no ambient portrait motion active.
    assert npc_idle_enabled("awaiting_player", True, True, False)
    assert not npc_idle_enabled("resolving_npcs", True, True, False)
    assert not npc_idle_enabled("autonomous", False, True, False)
    assert not npc_idle_enabled("ending", True, True, True)
    assert not npc_idle_enabled("awaiting_player", True, False, False)
    assert action_gesture_override_weight(0, 999.0, True) == 0.0
    assert action_gesture_override_weight(1, 0.0, True) == 1.0
    crossfade_weight = action_gesture_override_weight(1, 2.0, True)
    assert 0.0 < crossfade_weight < 1.0
    assert action_gesture_override_weight(1, 4.0, True) == 0.0
    assert action_gesture_override_weight(1, 99.0, False) == 1.0

    # The original chip voices make every public index audible, generate
    # deterministic bounded stereo PCM, and never consume simulation RNG.
    assert AUDIO_SAMPLE_RATE % AUDIO_SYNTH_DIVISOR == 0
    assert len(VOICE_PROFILES) == 4
    assert len({profile.key for profile in VOICE_PROFILES}) == 4
    assert all(-1.0 <= profile.pan <= 1.0 for profile in VOICE_PROFILES)
    assert {
        "talk", "listen", "compliment", "antagonize", "reflect", "fight",
        "flirt", "give", "steal", "use", "rest", "trade", "investigate", "elimination", "victory",
    } <= set(VOICE_CUE_CONTOURS)
    base_statuses = {key: 50 for key in RANGED_STATUS_KEYS}
    base_statuses["spiritual"] = 0
    base_mood = voice_mood_from_statuses(base_statuses)
    for status_key in STATUS_KEYS:
        changed_statuses = dict(base_statuses)
        changed_statuses[status_key] = 1 if status_key == "spiritual" else 100
        changed_mood = voice_mood_from_statuses(changed_statuses)
        assert changed_mood != base_mood
        assert changed_mood.cache_key != base_mood.cache_key
    low_statuses = {key: 20 for key in RANGED_STATUS_KEYS}
    low_statuses["spiritual"] = -1
    high_statuses = {key: 90 for key in RANGED_STATUS_KEYS}
    high_statuses["spiritual"] = 1
    low_mood = voice_mood_from_statuses(low_statuses)
    high_mood = voice_mood_from_statuses(high_statuses)
    assert low_mood.tone == "negative" and high_mood.tone == "positive"
    assert low_mood.energy < high_mood.energy
    assert low_mood.clarity < high_mood.clarity
    assert voice_nominal_pitch(0, "talk", high_mood) > voice_nominal_pitch(0, "talk", low_mood)
    generated_voice_buffers = set()
    for profile_index, cue_key in enumerate(("talk", "compliment", "fight", "victory")):
        generated = synthesize_voice_pcm(profile_index, cue_key, high_mood, 7)
        repeated = synthesize_voice_pcm(profile_index, cue_key, high_mood, 7)
        assert generated == repeated
        assert 0.16 <= generated.duration <= 0.90
        assert len(generated.buffer) % (2 * AUDIO_CHANNELS) == 0
        assert len(generated.mouth_windows) == len(VOICE_CUE_CONTOURS[cue_key])
        samples = array("h")
        samples.frombytes(generated.buffer)
        if sys.byteorder != "little":
            samples.byteswap()
        assert samples and max(abs(sample) for sample in samples) > 500
        assert max(abs(sample) for sample in samples) <= 32767
        generated_voice_buffers.add(generated.buffer)
    assert len(generated_voice_buffers) == len(VOICE_PROFILES)
    assert synthesize_voice_pcm(0, "talk", low_mood, 7).buffer != synthesize_voice_pcm(
        0, "talk", high_mood, 7
    ).buffer
    assert all(voice_caption(index, "talk", high_mood) for index in range(4))

    # Each NPC owns a distinct looping victory meme and seeded player finales
    # cover all four particle generators without touching the simulation RNG.
    assert len(NPC_VICTORY_GESTURES) == len(NPC_VICTORY_LABELS) == len(NPC_VICTORY_CAPTIONS) == 4
    for npc_index, gestures in enumerate(NPC_VICTORY_GESTURES):
        assert len(gestures) == len(set(gestures)) == 4
        assert all(gesture in GESTURE_LABELS for gesture in gestures)
        first_frame = victory_frame(811, npc_index, 3.75)
        assert first_frame == victory_frame(811, npc_index, 3.75)
        assert first_frame.gesture_key in gestures
        assert first_frame.next_gesture_key in gestures
        assert 0.0 <= first_frame.gesture_blend <= 1.0
        assert all(
            math.isfinite(value)
            for value in (
                first_frame.bounce,
                first_frame.body_x,
                first_frame.body_roll,
                first_frame.head_yaw,
                first_frame.head_nod,
                first_frame.pulse,
            )
        )
    assert {
        player_victory_effect_index(seed, "Nova Stone")
        for seed in range(80)
    } == set(range(len(PLAYER_VICTORY_EFFECTS)))
    victory_rng_probe = GameState(812)
    rng_before_victory_sampling = victory_rng_probe.rng.getstate()
    for npc_index in range(4):
        victory_frame(812, npc_index, 12.5)
    player_victory_effect_index(812, "Nova Stone", 3)
    assert victory_rng_probe.rng.getstate() == rng_before_victory_sampling

    # The presentation-side director consumes each immutable action once,
    # serializes a full five-participant round, bounds backlog, and queues one
    # terminal sting. These checks need no mixer or window.
    audio_state = GameState(813, "Audio Probe")
    for audio_participant in audio_state.participants:
        set_statuses(audio_participant, 100, 1)
    audio_state.interact("compliment")
    rng_after_audio_round = audio_state.rng.getstate()
    director = AudioDirector(813, False)
    director._ingest_rounds(audio_state, 10.0)
    assert director.seen_rounds == 1
    assert len(director.pending) == 5
    assert [cue.speaker_index for cue in director.pending] == [0, 0, 1, 2, 3]
    frozen_queue = tuple(director.pending)
    director._ingest_rounds(audio_state, 10.1)
    assert tuple(director.pending) == frozen_queue
    assert audio_state.rng.getstate() == rng_after_audio_round
    for _ in range(4):
        audio_state.interact("compliment")
        director._ingest_rounds(audio_state, 11.0)
    assert len(director.pending) == AUDIO_QUEUE_LIMIT
    assert max(cue.earliest_time for cue in director.pending) <= 13.25
    for audio_npc in audio_state.npcs:
        audio_npc.statuses["physical"] = 0
    audio_state._resolve_eliminations()
    assert audio_state.last_survivor is audio_state.player
    director._ingest_victory(audio_state, 12.0)
    victory_queue_size = len(director.pending)
    assert director.pending[-1].cue_key == "victory"
    assert director.pending[-1].speaker_index == -1
    assert director.pending[-1].earliest_time <= 13.50
    director._ingest_victory(audio_state, 12.1)
    assert len(director.pending) == victory_queue_size
    director.reset(813, 20.0)
    assert not director.pending and director.active is None and director.seen_rounds == 0

    paced_audio_state = GameState(8131, "Audio Pace", paced_rounds=True)
    for audio_participant in paced_audio_state.participants:
        set_statuses(audio_participant, 100, 1)
        for item_key in ITEM_KEYS:
            audio_participant.inventory[item_key] = 100
    paced_audio_state.interact("flirt")
    paced_director = AudioDirector(8131, False)
    paced_director._ingest_live_events(paced_audio_state, 40.0)
    assert len(paced_director.pending) == 1
    assert len(paced_director.seen_event_tokens) == 1
    for moment_index in range(1, 5):
        paced_audio_state.advance_resolution_moment()
        if paced_audio_state.phase == "resolving_npcs":
            paced_director._ingest_live_events(paced_audio_state, 40.0 + moment_index)
        else:
            paced_director._ingest_rounds(paced_audio_state, 40.0 + moment_index)
    assert len(paced_audio_state.round_history) == 1
    pending_before_archive_ingest = tuple(paced_director.pending)
    paced_director._ingest_rounds(paced_audio_state, 45.0)
    assert tuple(paced_director.pending) == pending_before_archive_ingest
    assert len(paced_director.seen_event_tokens) == 5

    idle_audio_state = GameState(814)
    idle_audio_state.idle_epoch = 0.0
    idle_director = AudioDirector(814, False)
    idle_director.reset(814, 0.0)
    rng_before_idle_audio = idle_audio_state.rng.getstate()
    for tick in range(16, 401):
        idle_director._schedule_idle(idle_audio_state, tick * 0.25, True)
    assert idle_director.pending
    assert all(cue.priority == 0 and cue.cue_key.startswith("idle_") for cue in idle_director.pending)
    assert len(idle_director.pending) <= AUDIO_QUEUE_LIMIT
    assert idle_audio_state.rng.getstate() == rng_before_idle_audio

    # Investigation targets and the six-index voice mood are frozen into each
    # record, so later selection/status changes cannot rewrite older audio.
    investigation_audio_state = GameState(815)
    for audio_participant in investigation_audio_state.participants:
        set_statuses(audio_participant, 100, 1)
    for item_key in ITEM_KEYS:
        investigation_audio_state.player.inventory[item_key] = 100
    investigation_audio_state.choose_story(0)
    investigation_audio_state.select_npc(1)
    investigation_audio_state.choose_story(0)
    first_investigation = investigation_audio_state.round_history[0].events[0]
    second_investigation = investigation_audio_state.round_history[1].events[0]
    assert first_investigation.target_index == 1
    assert second_investigation.target_index == 2
    expected_first_mood = voice_mood_from_statuses(
        dict(zip(STATUS_KEYS, first_investigation.voice_statuses))
    )
    set_statuses(investigation_audio_state.npcs[0], 1, -1)
    delayed_director = AudioDirector(815, False)
    delayed_director._ingest_rounds(investigation_audio_state, 30.0)
    delayed_cues = list(delayed_director.pending)
    assert delayed_cues[0].speaker_index == 0
    assert delayed_cues[5].speaker_index == 1
    assert delayed_cues[0].mood == expected_first_mood
    assert delayed_cues[0].mood != voice_mood_from_statuses(
        investigation_audio_state.npcs[0].statuses
    )

    gesture_probe = GameState(81)
    gesture_npc = gesture_probe.npcs[0]
    monotonic_gestures = {
        "physical": ("sprint_pose", "self_hug"),
        "emotional": ("heart_hands", "dramatic_turn"),
        "cognitive": ("temple_tap", "funky_ehh"),
        "social": ("open_palms", "guarded_cross"),
        "sentient": ("prayer_pose", "sprint_pose"),
        "spiritual": ("affirmation", "fist_clench"),
    }
    for status_key, (high_gesture, low_gesture) in monotonic_gestures.items():
        set_statuses(gesture_npc, 50, 0)
        if status_key == "spiritual":
            gesture_npc.set_spiritual(-1)
        else:
            gesture_npc.statuses[status_key] = 10
        low_weights = gesture_probe._npc_gesture_weights(gesture_npc, "talk")
        if status_key == "spiritual":
            gesture_npc.set_spiritual(1)
        else:
            gesture_npc.statuses[status_key] = 90
        high_weights = gesture_probe._npc_gesture_weights(gesture_npc, "talk")
        assert high_weights[high_gesture] > low_weights[high_gesture]
        assert low_weights[low_gesture] > high_weights[low_gesture]
        assert all(weight > 0.0 for weight in high_weights.values())
        assert all(weight > 0.0 for weight in low_weights.values())

    first_gestures = GameState(82)
    second_gestures = GameState(82)
    for probe in (first_gestures, second_gestures):
        set_statuses(probe.npcs[0], 63, -1)
    first_sequence = [
        first_gestures._choose_npc_gesture(first_gestures.npcs[0], action)
        for action in GameState.NPC_ACTIONS
    ]
    second_sequence = [
        second_gestures._choose_npc_gesture(second_gestures.npcs[0], action)
        for action in GameState.NPC_ACTIONS
    ]
    assert first_sequence == second_sequence
    assert all(key in NPC_GESTURE_KEYS for key in first_sequence)

    assert state.begin_trade()
    state.set_trade_item("offer", "dollars")
    state.set_trade_quantity("offer", 5)
    state.set_trade_item("request", "water_liters")
    state.set_trade_quantity("request", 1)
    for person in (state.player, npc):
        set_statuses(state.player, 50, 0)
        set_statuses(npc, 50, 0)
        chances = []
        for spiritual in (-1, 0, 1):
            person.set_spiritual(spiritual)
            chances.append(state.trade_willingness())
        assert chances[0] < chances[1] < chances[2]
    state.cancel_trade()

    set_statuses(participant, 50, 0)
    combat_scores = []
    for spiritual in (-1, 0, 1):
        participant.set_spiritual(spiritual)
        combat_scores.append(state._combat_score(participant))
    assert combat_scores[0] < combat_scores[1] < combat_scores[2]
    assert GameApp._status_color(None, -1, True) == RED
    assert GameApp._status_color(None, 0, True) == MUTED
    assert GameApp._status_color(None, 1, True) == CYAN
    state.validate()

    for item_key in ITEM_KEYS:
        state = GameState(120)
        for participant in state.participants:
            set_statuses(participant, 100, 1)
        state.selected_item = item_key
        state.player.inventory[item_key] = 100
        before = state.player.inventory[item_key]
        state.interact("use")
        assert state.player.inventory[item_key] == before - ITEM_BY_KEY[item_key].use_amount
        state.validate()

        state = GameState(121)
        for participant in state.participants:
            set_statuses(participant, 100, 1)
        state.selected_item = item_key
        state.player.inventory[item_key] = 100
        target = state.selected_npc
        target_before = target.inventory[item_key]
        state.interact("give")
        assert state.player.inventory[item_key] == 100 - ITEM_BY_KEY[item_key].give_amount
        assert target.inventory[item_key] == target_before + ITEM_BY_KEY[item_key].give_amount
        state.validate()

    # A gift transfers a future option instead of applying a free duplicate
    # dose. Only a later use action consumes the item and realizes its effects.
    transfer_probe = GameState(122)
    transfer_target = transfer_probe.selected_npc
    set_statuses(transfer_target, 30, 0)
    transfer_probe.player.inventory["food_pounds"] = 10
    target_food_before = transfer_target.inventory["food_pounds"]
    target_physical_before = transfer_target.statuses["physical"]
    summary, success = transfer_probe._resolve_actor_action(
        transfer_probe.player,
        transfer_target,
        "give",
        "food_pounds",
    )
    assert success and "gives" in summary
    assert transfer_target.inventory["food_pounds"] == target_food_before + 1
    assert transfer_target.statuses["physical"] == target_physical_before

    for action in (
        "talk", "listen", "compliment", "flirt", "antagonize", "fight",
        "give", "steal", "reflect", "use", "rest",
    ):
        state = GameState(200)
        for participant in state.participants:
            set_statuses(participant, 100, 1)
            for item_key in ITEM_KEYS:
                participant.inventory[item_key] = 100
        state.interact(action)
        actors = [event.actor_index for event in state.last_round_events]
        assert actors == [0, 1, 2, 3, 4]
        assert all(event.target_index is not None for event in state.last_round_events)
        assert all(event.action in GameState.NPC_ACTIONS for event in state.last_round_events[1:])
        assert state.last_round_events[0].gesture_key is None
        assert all(event.gesture_key in NPC_GESTURE_KEYS for event in state.last_round_events[1:])
        assert all(npc.gesture_turn == state.turn for npc in state.npcs)
        event_snapshot = state.public_status_snapshot()["last_round"]
        assert event_snapshot[0]["gesture_key"] is None
        assert [entry["gesture_key"] for entry in event_snapshot[1:]] == [
            event.gesture_key for event in state.last_round_events[1:]
        ]
        assert all(entry["voice_statuses"] is not None for entry in event_snapshot[1:])
        assert all("decision_score" not in entry for entry in event_snapshot)
        assert all(entry["reasoning"] for entry in event_snapshot[1:])
        assert event_snapshot[0]["reasoning"] == ""
        assert [entry["item_key"] for entry in event_snapshot] == [
            event.item_key for event in state.last_round_events
        ]
        assert [entry["impact"] for entry in event_snapshot] == [
            event.impact for event in state.last_round_events
        ]
        assert all(entry["impact"] for entry in event_snapshot)
        assert all(
            (entry["exchange"] is not None) == (event.exchange is not None)
            for entry, event in zip(event_snapshot, state.last_round_events)
        )
        assert (event_snapshot[0]["voice_statuses"] is not None) == (
            state.last_round_events[0].target_index is not None
            and state.last_round_events[0].target_index > 0
        )
        npc_snapshot = state.public_status_snapshot()["npcs"]
        assert all(view["gesture_key"] in NPC_GESTURE_KEYS for view in npc_snapshot)
        assert all(
            view["gesture_label"] == GESTURE_LABELS[view["gesture_key"]]
            for view in npc_snapshot
        )
        assert state.public_status_snapshot()["player"]["gesture_key"] is None
        state.validate()

    # Self-elimination is one player-only interaction. It consumes Tyler's
    # slot, is frozen in the activity archive, and does not prevent any living
    # NPC from receiving its later slot in the same round.
    state = GameState(201)
    for participant in state.participants:
        set_statuses(participant, 100, 1)
    force_npc_decisions(
        state,
        "compliment",
        lambda npc: next(person for person in state.living_participants if person is not npc),
    )
    state.interact("self_eliminate")
    assert state.player.eliminated and state.player.statuses["physical"] == 0
    assert not state.is_ending and state.phase == "autonomous"
    assert [event.actor_index for event in state.last_round_events] == [0, 1, 2, 3, 4]
    self_event = state.last_round_events[0]
    assert self_event.action == "self_eliminate"
    assert self_event.target_index == 0
    assert self_event.eliminated_indices == (0,)
    assert state.round_history[-1].events[0] == self_event
    assert all(event.action in GameState.NPC_ACTIONS for event in state.last_round_events[1:])
    assert "self_eliminate" not in GameState.NPC_ACTIONS

    # The resulting autonomous simulation retains the exact-one-survivor
    # terminal rule even when room strain reaches every remaining NPC at once.
    for npc in state.living_npcs:
        npc.statuses["physical"] = 1
    assert state.advance_autonomous_round()
    assert state.is_ending and len(state.living_participants) == 1
    assert state.last_survivor in state.npcs
    state.validate()

    # Internal calls cannot assign the player-only action to an NPC.
    state = GameState(202)
    npc = state.selected_npc
    npc_physical = npc.statuses["physical"]
    try:
        state._resolve_actor_action(npc, npc, "self_eliminate")
    except ValueError as error:
        assert "player-only" in str(error)
    else:
        raise AssertionError("An NPC was allowed to self-eliminate")
    assert npc.statuses["physical"] == npc_physical and npc.alive

    # The UI exposes exactly one wide, prominent player-only control in the
    # seven-column 1080p action grid, with room beneath it for the watermark.
    state = GameState(203, "Echo Seven")
    action_probe = GameApp.__new__(GameApp)
    action_probe.state = state
    action_probe.player_turn_cooldown = False
    drawn_actions = []
    action_probe._draw_button = lambda rect, label, kind, value, accent, prominent=False, enabled=True: drawn_actions.append(
        (rect, label, kind, value, accent, prominent, enabled)
    )
    GameApp._draw_action_choices(action_probe, TOP_HEIGHT + 392)
    assert len(drawn_actions) == len(GameState.ACTIONS)
    assert {entry[3] for entry in drawn_actions} == set(GameState.ACTIONS)
    assert {"flirt", "steal", "trade"} <= {entry[3] for entry in drawn_actions}
    self_buttons = [entry for entry in drawn_actions if entry[3] == "self_eliminate"]
    assert len(self_buttons) == 1
    self_button = self_buttons[0]
    assert "SELF ELIMINATE" in self_button[1]
    assert "ECHO SEVEN ONLY" in self_button[1]
    assert self_button[4] == RED and self_button[5] and self_button[6]
    assert len({entry[0][1] for entry in drawn_actions}) == 2
    assert max(rect[1] + rect[3] for rect, *_ in drawn_actions) <= HEIGHT - 26
    assert len([entry for entry in drawn_actions if entry[4] == RED and entry[5]]) == 1
    action_probe.player_turn_cooldown = True
    drawn_actions.clear()
    GameApp._draw_action_choices(action_probe, TOP_HEIGHT + 392)
    assert drawn_actions and not any(entry[6] for entry in drawn_actions)

    # All 25 item pairings, including unequal same-item exchanges, transfer
    # custom quantities atomically and conserve the party's inventory.
    for offered_key in ITEM_KEYS:
        for requested_key in ITEM_KEYS:
            state = GameState(500)
            for participant in state.participants:
                set_statuses(participant, 100, 1)
                for item_key in ITEM_KEYS:
                    participant.inventory[item_key] = 1_000
            target = state.selected_npc
            assert state.begin_trade()
            state.set_trade_item("offer", offered_key)
            state.set_trade_item("request", requested_key)
            state.set_trade_quantity("offer", 7)
            state.set_trade_quantity("request", 3)
            player_before = dict(state.player.inventory)
            target_before = dict(target.inventory)
            totals_before = {
                key: sum(participant.inventory[key] for participant in state.participants)
                for key in ITEM_KEYS
            }
            expected_player = dict(player_before)
            expected_target = dict(target_before)
            expected_player[offered_key] -= 7
            expected_target[offered_key] += 7
            expected_target[requested_key] -= 3
            expected_player[requested_key] += 3
            assert state.confirm_trade(roll=0.0)
            assert state.turn == 1
            assert state.trade_draft is None
            assert [event.actor_index for event in state.last_round_events] == [0, 1, 2, 3, 4]
            assert state.last_round_events[0].action == "trade"
            assert state.last_round_events[0].exchange == ExchangePlan(
                offered_key,
                7,
                requested_key,
                3,
            )
            assert state.last_round_events[0].impact
            assert state.player.inventory == expected_player
            assert target.inventory == expected_target
            for key in ITEM_KEYS:
                assert sum(participant.inventory[key] for participant in state.participants) == totals_before[key]
            state.validate()

    # Willingness responds independently and monotonically to every one of the
    # six indexes for both participants.
    state = GameState(600)
    assert state.begin_trade()
    state.set_trade_item("offer", "dollars")
    state.set_trade_quantity("offer", 5)
    state.set_trade_item("request", "water_liters")
    state.set_trade_quantity("request", 1)
    draft = state.trade_draft
    for participant in (state.player, state.selected_npc):
        set_statuses(state.player, 1, 0)
        set_statuses(state.selected_npc, 1, 0)
        baseline = state.trade_willingness(draft)
        for key in STATUS_KEYS:
            old = participant.statuses[key]
            participant.statuses[key] = 1 if key == "spiritual" else 100
            assert state.trade_willingness(draft) > baseline
            participant.statuses[key] = old
    set_statuses(state.player, 1, 0)
    set_statuses(state.selected_npc, 1, 0)
    low_willingness = state.trade_willingness(draft)
    set_statuses(state.player, 100, 1)
    set_statuses(state.selected_npc, 100, 1)
    high_willingness = state.trade_willingness(draft)
    assert 0.05 <= low_willingness < high_willingness <= 0.95
    state.set_trade_quantity("request", 20)
    unfair_willingness = state.trade_willingness()
    state.set_trade_quantity("request", 1)
    assert unfair_willingness < state.trade_willingness()

    # Locally invalid proposals consume no round. Once a valid private-stock
    # proposal is submitted, hidden shortage or refusal consumes Tyler's turn.
    state = GameState(610)
    force_npc_decisions(state, "talk", lambda npc: state.player)
    for participant in (state.player, state.selected_npc):
        for key in ITEM_KEYS:
            participant.inventory[key] = 100
    assert state.begin_trade()
    state.set_trade_item("offer", "water_liters")
    state.set_trade_item("request", "food_pounds")
    state.set_trade_quantity("offer", 10)
    state.set_trade_quantity("request", 10)
    turn_before = state.turn
    state.player.inventory["water_liters"] = 9
    shortage_before = [dict(participant.inventory) for participant in state.participants]
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == turn_before
    assert [participant.inventory for participant in state.participants] == shortage_before
    state.player.inventory["water_liters"] = 100
    state.selected_npc.inventory["food_pounds"] = 9
    private_shortage_before = [dict(participant.inventory) for participant in state.participants]
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == turn_before + 1
    assert [participant.inventory for participant in state.participants] == private_shortage_before
    assert state.trade_draft is None
    assert state.last_round_events[0].action == "trade"
    assert not state.last_round_events[0].success

    state = GameState(610)
    for participant in (state.player, state.selected_npc):
        for key in ITEM_KEYS:
            participant.inventory[key] = 100
    turn_before = state.turn
    assert state.begin_trade()
    state.selected_npc.inventory["food_pounds"] = 100
    state.trade_draft.offered_item = "unknown"
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == turn_before
    assert state.trade_draft is None
    assert state.begin_trade()
    state.trade_draft.offered_quantity = 0
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == turn_before
    assert state.trade_draft is None
    assert state.begin_trade()
    state.trade_draft.target_index = True
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == turn_before
    assert state.trade_draft is None
    assert state.begin_trade()
    state.set_trade_item("offer", "water_liters")
    state.set_trade_item("request", "food_pounds")
    state.set_trade_quantity("offer", 10)
    state.set_trade_quantity("request", 10)
    for invalid_quantity in (False, 1.5, "3"):
        try:
            state.set_trade_quantity("offer", invalid_quantity)
        except TypeError:
            pass
        else:
            raise AssertionError("Non-integer trade quantity was accepted")
    for invalid_quantity in (0, INVENTORY_CAP + 1):
        try:
            state.set_trade_quantity("offer", invalid_quantity)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-range trade quantity was accepted")
    try:
        state.confirm_trade(roll=2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range trade roll was accepted")
    assert state.turn == turn_before
    state.cancel_trade()
    assert state.turn == turn_before

    state = GameState(611)
    force_npc_decisions(state, "talk", lambda npc: state.player)
    for participant in (state.player, state.selected_npc):
        for key in ITEM_KEYS:
            participant.inventory[key] = 100
    assert state.begin_trade()
    state.set_trade_item("offer", "water_liters")
    state.set_trade_item("request", "food_pounds")
    state.set_trade_quantity("offer", 5)
    state.set_trade_quantity("request", 5)
    refusal_before = [dict(participant.inventory) for participant in state.participants]
    assert not state.confirm_trade(roll=1.0)
    assert state.turn == 1
    assert [participant.inventory for participant in state.participants] == refusal_before

    # Capacity checks use final net values, so they prevent overflow without
    # rejecting a safe same-item exchange whose intermediate value is high.
    state = GameState(612)
    force_npc_decisions(state, "talk", lambda npc: state.player)
    state.player.inventory["water_liters"] = 100
    state.selected_npc.inventory["water_liters"] = INVENTORY_CAP
    assert state.begin_trade()
    state.set_trade_item("offer", "water_liters")
    state.set_trade_item("request", "water_liters")
    state.set_trade_quantity("offer", 7)
    state.set_trade_quantity("request", 10)
    assert state.confirm_trade(roll=0.0)
    assert state.player.inventory["water_liters"] == 103
    assert state.selected_npc.inventory["water_liters"] == INVENTORY_CAP - 3

    state = GameState(613)
    state.player.inventory["water_liters"] = 100
    state.player.inventory["food_pounds"] = INVENTORY_CAP
    state.selected_npc.inventory["water_liters"] = 100
    state.selected_npc.inventory["food_pounds"] = 100
    assert state.begin_trade()
    state.set_trade_item("offer", "water_liters")
    state.set_trade_item("request", "food_pounds")
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == 0
    assert not state.last_round_events

    state = GameState(614)
    state.player.inventory["water_liters"] = 100
    state.selected_npc.inventory["water_liters"] = INVENTORY_CAP
    state.selected_npc.inventory["food_pounds"] = 100
    assert state.begin_trade()
    state.set_trade_item("offer", "water_liters")
    state.set_trade_item("request", "food_pounds")
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == 1
    assert state.last_round_events[0].action == "trade"

    state = GameState(615)
    state.player.inventory["water_liters"] = 100
    state.selected_npc.inventory["water_liters"] = 100
    assert state.begin_trade()
    state.set_trade_item("offer", "water_liters")
    state.set_trade_item("request", "water_liters")
    state.set_trade_quantity("offer", 4)
    state.set_trade_quantity("request", 4)
    assert not state.confirm_trade(roll=0.0)
    assert state.turn == 0

    state = GameState(616)
    assert state.begin_trade()
    state.trade_draft.target_index = True
    state.interact("trade")
    assert state.trade_draft is None
    assert state.turn == 0
    state.validate()

    story_corpus = " ".join(
        [scene.title + " " + scene.prompt for scene in SCENES.values()]
        + [choice.label + " " + choice.outcome for scene in SCENES.values() for choice in scene.choices]
    ).lower()
    assert "fifth signal" in story_corpus
    assert "no known escape" in story_corpus
    for forbidden in (
        "observatory",
        "market",
        "checkpoint",
        "aqueduct",
        "convoy",
        "maintenance passage",
        "open seam",
    ):
        assert forbidden not in story_corpus
    assert all(
        choice.next_scene in SCENES
        for scene in SCENES.values()
        for choice in scene.choices
    )

    for scene_id, scene in SCENES.items():
        assert len(scene.choices) == 3
        for choice_index, choice in enumerate(scene.choices):
            assert choice.next_scene in SCENES
            state = GameState(300 + choice_index)
            state.scene_id = scene_id
            for participant in state.participants:
                set_statuses(participant, 100, 1)
            for item_key in ITEM_KEYS:
                state.player.inventory[item_key] = 100
            state.choose_story(choice_index)
            assert state.last_round_events[0].actor_index == 0
            assert state.last_round_events[0].action == "investigate"
            assert state.last_round_events[0].target_index == state.selected_npc_index + 1
            assert [event.actor_index for event in state.last_round_events] == [0, 1, 2, 3, 4]
            state.validate()

    state = GameState(698)
    state.scene_id = "mechanism"
    state.player.inventory["dollars"] = 0
    for participant in state.participants:
        set_statuses(participant, 100, 1)
    state.choose_story(0)
    assert state.scene_id == "mechanism"
    assert not state.last_round_events[0].success
    assert state.turn == 1
    state.validate()

    state = GameState(699)
    for participant in state.participants:
        set_statuses(participant, 100, 1)
        for item_key in ITEM_KEYS:
            participant.inventory[item_key] = 100
    for choice_index in (0, 0, 2, 0, 0):
        state.choose_story(choice_index)
    assert not state.is_ending
    assert state.fifth_signal_found
    assert state.scene_id == "signal_refrain"
    assert state.turn == 5
    loop_ids = {"signal_refrain", "signal_afterimage", "signal_counterpoint"}
    visited = {state.scene_id}
    for choice_index in (0, 0, 1, 2, 1, 0, 2, 0, 1):
        state.choose_story(choice_index)
        assert not state.is_ending
        assert state.scene_id in loop_ids
        visited.add(state.scene_id)
    assert visited == loop_ids
    assert state.turn == 14
    state.validate()

    # Every status has a distinct, headlessly testable facial channel.
    state = GameState(700)
    npc = state.selected_npc
    set_statuses(npc, 50, 0)
    base_expression = face_expression_parameters(npc, 2.75, 0.83)
    for key in STATUS_KEYS:
        old = npc.statuses[key]
        npc.statuses[key] = 1 if key == "spiritual" else 100
        changed = face_expression_parameters(npc, 2.75, 0.83)
        assert changed != base_expression
        npc.statuses[key] = old

    # Positive and negative options move the two involved people in opposite
    # directions without granting either participant a second action.
    state = GameState(705)
    for participant in state.participants:
        set_statuses(participant, 50, 0)
    target = state.selected_npc
    target_before = dict(target.statuses)
    state._resolve_actor_action(state.player, target, "compliment")
    assert target.statuses["emotional"] > target_before["emotional"]
    assert target.statuses["social"] > target_before["social"]

    state = GameState(706)
    for participant in state.participants:
        set_statuses(participant, 50, 0)
    target = state.selected_npc
    target_before = dict(target.statuses)
    state._resolve_actor_action(state.player, target, "antagonize")
    assert target.statuses["emotional"] < target_before["emotional"]
    assert target.statuses["social"] < target_before["social"]

    # Low emotional/social indexes make autonomous hostility and fighting more
    # likely, while a stable NPC is more inclined to compliment.
    state = GameState(707)
    npc = state.selected_npc
    set_statuses(npc, 90, 1)
    stable_weights = state._npc_action_weights(npc)
    npc.statuses["emotional"] = 5
    npc.statuses["social"] = 5
    hostile_weights = state._npc_action_weights(npc)
    assert hostile_weights["antagonize"] > stable_weights["antagonize"]
    assert hostile_weights["fight"] > stable_weights["fight"]
    assert hostile_weights["compliment"] < stable_weights["compliment"]

    state = GameState(7071)
    set_statuses(state.player, 1, 0)
    set_statuses(state.selected_npc, 100, 1)
    defender_physical = state.selected_npc.statuses["physical"]
    state.rng = random.Random(2)
    _, hit = state._resolve_fight(state.player, state.selected_npc)
    assert not hit
    assert state.selected_npc.statuses["physical"] < defender_physical

    # Seeded rounds are reproducible and always place the player first.
    first = GameState(708)
    second = GameState(708)
    for state in (first, second):
        for participant in state.participants:
            set_statuses(participant, 100, 1)
        state.interact("talk")
    first_log = [(event.actor_index, event.action, event.target_index) for event in first.last_round_events]
    second_log = [(event.actor_index, event.action, event.target_index) for event in second.last_round_events]
    assert first_log == second_log
    assert [entry[0] for entry in first_log] == [0, 1, 2, 3, 4]

    # Completed rounds are frozen exactly once and the UI cursor can walk both
    # directions without losing a pinned older round when newer records arrive.
    state = GameState(709)
    for participant in state.participants:
        set_statuses(participant, 100, 1)
    force_npc_decisions(state, "compliment", lambda npc: state.player)
    for action in ("talk", "listen", "compliment"):
        state.interact(action)
    assert [record.round_number for record in state.round_history] == [1, 2, 3]
    assert all(isinstance(record.events, tuple) for record in state.round_history)
    assert state.round_history[-1].events == tuple(state.last_round_events)
    try:
        state.round_history[0].events = ()
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("Archived round records were mutable")
    activity_app = GameApp.__new__(GameApp)
    activity_app.state = state
    activity_app.activity_history_index = None
    assert activity_app._selected_activity_record().round_number == 3
    activity_app._scroll_activity(-1)
    assert activity_app._selected_activity_record().round_number == 2
    pinned_index = activity_app.activity_history_index
    state.interact("talk")
    assert activity_app.activity_history_index == pinned_index
    assert activity_app._selected_activity_record().round_number == 2
    activity_app._scroll_activity(1)
    assert activity_app._selected_activity_record().round_number == 3
    activity_app._scroll_activity(10)
    assert activity_app.activity_history_index is None
    assert activity_app._selected_activity_record().round_number == 4
    frozen_before_ending = tuple(state.round_history)
    for npc in state.npcs:
        npc.statuses["physical"] = 0
    state._resolve_eliminations()
    assert state.is_ending and state.last_survivor is state.player
    assert tuple(state.round_history) == frozen_before_ending
    # The exact same older/newer cursor remains active under the final overlay;
    # PageUp/PageDown, wheel input, and its two buttons all call this method.
    activity_app.activity_history_index = None
    assert activity_app._selected_activity_record().round_number == 4
    activity_app._scroll_activity(-1)
    assert activity_app._selected_activity_record().round_number == 3
    activity_app._scroll_activity(-99)
    assert activity_app._selected_activity_record().round_number == 1
    activity_app._scroll_activity(99)
    assert activity_app.activity_history_index is None
    assert activity_app._selected_activity_record().round_number == 4
    assert tuple(state.round_history) == frozen_before_ending
    state.validate()

    # A player-led round that produces the sole survivor is also archived once,
    # even though no later NPC slot or room-strain pass can run.
    state = GameState(7091)
    for participant in state.participants:
        set_statuses(participant, 100, 1)
    for npc in state.npcs[:3]:
        npc.statuses["physical"] = 0
    state._resolve_eliminations()
    state.selected_npc.statuses["physical"] = 1
    state.rng = random.Random(1)
    state.interact("fight")
    assert state.is_ending and state.last_survivor is state.player
    assert len(state.round_history) == 1
    assert state.round_history[0].round_number == 1
    assert state.round_history[0].events == tuple(state.last_round_events)
    terminal_archive = tuple(state.round_history)
    assert not state.advance_autonomous_round()
    assert tuple(state.round_history) == terminal_archive
    state.validate()

    # A player fight can eliminate an NPC before its slot; that NPC remains in
    # the portrait list but is skipped while later NPCs still act once.
    state = GameState(710)
    for participant in state.participants:
        set_statuses(participant, 100, 1)
    fallen = state.selected_npc
    fallen.statuses["physical"] = 1
    state.rng = random.Random(1)
    state.interact("fight")
    assert fallen.eliminated and fallen.statuses["physical"] == 0
    assert [event.actor_index for event in state.last_round_events] == [0, 2, 3, 4]
    assert state.last_round_events[0].eliminated_indices == (1,)
    assert fallen in state.npcs and len(state.npcs) == 4
    dead_statuses = dict(fallen.statuses)
    dead_inventory = dict(fallen.inventory)
    state.player.inventory["water_liters"] = 10
    state.player.inventory["food_pounds"] = 10
    state.interact("rest")
    assert fallen.statuses == dead_statuses
    assert fallen.inventory == dead_inventory
    eliminated_expression = face_expression_parameters(fallen, 2.75, 0.83)
    assert eliminated_expression["eliminated"] == 1.0
    assert eliminated_expression["eye_open"] > 1.4
    assert eliminated_expression["pupil_radius"] == 0.0
    state.validate()

    # If an NPC eliminates the player, later NPC slots still resolve and the
    # state switches to autonomous survivor rounds instead of ending.
    state = GameState(711)
    state.player.statuses["physical"] = 1
    for npc in state.npcs:
        set_statuses(npc, 100, 1)
    state.rng = random.Random(1)
    force_npc_decisions(
        state,
        "fight",
        lambda npc: (
            state.player
            if state.player.alive
            else next(person for person in state.living_participants if person is not npc)
        ),
    )
    state.interact("compliment")
    assert state.player.eliminated
    assert not state.is_game_over and not state.is_ending
    assert state.phase == "autonomous"
    assert [event.actor_index for event in state.last_round_events] == [0, 1, 2, 3, 4]
    player_death_turn = state.turn
    state.interact("rest")
    state.choose_story(0)
    assert state.turn == player_death_turn
    assert state.advance_autonomous_round()
    assert state.turn == player_death_turn + 1
    assert all(event.actor_index > 0 for event in state.last_round_events)
    for npc in state.living_npcs:
        npc.statuses["physical"] = min(npc.statuses["physical"], 8)
    autonomous_rounds = 1
    while not state.is_ending and autonomous_rounds < 20:
        assert state.advance_autonomous_round()
        autonomous_rounds += 1
    assert state.is_game_over and state.is_ending
    assert state.player.eliminated
    assert len(state.living_participants) == 1
    assert state.last_survivor in state.npcs
    assert "Last Survivor" in state.ending_title
    assert len(state.round_history) == state.turn
    assert state.round_history[-1].round_number == state.turn
    assert state.round_history[-1].events == tuple(state.last_round_events)
    terminal_archive = tuple(state.round_history)
    assert not state.advance_autonomous_round()
    assert tuple(state.round_history) == terminal_archive
    state.validate()

    # With no player choices, the surviving NPC rounds still carry the signal
    # mystery into discovery and then cycle its distracting three-scene loop.
    state = GameState(7111)
    for participant in state.participants:
        set_statuses(participant, 100, 0)
    state.player.statuses["physical"] = 0
    state._resolve_eliminations()
    force_npc_decisions(
        state,
        "compliment",
        lambda npc: next(person for person in state.living_participants if person is not npc),
    )
    autonomous_scenes = []
    for _ in range(7):
        assert state.advance_autonomous_round()
        autonomous_scenes.append(state.scene_id)
    assert autonomous_scenes[:4] == ["resonance", "mechanism", "pattern", "discovery"]
    assert autonomous_scenes[4:] == [
        "signal_refrain",
        "signal_afterimage",
        "signal_counterpoint",
    ]
    assert state.fifth_signal_found and not state.is_ending
    state.validate()

    state = GameState(712)
    for participant in state.participants:
        set_statuses(participant, 100, 1)
    state.turn = 3
    state.npcs[-1].statuses["physical"] = 1
    force_npc_decisions(state, "compliment", lambda npc: state.player)
    state.interact("compliment")
    assert state.npcs[-1].eliminated
    assert state.round_notices and "ROOM STRAIN" in state.round_notices[0]
    assert state.public_status_snapshot()["round_notices"] == state.round_notices
    state.validate()

    state = GameState(999)
    state.player.adjust("physical", -10_000)
    state.player.adjust("emotional", 10_000)
    state.player.set_spiritual(7)
    state.player.change_item("dollars", INVENTORY_CAP * 2)
    state._resolve_eliminations()
    assert state.player.statuses["physical"] == 0
    assert state.player.eliminated
    assert not state.is_ending and state.phase == "autonomous"
    assert state.player.statuses["emotional"] == 100
    assert state.player.statuses["spiritual"] == 1
    assert state.player.inventory["dollars"] == INVENTORY_CAP
    for npc in state.living_npcs:
        npc.statuses["physical"] = 0
    state._resolve_eliminations()
    assert state.is_ending
    assert len(state.living_participants) == 1
    assert state.last_survivor in state.npcs
    state.validate()

    # Four simultaneous NPC eliminations leave Tyler as the exact terminal
    # survivor; the game never ends with two survivors or with zero.
    state = GameState(1000)
    for npc in state.npcs:
        npc.statuses["physical"] = 0
    state._resolve_eliminations()
    assert state.is_ending and state.last_survivor is state.player
    assert len(state.living_participants) == 1
    state.validate()

    # The unforced Full Final Release planner, including sharing and rest, cannot settle
    # into an immortal cooperation loop after the player is gone. Escalating
    # survival pressure still carries every seeded room to one living signal.
    for lifecycle_seed in range(12):
        lifecycle = GameState(20_000 + lifecycle_seed)
        lifecycle.player.statuses["physical"] = 0
        lifecycle._resolve_eliminations()
        lifecycle_rounds = 0
        while not lifecycle.is_ending and lifecycle_rounds < 90:
            assert lifecycle.advance_autonomous_round()
            lifecycle.validate()
            lifecycle_rounds += 1
        assert lifecycle.is_ending
        assert len(lifecycle.living_participants) == 1
        assert lifecycle.last_survivor in lifecycle.npcs

    for lifecycle_seed in range(6):
        lifecycle = GameState(21_000 + lifecycle_seed)
        lifecycle_rounds = 0
        while not lifecycle.is_ending and lifecycle_rounds < 110:
            if lifecycle.player.alive:
                lifecycle.interact("compliment")
            else:
                assert lifecycle.advance_autonomous_round()
            lifecycle.validate()
            lifecycle_rounds += 1
        assert lifecycle.is_ending
        assert len(lifecycle.living_participants) == 1

    print(
        "OK: Fifth Signal Full Final Release at 1920x1080, hidden endless room arc, true one-second nonblocking "
        "participant beats, player-first and post-player autonomous rounds, exact player/NPC action "
        "parity except player-only self-elimination, flirt and five-item steal resolution, NPC-initiated "
        "atomic exchange planning, exact-one-survivor endings, immutable scrollable descriptive round "
        "history with frozen outcomes, impacts, motives, and final-screen review, validated persistent "
        "custom player names, ternary spirituality with black flaming horns, neutral-window dimming, "
        "and layered glowing halos, 25 player trade pairings, 6-index "
        "expressions, six-index/five-supply NPC utility reasoning, 22-state NPC gesture logic, five new "
        "full-body emotes, four persistent idle personalities, original emotion-synced "
        "procedural chip voices, four NPC victory memes, four player particle finales, articulated portrait "
        f"meshes, {len(TEXTURE_ASSET_FILES)} licensed texture channels, "
        f"and {len(SCENES)} room scenes validated."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The Fifth Signal sealed-room PyOpenGL Full Final Release"
    )
    parser.add_argument("--check", action="store_true", help="validate game logic without opening a window")
    parser.add_argument("--seed", type=int, default=None, help="use a repeatable randomized starting state")
    parser.add_argument(
        "--name",
        type=normalize_player_name,
        default=None,
        help="start immediately with a validated custom player name",
    )
    parser.add_argument("--mute", action="store_true", help="start with procedural voices muted")
    parser.add_argument("--capture", metavar="PATH", help="render one frame to an image and exit")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.check:
        run_checks()
        return
    load_graphics()
    GameApp(
        args.seed,
        args.name,
        audio_enabled=args.capture is None,
        start_muted=args.mute,
    ).run(args.capture)


if __name__ == "__main__":
    main()
