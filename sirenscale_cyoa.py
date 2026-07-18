import json
import math
import os
import random
import sys
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List

import pygame

WIDTH, HEIGHT = 800, 480
FPS = 60
MAX_DEPTH = 8  # 2^8 = 256 full routes
SAVE_FILE = "sirenscale_save.json"

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sirenscale: The Hundred Paths")
clock = pygame.time.Clock()

TITLE_FONT = pygame.font.SysFont("georgia", 28, bold=True)
SUBTITLE_FONT = pygame.font.SysFont("georgia", 18, bold=True)
BODY_FONT = pygame.font.SysFont("georgia", 18)
SMALL_FONT = pygame.font.SysFont("consolas", 14)
BUTTON_FONT = pygame.font.SysFont("georgia", 18, bold=True)

BG_TOP = (10, 18, 30)
BG_BOTTOM = (5, 8, 14)
PANEL = (12, 20, 28)
PANEL_ALT = (21, 31, 45)
PANEL_EDGE = (74, 137, 157)
TEXT = (227, 238, 240)
MUTED = (149, 178, 182)
ACCENT = (94, 224, 207)
ACCENT_2 = (141, 179, 255)
WARNING = (255, 193, 120)
DANGER = (225, 110, 110)
GOOD = (153, 224, 159)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

ACT_NAMES = [
    "The Rotwater Gate",
    "The Black Causeway",
    "The Thorn Mire",
    "The Salt-Glass Warrens",
    "The Sunken Choir",
    "The Pale Observatory",
    "The Iron Estuary",
    "The Last Flood",
]

REGIONS = [
    "a drowned boardwalk stitched together with iron nails",
    "a marsh cathedral half-swallowed by reeds",
    "a bone-white ferry landing drifting in mist",
    "a rusted lock gate breathing brine through broken valves",
    "a drowned orchard where every trunk wears hanging lanterns",
    "a causeway paved in wet black stone",
    "a graveyard of skiffs tangled in eelgrass",
    "a wind-carved dune of powdered glass",
    "a flooded archive whose shelves float like rafts",
    "a chamber of cracked mirrors beneath the marsh",
    "a chain bridge trembling above luminous water",
    "a collapsed watchtower ringed by thorn-vines",
]

OMENS = [
    "moonlight catches in Sirenscale's squid-shimmer skin like oil on water",
    "the old flood bells ring once with no visible hand",
    "tiny silver fish circle your boots in a perfect spiral",
    "Sirenscale's crocodile lids blink one set after the other",
    "the air smells of salt, rust, and wet paper",
    "a distant choir rises from somewhere below the waterline",
    "the reeds lean the wrong way, as if a hidden tide is inhaling",
    "black rain begins and ends in the span of a breath",
    "the shadows of birds pass overhead, though the sky is empty",
    "your reflection lags behind you in the nearest puddle",
    "Sirenscale's iron arm leaks a line of dark oil into the water",
    "a pale current threads around your ankles like a warning",
]

TENSIONS = [
    "scavengers with lantern masks are stripping the place for relic wire",
    "a marsh knight drags a chain-net across the only dry path",
    "something beneath the boards keeps matching your footsteps",
    "a nest of leech-hounds waits in the reeds, too still to be natural",
    "an old shrine is calling names in a human voice that is not human",
    "a ferryman made of stitched raincoat leather blocks the route",
    "glass-eyed sentries patrol the shoreline in pairs",
    "a ruined lift groans upward from the flooded dark with no operator",
    "a sickly green fire burns in the windows of a dead chapel",
    "the tide itself seems to be testing you, rising only where you stand",
    "a pack of bone divers is digging for something under the mud",
    "a procession of faceless pilgrims is crossing the route in silence",
]

DISCOVERIES = [
    "a bronze seal stamped with a queen no history remembers",
    "a waterlogged map that redraws itself whenever you look away",
    "a glass tooth the size of your thumb, warm to the touch",
    "a vault key grown over with salt crystals",
    "a reliquary filled with black seeds that click like teeth",
    "a spool of copper thread humming with trapped weather",
    "a compass that points toward grief instead of north",
    "a pearl of smoky light trapped in a cage of wire",
    "a child-sized crown made of reeds and silver hooks",
    "a page torn from an impossible book, still dry despite the flood",
    "a lantern filled with seawater and a living star",
    "a lockbox whose hinges open only underwater",
]

SIRENSCALE_LINES = [
    "\"The marsh never lies. It just edits.\"",
    "\"Pick a road and the flood will judge what kind of person chose it.\"",
    "\"I can smell fear, rust, and bad luck. Tonight we've got all three.\"",
    "\"Careful. Dead places love an audience.\"",
    "\"If this goes badly, I reserve the right to look unimpressed.\"",
    "\"You walk like someone who still thinks outcomes can be clean.\"",
    "\"I have seen this kind of silence before. Something is hiding inside it.\"",
    "\"There are softer ways through. There are louder ways too.\"",
    "\"I can guide you through the dark, but I cannot choose who you become in it.\"",
    "\"The tide keeps receipts. Try not to owe it too much.\"",
]

THEMES = [
    {
        "name": "stealth",
        "challenge": [
            "The safest line forward runs through narrow reed tunnels and beneath sagging ropes of moss.",
            "A hidden route appears where flooded beams form a dark tunnel under the walkway.",
            "Sirenscale spots a tide-channel wide enough for both of you if you move in silence.",
        ],
        "choices": [
            {
                "label": "Follow Sirenscale through the hidden channel",
                "effects": {"trust": 2, "insight": 1, "storm": 0, "feral": -1, "relic": 0},
            },
            {
                "label": "Cut across the open path before the sentries close in",
                "effects": {"trust": -1, "insight": 0, "storm": 1, "feral": 2, "relic": 0},
            },
        ],
    },
    {
        "name": "diplomacy",
        "challenge": [
            "The figures ahead are armed, frightened, and close enough to become allies or enemies.",
            "A tense stand-off blocks the route, and both sides keep waiting for someone else to make the first mistake.",
            "The marshfolk here respect force, but they remember courtesy even longer.",
        ],
        "choices": [
            {
                "label": "Offer words first and let Sirenscale read the room",
                "effects": {"trust": 2, "insight": 2, "storm": 0, "feral": -1, "relic": 0},
            },
            {
                "label": "Push through with cold certainty and a hand on your weapon",
                "effects": {"trust": -1, "insight": 0, "storm": 1, "feral": 2, "relic": 1},
            },
        ],
    },
    {
        "name": "hunt",
        "challenge": [
            "Whatever is tracking you has finally decided to stop being shy.",
            "The water ahead ripples in circles, and the circles are moving against the current.",
            "Something lean and patient is using the mist like camouflage.",
        ],
        "choices": [
            {
                "label": "Set an ambush with Sirenscale and wait for the strike",
                "effects": {"trust": 1, "insight": 2, "storm": 0, "feral": 1, "relic": 1},
            },
            {
                "label": "Charge the threat before it controls the ground",
                "effects": {"trust": 0, "insight": -1, "storm": 1, "feral": 3, "relic": 0},
            },
        ],
    },
    {
        "name": "ritual",
        "challenge": [
            "An altar stands half-submerged nearby, still active despite the centuries of rot.",
            "A sequence of bells, glyphs, and tides forms a ritual pattern only barely intact.",
            "Sirenscale recognizes a shrine mechanism that can open the deeper path if you dare touch it.",
        ],
        "choices": [
            {
                "label": "Complete the rite carefully with Sirenscale guiding you",
                "effects": {"trust": 2, "insight": 3, "storm": 1, "feral": -1, "relic": 1},
            },
            {
                "label": "Break the shrine and take what survives the backlash",
                "effects": {"trust": -2, "insight": 0, "storm": 2, "feral": 2, "relic": 2},
            },
        ],
    },
    {
        "name": "rescue",
        "challenge": [
            "Someone is trapped ahead, and the marsh is deciding whether to keep them.",
            "A stranger hangs between rescue and drowning beneath splintered boards.",
            "A small group is pinned in a rising pocket of floodwater just off the main route.",
        ],
        "choices": [
            {
                "label": "Help them, even if it costs time and safety",
                "effects": {"trust": 2, "insight": 1, "storm": 0, "feral": -1, "relic": 0},
            },
            {
                "label": "Move on before the marsh turns their crisis into yours",
                "effects": {"trust": -2, "insight": 0, "storm": 1, "feral": 1, "relic": 1},
            },
        ],
    },
    {
        "name": "relic",
        "challenge": [
            "A buried chamber offers a prize, but the air around it feels watched.",
            "The route splits around a cache of old-world salvage no one else has found yet.",
            "A vault beneath the mud is opening itself by inches, almost inviting you in.",
        ],
        "choices": [
            {
                "label": "Take only what you can understand",
                "effects": {"trust": 1, "insight": 2, "storm": 0, "feral": 0, "relic": 2},
            },
            {
                "label": "Strip the place fast before the marsh reclaims it",
                "effects": {"trust": -1, "insight": -1, "storm": 1, "feral": 2, "relic": 3},
            },
        ],
    },
    {
        "name": "storm",
        "challenge": [
            "Weather gathers over the marsh with the intent of a living thing.",
            "A crackling front of black rain is rolling directly toward you.",
            "The air starts to glow around old metal and standing water at once.",
        ],
        "choices": [
            {
                "label": "Shelter with Sirenscale and wait for the pattern in the storm",
                "effects": {"trust": 2, "insight": 2, "storm": 1, "feral": 0, "relic": 0},
            },
            {
                "label": "Walk into it and force the storm to make way",
                "effects": {"trust": -1, "insight": 0, "storm": 3, "feral": 2, "relic": 0},
            },
        ],
    },
    {
        "name": "revelation",
        "challenge": [
            "The truth is close enough to hurt, and Sirenscale knows more than is comfortable.",
            "A hidden history surfaces here, one that links you, the flood, and your companion.",
            "The marsh offers an answer, but answers here always arrive with teeth.",
        ],
        "choices": [
            {
                "label": "Ask Sirenscale for the full truth, no matter the cost",
                "effects": {"trust": 3, "insight": 3, "storm": 0, "feral": -1, "relic": 0},
            },
            {
                "label": "Bury the truth for now and keep moving",
                "effects": {"trust": -1, "insight": -1, "storm": 1, "feral": 1, "relic": 1},
            },
        ],
    },
    {
        "name": "defense",
        "challenge": [
            "A chokepoint gives you one chance to hold the line and one chance to lose it forever.",
            "The route narrows to a single defensible platform above the flood.",
            "You can hear the enemy before you see them; too many feet, too little mercy.",
        ],
        "choices": [
            {
                "label": "Hold together with Sirenscale and make the ground matter",
                "effects": {"trust": 2, "insight": 1, "storm": 1, "feral": 1, "relic": 0},
            },
            {
                "label": "Break their momentum with a brutal first strike",
                "effects": {"trust": -1, "insight": 0, "storm": 1, "feral": 3, "relic": 0},
            },
        ],
    },
    {
        "name": "bargain",
        "challenge": [
            "A power older than the road itself is willing to make a deal.",
            "Something in the water knows your name and speaks it like an offer.",
            "The tide asks for a price in exchange for safe passage or forbidden leverage.",
        ],
        "choices": [
            {
                "label": "Negotiate carefully and refuse any hidden hooks",
                "effects": {"trust": 2, "insight": 2, "storm": 0, "feral": 0, "relic": 1},
            },
            {
                "label": "Take the dangerous bargain and trust yourself to survive it",
                "effects": {"trust": -2, "insight": 1, "storm": 2, "feral": 2, "relic": 2},
            },
        ],
    },
]

ENDING_DATA = {
    "tidebound_crown": {
        "title": "Ending: The Tidebound Crown",
        "body": [
            "Because you trusted Sirenscale and kept reaching for understanding instead of easy force, the marsh opens rather than resists. Hidden channels align, dead bells ring in harmony, and the drowned roads rise long enough for the two of you to cross them like royalty returning to a lost city.",
            "At the heart of the flood you find a throne grown from reeds, hooks, and old iron. You do not take it alone. Sirenscale stands beside you, unreadable and shimmering, and together you bind the wild tide to a wiser order.",
            "Travelers later speak of two figures moving through stormlight: one human, one not, both carrying law into a land that had forgotten it. The marsh still hungers, but now it answers to a crown shared rather than stolen.",
        ],
    },
    "iron_covenant": {
        "title": "Ending: The Iron Covenant",
        "body": [
            "You and Sirenscale become something sharper than companionship: a pact forged in blood, rust, and mutually chosen violence. The enemies of the marsh learn quickly that neither of you breaks formation, and every ambush becomes a lesson told by the few who flee.",
            "When the last gate falls, Sirenscale clasps your wrist with the heavy iron hand and names you kin by action, not sentiment. It is the closest thing to devotion either of you trusts.",
            "The world beyond the marsh comes to fear the rumor of your alliance. Not because it is cruel for cruelty's sake, but because it is disciplined, patient, and impossible to intimidate.",
        ],
    },
    "glass_oracle": {
        "title": "Ending: The Glass Oracle",
        "body": [
            "You pursued truth past the point where most travelers would have turned back. In the Pale Observatory, beneath cracked lenses and moon-drowned mirrors, the flood finally yields its pattern to you.",
            "Sirenscale watches as you learn to read storms, memories, and lies the same way: by the distortions they leave behind. Knowledge remakes you before dawn. The marsh no longer feels random. It feels legible.",
            "You leave carrying no throne and no army, only vision. Yet that proves to be enough. Kingdoms far from the flood start changing because you can now see the fractures in them before they break.",
        ],
    },
    "storm_warden": {
        "title": "Ending: The Storm Warden",
        "body": [
            "You walked into weather until weather learned your name. By the time the last flood rises, lightning is no longer a threat to you so much as a language you have forced yourself to understand.",
            "Sirenscale laughs once, short and disbelieving, when the black rain parts around your shoulders. Together you drive through the marsh like a moving front: impossible, electric, untouchable.",
            "People later swear they saw a figure patrolling the causeways with a monster at their side and thunder pacing just overhead. They are almost right. The only mistake is calling Sirenscale a monster.",
        ],
    },
    "mire_throne": {
        "title": "Ending: The Mire Throne",
        "body": [
            "You chose force so often that the marsh eventually stopped testing your restraint and started rewarding your appetite. One gate after another gives way, not because you were worthy, but because you were relentless.",
            "Sirenscale stays with you, though the distance between you never fully closes. By the time you claim the drowned throne, the bond is real but uneasy, built more on usefulness than trust.",
            "You rule, if that is the word, from a palace of wet iron and silence. The roads become safe under your gaze. They also become hard. Mercy survives there only when you deliberately remember to allow it.",
        ],
    },
    "drowned_archive": {
        "title": "Ending: The Drowned Archive",
        "body": [
            "Every relic you carried led to another door, another hidden room, another answer someone had tried to bury. By journey's end, you and Sirenscale descend beneath the flood into a library that should not still exist.",
            "Shelves stretch into black water. Lantern-stars float between them. Somewhere in that impossible archive is the record of how the marsh was made and how it may one day end.",
            "You choose not to leave. Not yet. The world above keeps moving, but below it, you and Sirenscale become keepers of forbidden memory, guarding truths powerful enough to drown empires.",
        ],
    },
    "lantern_road": {
        "title": "Ending: The Lantern Road",
        "body": [
            "You never became a conqueror, prophet, or storm-lord. Instead you chose something harder to brag about and often harder to maintain: balance. Enough mercy to be human. Enough steel to survive.",
            "Sirenscale approves in the only way that matters: by continuing to walk beside you when the danger is over. Together you mark a safe route across the drowned world, hanging lanterns where there had only been fear.",
            "Years later, travelers still speak your names together. Not as rulers, but as the pair who carved a road through despair and proved the flood was not the end of every story.",
        ],
    },
    "quiet_escape": {
        "title": "Ending: The Quiet Escape",
        "body": [
            "You survive. That should not sound small, but in a place like this survival is a stubborn, meaningful victory. You never forced the marsh to reveal its deepest heart, and perhaps that saved you from becoming one more thing buried inside it.",
            "Sirenscale leads you out by roads that seem to appear only for those willing to leave power behind. At dawn the fog thins, the flood loosens its grip, and the world begins to look ordinary again.",
            "You do not return unchanged. Neither does Sirenscale. But the two of you carry the memory of the drowned roads like a private scar: proof that not every worthy ending requires dominion.",
        ],
    },
    "hollow_eclipse": {
        "title": "Ending: The Hollow Eclipse",
        "body": [
            "Somewhere along the path, mistrust and hunger hollowed the center out of things. By the time you reach the final basin, the marsh does not greet you like a traveler anymore. It recognizes you as damage.",
            "Sirenscale stands close, then not close enough, then impossibly far away all at once. Whatever bond might have steadied the journey collapses under the weight of choices neither of you can quite forgive.",
            "When the eclipse comes, it is not in the sky but in the self. You leave the marsh altered, powerful in fragments, but followed forever by the sense that something essential drowned behind you and never surfaced.",
        ],
    },
    "sirenscale_unbound": {
        "title": "Ending: Sirenscale Unbound",
        "body": [
            "You reached the end not by mastering the marsh, but by changing what Sirenscale believed was possible. Trust, risk, hunger, and wonder twisted together until your companion stopped behaving like a creature tied to old scripts.",
            "At the Last Flood, Sirenscale steps into the deepest water and emerges transformed: not tamed, not redeemed, but fully self-chosen. The marsh recoils as if one of its own laws has just resigned.",
            "When the two of you walk back into the world, every road feels slightly less fixed than before. That is terrifying. It is also, perhaps, the first real freedom either of you has seen.",
        ],
    },
}


@dataclass
class StoryState:
    path: str = ""
    stats: Dict[str, int] = field(default_factory=lambda: {
        "trust": 0,
        "insight": 0,
        "storm": 0,
        "feral": 0,
        "relic": 0,
    })
    journal: List[str] = field(default_factory=list)
    ended: bool = False
    ending_key: str = ""

    def depth(self) -> int:
        return len(self.path)


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def pick(seq, seed: int, salt: str):
    if not seq:
        return None
    idx = stable_seed(f"{seed}:{salt}") % len(seq)
    return seq[idx]


def wrap_text(text: str, font: pygame.font.Font, width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if font.size(trial)[0] <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def lerp(a, b, t):
    return a + (b - a) * t


def draw_vertical_gradient(surface, top_color, bottom_color):
    for y in range(surface.get_height()):
        t = y / max(1, surface.get_height() - 1)
        color = (
            int(lerp(top_color[0], bottom_color[0], t)),
            int(lerp(top_color[1], bottom_color[1], t)),
            int(lerp(top_color[2], bottom_color[2], t)),
        )
        pygame.draw.line(surface, color, (0, y), (surface.get_width(), y))


class Button:
    def __init__(self, rect, text, accent=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.accent = accent
        self.hovered = False

    def draw(self, surf):
        bg = (24, 36, 49) if not self.accent else (27, 62, 63)
        if self.hovered:
            bg = (36, 54, 71) if not self.accent else (34, 86, 86)
        pygame.draw.rect(surf, bg, self.rect, border_radius=14)
        pygame.draw.rect(surf, ACCENT if self.accent else PANEL_EDGE, self.rect, 2, border_radius=14)
        lines = wrap_text(self.text, BUTTON_FONT, self.rect.width - 24)
        total_h = len(lines) * BUTTON_FONT.get_linesize()
        y = self.rect.centery - total_h // 2
        for line in lines:
            img = BUTTON_FONT.render(line, True, TEXT)
            surf.blit(img, (self.rect.centerx - img.get_width() // 2, y))
            y += BUTTON_FONT.get_linesize()

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)


class StoryEngine:
    def __init__(self):
        self.state = StoryState()
        self.mode = "menu"
        self.scroll = 0
        self.bg_timer = 0.0
        self.choice_buttons: List[Button] = []
        self.menu_buttons = [
            Button((40, 408, 220, 44), "Start a New Story", accent=True),
            Button((290, 408, 220, 44), "Load Saved Story"),
            Button((540, 408, 220, 44), "Quit"),
        ]
        self.scene = self.generate_scene()
        self.rebuild_buttons()

    def reset(self):
        self.state = StoryState()
        self.mode = "story"
        self.scroll = 0
        self.scene = self.generate_scene()
        self.rebuild_buttons()

    def save_game(self):
        payload = {
            "path": self.state.path,
            "stats": self.state.stats,
            "journal": self.state.journal,
            "ended": self.state.ended,
            "ending_key": self.state.ending_key,
        }
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_game(self):
        if not os.path.exists(SAVE_FILE):
            return False
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.state = StoryState(
            path=payload.get("path", ""),
            stats=payload.get("stats", {
                "trust": 0,
                "insight": 0,
                "storm": 0,
                "feral": 0,
                "relic": 0,
            }),
            journal=payload.get("journal", []),
            ended=payload.get("ended", False),
            ending_key=payload.get("ending_key", ""),
        )
        self.mode = "ending" if self.state.ended else "story"
        self.scroll = 0
        self.scene = self.generate_scene()
        self.rebuild_buttons()
        return True

    def rebuild_buttons(self):
        self.choice_buttons = []
        if self.mode == "story" and not self.state.ended:
            y = 366
            for i, choice in enumerate(self.scene["choices"]):
                self.choice_buttons.append(
                    Button((40, y + i * 50, 720, 44), f"{i + 1}. {choice['label']}", accent=(i == 0))
                )
        elif self.mode == "ending":
            self.choice_buttons = [
                Button((40, 408, 220, 44), "Play Again", accent=True),
                Button((290, 408, 220, 44), "Return to Title"),
                Button((540, 408, 220, 44), "Quit"),
            ]

    def relationship_status(self) -> str:
        trust = self.state.stats["trust"]
        if trust >= 11:
            return "Sirenscale trusts you like chosen family."
        if trust >= 7:
            return "Sirenscale is openly committed to your survival."
        if trust >= 3:
            return "Sirenscale watches you with wary respect."
        if trust >= 0:
            return "Sirenscale stays close, but not unguarded."
        return "Sirenscale follows, yet the bond is fraying."

    def stat_line(self) -> str:
        s = self.state.stats
        return (
            f"Trust {s['trust']}   Insight {s['insight']}   Storm {s['storm']}   "
            f"Feral {s['feral']}   Relics {s['relic']}"
        )

    def choose_ending(self) -> str:
        s = self.state.stats
        if s["trust"] >= 9 and s["insight"] >= 8 and s["relic"] >= 4:
            return "tidebound_crown"
        if s["trust"] >= 9 and s["feral"] >= 8:
            return "iron_covenant"
        if s["insight"] >= 12 and s["trust"] >= 5:
            return "glass_oracle"
        if s["storm"] >= 12:
            return "storm_warden"
        if s["feral"] >= 12 and s["trust"] < 5:
            return "mire_throne"
        if s["relic"] >= 7 and s["insight"] >= 7 and s["trust"] < 6:
            return "drowned_archive"
        if s["trust"] >= 7 and 3 <= s["feral"] <= 8 and 3 <= s["insight"] <= 9:
            return "lantern_road"
        if s["trust"] < 7 and s["feral"] < 6 and s["insight"] < 6 and s["storm"] < 6:
            return "quiet_escape"
        if s["trust"] <= 1 or (s["feral"] >= 8 and s["insight"] >= 8 and s["storm"] >= 8 and s["trust"] < 6):
            return "hollow_eclipse"
        return "sirenscale_unbound"

    def generate_scene(self):
        if self.state.depth() >= MAX_DEPTH or self.state.ended:
            ending_key = self.state.ending_key or self.choose_ending()
            self.state.ended = True
            self.state.ending_key = ending_key
            data = ENDING_DATA[ending_key]
            journal_preview = "\n".join(f"• {entry}" for entry in self.state.journal[-8:]) or "• You stepped into the flood."
            ending_text = "\n\n".join(data["body"]) + "\n\nRoute remembered:\n" + journal_preview
            return {
                "title": data["title"],
                "body": ending_text,
                "choices": [],
            }

        depth = self.state.depth()
        act_name = ACT_NAMES[min(depth, len(ACT_NAMES) - 1)]
        seed = stable_seed(self.state.path or "root")
        region = pick(REGIONS, seed, "region")
        omen = pick(OMENS, seed, "omen")
        tension = pick(TENSIONS, seed, "tension")
        discovery = pick(DISCOVERIES, seed, "discovery")
        siren_line = pick(SIRENSCALE_LINES, seed, "line")
        theme = THEMES[stable_seed(f"{seed}:theme") % len(THEMES)]
        challenge = pick(theme["challenge"], seed, "challenge")

        intro = (
            f"Act {depth + 1}: {act_name}\n\n"
            f"You and Sirenscale reach {region}. {omen}. Ahead, {tension}."
        )

        middle = (
            f"Nearby lies {discovery}. {challenge} "
            f"Sirenscale moves with that irreverent patrol-step of theirs, dried marsh-hair swaying, shark-snake grin half hidden in the mist. {siren_line}"
        )

        summary = (
            f"{self.relationship_status()} The flood remembers every decision. "
            f"At this point in the journey, your path feels shaped by patience, appetite, and the kind of truth you can still bear."
        )

        body = "\n\n".join([intro, middle, summary])
        return {
            "title": f"Sirenscale: The Hundred Paths",
            "body": body,
            "choices": theme["choices"],
        }

    def apply_choice(self, idx: int):
        if idx < 0 or idx >= len(self.scene["choices"]):
            return
        choice = self.scene["choices"][idx]
        for key, value in choice["effects"].items():
            self.state.stats[key] = self.state.stats.get(key, 0) + value
        self.state.path += str(idx)
        act_name = ACT_NAMES[min(self.state.depth() - 1, len(ACT_NAMES) - 1)]
        self.state.journal.append(f"{act_name}: {choice['label']}")
        self.scroll = 0
        if self.state.depth() >= MAX_DEPTH:
            self.state.ending_key = self.choose_ending()
            self.state.ended = True
            self.mode = "ending"
        else:
            self.mode = "story"
        self.scene = self.generate_scene()
        self.rebuild_buttons()

    def update(self, dt: float):
        self.bg_timer += dt

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False

        if self.mode == "menu":
            for button in self.menu_buttons:
                if button.handle_event(event):
                    if button.text == "Start a New Story":
                        self.reset()
                    elif button.text == "Load Saved Story":
                        if not self.load_game():
                            pass
                    else:
                        return False
            return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.mode in ("story", "ending"):
                    self.mode = "menu"
                    self.rebuild_buttons()
                    return True
                return False
            if event.key == pygame.K_s and self.mode in ("story", "ending"):
                self.save_game()
            if event.key == pygame.K_l and self.mode == "menu":
                self.load_game()
            if event.key == pygame.K_r:
                self.reset()
            if event.key == pygame.K_UP:
                self.scroll = max(0, self.scroll - 30)
            if event.key == pygame.K_DOWN:
                self.scroll += 30
            if self.mode == "story":
                if event.key == pygame.K_1:
                    self.apply_choice(0)
                elif event.key == pygame.K_2:
                    self.apply_choice(1)
            elif self.mode == "ending":
                if event.key == pygame.K_RETURN:
                    self.reset()

        if event.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, self.scroll - event.y * 40)

        if self.mode == "story":
            for i, button in enumerate(self.choice_buttons):
                if button.handle_event(event):
                    self.apply_choice(i)
                    break
        elif self.mode == "ending":
            for i, button in enumerate(self.choice_buttons):
                if button.handle_event(event):
                    if i == 0:
                        self.reset()
                    elif i == 1:
                        self.mode = "menu"
                        self.rebuild_buttons()
                    else:
                        return False
        return True

    def draw_particles(self, surf):
        for i in range(36):
            t = self.bg_timer * (0.15 + i * 0.01)
            x = (i * 97 + math.sin(t + i) * 120 + 180 * math.sin(i * 0.3)) % WIDTH
            y = (i * 53 + t * 40) % HEIGHT
            r = 1 + (i % 3)
            color = (90 + i * 2 % 120, 140 + i * 3 % 100, 150 + i * 2 % 80)
            pygame.draw.circle(surf, color, (int(x), int(y)), r)

    def draw_header(self, surf, subtitle):
        panel = pygame.Rect(30, 16, WIDTH - 60, 72)
        pygame.draw.rect(surf, PANEL, panel, border_radius=18)
        pygame.draw.rect(surf, PANEL_EDGE, panel, 2, border_radius=18)
        title_img = TITLE_FONT.render("Sirenscale: The Hundred Paths", True, TEXT)
        subtitle_img = SUBTITLE_FONT.render(subtitle, True, MUTED)
        surf.blit(title_img, (panel.x + 18, panel.y + 10))
        surf.blit(subtitle_img, (panel.x + 20, panel.y + 42))

    def draw_story(self, surf):
        self.draw_header(surf, f"256 full routes • roughly 10 endings • S to save • Esc for title")

        story_panel = pygame.Rect(30, 100, WIDTH - 60, 210)
        pygame.draw.rect(surf, PANEL, story_panel, border_radius=20)
        pygame.draw.rect(surf, PANEL_EDGE, story_panel, 2, border_radius=20)

        stat_panel = pygame.Rect(30, 316, WIDTH - 60, 38)
        pygame.draw.rect(surf, PANEL_ALT, stat_panel, border_radius=16)
        pygame.draw.rect(surf, PANEL_EDGE, stat_panel, 2, border_radius=16)
        stat_img = SMALL_FONT.render(self.stat_line(), True, ACCENT)
        surf.blit(stat_img, (stat_panel.x + 12, stat_panel.y + 11))
        act_img = SMALL_FONT.render(f"Depth {self.state.depth()} / {MAX_DEPTH}", True, MUTED)
        surf.blit(act_img, (stat_panel.right - act_img.get_width() - 12, stat_panel.y + 11))

        clip = surf.get_clip()
        surf.set_clip(story_panel.inflate(-30, -24))
        wrapped = wrap_text(self.scene["body"], BODY_FONT, story_panel.width - 60)
        y = story_panel.y + 24 - self.scroll
        for line in wrapped:
            color = TEXT if line else TEXT
            img = BODY_FONT.render(line, True, color)
            surf.blit(img, (story_panel.x + 24, y))
            y += BODY_FONT.get_linesize() + (8 if line == "" else 0)
        surf.set_clip(clip)

        content_height = max(0, y - (story_panel.y + 24 - self.scroll))
        visible_h = story_panel.height - 48
        max_scroll = max(0, content_height - visible_h)
        self.scroll = max(0, min(self.scroll, max_scroll))
        if max_scroll > 0:
            bar_h = max(40, int(visible_h * visible_h / max(content_height, visible_h + 1)))
            bar_y = story_panel.y + 24 + int((visible_h - bar_h) * (self.scroll / max_scroll))
            pygame.draw.rect(surf, (28, 47, 54), (story_panel.right - 16, story_panel.y + 24, 8, visible_h), border_radius=4)
            pygame.draw.rect(surf, ACCENT_2, (story_panel.right - 16, bar_y, 8, bar_h), border_radius=4)

        for button in self.choice_buttons:
            button.draw(surf)

    def draw_menu(self, surf):
        self.draw_header(surf, "A branching text adventure in pygame starring Sirenscale as your companion")
        panel = pygame.Rect(30, 100, WIDTH - 60, 280)
        pygame.draw.rect(surf, PANEL, panel, border_radius=20)
        pygame.draw.rect(surf, PANEL_EDGE, panel, 2, border_radius=20)

        blurb = (
            "Cross the drowned world with Sirenscale: a shimmering, iron-armed wanderer with crocodile lids, "
            "squid-skin iridescence, dried marsh-hair, and a smile full of shark and snake.\n\n"
            "Every scene offers a real branch. Your choices change trust, insight, storm, feral instinct, and relic gain. "
            "After eight major acts, those choices push the story into one of roughly ten endings.\n\n"
            "Controls:\n"
            "• Mouse or 1/2 to choose\n"
            "• Mouse wheel / Up / Down to scroll\n"
            "• S to save during a run\n"
            "• L to load from the title\n"
            "• R to restart anytime\n"
            "• Esc returns to the title screen"
        )
        wrapped = wrap_text(blurb, BODY_FONT, panel.width - 60)
        y = panel.y + 30
        for line in wrapped:
            img = BODY_FONT.render(line, True, TEXT)
            surf.blit(img, (panel.x + 24, y))
            y += BODY_FONT.get_linesize() + (8 if line == "" else 0)

        if os.path.exists(SAVE_FILE):
            tip = SMALL_FONT.render("Save file detected: sirenscale_save.json", True, GOOD)
            surf.blit(tip, (panel.x + 18, panel.bottom - 26))
        else:
            tip = SMALL_FONT.render("No save file found yet.", True, MUTED)
            surf.blit(tip, (panel.x + 18, panel.bottom - 26))

        for button in self.menu_buttons:
            button.draw(surf)

    def draw_ending(self, surf):
        self.draw_header(surf, "Ending reached • Enter or click Play Again to start over")
        panel = pygame.Rect(30, 100, WIDTH - 60, 280)
        pygame.draw.rect(surf, PANEL, panel, border_radius=20)
        pygame.draw.rect(surf, PANEL_EDGE, panel, 2, border_radius=20)

        title_img = SUBTITLE_FONT.render(self.scene["title"], True, WARNING)
        surf.blit(title_img, (panel.x + 18, panel.y + 16))
        stat_img = SMALL_FONT.render(self.stat_line(), True, ACCENT)
        surf.blit(stat_img, (panel.right - stat_img.get_width() - 18, panel.y + 18))

        wrapped = wrap_text(self.scene["body"], BODY_FONT, panel.width - 60)
        clip = surf.get_clip()
        text_rect = pygame.Rect(panel.x + 18, panel.y + 52, panel.width - 36, panel.height - 70)
        surf.set_clip(text_rect)
        y = text_rect.y - self.scroll
        for line in wrapped:
            img = BODY_FONT.render(line, True, TEXT)
            surf.blit(img, (text_rect.x, y))
            y += BODY_FONT.get_linesize() + (8 if line == "" else 0)
        surf.set_clip(clip)

        content_height = max(0, y - (text_rect.y - self.scroll))
        visible_h = text_rect.height
        max_scroll = max(0, content_height - visible_h)
        self.scroll = max(0, min(self.scroll, max_scroll))
        if max_scroll > 0:
            bar_h = max(40, int(visible_h * visible_h / max(content_height, visible_h + 1)))
            bar_y = text_rect.y + int((visible_h - bar_h) * (self.scroll / max_scroll))
            pygame.draw.rect(surf, (28, 47, 54), (panel.right - 16, text_rect.y, 8, visible_h), border_radius=4)
            pygame.draw.rect(surf, ACCENT_2, (panel.right - 16, bar_y, 8, bar_h), border_radius=4)

        for button in self.choice_buttons:
            button.draw(surf)

    def draw(self, surf):
        draw_vertical_gradient(surf, BG_TOP, BG_BOTTOM)
        self.draw_particles(surf)

        if self.mode == "menu":
            self.draw_menu(surf)
        elif self.mode == "ending":
            self.draw_ending(surf)
        else:
            self.draw_story(surf)

        pygame.draw.rect(surf, (255, 255, 255, 10), (0, 0, WIDTH, HEIGHT), 1)


def main():
    engine = StoryEngine()
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            running = engine.handle_event(event)
            if not running:
                break
        engine.update(dt)
        engine.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
