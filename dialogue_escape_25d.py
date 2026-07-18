#!/usr/bin/env python3
"""
The Third Key

A no-combat 2.5D dialogue RPG made with Pygame.

You are locked in one room with Mara, a locksmith, and Ivo, an archivist.
The exits are all possible, but none of them are simple. Talk, investigate,
trade trust for information, and decide who gets out.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pygame


WIDTH, HEIGHT = 960, 600
FPS = 60

ROOM_W = 18.0
ROOM_D = 12.0
ISO_X = 32
ISO_Y = 17
ORIGIN_X = WIDTH // 2
ORIGIN_Y = 128
WALL_H = 118

PLAYER_SPEED = 4.2
INTERACT_RANGE = 1.75

BG = (14, 16, 22)
TEXT = (234, 238, 232)
MUTED = (153, 160, 158)
PANEL = (20, 24, 30)
PANEL_EDGE = (76, 87, 98)
GOLD = (231, 180, 76)
GREEN = (96, 184, 128)
BLUE = (88, 154, 218)
RED = (205, 94, 89)
VIOLET = (154, 112, 210)
SHADOW = (3, 4, 7)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def wrap_text(font: pygame.font.Font, text: str, width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            test = f"{line} {word}"
            if font.size(test)[0] <= width:
                line = test
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_wrapped(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: Tuple[int, int, int],
    rect: pygame.Rect,
    line_gap: int = 4,
) -> int:
    y = rect.y
    line_h = font.get_linesize() + line_gap
    for line in wrap_text(font, text, rect.w):
        if y + line_h > rect.bottom:
            break
        surface.blit(font.render(line, True, color), (rect.x, y))
        y += line_h
    return y


def polygon(surface: pygame.Surface, color: Tuple[int, int, int], points: Sequence[Tuple[float, float]]) -> None:
    pygame.draw.polygon(surface, color, [(int(x), int(y)) for x, y in points])


@dataclass
class Actor:
    key: str
    name: str
    x: float
    z: float
    color: Tuple[int, int, int]
    trim: Tuple[int, int, int]
    facing: int = 1


@dataclass
class Hotspot:
    key: str
    name: str
    x: float
    z: float
    radius: float


@dataclass
class MenuOption:
    label: str
    action: Callable[[], None]
    enabled: bool = True
    hint: str = ""


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("The Third Key - no combat dialogue RPG")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        self.tiny = pygame.font.SysFont("consolas", 12)
        self.bold = pygame.font.SysFont("consolas", 20, bold=True)
        self.big = pygame.font.SysFont("consolas", 42, bold=True)
        self.reset_game()

    def reset_game(self) -> None:
        self.player = Actor("player", "You", 9.0, 9.2, (226, 214, 139), (255, 246, 171))
        self.npcs: Dict[str, Actor] = {
            "mara": Actor("mara", "Mara", 5.0, 6.7, (78, 151, 176), (198, 232, 240)),
            "ivo": Actor("ivo", "Ivo", 12.5, 6.4, (136, 103, 181), (222, 205, 250), -1),
        }
        self.hotspots = [
            Hotspot("door", "Sealed Door", 17.25, 5.9, 1.85),
            Hotspot("console", "Wall Console", 14.8, 2.45, 1.45),
            Hotspot("vent", "High Vent", 6.1, 0.8, 1.45),
            Hotspot("mirror", "Black Mirror", 1.1, 3.8, 1.45),
            Hotspot("grate", "Drain Grate", 4.0, 9.2, 1.25),
            Hotspot("table", "Scratched Table", 9.5, 7.0, 1.6),
        ]
        self.items: set[str] = set()
        self.flags: set[str] = set()
        self.trust = {"mara": 0, "ivo": 0}
        self.mode = "explore"
        self.panel_title = ""
        self.panel_body = ""
        self.options: List[MenuOption] = []
        self.current_npc: Optional[str] = None
        self.notice = "The lock clicks behind you. Two strangers look up."
        self.notice_timer = 4.0
        self.ending_title = ""
        self.ending_body = ""

    def run(self) -> None:
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self.handle_events()
            self.update(dt)
            self.render()
            pygame.display.flip()

    def iso(self, x: float, z: float, height: float = 0.0) -> Tuple[int, int]:
        sx = ORIGIN_X + (x - z) * ISO_X
        sy = ORIGIN_Y + (x + z) * ISO_Y - height
        return int(sx), int(sy)

    def has(self, item: str) -> bool:
        return item in self.items

    def flag(self, name: str) -> bool:
        return name in self.flags

    def add_item(self, item: str) -> None:
        if item not in self.items:
            self.items.add(item)
            self.toast(f"Added: {item}")

    def add_flag(self, flag: str) -> None:
        self.flags.add(flag)

    def add_trust(self, npc: str, amount: int) -> None:
        self.trust[npc] = clamp(self.trust[npc] + amount, -3, 5)

    def toast(self, text: str, seconds: float = 3.0) -> None:
        self.notice = text
        self.notice_timer = seconds

    def opt(
        self,
        label: str,
        action: Callable[[], None],
        enabled: bool = True,
        hint: str = "",
    ) -> MenuOption:
        return MenuOption(label, action, enabled, hint)

    def set_panel(self, title: str, body: str, options: Sequence[MenuOption], mode: str = "menu") -> None:
        self.mode = mode
        self.panel_title = title
        self.panel_body = body
        self.options = list(options)[:9]

    def reply(self, title: str, body: str, back: Callable[[], None]) -> None:
        self.set_panel(title, body, [self.opt("Continue", back)], self.mode)

    def close_panel(self) -> None:
        self.mode = "explore"
        self.options = []
        self.current_npc = None

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                if self.mode == "ending":
                    pygame.quit()
                    sys.exit()
                if self.mode in {"menu", "dialogue"}:
                    self.close_panel()
                else:
                    pygame.quit()
                    sys.exit()
                continue

            if self.mode == "ending":
                if event.key == pygame.K_r:
                    self.reset_game()
                continue

            if self.mode in {"menu", "dialogue"}:
                if pygame.K_1 <= event.key <= pygame.K_9:
                    index = event.key - pygame.K_1
                    self.choose_option(index)
                elif event.key in {pygame.K_RETURN, pygame.K_SPACE} and len(self.options) == 1:
                    self.choose_option(0)
                continue

            if event.key == pygame.K_e:
                self.interact()
            elif event.key == pygame.K_TAB:
                self.show_status()

    def choose_option(self, index: int) -> None:
        if not 0 <= index < len(self.options):
            return
        option = self.options[index]
        if option.enabled:
            option.action()
        else:
            self.toast(option.hint or "That is not available yet.")

    def update(self, dt: float) -> None:
        if self.notice_timer > 0:
            self.notice_timer -= dt

        if self.mode != "explore":
            return

        keys = pygame.key.get_pressed()
        dx = 0.0
        dz = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += 1.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dz -= 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dz += 1.0

        if dx or dz:
            mag = math.hypot(dx, dz)
            dx /= mag
            dz /= mag
            self.player.x = clamp(self.player.x + dx * PLAYER_SPEED * dt, 1.0, ROOM_W - 1.0)
            self.player.z = clamp(self.player.z + dz * PLAYER_SPEED * dt, 1.0, ROOM_D - 1.0)
            if abs(dx) > 0.05:
                self.player.facing = 1 if dx > 0 else -1

    def nearest_interactable(self) -> Optional[Tuple[str, str, float]]:
        pxz = (self.player.x, self.player.z)
        best: Optional[Tuple[str, str, float]] = None
        for key, npc in self.npcs.items():
            d = dist(pxz, (npc.x, npc.z))
            if d <= INTERACT_RANGE and (best is None or d < best[2]):
                best = ("npc", key, d)
        for spot in self.hotspots:
            d = dist(pxz, (spot.x, spot.z))
            if d <= spot.radius and (best is None or d < best[2]):
                best = ("hotspot", spot.key, d)
        return best

    def interact(self) -> None:
        nearest = self.nearest_interactable()
        if nearest is None:
            self.toast("Nothing is close enough to use.")
            return
        kind, key, _ = nearest
        if kind == "npc":
            self.talk_to(key)
        else:
            getattr(self, f"show_{key}")()

    # Dialogue: Mara

    def talk_to(self, npc: str) -> None:
        self.current_npc = npc
        if npc == "mara":
            self.show_mara_root()
        else:
            self.show_ivo_root()

    def show_mara_root(self) -> None:
        trust = self.trust["mara"]
        mood = "guarded"
        if trust >= 4:
            mood = "ready to bet her life on the plan"
        elif trust >= 2:
            mood = "focused and almost trusting"
        elif trust < 0:
            mood = "watching you like another lock"

        body = (
            f"Mara crouches beside the console with a brass hairpin between her teeth. "
            f"She is {mood}. Her eyes keep returning to the door seam."
        )
        options = [
            self.opt("Who are you?", self.mara_who),
            self.opt("Read the sealed door with her.", self.mara_door),
            self.opt("Ask for tools or a bypass.", self.mara_tools),
            self.opt("Ask what she is hiding.", self.mara_secret),
            self.opt("Talk about Ivo.", self.mara_ivo),
            self.opt("Promise nobody gets left.", self.mara_promise),
            self.opt("Suggest the vent if only one person fits.", self.mara_vent),
            self.opt("Show her something from your inventory.", self.mara_show_items, bool(self.items), "Your inventory is empty."),
            self.opt("Step away.", self.close_panel),
        ]
        self.set_panel("Mara Vale", body, options, "dialogue")

    def mara_who(self) -> None:
        if not self.flag("mara_intro"):
            self.add_flag("mara_intro")
            self.add_trust("mara", 1)
            body = (
                "'Mara Vale. Doorwright, locksmith, occasional criminal when the lock deserves it.' "
                "She taps the console. 'This room was built to judge intentions, not strength.'"
            )
        else:
            body = (
                "'Still Mara. Still annoyed. Still alive, which means the room has not finished "
                "asking its question.'"
            )
        self.reply("Mara", body, self.show_mara_root)

    def mara_door(self) -> None:
        self.add_flag("mara_door_theory")
        body = (
            "Mara points at three metal plates around the sealed door. 'Power, phrase, witness. "
            "Power comes from the console. The phrase is probably Ivo's department. Witness means "
            "the door wants to know who you are willing to leave behind.'"
        )
        self.reply("Mara", body, self.show_mara_root)

    def mara_tools(self) -> None:
        if not self.has("hairpin") and self.trust["mara"] >= 1:
            self.add_item("hairpin")
            body = (
                "Mara studies your hands, then passes you her brass hairpin. 'Do not snap it. "
                "Thin tool, wide patience. It can open the console panel or pry a vent screw.'"
            )
        elif not self.has("hairpin"):
            body = (
                "'Tools go to people with plans,' Mara says. 'Convince me you have one, and I will "
                "share.'"
            )
        elif not self.flag("wire_prepped") and self.has("copper wire"):
            self.add_flag("wire_prepped")
            self.add_trust("mara", 1)
            body = (
                "She twists the copper wire into a neat bypass lead. 'There. That should carry "
                "enough current to wake the door without cooking us.'"
            )
        else:
            body = (
                "'The hairpin is the delicate tool. The wire is the bridge. The console is the liar. "
                "Make those three agree and the door will listen.'"
            )
        self.reply("Mara", body, self.show_mara_root)

    def mara_secret(self) -> None:
        if self.trust["mara"] >= 2 and not self.has("red keycard"):
            self.add_item("red keycard")
            self.add_flag("mara_confessed")
            self.add_trust("mara", 1)
            body = (
                "Mara exhales and pulls a cracked red keycard from her boot. 'I tried to leave "
                "before either of you woke up. The room punished the card, not me. Maybe it still "
                "counts as a false witness.'"
            )
        elif self.has("red keycard"):
            body = (
                "'The card is not a key anymore,' she says. 'It is evidence. Some locks hate evidence.'"
            )
        else:
            self.add_trust("mara", -1)
            body = (
                "'I am hiding the same thing you are,' Mara says. 'A first thought I am not proud of.'"
            )
        self.reply("Mara", body, self.show_mara_root)

    def mara_ivo(self) -> None:
        self.add_flag("mara_knows_ivo")
        if not self.flag("mara_ivo_talk"):
            self.add_flag("mara_ivo_talk")
            self.add_trust("mara", 1)
        body = (
            "'Ivo reads rooms the way I read locks. He also thinks every exit charges a moral fee.' "
            "She glances toward him. 'He may be right here. Be careful what you promise.'"
        )
        self.reply("Mara", body, self.show_mara_root)

    def mara_promise(self) -> None:
        if not self.flag("mara_promised"):
            self.add_flag("mara_promised")
            self.add_trust("mara", 1)
        if self.flag("door_powered") and self.flag("passphrase_known"):
            self.add_flag("mara_ready")
            body = (
                "Mara stands and pockets her fear with visible effort. 'Power and phrase. Good. "
                "When the door opens, I move on your word. Do not make me regret hearing it.'"
            )
        else:
            body = (
                "'Then make it practical,' she says. 'Find power for the door and get the phrase "
                "from Ivo. Promises are easier to believe when they have parts.'"
            )
        self.reply("Mara", body, self.show_mara_root)

    def mara_vent(self) -> None:
        self.add_flag("mara_will_take_vent")
        if self.trust["mara"] >= 1:
            body = (
                "'I can fit,' Mara says after measuring it with her eyes. 'Maybe I can even unjam "
                "something from the other side. But if I go first, the duct will collapse behind me.'"
            )
        else:
            self.add_trust("mara", -1)
            body = (
                "'So that is your shape,' Mara says. 'Find a hole and put someone else into it.'"
            )
        self.reply("Mara", body, self.show_mara_root)

    def mara_show_items(self) -> None:
        options: List[MenuOption] = []
        if self.has("copper wire"):
            options.append(self.opt("Show the copper wire.", self.mara_show_wire))
        if self.has("vent handle"):
            options.append(self.opt("Show the vent handle.", self.mara_show_handle))
        if self.has("mirror shard"):
            options.append(self.opt("Show the mirror shard.", self.mara_show_shard))
        if self.has("red keycard"):
            options.append(self.opt("Show the red keycard.", self.mara_show_card))
        if not options:
            options.append(self.opt("Nothing useful right now.", self.show_mara_root))
        options.append(self.opt("Back.", self.show_mara_root))
        self.set_panel("Mara - inventory", "Mara holds out a grease-stained palm.", options, "dialogue")

    def mara_show_wire(self) -> None:
        self.add_flag("wire_prepped")
        self.add_trust("mara", 1)
        body = (
            "She strips the wire with her teeth and folds it into a cleaner lead. 'Now it can carry "
            "signal instead of just hope.'"
        )
        self.reply("Mara", body, self.show_mara_root)

    def mara_show_handle(self) -> None:
        self.add_flag("vent_known")
        body = (
            "'That fits the vent screws,' Mara says. 'The bad news is the duct is a one-person exit. "
            "Old collapse seal. It closes after a body passes.'"
        )
        self.reply("Mara", body, self.show_mara_root)

    def mara_show_shard(self) -> None:
        self.add_flag("mara_mirror_warning")
        body = (
            "Mara refuses to touch the shard. 'Mirrors are locks that do not admit they have hinges. "
            "Ask Ivo before you bleed a decision into that thing.'"
        )
        self.reply("Mara", body, self.show_mara_root)

    def mara_show_card(self) -> None:
        body = (
            "'It failed because I used it alone,' Mara says. 'With the door powered, it might spoof "
            "the witness plate. That could save us, or make the room count very strangely.'"
        )
        self.add_flag("card_hint")
        self.reply("Mara", body, self.show_mara_root)

    # Dialogue: Ivo

    def show_ivo_root(self) -> None:
        trust = self.trust["ivo"]
        mood = "folded into his oversized coat"
        if trust >= 4:
            mood = "scared but steady"
        elif trust >= 2:
            mood = "watching you with cautious hope"
        elif trust < 0:
            mood = "keeping the table between you and him"

        body = (
            f"Ivo Sen stands near the black mirror, {mood}. Chalk dust marks his sleeves. "
            "He looks like he has already read the last page and disliked it."
        )
        options = [
            self.opt("Who are you?", self.ivo_who),
            self.opt("Ask about the mirror.", self.ivo_mirror),
            self.opt("Ask him to decode the wall scratches.", self.ivo_decode),
            self.opt("Ask for chalk.", self.ivo_chalk),
            self.opt("Promise you will not abandon anyone.", self.ivo_promise),
            self.opt("Admit you may leave alone if you must.", self.ivo_hard_truth),
            self.opt("Ask what Mara is not saying.", self.ivo_mara),
            self.opt("Invite him into the door plan.", self.ivo_ready),
            self.opt("Step away.", self.close_panel),
        ]
        self.set_panel("Ivo Sen", body, options, "dialogue")

    def ivo_who(self) -> None:
        if not self.flag("ivo_intro"):
            self.add_flag("ivo_intro")
            self.add_trust("ivo", 1)
            body = (
                "'Ivo Sen. Archivist, translator, coward under pressure, depending on who writes "
                "the report.' He gives a small bow. 'This room is written in bargains.'"
            )
        else:
            body = (
                "'Still Ivo,' he says. 'Still hoping the room accepts footnotes.'"
            )
        self.reply("Ivo", body, self.show_ivo_root)

    def ivo_mirror(self) -> None:
        self.add_flag("ivo_mirror_lesson")
        if not self.flag("ivo_mirror_first"):
            self.add_flag("ivo_mirror_first")
            self.add_trust("ivo", 1)
        body = (
            "'The mirror is an exit for meanings, not bodies,' Ivo says. 'But sometimes bodies "
            "are mostly meanings. It needs chalk, a shard, and a phrase spoken with intent.'"
        )
        self.reply("Ivo", body, self.show_ivo_root)

    def ivo_decode(self) -> None:
        if self.flag("passphrase_known"):
            body = (
                "'The phrase remains the same,' Ivo says. 'The third key is witness. I wish it "
                "sounded less hungry.'"
            )
        elif self.trust["ivo"] >= 1 or self.flag("scratches_seen"):
            self.add_flag("passphrase_known")
            self.add_trust("ivo", 1)
            body = (
                "Ivo traces the scratches in the air. 'It is not a riddle, it is a contract line: "
                "the third key is witness. Say it at the console after the door has power.'"
            )
        else:
            body = (
                "'I can decode it,' Ivo says, 'but I need to know you are not just collecting ways "
                "to abandon us.'"
            )
        self.reply("Ivo", body, self.show_ivo_root)

    def ivo_chalk(self) -> None:
        if self.has("chalk"):
            body = "'You already have my chalk,' he says. 'Use a closed circle, not a line.'"
        elif self.trust["ivo"] >= 1 or self.flag("ivo_mirror_lesson"):
            self.add_item("chalk")
            body = (
                "Ivo breaks his chalk in half and gives you the longer piece. 'For circles. "
                "Never draw a door unless you know who is expected to answer.'"
            )
        else:
            body = (
                "Ivo grips the chalk tighter. 'Not yet. Chalk is how bad ideas become architecture.'"
            )
        self.reply("Ivo", body, self.show_ivo_root)

    def ivo_promise(self) -> None:
        if not self.flag("ivo_promised"):
            self.add_flag("ivo_promised")
            self.add_trust("ivo", 1)
        if self.flag("door_powered") and self.flag("passphrase_known"):
            self.add_flag("ivo_ready")
            body = (
                "Ivo nods once. 'Then I will stand at the witness plate. If the room asks what we "
                "are, I will answer: temporary allies.'"
            )
        else:
            body = (
                "'Then prove it in sequence,' Ivo says. 'Power, phrase, witness. The room respects "
                "order more than bravery.'"
            )
        self.reply("Ivo", body, self.show_ivo_root)

    def ivo_hard_truth(self) -> None:
        self.add_flag("selfish_truth")
        if not self.flag("ivo_heard_hard_truth"):
            self.add_flag("ivo_heard_hard_truth")
            self.add_trust("ivo", -1)
        body = (
            "Ivo looks hurt, then oddly relieved. 'A cruel truth is still a truth. The door may "
            "like you more for saying it. I do not.'"
        )
        self.reply("Ivo", body, self.show_ivo_root)

    def ivo_mara(self) -> None:
        self.add_flag("ivo_knows_mara")
        body = (
            "'Mara tried the red card before we woke,' Ivo whispers. 'I do not blame her. I do "
            "blame the room for remembering.'"
        )
        if self.trust["ivo"] >= 1 and not self.has("red keycard") and self.flag("mara_confessed"):
            body += " He adds, 'If she gave it to you, treat it like testimony, not a tool.'"
        self.reply("Ivo", body, self.show_ivo_root)

    def ivo_ready(self) -> None:
        if self.trust["ivo"] >= 2 or self.flag("ivo_promised"):
            self.add_flag("ivo_ready")
            body = (
                "Ivo presses his hands flat to stop them shaking. 'I will come. Or witness. Or "
                "whatever the door demands, as long as you do not make the demand alone.'"
            )
        else:
            body = (
                "'Not yet,' he says. 'I need a reason to believe your plan includes more than your "
                "own silhouette leaving through a gap.'"
            )
        self.reply("Ivo", body, self.show_ivo_root)

    # Hotspots

    def show_table(self) -> None:
        body = (
            "The table is bolted to the floor. Its top is carved with tally marks, partial phrases, "
            "and one brass handle hidden under a loose strip of tape."
        )
        options = [
            self.opt("Take the brass vent handle.", self.table_take_handle, not self.has("vent handle"), "You already took it."),
            self.opt("Read the scratches.", self.table_read_scratches),
            self.opt("Check underneath.", self.table_underneath),
            self.opt("Back.", self.close_panel),
        ]
        self.set_panel("Scratched Table", body, options)

    def table_take_handle(self) -> None:
        self.add_item("vent handle")
        self.add_flag("vent_known")
        self.reply("Scratched Table", "The handle comes free with a quiet snap. Its square tip matches the high vent screws.", self.show_table)

    def table_read_scratches(self) -> None:
        self.add_flag("scratches_seen")
        body = (
            "Most scratches are failed counts. One line repeats in three hands: 'ONE GOES NOWHERE. "
            "TWO GO APART. THREE GO THROUGH.' Ivo may be able to decode the rest."
        )
        self.reply("Scratched Table", body, self.show_table)

    def table_underneath(self) -> None:
        if not self.flag("under_table_note"):
            self.add_flag("under_table_note")
            body = (
                "A torn note is wedged under the table lip: 'The room accepts sacrifice, fraud, "
                "consensus, or poetry. It cannot tell which is worst.'"
            )
        else:
            body = "Only old gum, bolt heads, and the memory of the note remain under the table."
        self.reply("Scratched Table", body, self.show_table)

    def show_grate(self) -> None:
        body = "A drain grate hums with cold air. Something copper glints below the mesh."
        options = [
            self.opt("Hook out the copper wire.", self.grate_wire, not self.has("copper wire"), "The grate is already empty."),
            self.opt("Listen at the drain.", self.grate_listen),
            self.opt("Back.", self.close_panel),
        ]
        self.set_panel("Drain Grate", body, options)

    def grate_wire(self) -> None:
        self.add_item("copper wire")
        body = "You fish out a coil of copper wire. It smells like ozone and old rain."
        self.reply("Drain Grate", body, self.show_grate)

    def grate_listen(self) -> None:
        self.add_flag("vent_known")
        body = (
            "Air whispers through the drain in pulses. It is connected to the vent, but not wide "
            "enough for a person. The room has many ways to move air and very few to move mercy."
        )
        self.reply("Drain Grate", body, self.show_grate)

    def show_console(self) -> None:
        power = "awake" if self.flag("door_powered") else "dark"
        phrase = "entered" if self.flag("door_unlocked") else "waiting"
        body = (
            f"The console is {power}. Its phrase line is {phrase}. Three sockets are labeled "
            "POWER, PHRASE, WITNESS."
        )
        options = [
            self.opt("Open the service panel.", self.console_open_panel, not self.flag("console_panel_open"), "The panel is already open."),
            self.opt("Reroute power to the door.", self.console_power, not self.flag("door_powered"), "The door already has power."),
            self.opt("Enter the passphrase.", self.console_phrase, not self.flag("door_unlocked"), "The phrase has already been accepted."),
            self.opt("Swipe the red keycard as a witness.", self.console_card, self.has("red keycard"), "You do not have the red keycard."),
            self.opt("Back.", self.close_panel),
        ]
        self.set_panel("Wall Console", body, options)

    def console_open_panel(self) -> None:
        if self.has("hairpin") or self.has("copper wire") or self.has("red keycard"):
            self.add_flag("console_panel_open")
            body = (
                "The panel pops open. Inside, three leads have been deliberately crossed so the "
                "door can hear the room but not the people in it."
            )
        else:
            body = "The service panel needs something thin: a hairpin, a wire, or a piece of reckless luck."
        self.reply("Wall Console", body, self.show_console)

    def console_power(self) -> None:
        if not self.flag("console_panel_open"):
            body = "The panel is still sealed. You need to open it before you can reroute anything."
        elif self.has("copper wire") or self.flag("wire_prepped") or self.trust["mara"] >= 2:
            self.add_flag("door_powered")
            body = (
                "You bridge the power lead. The sealed door answers with a low, patient chime. "
                "Mara gives a tiny, involuntary smile."
            )
        else:
            body = "The panel is open, but you need conductive wire or Mara's practiced hands to make the bridge."
        self.reply("Wall Console", body, self.show_console)

    def console_phrase(self) -> None:
        if not self.flag("door_powered"):
            body = "The phrase field is blank. The door needs power before it can listen."
        elif not self.flag("passphrase_known"):
            body = "You do not know the phrase. The scratches on the table and Ivo's memory may help."
        else:
            self.add_flag("door_unlocked")
            body = (
                "You enter: THE THIRD KEY IS WITNESS. The console accepts every word. The door seam "
                "fills with warm light."
            )
        self.reply("Wall Console", body, self.show_console)

    def console_card(self) -> None:
        if not self.flag("door_powered"):
            body = "The red card refuses to wake while the console has no power."
        else:
            self.add_flag("card_witness")
            body = (
                "The console reads the cracked red keycard as a fourth, false witness. Somewhere "
                "inside the door, arithmetic becomes suspicious."
            )
        self.reply("Wall Console", body, self.show_console)

    def show_door(self) -> None:
        if self.flag("door_unlocked"):
            body = (
                "The sealed door is open by an inch. It will widen for a decision, not for hesitation. "
                "The witness plate glows like a held breath."
            )
        else:
            body = (
                "The sealed door has no handle, only three plates: POWER, PHRASE, WITNESS. It looks "
                "less locked than unconvinced."
            )
        options = [
            self.opt("Inspect the witness plate.", self.door_inspect),
            self.opt("Push the door.", self.door_push),
            self.opt("Leave alone through the main door.", self.end_door_solo, self.flag("door_unlocked"), "The door is not unlocked."),
            self.opt("Leave with Mara.", self.end_door_mara, self.flag("door_unlocked") and (self.flag("mara_ready") or self.trust["mara"] >= 2), "Mara is not ready to trust this."),
            self.opt("Leave with Ivo.", self.end_door_ivo, self.flag("door_unlocked") and (self.flag("ivo_ready") or self.trust["ivo"] >= 2), "Ivo is not ready to trust this."),
            self.opt("Call all three through.", self.end_door_all, self.flag("door_unlocked") and self.flag("mara_ready") and self.flag("ivo_ready"), "Both NPCs need to be ready."),
            self.opt("Let the false witness decide.", self.end_false_witness, self.flag("door_unlocked") and self.flag("card_witness"), "The red keycard has not been counted."),
            self.opt("Back.", self.close_panel),
        ]
        self.set_panel("Sealed Door", body, options)

    def door_inspect(self) -> None:
        self.add_flag("mara_door_theory")
        body = (
            "The witness plate is warm. It is not testing whether someone stays behind. It is "
            "testing whether the choice is named honestly."
        )
        self.reply("Sealed Door", body, self.show_door)

    def door_push(self) -> None:
        if self.flag("door_unlocked"):
            body = "The door gives an inch, then waits for you to choose who crosses the threshold."
        elif not self.flag("door_powered"):
            body = "The door remains cold. The console has not powered it."
        elif not self.flag("passphrase_known"):
            body = "The door hums, awake but unconvinced. It still wants the phrase."
        else:
            body = "The door has heard the phrase, but the final witness decision has not been made."
        self.reply("Sealed Door", body, self.show_door)

    def show_vent(self) -> None:
        state = "open" if self.flag("vent_open") else "screwed shut"
        body = (
            f"The high vent is {state}. The duct behind it is narrow, angled upward, and rigged with "
            "a collapse seal. It is an exit for one body unless you enjoy becoming a cork."
        )
        options = [
            self.opt("Open the vent cover.", self.vent_open, not self.flag("vent_open"), "The vent is already open."),
            self.opt("Crawl out alone.", self.end_vent_solo, self.flag("vent_open"), "The vent is still shut."),
            self.opt("Send Mara through.", self.end_vent_mara, self.flag("vent_open"), "The vent is still shut."),
            self.opt("Send Ivo through.", self.end_vent_ivo, self.flag("vent_open"), "The vent is still shut."),
            self.opt("Brace it for both NPCs.", self.end_vent_both_npcs, self.flag("vent_open") and self.has("copper wire") and self.flag("mara_ready") and self.flag("ivo_ready"), "Needs wire and both NPCs ready."),
            self.opt("Back.", self.close_panel),
        ]
        self.set_panel("High Vent", body, options)

    def vent_open(self) -> None:
        if self.has("vent handle") or self.has("hairpin"):
            self.add_flag("vent_open")
            body = (
                "The cover unscrews and swings down. Dust spills over your shoulders. The duct "
                "smells like outside rain."
            )
        else:
            body = "The screws need the brass handle from the table or a thin tool like Mara's hairpin."
        self.reply("High Vent", body, self.show_vent)

    def show_mirror(self) -> None:
        circle = "A chalk circle waits below it." if self.flag("chalk_circle") else "The floor before it is bare."
        body = (
            "The black mirror reflects the room from the wrong corner. In it, the door is open, "
            f"closed, and missing. {circle}"
        )
        options = [
            self.opt("Take a mirror shard.", self.mirror_shard, not self.has("mirror shard"), "You already have a shard."),
            self.opt("Draw a chalk circle.", self.mirror_chalk, self.has("chalk") and not self.flag("chalk_circle"), "You need chalk or already drew the circle."),
            self.opt("Ask Ivo to read the reflection.", self.mirror_ask_ivo, self.trust["ivo"] >= 2 or self.flag("ivo_mirror_lesson"), "Ivo does not trust this enough yet."),
            self.opt("Step through alone.", self.end_mirror_solo, self.mirror_ready_basic(), "Needs shard, chalk circle, and passphrase."),
            self.opt("Send Ivo through the reflection.", self.end_mirror_ivo, self.flag("mirror_ready"), "The mirror has not been interpreted."),
            self.opt("Lead all three into the wrong reflection.", self.end_mirror_all, self.flag("mirror_ready") and self.flag("mara_ready") and self.flag("ivo_ready"), "Needs mirror ready and both NPCs ready."),
            self.opt("Back.", self.close_panel),
        ]
        self.set_panel("Black Mirror", body, options)

    def mirror_shard(self) -> None:
        self.add_item("mirror shard")
        body = (
            "The shard comes loose without breaking the rest of the mirror. It reflects your face "
            "with a half-second delay."
        )
        self.reply("Black Mirror", body, self.show_mirror)

    def mirror_chalk(self) -> None:
        self.add_flag("chalk_circle")
        body = (
            "You draw a closed chalk circle. The line absorbs light until it looks cut into the floor."
        )
        self.reply("Black Mirror", body, self.show_mirror)

    def mirror_ask_ivo(self) -> None:
        if self.mirror_ready_basic():
            self.add_flag("mirror_ready")
            self.add_trust("ivo", 1)
            body = (
                "Ivo reads the reversed passphrase in the shard. 'It will open, but not to the hall. "
                "It opens to the version of the room where the decision already happened.'"
            )
        else:
            body = (
                "Ivo studies the mirror. 'It still lacks grammar. Shard, chalk circle, phrase. Then "
                "it can become dangerous in a useful way.'"
            )
        self.reply("Black Mirror", body, self.show_mirror)

    def mirror_ready_basic(self) -> bool:
        return self.has("mirror shard") and self.flag("chalk_circle") and self.flag("passphrase_known")

    # Status and endings

    def show_status(self) -> None:
        inventory = ", ".join(sorted(self.items)) if self.items else "nothing"
        known = []
        if self.flag("door_powered"):
            known.append("door powered")
        if self.flag("passphrase_known"):
            known.append("phrase known")
        if self.flag("door_unlocked"):
            known.append("main door unlocked")
        if self.flag("vent_open"):
            known.append("vent open")
        if self.flag("mirror_ready"):
            known.append("mirror interpreted")
        leads = ", ".join(known) if known else "no exits solved yet"
        body = (
            f"Inventory: {inventory}\n"
            f"Mara trust: {self.trust['mara']} | Ivo trust: {self.trust['ivo']}\n"
            f"Leads: {leads}"
        )
        self.set_panel("State", body, [self.opt("Continue", self.close_panel)])

    def set_ending(self, title: str, body: str) -> None:
        self.mode = "ending"
        self.ending_title = title
        self.ending_body = body
        self.options = []

    def end_door_solo(self) -> None:
        self.set_ending(
            "Ending: Honest Cowardice",
            (
                "You name the choice and step through alone. The witness plate accepts the truth "
                "because you do not decorate it. Behind you, Mara swears, Ivo goes silent, and the "
                "door becomes a wall. You escape. The room keeps the rest."
            ),
        )

    def end_door_mara(self) -> None:
        self.set_ending(
            "Ending: Two Practical Survivors",
            (
                "You call Mara and move before doubt can vote. She slips through beside you, shoulder "
                "first, dragging a laugh out of herself when the hall is real. Ivo remains behind, "
                "already writing the moral appendix in his head."
            ),
        )

    def end_door_ivo(self) -> None:
        self.set_ending(
            "Ending: Footnote To Freedom",
            (
                "You take Ivo through the main door. He keeps one hand on the frame until the last "
                "second, whispering an apology to Mara. The door shuts softly, like a book deciding "
                "where the chapter ends."
            ),
        )

    def end_door_all(self) -> None:
        self.set_ending(
            "Ending: Three Go Through",
            (
                "Mara bridges the witness plate with one hand. Ivo speaks the phrase with the other "
                "on your shoulder. You cross together. The room tries to count sacrifice, fraud, and "
                "consensus, then gives up. All three of you escape into a hallway that smells like rain."
            ),
        )

    def end_false_witness(self) -> None:
        self.set_ending(
            "Ending: The Door Counts Four",
            (
                "The cracked red keycard volunteers as a person. The door accepts this terrible "
                "bookkeeping and opens too wide. You, Mara, and Ivo stumble out, followed by a fourth "
                "shadow with Mara's old fear printed on its face. It smiles and walks the other way."
            ),
        )

    def end_vent_solo(self) -> None:
        self.set_ending(
            "Ending: The Narrow Answer",
            (
                "You climb into the duct and do not look back until the collapse seal drops. The "
                "outside air is cold and real. You fit through because the exit was made for one, "
                "and because you let that be enough."
            ),
        )

    def end_vent_mara(self) -> None:
        self.set_ending(
            "Ending: Mara Takes The Duct",
            (
                "Mara climbs fast, all elbows and certainty. The collapse seal slams after her boots. "
                "A minute later the door lights flicker, maybe from her work outside, maybe from the "
                "room laughing. Only Mara escapes."
            ),
        )

    def end_vent_ivo(self) -> None:
        self.set_ending(
            "Ending: Ivo's Thin Mercy",
            (
                "Ivo hates the duct, which is why he survives it. He whispers the passphrase backward "
                "as he crawls. The vent seals behind him. Somewhere beyond the wall, he begins shouting "
                "for help that may or may not arrive."
            ),
        )

    def end_vent_both_npcs(self) -> None:
        self.set_ending(
            "Ending: The Brace Holds Once",
            (
                "You twist the copper wire around the collapse catch while Mara boosts Ivo into the "
                "duct. She follows. The brace holds for two bodies and then fails. Their voices fade "
                "toward rain. You remain in the room, alone but not useless."
            ),
        )

    def end_mirror_solo(self) -> None:
        self.set_ending(
            "Ending: You Leave Sideways",
            (
                "You step into the chalk circle with the mirror shard against your palm. The room "
                "folds. You emerge into the same room, empty, dusty, and unlocked from the inside. "
                "No one is trapped here anymore. No one remembers being with you."
            ),
        )

    def end_mirror_ivo(self) -> None:
        self.set_ending(
            "Ending: Ivo Becomes The Exit",
            (
                "Ivo enters the wrong reflection and the mirror goes clear. The sealed door opens "
                "without a sound, but only for Mara and you. On the other side of the glass, Ivo waves "
                "from a room full of doors and starts cataloging them."
            ),
        )

    def end_mirror_all(self) -> None:
        self.set_ending(
            "Ending: The Room Escapes",
            (
                "All three of you step into the circle. The mirror does not release you into a hall. "
                "It releases the room into the world. Dawn pours through the ceiling. The sealed door, "
                "the vent, the console, and every bad bargain vanish, leaving three people standing "
                "in an empty lot with chalk on their shoes."
            ),
        )

    # Drawing

    def render(self) -> None:
        self.screen.fill(BG)
        self.draw_room()
        self.draw_scene_objects()
        self.draw_hud()
        if self.mode in {"menu", "dialogue"}:
            self.draw_panel()
        elif self.mode == "ending":
            self.draw_ending()

    def draw_room(self) -> None:
        floor = [self.iso(0, 0), self.iso(ROOM_W, 0), self.iso(ROOM_W, ROOM_D), self.iso(0, ROOM_D)]
        polygon(self.screen, (42, 47, 53), floor)

        back_wall = [self.iso(0, 0), self.iso(ROOM_W, 0), self.iso(ROOM_W, 0, WALL_H), self.iso(0, 0, WALL_H)]
        left_wall = [self.iso(0, 0), self.iso(0, ROOM_D), self.iso(0, ROOM_D, WALL_H), self.iso(0, 0, WALL_H)]
        right_wall = [self.iso(ROOM_W, 0), self.iso(ROOM_W, ROOM_D), self.iso(ROOM_W, ROOM_D, WALL_H), self.iso(ROOM_W, 0, WALL_H)]
        polygon(self.screen, (35, 39, 48), back_wall)
        polygon(self.screen, (31, 35, 43), left_wall)
        polygon(self.screen, (37, 42, 50), right_wall)

        for x in range(int(ROOM_W) + 1):
            pygame.draw.line(self.screen, (55, 61, 67), self.iso(x, 0), self.iso(x, ROOM_D), 1)
        for z in range(int(ROOM_D) + 1):
            pygame.draw.line(self.screen, (55, 61, 67), self.iso(0, z), self.iso(ROOM_W, z), 1)

        pygame.draw.line(self.screen, (83, 91, 98), self.iso(0, ROOM_D), self.iso(ROOM_W, ROOM_D), 3)
        pygame.draw.line(self.screen, (62, 69, 77), self.iso(0, 0, WALL_H), self.iso(ROOM_W, 0, WALL_H), 2)
        pygame.draw.line(self.screen, (62, 69, 77), self.iso(0, 0, WALL_H), self.iso(0, ROOM_D, WALL_H), 2)
        pygame.draw.line(self.screen, (62, 69, 77), self.iso(ROOM_W, 0, WALL_H), self.iso(ROOM_W, ROOM_D, WALL_H), 2)

    def draw_scene_objects(self) -> None:
        drawables: List[Tuple[float, Callable[[], None]]] = []
        for spot in self.hotspots:
            drawables.append((spot.x + spot.z, lambda s=spot: self.draw_hotspot(s)))
        for npc in self.npcs.values():
            drawables.append((npc.x + npc.z, lambda a=npc: self.draw_actor(a, False)))
        drawables.append((self.player.x + self.player.z, lambda: self.draw_actor(self.player, True)))

        for _, draw in sorted(drawables, key=lambda item: item[0]):
            draw()

        nearest = self.nearest_interactable()
        if nearest:
            kind, key, _ = nearest
            if kind == "npc":
                name = self.npcs[key].name
            else:
                name = next(s.name for s in self.hotspots if s.key == key)
            self.draw_interaction_prompt(name)

    def draw_hotspot(self, spot: Hotspot) -> None:
        if spot.key == "door":
            self.draw_door_prop()
        elif spot.key == "console":
            self.draw_console_prop(spot)
        elif spot.key == "vent":
            self.draw_vent_prop()
        elif spot.key == "mirror":
            self.draw_mirror_prop()
        elif spot.key == "grate":
            self.draw_grate_prop(spot)
        elif spot.key == "table":
            self.draw_table_prop(spot)

    def draw_door_prop(self) -> None:
        z1, z2 = 4.65, 7.2
        h = 102
        pts = [self.iso(ROOM_W, z1), self.iso(ROOM_W, z2), self.iso(ROOM_W, z2, h), self.iso(ROOM_W, z1, h)]
        color = (74, 83, 88) if not self.flag("door_unlocked") else (83, 121, 100)
        polygon(self.screen, color, pts)
        pygame.draw.line(self.screen, GOLD if self.flag("door_unlocked") else (126, 103, 71), self.iso(ROOM_W, z1, 48), self.iso(ROOM_W, z2, 48), 3)
        pygame.draw.line(self.screen, (18, 20, 24), self.iso(ROOM_W, (z1 + z2) / 2), self.iso(ROOM_W, (z1 + z2) / 2, h), 2)

    def draw_console_prop(self, spot: Hotspot) -> None:
        x, y = self.iso(spot.x, spot.z, 28)
        body = pygame.Rect(x - 24, y - 28, 48, 38)
        pygame.draw.rect(self.screen, (39, 53, 64), body, border_radius=4)
        light = GREEN if self.flag("door_powered") else RED
        pygame.draw.rect(self.screen, light, (body.x + 8, body.y + 8, 32, 8), border_radius=3)
        pygame.draw.rect(self.screen, (12, 15, 18), body, 2, border_radius=4)

    def draw_vent_prop(self) -> None:
        x, y = self.iso(6.2, 0.0, 78)
        rect = pygame.Rect(x - 36, y - 16, 72, 30)
        pygame.draw.rect(self.screen, (54, 61, 66), rect, border_radius=3)
        pygame.draw.rect(self.screen, (14, 17, 20), rect, 2, border_radius=3)
        for i in range(5):
            lx = rect.x + 10 + i * 13
            pygame.draw.line(self.screen, (97, 108, 113), (lx, rect.y + 5), (lx, rect.bottom - 5), 2)
        if self.flag("vent_open"):
            pygame.draw.rect(self.screen, (10, 12, 15), rect.inflate(-12, -8), border_radius=2)

    def draw_mirror_prop(self) -> None:
        x, y = self.iso(0.1, 3.8, 72)
        pts = [(x, y - 42), (x + 30, y - 18), (x + 30, y + 42), (x, y + 18)]
        polygon(self.screen, (18, 20, 28), pts)
        polygon(self.screen, (43, 35, 60), [(x + 5, y - 28), (x + 24, y - 13), (x + 24, y + 28), (x + 5, y + 13)])
        pygame.draw.line(self.screen, VIOLET, (x + 8, y - 24), (x + 22, y + 22), 1)
        if self.flag("chalk_circle"):
            cx, cy = self.iso(1.6, 4.45)
            pygame.draw.ellipse(self.screen, (210, 214, 201), (cx - 38, cy - 12, 76, 24), 2)

    def draw_grate_prop(self, spot: Hotspot) -> None:
        x, y = self.iso(spot.x, spot.z)
        pygame.draw.ellipse(self.screen, (18, 21, 24), (x - 38, y - 13, 76, 26))
        pygame.draw.ellipse(self.screen, (89, 96, 101), (x - 38, y - 13, 76, 26), 2)
        for i in range(-2, 3):
            pygame.draw.line(self.screen, (76, 84, 89), (x + i * 12, y - 10), (x + i * 12, y + 10), 1)

    def draw_table_prop(self, spot: Hotspot) -> None:
        top = [
            self.iso(spot.x - 1.25, spot.z - 0.7, 22),
            self.iso(spot.x + 1.25, spot.z - 0.7, 22),
            self.iso(spot.x + 1.25, spot.z + 0.7, 22),
            self.iso(spot.x - 1.25, spot.z + 0.7, 22),
        ]
        polygon(self.screen, (102, 80, 57), top)
        polygon(self.screen, (68, 51, 38), [self.iso(spot.x - 1.25, spot.z + 0.7, 22), self.iso(spot.x + 1.25, spot.z + 0.7, 22), self.iso(spot.x + 1.25, spot.z + 0.7), self.iso(spot.x - 1.25, spot.z + 0.7)])
        for lx, lz in [(spot.x - 0.9, spot.z - 0.4), (spot.x + 0.9, spot.z - 0.4), (spot.x - 0.9, spot.z + 0.4), (spot.x + 0.9, spot.z + 0.4)]:
            pygame.draw.line(self.screen, (52, 39, 30), self.iso(lx, lz, 21), self.iso(lx, lz), 3)

    def draw_actor(self, actor: Actor, is_player: bool) -> None:
        x, y = self.iso(actor.x, actor.z)
        scale = 0.96 + actor.z / ROOM_D * 0.12
        shadow_w = int(38 * scale)
        pygame.draw.ellipse(self.screen, SHADOW, (x - shadow_w // 2, y - 8, shadow_w, 14))

        body_h = int(48 * scale)
        body_w = int(28 * scale)
        body = pygame.Rect(x - body_w // 2, y - body_h, body_w, body_h)
        pygame.draw.rect(self.screen, actor.color, body, border_radius=9)
        pygame.draw.rect(self.screen, actor.trim, (body.x + 5, body.y + 8, body.w - 10, 8), border_radius=3)
        head_r = int(12 * scale)
        pygame.draw.circle(self.screen, (207, 174, 139), (x, body.y - head_r + 2), head_r)
        eye_x = x + actor.facing * int(5 * scale)
        pygame.draw.circle(self.screen, (18, 22, 24), (eye_x, body.y - head_r), 2)

        if is_player:
            pygame.draw.circle(self.screen, GOLD, (x, y - body_h - 22), 5)
        else:
            label = self.small.render(actor.name, True, TEXT)
            bg = label.get_rect(center=(x, y - body_h - 32)).inflate(8, 4)
            pygame.draw.rect(self.screen, (11, 13, 17), bg, border_radius=4)
            self.screen.blit(label, label.get_rect(center=bg.center))

    def draw_hud(self) -> None:
        pygame.draw.rect(self.screen, (9, 11, 15), (14, 12, 342, 64), border_radius=6)
        pygame.draw.rect(self.screen, (55, 62, 70), (14, 12, 342, 64), 1, border_radius=6)
        inv = ", ".join(sorted(self.items)) if self.items else "empty"
        line1 = f"Inventory: {inv}"
        line2 = f"Mara {self.trust['mara']} | Ivo {self.trust['ivo']}"
        self.screen.blit(self.small.render(line1[:48], True, TEXT), (26, 24))
        self.screen.blit(self.small.render(line2, True, MUTED), (26, 48))

        if self.notice_timer > 0 and self.notice:
            surface = self.small.render(self.notice, True, GOLD)
            rect = surface.get_rect(midtop=(WIDTH // 2, 18))
            bg = rect.inflate(18, 10)
            pygame.draw.rect(self.screen, (12, 14, 18), bg, border_radius=5)
            pygame.draw.rect(self.screen, (91, 78, 45), bg, 1, border_radius=5)
            self.screen.blit(surface, rect)

    def draw_interaction_prompt(self, name: str) -> None:
        text = self.bold.render(f"E  {name}", True, TEXT)
        rect = text.get_rect(center=(WIDTH // 2, HEIGHT - 34))
        bg = rect.inflate(26, 14)
        pygame.draw.rect(self.screen, (13, 15, 19), bg, border_radius=6)
        pygame.draw.rect(self.screen, GOLD, bg, 1, border_radius=6)
        self.screen.blit(text, rect)

    def draw_panel(self) -> None:
        rect = pygame.Rect(34, HEIGHT - 238, WIDTH - 68, 214)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 2, border_radius=8)

        title = self.bold.render(self.panel_title, True, GOLD)
        self.screen.blit(title, (rect.x + 18, rect.y + 14))

        body_rect = pygame.Rect(rect.x + 18, rect.y + 44, rect.w - 36, 78)
        draw_wrapped(self.screen, self.font, self.panel_body, TEXT, body_rect)

        option_y = rect.y + 126
        col_w = (rect.w - 48) // 2
        for index, option in enumerate(self.options):
            col = index // 5
            row = index % 5
            x = rect.x + 18 + col * col_w
            y = option_y + row * 18
            label = f"{index + 1}. {option.label}"
            if not option.enabled and option.hint:
                label = f"{label} [{option.hint}]"
            color = TEXT if option.enabled else MUTED
            self.screen.blit(self.small.render(label[:58], True, color), (x, y))

    def draw_ending(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 178))
        self.screen.blit(overlay, (0, 0))

        rect = pygame.Rect(110, 112, WIDTH - 220, 356)
        pygame.draw.rect(self.screen, (18, 22, 28), rect, border_radius=8)
        pygame.draw.rect(self.screen, GOLD, rect, 2, border_radius=8)
        title = self.big.render(self.ending_title, True, GOLD)
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, rect.y + 54)))
        body_rect = pygame.Rect(rect.x + 46, rect.y + 112, rect.w - 92, 168)
        draw_wrapped(self.screen, self.font, self.ending_body, TEXT, body_rect, 6)
        footer = self.bold.render("R restart    Esc quit", True, MUTED)
        self.screen.blit(footer, footer.get_rect(center=(WIDTH // 2, rect.bottom - 44)))


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
