#!/usr/bin/env python3
"""
The Glass City
A text-based choose-your-own-adventure game with complex branching
and 12 different endings.

How to play:
- Run this file with Python 3.
- Type the number of your choice and press Enter.
- Some choices change your stats, inventory, or story flags.
"""

from textwrap import fill


WRAP = 78


def say(text=""):
    print(fill(text, WRAP))
    print()


class Game:
    def __init__(self):
        self.state = {
            "courage": 0,
            "wit": 0,
            "trust": 0,
            "shadow": 0,
            "inventory": set(),
            "flags": set(),
            "name": "",
            "alive": True,
        }
        self.endings = []

    def add_item(self, item):
        self.state["inventory"].add(item)

    def has_item(self, item):
        return item in self.state["inventory"]

    def set_flag(self, flag):
        self.state["flags"].add(flag)

    def has_flag(self, flag):
        return flag in self.state["flags"]

    def mod(self, stat, amount):
        self.state[stat] += amount

    def show_status(self):
        inv = ", ".join(sorted(self.state["inventory"])) or "nothing"
        flags = ", ".join(sorted(self.state["flags"])) or "none"
        print("-" * WRAP)
        print(
            f"Courage: {self.state['courage']} | "
            f"Wit: {self.state['wit']} | "
            f"Trust: {self.state['trust']} | "
            f"Shadow: {self.state['shadow']}"
        )
        print(f"Inventory: {inv}")
        print(f"Key Flags: {flags}")
        print("-" * WRAP)
        print()

    def choose(self, prompt, options):
        """
        options = list of tuples: (label, function)
        """
        say(prompt)
        for i, (label, _) in enumerate(options, start=1):
            print(f"{i}. {label}")
        print()

        while True:
            choice = input("> ").strip()
            if choice.lower() in {"status", "stats"}:
                self.show_status()
                continue
            if choice.lower() in {"inventory", "inv"}:
                inv = ", ".join(sorted(self.state["inventory"])) or "nothing"
                say(f"You are carrying: {inv}.")
                continue
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    print()
                    return options[idx][1]()
            print("Please enter a valid number, or type 'status' to inspect your state.\n")

    def ending(self, number, title, text):
        self.endings.append((number, title))
        print("=" * WRAP)
        print(f"ENDING {number}: {title}")
        print("=" * WRAP)
        print()
        say(text)
        print("Thanks for playing The Glass City.")
        print()

    def start(self):
        say("Welcome to THE GLASS CITY.")
        say(
            "They say the city appears only once every thirteen years, rising from the "
            "fog with towers like crystal knives. Inside waits the Heart Mirror, an "
            "artifact that can reveal truth, rewrite memory, or split a person into the "
            "self they are and the self they pretend to be."
        )
        name = input("What is your name, traveler? ").strip() or "Traveler"
        self.state["name"] = name
        print()
        say(
            f"{name}, you arrive at dusk carrying a sealed letter, a half-burned map, "
            "and too many reasons to enter a place most people flee."
        )
        self.gate()

    # ---------- STORY NODES ----------

    def gate(self):
        self.choose(
            "At the broken gate of the city, three paths open before you.",
            [
                ("Bribe the old ferryman to take you through the flooded lower canal.", self.canal),
                ("Climb the sunken wall and slip into the upper ruins alone.", self.wall),
                ("Follow a lantern carried by a masked stranger into a side passage.", self.stranger),
            ],
        )

    def canal(self):
        say(
            "The ferryman squints at your letter, then at your face. 'Everyone enters for "
            "one reason and leaves for another,' he mutters. His flatboat glides through a "
            "street drowned in black water. Below the surface, statues stare upward."
        )
        self.mod("trust", 1)
        self.choose(
            "Halfway through the canal, something bumps the boat.",
            [
                ("Lean over and inspect the water.", self.inspect_water),
                ("Keep your eyes ahead and question the ferryman.", self.question_ferryman),
                ("Draw the hook-knife from the boat and prepare for trouble.", self.fight_canal),
            ],
        )

    def inspect_water(self):
        say(
            "You peer into the water and see your own face looking back—except its eyes are "
            "closed. A pale hand rises and offers you a silver key."
        )
        self.add_item("silver key")
        self.mod("shadow", 1)
        self.set_flag("touched_blackwater")
        say("You take the silver key. The hand sinks without a ripple.")
        self.market()

    def question_ferryman(self):
        say(
            "The ferryman admits he once served the city's ruler, the Lady of Glass, until "
            "he saw the Heart Mirror consume a man's memory and leave him smiling."
        )
        self.mod("wit", 1)
        self.set_flag("learned_mirror_cost")
        self.market()

    def fight_canal(self):
        say(
            "You seize the hook-knife just as a glass-scaled creature bursts from the water. "
            "You slash wildly. It retreats, but not before its tail cuts your arm."
        )
        self.add_item("hook-knife")
        self.mod("courage", 1)
        self.mod("shadow", 1)
        self.market()

    def wall(self):
        say(
            "You scale the broken wall and cut your palm on a jagged edge of green glass. "
            "The upper ruins are wind-scoured and silent. A flock of mirrored crows watches "
            "you from a roofline without making a sound."
        )
        self.mod("courage", 1)
        self.choose(
            "A bell tolls somewhere deeper in the city.",
            [
                ("Follow the crows toward the sound.", self.bell_tower),
                ("Search the ruined guard post before moving on.", self.guard_post),
                ("Ignore everything and head straight for the central spire.", self.direct_spire),
            ],
        )

    def bell_tower(self):
        say(
            "Inside the bell tower you find a journal sealed beneath cracked stone. It "
            "describes secret passages under the palace and warns: 'The Mirror grants truth "
            "only to those who already know what they fear.'"
        )
        self.add_item("tower journal")
        self.mod("wit", 2)
        self.market()

    def guard_post(self):
        say(
            "In the guard post you find a rusted insignia and a loaded flare pistol. Scratched "
            "into the table is a map fragment pointing toward the market square."
        )
        self.add_item("flare pistol")
        self.add_item("map fragment")
        self.mod("wit", 1)
        self.market()

    def direct_spire(self):
        say(
            "You march toward the central spire too quickly and step onto a singing glass "
            "bridge. Invisible sentries awaken, turning the air sharp and bright."
        )
        self.choose(
            "The sentries close in.",
            [
                ("Run.", self.run_from_sentries),
                ("Hide under the bridge supports.", self.hide_from_sentries),
                ("Stand your ground and speak your true name aloud.", self.speak_true_name),
            ],
        )

    def run_from_sentries(self):
        say(
            "You sprint across the bridge as light spears crack around your heels. You dive "
            "through an archway and tumble into the abandoned market."
        )
        self.mod("courage", 1)
        self.mod("shadow", 1)
        self.market()

    def hide_from_sentries(self):
        say(
            "You wedge yourself beneath the bridge. The sentries sweep past, but the effort "
            "leaves you shaken and ashamed."
        )
        self.mod("wit", 1)
        self.mod("trust", -1)
        self.market()

    def speak_true_name(self):
        say(
            "You speak your full name into the cold air. The sentries pause. One bows, as if "
            "recognizing a claim older than the city itself."
        )
        self.set_flag("recognized_by_sentries")
        self.mod("trust", 1)
        self.mod("courage", 1)
        self.market()

    def stranger(self):
        say(
            "The masked stranger says nothing, only beckons. The side passage smells of dust, "
            "wax, and old roses. Finally, in a candlelit alcove, the stranger removes her mask."
        )
        say(
            "She is a young archivist named Ilya, and she claims the letter you carry was meant "
            "for her brother, who vanished in the palace years ago."
        )
        self.set_flag("met_ilya")
        self.mod("trust", 2)
        self.choose(
            "Ilya offers an alliance.",
            [
                ("Trust her and show her the sealed letter.", self.share_letter),
                ("Keep the letter hidden but travel with her anyway.", self.travel_with_ilya),
                ("Refuse her help and continue alone.", self.refuse_ilya),
            ],
        )

    def share_letter(self):
        say(
            "Ilya breaks the seal. Inside is a warning: 'Do not let the Heart Mirror choose "
            "for you. Break it, bury it, or master it—but never ask it who you are.'"
        )
        self.set_flag("read_letter")
        self.mod("wit", 1)
        self.mod("trust", 1)
        self.market()

    def travel_with_ilya(self):
        say(
            "Ilya nods, though she clearly notices your caution. She teaches you how to read "
            "the colored reflections in the paving stones—safe routes, unstable routes, and traps."
        )
        self.set_flag("ilya_ally")
        self.mod("wit", 1)
        self.market()

    def refuse_ilya(self):
        say(
            "Ilya steps aside with a look that is not anger but disappointment. 'Then at least "
            "remember this,' she says. 'In the palace, every locked door protects something, "
            "but sometimes it protects the city from you.'"
        )
        self.mod("trust", -1)
        self.market()

    def market(self):
        say(
            "You reach the abandoned market square, where dozens of glass-faced mannequins stand "
            "in merchant stalls as if the city paused mid-breath. Three landmarks draw you in: "
            "the House of Records, the Moon Well, and the shattered theater."
        )
        self.choose(
            "Where do you go first?",
            [
                ("Search the House of Records for the city's history.", self.records),
                ("Descend to the Moon Well, where whispers rise from below.", self.moon_well),
                ("Enter the shattered theater, where music still plays at night.", self.theater),
            ],
        )

    def records(self):
        say(
            "Dust sheets the records hall. Thousands of names fill shelves and ledgers, some crossed "
            "out, some rewritten in a different hand. You find your own name in a book that should "
            "be older than your grandparents."
        )
        self.set_flag("found_name_in_records")
        self.mod("wit", 2)
        self.choose(
            "The discovery leaves you cold.",
            [
                ("Rip the page out and keep it.", self.take_page),
                ("Study nearby books for more clues.", self.study_ledgers),
                ("Burn the ledger and flee the building.", self.burn_ledger),
            ],
        )

    def take_page(self):
        say(
            "You tear out the page bearing your name. On the back is a sigil matching the seal on "
            "your letter."
        )
        self.add_item("record page")
        self.mod("shadow", 1)
        self.palace_approach()

    def study_ledgers(self):
        say(
            "By cross-referencing dates, you realize the Lady of Glass preserves the city by stealing "
            "possible futures from those who enter. Some become counselors. Some become statues. "
            "Some are sent back out, changed in ways they cannot remember."
        )
        self.set_flag("understands_city")
        self.mod("wit", 2)
        self.palace_approach()

    def burn_ledger(self):
        say(
            "Flames rush unnaturally fast through the paper. The hall screams—not with voices, but "
            "with cracking glass. You escape with smoke in your lungs."
        )
        self.set_flag("records_burned")
        self.mod("courage", 1)
        self.mod("shadow", 2)
        self.palace_approach()

    def moon_well(self):
        say(
            "The Moon Well is a circular shaft lined in mirrored stone. Below, moonlight glows where "
            "no moon can reach. A rope hangs from an iron ring."
        )
        self.choose(
            "What do you do?",
            [
                ("Climb down into the well.", self.climb_well),
                ("Drop a coin and listen to the echo.", self.listen_well),
                ("Call into the dark and ask who waits below.", self.call_well),
            ],
        )

    def climb_well(self):
        say(
            "At the bottom, you discover a hidden shrine and a sleeping figure trapped in a shell of "
            "clear glass. The figure is alive. Barely."
        )
        self.add_item("moon sigil")
        self.set_flag("found_shrine")
        self.mod("courage", 1)
        self.choose(
            "The trapped figure opens one eye.",
            [
                ("Try to free them.", self.free_prisoner),
                ("Ask who imprisoned them.", self.ask_prisoner),
                ("Take the sigil and leave quietly.", self.leave_prisoner),
            ],
        )

    def free_prisoner(self):
        say(
            "You crack the shell with a fallen iron hook. The prisoner collapses into your arms. "
            "He introduces himself as Orren, a former steward of the palace."
        )
        self.set_flag("orren_ally")
        self.mod("trust", 1)
        self.mod("courage", 1)
        self.palace_approach()

    def ask_prisoner(self):
        say(
            "The prisoner whispers, 'I chose memory over obedience.' Before he can say more, the glass "
            "seals over his mouth again. You leave with the words lodged in your mind."
        )
        self.set_flag("heard_memory_over_obedience")
        self.mod("wit", 1)
        self.palace_approach()

    def leave_prisoner(self):
        say(
            "You pocket the moon sigil and leave, pretending not to hear the faint tapping behind you."
        )
        self.mod("shadow", 1)
        self.palace_approach()

    def listen_well(self):
        say(
            "The coin falls a long time. When the echo returns, it is your own voice saying: 'Do not "
            "trade grief for comfort.'"
        )
        self.set_flag("heard_warning")
        self.mod("wit", 1)
        self.palace_approach()

    def call_well(self):
        say(
            "Something answers in perfect imitation of your voice, but a half-second too late. The sound "
            "crawls over your skin. You step back, changed by it."
        )
        self.mod("shadow", 2)
        self.set_flag("echo_marked")
        self.palace_approach()

    def theater(self):
        say(
            "In the shattered theater, a phantom orchestra rehearses for an audience of dust. Center stage "
            "stands a woman in mirrored armor. She calls herself the Actress and claims to be one of the "
            "Lady's discarded selves."
        )
        self.choose(
            "The Actress offers knowledge for a price.",
            [
                ("Ask what price she wants.", self.actress_price),
                ("Challenge her to prove she knows the palace.", self.challenge_actress),
                ("Attack before she can deceive you.", self.attack_actress),
            ],
        )

    def actress_price(self):
        say(
            "'A memory,' she says. 'A small one. Nothing fatal.' If you agree, she tells you of a hidden "
            "door beneath the throne and how to silence the palace sentries."
        )
        self.choose(
            "Do you pay?",
            [
                ("Yes. Give up a memory.", self.pay_memory),
                ("No. Refuse the bargain.", self.refuse_bargain),
            ],
        )

    def pay_memory(self):
        say(
            "The Actress kisses your forehead. Suddenly you cannot remember the face of the first person "
            "who ever loved you. In return, she teaches you the palace's hush-sign."
        )
        self.set_flag("knows_hush_sign")
        self.mod("shadow", 1)
        self.mod("wit", 2)
        self.palace_approach()

    def refuse_bargain(self):
        say(
            "The Actress smiles sadly. 'Good,' she says. 'Then perhaps you are still dangerous.' She vanishes, "
            "but leaves behind a shard polished like a mirror."
        )
        self.add_item("mirror shard")
        self.mod("courage", 1)
        self.palace_approach()

    def challenge_actress(self):
        say(
            "You ask for proof. She recites the last words of three rulers and the secret nickname your mother "
            "used when you were a child. You never told anyone that name."
        )
        self.add_item("mirror shard")
        self.set_flag("actress_truth")
        self.mod("wit", 1)
        self.mod("shadow", 1)
        self.palace_approach()

    def attack_actress(self):
        say(
            "You lunge. Your blade passes through her, but the stage floor cracks beneath you. You barely escape "
            "with a cut cheek and a piece of mirrored armor she chose to leave behind."
        )
        self.add_item("armor shard")
        self.mod("courage", 2)
        self.mod("shadow", 1)
        self.palace_approach()

    def palace_approach(self):
        say(
            "Night deepens. All roads now tilt toward the Palace of Refractions, where the Lady of Glass keeps "
            "the Heart Mirror. At the palace steps you find one more chance to prepare."
        )
        options = [
            ("Enter through the grand front gate.", self.front_gate),
            ("Search for a hidden servant passage.", self.hidden_passage),
            ("Wait and see who else approaches the palace.", self.wait_steps),
        ]
        self.choose("How do you approach the palace?", options)

    def front_gate(self):
        say(
            "Glass sentries lower spears as you approach the main doors."
        )
        if self.has_flag("recognized_by_sentries") or self.has_flag("knows_hush_sign"):
            say(
                "You use what you learned earlier, and the sentries part without violence."
            )
            self.throne_hall()
        else:
            self.choose(
                "They demand a reason to admit you.",
                [
                    ("Show them your sealed letter or record page.", self.show_proof_gate),
                    ("Fight your way in.", self.force_gate),
                    ("Retreat and search for another way.", self.hidden_passage),
                ],
            )

    def show_proof_gate(self):
        if self.has_item("record page") or self.has_flag("read_letter"):
            say(
                "The sigil or letter is enough. The sentries recognize old authority and permit entry."
            )
            self.mod("trust", 1)
            self.throne_hall()
        else:
            say(
                "You fumble for proof you do not truly have. The sentries see through the act."
            )
            self.force_gate()

    def force_gate(self):
        say(
            "You crash into the sentries. Whether by hook-knife, flare pistol, or sheer desperation, you make "
            "it inside—but not unnoticed."
        )
        self.mod("courage", 2)
        self.mod("shadow", 2)
        self.set_flag("entered_by_force")
        self.throne_hall()

    def hidden_passage(self):
        say(
            "You circle the palace and search beneath collapsed stairways and drained fountains. At last you find "
            "a concealed hatch."
        )
        if self.has_item("silver key"):
            say(
                "The silver key from the black water fits perfectly. The hatch opens into a narrow passage of whispering glass."
            )
            self.mod("wit", 1)
            self.throne_hall(secret=True)
        elif self.has_flag("orren_ally") or self.has_item("moon sigil"):
            say(
                "With guidance from Orren or by matching the moon sigil to a socket in the stone, you open the way."
            )
            self.throne_hall(secret=True)
        else:
            say(
                "You cannot open it. While searching, you spot another figure approaching the palace."
            )
            self.wait_steps()

    def wait_steps(self):
        say(
            "You wait in the shadow of a broken lion statue. Soon another traveler arrives—"
        )
        if self.has_flag("met_ilya") or self.has_flag("ilya_ally"):
            say(
                "it is Ilya, carrying a satchel of stolen palace keys and looking like she expected to find you."
            )
            self.choose(
                "Ilya offers to help you reach the Mirror chamber.",
                [
                    ("Accept her help fully.", self.accept_ilya_palace),
                    ("Use her keys but keep your own agenda.", self.use_ilya_keys),
                    ("Turn her away and continue alone.", self.turn_ilya_away),
                ],
            )
        elif self.has_flag("orren_ally"):
            say(
                "it is Orren, weak but determined. He says there is still time to choose what kind of damage you will do."
            )
            self.choose(
                "Orren offers to guide you.",
                [
                    ("Follow Orren through the servant halls.", self.orren_guide),
                    ("Ask Orren how to destroy the Mirror.", self.orren_destroy),
                    ("Leave him behind.", self.leave_orren),
                ],
            )
        else:
            say(
                "it is a silent child carrying a glass lantern. The child points to a side door, then vanishes."
            )
            self.set_flag("lantern_child")
            self.throne_hall(secret=True)

    def accept_ilya_palace(self):
        say(
            "You and Ilya enter together. Trust, once given, lightens every corridor."
        )
        self.set_flag("ilya_ally")
        self.mod("trust", 2)
        self.throne_hall(secret=True)

    def use_ilya_keys(self):
        say(
            "You take the keys and a half-promise. Ilya notices, but says nothing."
        )
        self.add_item("palace keys")
        self.mod("trust", -1)
        self.throne_hall(secret=True)

    def turn_ilya_away(self):
        say(
            "Ilya stares at you for a long moment. 'Then do not ask me to mourn who returns,' she says."
        )
        self.mod("trust", -2)
        self.throne_hall()

    def orren_guide(self):
        say(
            "Orren leads you through forgotten service corridors into the palace's heart."
        )
        self.mod("trust", 1)
        self.set_flag("orren_ally")
        self.throne_hall(secret=True)

    def orren_destroy(self):
        say(
            "Orren tells you the Mirror can be broken only by striking it with a shard that already reflects it."
        )
        self.set_flag("knows_break_method")
        self.throne_hall(secret=True)

    def leave_orren(self):
        say(
            "You step past Orren. He does not follow."
        )
        self.mod("shadow", 1)
        self.throne_hall()

    def throne_hall(self, secret=False):
        say(
            "You reach the throne hall, a cavern of crystal ribs and floating reflections. At the far end stands "
            "the Lady of Glass, serene and terrible. Behind her hovers the Heart Mirror, shaped like a vertical pool "
            "of silver fire."
        )
        if secret:
            say(
                "Because you entered unseen, you have a brief moment before she notices you."
            )
            self.choose(
                "What do you do first?",
                [
                    ("Study the Mirror instead of the Lady.", self.study_mirror),
                    ("Confront the Lady immediately.", self.confront_lady),
                    ("Search for a way beneath the throne.", self.search_throne),
                ],
            )
        else:
            self.confront_lady()

    def study_mirror(self):
        say(
            "The Mirror shows not your face, but twelve possible versions of you, each tied to a different fate."
        )
        self.mod("wit", 1)
        self.set_flag("saw_possible_selves")
        self.final_choice()

    def confront_lady(self):
        say(
            "The Lady of Glass turns toward you as if she has always known the precise moment you would arrive."
        )
        if self.has_flag("understands_city") or self.has_flag("learned_mirror_cost"):
            say(
                "'You steal futures,' you accuse. She does not deny it."
            )
        else:
            say(
                "'You have come for answers,' she says. 'Most people do. They only regret which answer they pick.'"
            )
        self.final_choice()

    def search_throne(self):
        say(
            "Beneath the throne you find a concealed recess."
        )
        if self.has_item("mirror shard") or self.has_item("armor shard"):
            say(
                "Inside lies a socket clearly meant for a reflective shard. You begin to understand how the Mirror might be broken."
            )
            self.set_flag("knows_break_method")
            self.final_choice()
        else:
            say(
                "You find only dust and old blood before the Lady notices you."
            )
            self.final_choice()

    # ---------- FINAL DECISION TREE ----------

    def final_choice(self):
        options = [
            ("Ask the Heart Mirror to reveal the truth about your life.", self.ask_truth),
            ("Ask the Heart Mirror to restore someone or something you lost.", self.ask_restoration),
            ("Attempt to seize control of the Heart Mirror.", self.take_mirror),
            ("Attempt to destroy the Heart Mirror.", self.destroy_mirror),
            ("Reject the Mirror and leave the city with what you've learned.", self.reject_mirror),
        ]
        self.choose("Before you, the city balances on a single decision.", options)

    def ask_truth(self):
        say(
            "You step before the Heart Mirror and ask for truth."
        )
        if self.state["wit"] >= 4 and self.state["shadow"] <= 2:
            self.ending(
                1,
                "The Unvarnished Self",
                f"The Mirror reveals your life without flattery or excuse. You see your wounds, your habits, "
                f"your vanities, your loyalties, and the shape of your future if you remain unchanged. The truth "
                f"does not destroy you. It hardens into wisdom. You leave the Glass City carrying no treasure, but "
                f"at last you are difficult to fool—including by yourself."
            )
        elif self.state["shadow"] >= 4:
            self.ending(
                2,
                "The Splintered Mind",
                "The Mirror gives you truth in fragments too sharp to hold. You leave the palace alive, but every "
                "reflection now shows a slightly different version of you. In time, the city gains another whispering "
                "ghost: a traveler who knows too much and trusts nothing, not even his own face."
            )
        else:
            self.ending(
                3,
                "The Quiet Clerk",
                "The Lady smiles as the Mirror answers. You learn a terrible truth: years ago, part of you already "
                "came here and chose service over uncertainty. The person who entered tonight was the remainder. You "
                "do not resist when palace attendants bring you robes and ledgers. At dawn, the city has a new clerk."
            )

    def ask_restoration(self):
        say(
            "You ask the Mirror to return what was lost."
        )
        if self.has_flag("heard_warning"):
            self.ending(
                4,
                "Grief Kept Honest",
                "At the last moment, you remember the warning from the Moon Well: do not trade grief for comfort. "
                "You withdraw your request. The refusal costs you tears, but saves your life. You leave carrying pain "
                "that is truly yours, and that becomes the beginning of healing."
            )
        elif self.has_flag("ilya_ally") and self.state["trust"] >= 3:
            self.ending(
                5,
                "The Borrowed Return",
                "The Mirror grants your wish—but imperfectly. Someone beloved returns, yet altered, as if rebuilt from "
                "memory rather than life. Ilya sees what you have done and stays anyway, because mercy can look like "
                "complicity when love is involved. You leave together with a miracle that may one day turn on you."
            )
        else:
            self.ending(
                6,
                "The Price of Comfort",
                "The Mirror restores what you begged for by taking an equal weight from your future. You walk out with "
                "your loss reversed and your years mysteriously shortened. The city keeps the remainder. It is a fair "
                "trade only in the cruel arithmetic of desperate people."
            )

    def take_mirror(self):
        say(
            "You try to master the Heart Mirror instead of kneeling to it."
        )
        if self.state["courage"] >= 4 and self.state["wit"] >= 4 and self.state["trust"] >= 2:
            self.ending(
                7,
                "Ruler of Refractions",
                "Against every sane probability, you outplay the Lady of Glass. The Mirror bends to will, but only because "
                "you understand its trap: it cannot define someone who keeps choosing himself consciously. The city accepts "
                "you as its new sovereign. Whether that makes you a guardian or merely a better tyrant will take years to learn."
            )
        elif self.state["shadow"] >= 5:
            self.ending(
                8,
                "The New Monster",
                "You seize the Mirror, and it loves you for all the wrong reasons. Your hungers, grievances, and hidden vanity "
                "expand until they eclipse your original purpose. The Lady falls. The city kneels. By winter, travelers whisper "
                "that the old ruler was merciful compared to the one who came after."
            )
        else:
            self.ending(
                9,
                "Statue in the Hall",
                "You reach for power without enough leverage to hold it. The Mirror freezes around you in a sheath of perfect glass. "
                "Centuries later, visitors will admire the realism of your expression and never know it captured the exact second you "
                "understood ambition and readiness are not the same thing."
            )

    def destroy_mirror(self):
        say(
            "You choose destruction."
        )
        can_break = (
            self.has_flag("knows_break_method")
            or (self.has_item("mirror shard") and self.has_item("armor shard"))
            or (self.has_item("mirror shard") and self.has_flag("actress_truth"))
        )

        if can_break and self.state["courage"] >= 3:
            if self.has_flag("orren_ally") or self.has_flag("ilya_ally"):
                self.ending(
                    10,
                    "The City Set Free",
                    "With a reflective shard and brutal resolve, you strike the Heart Mirror at its own image. It breaks with a sound "
                    "like a thousand held breaths released at once. The Lady of Glass collapses. Across the city, statues crack open, "
                    "memories return, and the trapped futures of countless people spill back into the world. The Glass City does not vanish, "
                    "but it becomes mortal at last."
                )
            else:
                self.ending(
                    11,
                    "The Lone Liberator",
                    "You smash the Mirror and survive the backlash alone. The city howls as its magic unravels. You stagger out at dawn, "
                    "bleeding, carrying proof no one will fully believe. Still, the disappearances stop. Sometimes history is changed by people "
                    "who never get the credit."
                )
        else:
            self.ending(
                12,
                "Shattered, Not Broken",
                "You attack the Heart Mirror without the right knowledge or tool. A crack appears—but only one. The backlash tears through the "
                "throne hall, throwing you into darkness. When you wake outside the city, you cannot remember your own name, only that something "
                "inside still waits unfinished."
            )

    def reject_mirror(self):
        say(
            "You refuse the Mirror entirely."
        )
        if self.has_flag("understands_city") and self.state["trust"] >= 2:
            self.ending(
                13,
                "The Witness",
                "You deny the Lady, the Mirror, and the seduction of certainty. Instead you leave with knowledge, names, methods, and proof enough "
                "to expose the city over time. It is the least dramatic ending and perhaps the wisest. Some evils are not defeated in one night; "
                "they are starved by being seen clearly and spoken of without fear."
            )
        elif self.has_flag("records_burned"):
            self.ending(
                14,
                "Ashes Over Glass",
                "You turn your back on the Mirror and flee while the records hall still burns in the distance. By morning, the fire has spread farther "
                "than anyone expected. The city survives, but wounded and diminished. You become both coward and catalyst, depending on who tells it."
            )
        else:
            self.ending(
                15,
                "The Door Left Closed",
                "You walk away. No bargain, no revelation, no miracle. Years later, you will still wonder what answer the Heart Mirror would have given. "
                "Yet you keep your uncertainty, and in that uncertainty your freedom remains intact."
            )


def main():
    game = Game()
    game.start()


if __name__ == "__main__":
    main()
