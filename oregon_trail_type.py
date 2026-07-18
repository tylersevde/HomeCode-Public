import random  
import json  
import os  
  
SAVE_FILE = "oregon_polished_save.json"  
GOAL_DISTANCE = 1200  
  
LANDMARKS = [  
    {"mile": 150, "name": "Kansas River", "type": "river",  
     "text": "A wide river cuts across the trail."},  
    {"mile": 300, "name": "Fort Kearny", "type": "fort",  
     "text": "A busy fort where travelers rest and trade."},  
    {"mile": 520, "name": "Chimney Rock", "type": "landmark",  
     "text": "The tall stone spire rises over the plains."},  
    {"mile": 700, "name": "Fort Laramie", "type": "fort",  
     "text": "A major stop for supplies, repairs, and news."},  
    {"mile": 920, "name": "South Pass", "type": "pass",  
     "text": "A broad mountain pass opens the way west."},  
    {"mile": 1080, "name": "Snake River", "type": "river",  
     "text": "Cold fast water rushes through the rocks."},  
    {"mile": 1200, "name": "Oregon City", "type": "goal",  
     "text": "You finally see the end of the trail ahead."},  
]  
  
  
# ---------------------------  
# Color helpers  
# ---------------------------  
def color(text, code):  
    return f"\033[{code}m{text}\033[0m"  
  
def red(text): return color(text, "31")  
def green(text): return color(text, "32")  
def yellow(text): return color(text, "33")  
def blue(text): return color(text, "34")  
def magenta(text): return color(text, "35")  
def cyan(text): return color(text, "36")  
  
  
# ---------------------------  
# Utility  
# ---------------------------  
def clamp(value, low, high):  
    return max(low, min(value, high))  
  
def living_members(party):  
    return [m for m in party if m["alive"]]  
  
def random_living_member(party):  
    alive = living_members(party)  
    return random.choice(alive) if alive else None  
  
def party_size(state):  
    return max(1, len(living_members(state["party"])))  
  
def heal_member(member, amount):  
    if member and member["alive"]:  
        member["health"] = clamp(member["health"] + amount, 0, 100)  
  
def damage_member(member, amount):  
    if member and member["alive"]:  
        member["health"] -= amount  
        if member["health"] <= 0:  
            member["health"] = 0  
            member["alive"] = False  
            print(red(f"{member['name']} has died."))  
  
def apply_food_cost(state, base_per_person):  
    used = base_per_person * party_size(state)  
    state["food"] -= used  
    return used  
  
  
# ---------------------------  
# Save / Load  
# ---------------------------  
def save_game(state):  
    with open(SAVE_FILE, "w") as f:  
        json.dump(state, f, indent=2)  
    print(green("Game saved."))  
  
def load_game():  
    if not os.path.exists(SAVE_FILE):  
        return None  
    with open(SAVE_FILE, "r") as f:  
        return json.load(f)  
  
  
# ---------------------------  
# Setup  
# ---------------------------  
def choose_party():  
    leader = input("What is your name, wagon leader? ").strip()  
    if not leader:  
        leader = "Leader"  
  
    party = [{"name": leader, "health": 100, "alive": True}]  
  
    print("\nEnter 3 party member names.")  
    for i in range(3):  
        name = input(f"Party member {i + 1}: ").strip()  
        if not name:  
            name = f"Traveler{i + 1}"  
        party.append({"name": name, "health": 100, "alive": True})  
  
    return leader, party  
  
def new_game():  
    leader, party = choose_party()  
    return {  
        "leader": leader,  
        "party": party,  
        "day": 1,  
        "miles": 0,  
        "food": 140,  
        "money": 60,  
        "bullets": 36,  
        "medkits": 2,  
        "wagon": 100,  
        "weather": "Clear",  
        "visited_landmarks": []  
    }  
  
  
# ---------------------------  
# Display  
# ---------------------------  
def show_status(state):  
    print("\n" + "=" * 60)  
    print(cyan(f"{state['leader']}'s Wagon"))  
    print(f"Day: {state['day']}")  
    print(f"Distance: {state['miles']}/{GOAL_DISTANCE} miles")  
    print(f"Weather: {state['weather']}")  
    print(f"Food: {state['food']}")  
    print(f"Money: ${state['money']}")  
    print(f"Bullets: {state['bullets']}")  
    print(f"Medkits: {state['medkits']}")  
    print(f"Wagon condition: {state['wagon']}/100")  
    print("Party:")  
    for member in state["party"]:  
        status = green("Alive") if member["alive"] else red("Dead")  
        hp = member["health"]  
        hp_text = green(hp) if hp > 60 else yellow(hp) if hp > 25 else red(hp)  
        print(f"  - {member['name']} | Health: {hp_text} | {status}")  
    print("=" * 60)  
  
  
# ---------------------------  
# Weather  
# ---------------------------  
def roll_weather():  
    roll = random.randint(1, 100)  
    if roll <= 25:  
        return "Clear"  
    elif roll <= 45:  
        return "Hot"  
    elif roll <= 65:  
        return "Rain"  
    elif roll <= 85:  
        return "Cold"  
    else:  
        return "Storm"  
  
def start_day_weather(state):  
    state["weather"] = roll_weather()  
    print(blue(f"\nToday's weather: {state['weather']}"))  
  
    if state["weather"] == "Hot":  
        extra = 2 * party_size(state)  
        state["food"] -= extra  
        print(yellow(f"The heat spoils food and tires the party. Extra food used: {extra}"))  
  
    elif state["weather"] == "Rain":  
        state["wagon"] -= 3  
        print(yellow("Mud and rain wear down the wagon."))  
  
    elif state["weather"] == "Cold":  
        member = random_living_member(state["party"])  
        if member:  
            damage_member(member, 5)  
            print(yellow(f"{member['name']} suffers from the cold."))  
  
    elif state["weather"] == "Storm":  
        state["wagon"] -= 8  
        member = random_living_member(state["party"])  
        if member:  
            damage_member(member, 8)  
        print(red("A storm batters the wagon and the party."))  
  
    state["wagon"] = clamp(state["wagon"], 0, 100)  
  
  
# ---------------------------  
# Combat  
# ---------------------------  
def bandit_combat(state):  
    print(red("\nBandits attack!"))  
    enemy_hp = random.randint(28, 45)  
    aim_bonus = 0  
    cover = False  
  
    while enemy_hp > 0 and living_members(state["party"]):  
        print("\n" + "-" * 40)  
        print(f"Bandit health: {enemy_hp}")  
        print(f"Your bullets: {state['bullets']}")  
        print("1. Shoot")  
        print("2. Steady aim")  
        print("3. Take cover")  
        print("4. Run")  
  
        choice = input("> ").strip()  
  
        if choice == "1":  
            if state["bullets"] <= 0:  
                print(red("You have no bullets!"))  
            else:  
                state["bullets"] -= 1  
                hit_chance = 55 + aim_bonus  
                if state["weather"] in ("Rain", "Storm"):  
                    hit_chance -= 10  
  
                if random.randint(1, 100) <= hit_chance:  
                    dmg = random.randint(12, 24)  
                    enemy_hp -= dmg  
                    print(green(f"You hit the bandit for {dmg} damage."))  
                else:  
                    print(yellow("You missed."))  
  
                aim_bonus = 0  
                cover = False  
  
        elif choice == "2":  
            aim_bonus = min(40, aim_bonus + 20)  
            cover = False  
            print(cyan("You steady your aim for the next shot."))  
  
        elif choice == "3":  
            cover = True  
            aim_bonus = 0  
            print(cyan("You duck behind the wagon for cover."))  
  
        elif choice == "4":  
            escape = 35  
            if state["wagon"] < 30:  
                escape -= 10  
            if random.randint(1, 100) <= escape:  
                print(yellow("You escaped the bandits."))  
                return  
            else:  
                print(red("You failed to escape!"))  
        else:  
            print(red("Invalid choice."))  
            continue  
  
        if enemy_hp > 0:  
            enemy_hit = 55  
            if cover:  
                enemy_hit -= 25  
  
            if random.randint(1, 100) <= enemy_hit:  
                victim = random_living_member(state["party"])  
                if victim:  
                    dmg = random.randint(8, 18)  
                    damage_member(victim, dmg)  
                    print(red(f"The bandit hits {victim['name']} for {dmg} damage."))  
            else:  
                print(green("The bandit misses."))  
  
    if enemy_hp <= 0:  
        found_money = random.randint(6, 18)  
        found_food = random.randint(8, 22)  
        found_bullets = random.randint(2, 8)  
        state["money"] += found_money  
        state["food"] += found_food  
        state["bullets"] += found_bullets  
        print(green("\nYou defeated the bandit."))  
        print(green(f"Loot: ${found_money}, {found_food} food, {found_bullets} bullets"))  
  
  
# ---------------------------  
# Events  
# ---------------------------  
def disease_event(state):  
    if random.randint(1, 100) <= 18:  
        victim = random_living_member(state["party"])  
        if victim:  
            disease = random.choice(["fever", "dysentery", "cholera", "infection"])  
            print(red(f"\nDisease strikes! {victim['name']} gets {disease}."))  
            damage_member(victim, random.randint(10, 26))  
  
            if victim["alive"] and state["medkits"] > 0:  
                use = input("Use a medkit now? (y/n): ").strip().lower()  
                if use == "y":  
                    state["medkits"] -= 1  
                    heal_member(victim, 20)  
                    print(green(f"{victim['name']} was treated."))  
  
def snake_bite_event(state):  
    victim = random_living_member(state["party"])  
    if victim:  
        dmg = random.randint(8, 20)  
        print(red(f"\nA snake bites {victim['name']}!"))  
        damage_member(victim, dmg)  
  
        if victim["alive"] and state["medkits"] > 0:  
            use = input("Use a medkit for the snake bite? (y/n): ").strip().lower()  
            if use == "y":  
                state["medkits"] -= 1  
                heal_member(victim, 15)  
                print(green(f"{victim['name']} feels better."))  
  
def random_travel_event(state):  
    roll = random.randint(1, 100)  
  
    if roll <= 15:  
        bandit_combat(state)  
  
    elif roll <= 25:  
        snake_bite_event(state)  
  
    elif roll <= 40:  
        print(yellow("\nA wagon wheel cracks on rough trail."))  
        state["wagon"] -= random.randint(8, 16)  
  
    elif roll <= 52:  
        print(green("\nYou find berries and edible plants."))  
        state["food"] += random.randint(10, 20)  
  
    elif roll <= 62:  
        print(yellow("\nSome food spoils in the heat and dust."))  
        state["food"] -= random.randint(8, 16)  
  
    elif roll <= 70:  
        victim = random_living_member(state["party"])  
        if victim:  
            print(red(f"\n{victim['name']} falls and is injured on the trail."))  
            damage_member(victim, random.randint(6, 16))  
  
    state["wagon"] = clamp(state["wagon"], 0, 100)  
  
def random_death_event(state):  
    if random.randint(1, 100) <= 4:  
        victim = random_living_member(state["party"])  
        if victim:  
            print(red(f"\nA terrible accident kills {victim['name']}."))  
            victim["health"] = 0  
            victim["alive"] = False  
  
  
# ---------------------------  
# Rivers and landmarks  
# ---------------------------  
def river_crossing(state, river_name):  
    print(blue(f"\nYou must cross {river_name}."))  
    depth = random.randint(1, 3)  
    depth_text = {1: "shallow", 2: "medium", 3: "deep"}[depth]  
    print(f"The river looks {depth_text}.")  
    print("1. Ford the river")  
    print("2. Ferry across ($10)")  
    print("3. Wait for calmer conditions")  
  
    choice = input("> ").strip()  
  
    if choice == "1":  
        success = 80 if depth == 1 else 55 if depth == 2 else 25  
        if random.randint(1, 100) <= success:  
            print(green("You crossed safely."))  
        else:  
            print(red("The crossing goes badly."))  
            state["food"] -= random.randint(10, 25)  
            state["wagon"] -= random.randint(10, 20)  
            victim = random_living_member(state["party"])  
            damage_member(victim, random.randint(8, 25))  
  
    elif choice == "2":  
        if state["money"] < 10:  
            print(red("Not enough money for the ferry."))  
            print(yellow("You are forced to ford the river instead."))  
            success = 55  
            if random.randint(1, 100) <= success:  
                print(green("You crossed safely."))  
            else:  
                print(red("The river crossing goes badly."))  
                state["food"] -= random.randint(10, 25)  
                state["wagon"] -= random.randint(10, 20)  
                victim = random_living_member(state["party"])  
                damage_member(victim, random.randint(8, 25))  
        else:  
            state["money"] -= 10  
            print(green("The ferry gets you across safely."))  
  
    else:  
        print(yellow("You wait a day for better conditions."))  
        used = apply_food_cost(state, 4)  
        print(f"The party uses {used} food while waiting.")  
  
    state["wagon"] = clamp(state["wagon"], 0, 100)  
  
def fort_shop(state, fort_name):  
    print(cyan(f"\nWelcome to {fort_name}."))  
    while True:  
        print("\nShop menu:")  
        print("1. Buy 25 food for $10")  
        print("2. Buy 10 bullets for $5")  
        print("3. Buy 1 medkit for $15")  
        print("4. Repair wagon +20 for $12")  
        print("5. Rest at the fort (heal party) for $20") 
        print("6. Sell 25 food for for $5")  
        print("7. Leave")  
  
        choice = input("> ").strip()  
  
        if choice == "1":  
            if state["money"] >= 10:  
                state["money"] -= 10  
                state["food"] += 25  
                print(green("You bought 25 food."))  
            else:  
                print(red("Not enough money."))  
  
        elif choice == "2":  
            if state["money"] >= 5:  
                state["money"] -= 5  
                state["bullets"] += 10  
                print(green("You bought 10 bullets."))  
            else:  
                print(red("Not enough money."))  
  
        elif choice == "3":  
            if state["money"] >= 15:  
                state["money"] -= 15  
                state["medkits"] += 1  
                print(green("You bought 1 medkit."))  
            else:  
                print(red("Not enough money."))  
  
        elif choice == "4":  
            if state["money"] >= 12:  
                state["money"] -= 12  
                state["wagon"] = clamp(state["wagon"] + 20, 0, 100)  
                print(green("The wagon was repaired."))  
            else:  
                print(red("Not enough money."))  
  
        elif choice == "5":  
            if state["money"] >= 20:  
                state["money"] -= 20  
                for member in living_members(state["party"]):  
                    heal_member(member, 25)  
                print(green("The party rests and recovers health."))  
            else:  
                print(red("Not enough money."))  
  
        elif choice == "6":  
            if state["food"] >= 25:  
                state["food"] -= 25  
                state["money"] += 5
                print(green("You sold 25 food."))  
            else:  
                print(red("Not enough food."))
                
        elif choice == "7":  
            break  
  
        else:  
            print(red("Invalid choice."))  
  
def check_landmarks(state):  
    for landmark in LANDMARKS:  
        if state["miles"] >= landmark["mile"] and landmark["name"] not in state["visited_landmarks"]:  
            state["visited_landmarks"].append(landmark["name"])  
            print(magenta(f"\nLandmark reached: {landmark['name']}"))  
            print(landmark["text"])  
  
            if landmark["type"] == "river":  
                river_crossing(state, landmark["name"])  
  
            elif landmark["type"] == "fort":  
                fort_shop(state, landmark["name"])  
  
            elif landmark["type"] == "pass":  
                print(yellow("The steep climb strains the wagon."))  
                state["wagon"] -= 8  
  
            elif landmark["type"] == "goal":  
                return  
  
    state["wagon"] = clamp(state["wagon"], 0, 100)  
  
  
# ---------------------------  
# Actions  
# ---------------------------  
def travel(state):  
    move = random.randint(70, 120)  
  
    if state["wagon"] < 35:  
        move = int(move * 0.65)  
        print(yellow("Your damaged wagon slows the party down."))  
  
    if state["weather"] == "Rain":  
        move = int(move * 0.85)  
    elif state["weather"] == "Storm":  
        move = int(move * 0.70)  
  
    food_used = apply_food_cost(state, random.randint(3, 5))  
    state["wagon"] -= random.randint(4, 9)  
    state["miles"] += move  
  
    print(green(f"\nYou traveled {move} miles."))  
    print(f"The party used {food_used} food.")  
  
    random_travel_event(state)  
    check_landmarks(state)  
  
    state["wagon"] = clamp(state["wagon"], 0, 100)  
    state["day"] += 1  
  
def rest(state):  
    print("\nThe party rests for the day.")  
    used = apply_food_cost(state, 3)  
    for member in living_members(state["party"]):  
        heal_member(member, random.randint(8, 18))  
    state["wagon"] = clamp(state["wagon"] + 4, 0, 100)  
    print(green(f"The party recovers. Food used: {used}"))  
    state["day"] += 1  
  
def hunt(state):  
    print("\nYou go hunting.")  
    if state["bullets"] < 3:  
        print(red("Not enough bullets to hunt."))  
        return  
  
    print("1. Quick hunt")  
    print("2. Careful hunt")  
    choice = input("> ").strip()  
  
    if choice == "1":  
        state["bullets"] -= 3  
        if random.randint(1, 100) <= 60:  
            gained = random.randint(16, 35)  
            state["food"] += gained  
            print(green(f"You got some game. Food gained: {gained}"))  
        else:  
            print(yellow("You came back empty-handed."))  
  
    elif choice == "2":  
        state["bullets"] -= 5  
        if random.randint(1, 100) <= 80:  
            gained = random.randint(28, 60)  
            state["food"] += gained  
            print(green(f"Careful hunting paid off. Food gained: {gained}"))  
        else:  
            print(yellow("Even after waiting, you found nothing."))  
  
    else:  
        print(red("Invalid choice."))  
        return  
  
    state["day"] += 1  
  
def repair_wagon(state):  
    print("\nYou spend the day repairing the wagon.")  
    used = apply_food_cost(state, 2)  
    repair = random.randint(12, 26)  
    state["wagon"] = clamp(state["wagon"] + repair, 0, 100)  
    print(green(f"You repair the wagon by {repair}. Food used: {used}"))  
    state["day"] += 1  
  
def use_medkit(state):  
    if state["medkits"] <= 0:  
        print(red("\nYou have no medkits."))  
        return  
  
    alive = living_members(state["party"])  
    if not alive:  
        return  
  
    print("\nWho gets the medkit?")  
    for i, member in enumerate(alive, start=1):  
        print(f"{i}. {member['name']} (Health: {member['health']})")  
  
    choice = input("> ").strip()  
    if not choice.isdigit():  
        print(red("Invalid choice."))  
        return  
  
    idx = int(choice) - 1  
    if 0 <= idx < len(alive):  
        state["medkits"] -= 1  
        heal_member(alive[idx], 35)  
        print(green(f"{alive[idx]['name']} was treated."))  
        state["day"] += 1  
    else:  
        print(red("Invalid choice."))  
  
  
# ---------------------------  
# Endgame  
# ---------------------------  
def calculate_score(state):  
    survivors = len(living_members(state["party"]))  
    return (  
        state["miles"]  
        + survivors * 250  
        + state["food"]  
        + state["money"] * 2  
        + state["bullets"]  
        + state["medkits"] * 20  
        + state["wagon"]  
        - state["day"] * 4  
    )  
  
def game_over_checks(state):  
    if len(living_members(state["party"])) == 0:  
        print(red("\nEveryone in the wagon party has died."))  
        print(red("Game over."))  
        return True  
  
    if state["food"] <= 0:  
        print(red("\nYour party has run out of food."))  
        print(red("Game over."))  
        return True  
  
    if state["miles"] >= GOAL_DISTANCE:  
        print(green("\nYou made it to Oregon City!"))  
        score = calculate_score(state)  
        print(cyan(f"Final score: {score}"))  
        print(f"Survivors: {len(living_members(state['party']))}/{len(state['party'])}")  
        print(f"Landmarks reached: {len(state['visited_landmarks'])}")  
        return True  
  
    return False  
  
  
# ---------------------------  
# Main  
# ---------------------------  
def main():  
    print(cyan("OREGON TRAIL - POLISHED TEXT EDITION"))  
    print("1. New game")  
    print("2. Load game")  
  
    start = input("> ").strip()  
    if start == "2":  
        state = load_game()  
        if state is None:  
            print(red("No save file found. Starting new game."))  
            state = new_game()  
        else:  
            print(green("Game loaded."))  
    else:  
        state = new_game()  
  
    while True:  
        if game_over_checks(state):  
            break  
  
        show_status(state)  
        start_day_weather(state)  
  
        if game_over_checks(state):  
            break  
  
        print("\nChoose an action:")  
        print("1. Travel")  
        print("2. Rest")  
        print("3. Hunt")  
        print("4. Repair wagon")  
        print("5. Use medkit")  
        print("6. Save game")  
        print("7. Quit")  
  
        choice = input("> ").strip()  
  
        if choice == "1":  
            travel(state)  
            disease_event(state)  
            random_death_event(state)  
  
        elif choice == "2":  
            rest(state)  
            disease_event(state)  
  
        elif choice == "3":  
            hunt(state)  
            disease_event(state)  
  
        elif choice == "4":  
            repair_wagon(state)  
  
        elif choice == "5":  
            use_medkit(state)  
  
        elif choice == "6":  
            save_game(state)  
            continue  
  
        elif choice == "7":  
            print(yellow("\nYou quit the game."))  
            break  
  
        else:  
            print(red("Invalid choice."))  
  
    print("\nThanks for playing.")  
  
main()  
