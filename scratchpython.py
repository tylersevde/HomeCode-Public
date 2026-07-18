# Tiny Decision Game

print("You find two doors.")
print("One door is red.")
print("One door is blue.")

choice = input("Which door do you choose? ").lower()

if choice == "red":
    print("You found a sword.")
elif choice == "blue":
    print("You found a shield.")
