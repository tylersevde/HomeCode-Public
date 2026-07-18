import random

secret = random.randint(1, 100)
guess = 0

print("Guess the number from 1 to 100!")

while guess != secret:
	guess = int(input("Enter your guess: "))
	
	if guess < secret:
		print("Too low!")
	elif guess > secret:
		print("Too high!")
	else:
		print("You got it!")
