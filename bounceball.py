#   
import tkinter as tk  
  
WIDTH = 600  
HEIGHT = 400  
  
ball_x = 300  
ball_y = 200  
ball_dx = 4  
ball_dy = 4  
ball_size = 50 
  
paddle_x = 250  
paddle_y = 360  
paddle_width =  150
paddle_height = 15  
paddle_speed = 10
  
score = 0  
left_pressed = False  
right_pressed = False  
game_over = False  
  
root = tk.Tk()  
root.title("Bouncing Ball Game")  
  
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")  
canvas.pack()  
  
score_text = canvas.create_text(60, 20, text="Score: 0", fill="white", font=("Arial", 16))  
  
def key_press(event):  
    global left_pressed, right_pressed  
    if event.keysym == "Left":  
        left_pressed = True  
    elif event.keysym == "Right":  
        right_pressed = True  
  
def key_release(event):  
    global left_pressed, right_pressed  
    if event.keysym == "Left":  
        left_pressed = False  
    elif event.keysym == "Right":  
        right_pressed = False  
  
def draw():  
    canvas.delete("ball")  
    canvas.delete("paddle")  
  
    canvas.create_oval(  
        ball_x, ball_y,  
        ball_x + ball_size, ball_y + ball_size,  
        fill="red", tags="ball"  
    )  
  
    canvas.create_rectangle(  
        paddle_x, paddle_y,  
        paddle_x + paddle_width, paddle_y + paddle_height,  
        fill="white", tags="paddle"  
    )  
  
def update():  
    global ball_x, ball_y, ball_dx, ball_dy  
    global paddle_x, score, game_over  
  
    if game_over:  
        return  
  
    if left_pressed and paddle_x > 0:  
        paddle_x -= paddle_speed  
    if right_pressed and paddle_x + paddle_width < WIDTH:  
        paddle_x += paddle_speed  
  
    ball_x += ball_dx  
    ball_y += ball_dy  
  
    if ball_x <= 0 or ball_x + ball_size >= WIDTH:  
        ball_dx = -ball_dx  
  
    if ball_y <= 0:  
        ball_dy = -ball_dy  
  
    if (  
        ball_y + ball_size >= paddle_y  
        and ball_y + ball_size <= paddle_y + paddle_height  
        and ball_x + ball_size >= paddle_x  
        and ball_x <= paddle_x + paddle_width  
    ):  
        ball_dy = -ball_dy  
        score += 1  
        canvas.itemconfig(score_text, text=f"Score: {score}")  
  
    if ball_y > HEIGHT:  
        game_over = True  
        canvas.create_text(  
            WIDTH // 2, HEIGHT // 2,  
            text="GAME OVER",  
            fill="yellow",  
            font=("Arial", 24)  
        )  
        return  
  
    draw()  
    root.after(20, update)  
  
root.bind("<KeyPress>", key_press)  
root.bind("<KeyRelease>", key_release)  
  
draw()  
update()  
root.mainloop()  
