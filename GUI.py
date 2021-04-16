import tkinter as tk
from PIL import ImageTk, Image
import requests

window = tk.Tk()
window.title('Visual IQ Test')
window.geometry('1000x600')
window.configure(bg='#e9f5ff')
window.resizable(False, False)      # make window non-resizable


# ----------- CONTEXT PANEL FRAME -----------
context_panel_frame = tk.LabelFrame(window, font=('Helvatical bold', 24), text='Context Panels', height=400, width=580, bg='#e9f5ff', padx=10, pady=10)
context_panel_frame.grid(row=0, column=0, padx=10, pady=10)

# get example image in place of context panels
img_url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/Raven_Progressive_Matrix.jpg'
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((100, 100), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)

# the positions to place the 8 context panels
x_positions = [0, 180, 360]
y_positions = [0, 120, 240]

for y in range(3):
    for x in range(3):
        # if last position (bottom-right), put question mark
        if(x == 2 and y == 2):
            label = tk.Label(context_panel_frame, text='?', font=('Helvatical bold', 24), bg='#e9f5ff')
            label.place(x = 400, y = 270)
        else:
            img_panel = tk.Label(context_panel_frame, image=img)
            img_panel.place(x = x_positions[x], y = y_positions[y])
            

# ----------- SCORE FRAME ----------- 
score_frame = tk.LabelFrame(window, text='Scores', font=('Helvatical bold', 24), height=400, width=380, bg='#e9f5ff', padx=10, pady=10)
score_frame.grid(row=0, column=1, padx=10, pady=10)

def get_next_question():
    print('getting next question ...')

score1 = tk.Label(score_frame, text = 'Human: ', bg='#e9f5ff', font=('Helvatical bold', 20))
score1.place(x = 0, y = 5)
  
score2 = tk.Label(score_frame, text = 'AI: ', bg='#e9f5ff', font=('Helvatical bold', 20))
score2.place(x = 0, y = 35)

button = tk.Button(score_frame, text='Next question ->', font=('Helvatical bold', 18), highlightbackground='#E9F5FF', padx=5, pady=5, command=get_next_question)
button.place(x=0, y=300)


# ----------- CHOICE PANEL FRAME -----------
choice_panel_frame = tk.LabelFrame(window, text='Choice Panels', font=('Helvatical bold', 24), height=260, width=980, bg='#e9f5ff', padx=10, pady=10)
choice_panel_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# choices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

v = tk.IntVar()     # holds the value of the currently selected choice panel
v.set(1)            # default to choice 1

def update_choice():
    print(v.get())

# get example choice panel
img_url2 = 'https://www.researchgate.net/profile/Steven-Thorne/publication/222416020/figure/fig4/AS:767804156411904@1560070184385/Problem-illustrating-the-Ravens-Progressive-Matrices-Test.png'
img2 = Image.open(requests.get(img_url2, stream=True).raw)
img2 = img2.resize((80, 80), Image.ANTIALIAS)
img2 = ImageTk.PhotoImage(img2)

for i in range(8):
    radio_button = tk.Radiobutton(choice_panel_frame, 
                #    text=choices[i],
                   padx = 20, 
                   variable = v, 
                   value = i+1,
                   command = update_choice,
                   bg = '#e9f5ff')
    radio_button.config(image = img2)
    radio_button.pack(side = tk.LEFT, padx = (4, 10))


window.mainloop()