import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import ImageTk, Image
import requests
import os


def update_choice(v):
    global choice
    choice= v.get()
    print(v.get())

def get_next_question(window, human_correct, human_total, target, v, RAVEN_folder, file, file_number):
    file_type = ["train", "test", "val"]
    
    file = "RAVEN_"+str(file_number)+"_"+file_type[0]+".npz"
    
    exists = False
    for i in range(3):
        file = "RAVEN_"+str(file_number)+"_"+file_type[i]+".npz"
        if os.path.exists(os.path.join(RAVEN_folder, file)):
            break
        
    
    file_number = file_number+1
    global choice
    choice = v
    print(choice)
    if target==choice:
        human_correct = human_correct + 1
    human_total = human_total + 1
    
    window.destroy()
 
    window = tk.Tk()

    window.title('Visual IQ Test')
    window.geometry('1000x600')
    window.configure(bg='#e9f5ff')
    window.resizable(False, False)      # make window non-resizable

    ai_total = 0
    ai_correct = 0



    # ----------- CONTEXT PANEL FRAME -----------
    context_panel_frame = tk.LabelFrame(window, font=('Helvatical bold', 24), text='Context Panels', height=400, width=580, bg='#e9f5ff', padx=10, pady=10)
    context_panel_frame.grid(row=0, column=0, padx=10, pady=10)
    
    npz_file = np.load(os.path.join(RAVEN_folder,file))
    npz_subfile_image = npz_file["image"]
    npz_subfile_target = npz_file["target"]
    target = npz_subfile_target
    print(target)

    #---------------------------------------------

    # get example image in place of context panels
    #    img_url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/Raven_Progressive_Matrix.jpg'
    #    img = Image.open(requests.get(img_url, stream=True).raw)
    #    img = img.resize((100, 100), Image.ANTIALIAS)
    #    img = ImageTk.PhotoImage(img)

    # the positions to place the 8 context panels
    x_positions = [0, 180, 360]
    y_positions = [0, 120, 240]


    img_position = 0
    images = np.empty(16, dtype=object)
    for y in range(3):
        for x in range(3):
            img = Image.fromarray(npz_subfile_image[img_position])
            img.save(str(img_position)+".jpg")
            img = Image.open(os.path.join(os.getcwd(),str(img_position)+".jpg"))
            img = img.resize((100, 100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            images[img_position] = img
            # if last position (bottom-right), put question mark
            if(x == 2 and y == 2):
                label = tk.Label(context_panel_frame, text='?', font=('Helvatical bold', 24), bg='#e9f5ff')
                label.place(x = 400, y = 270)
            else:
                img_panel = tk.Label(context_panel_frame, image=images[img_position])
                img_panel.place(x = x_positions[x], y = y_positions[y])
            img_position = img_position+1
    img_position = img_position-1            

    # ----------- SCORE FRAME ----------- 
    if human_total!=0:
        human_score = human_correct/human_total
    else:
        human_score = ""
    if ai_total!=0:
        ai_score = ai_correct/ai_total
    else:
        ai_score = ""


    score_frame = tk.LabelFrame(window, text='Scores', font=('Helvatical bold', 24), height=400, width=380, bg='#e9f5ff', padx=10, pady=10)
    score_frame.grid(row=0, column=1, padx=10, pady=10)




    score1 = tk.Label(score_frame, text = 'Human: '+str(human_score), bg='#e9f5ff', font=('Helvatical bold', 20))
    score1.place(x = 0, y = 5)

    score2 = tk.Label(score_frame, text = 'AI: '+str(ai_score), bg='#e9f5ff', font=('Helvatical bold', 20))
    score2.place(x = 0, y = 35)

    button = tk.Button(score_frame, text='Next question ->', font=('Helvatical bold', 18), highlightbackground='#E9F5FF', padx=5, pady=5, command=lambda: get_next_question(window, human_correct, human_total, target, choice, RAVEN_folder, file, file_number))
    button.place(x=0, y=300)


    # ----------- CHOICE PANEL FRAME -----------
    choice_panel_frame = tk.LabelFrame(window, text='Choice Panels', font=('Helvatical bold', 24), height=260, width=980, bg='#e9f5ff', padx=10, pady=10)
    choice_panel_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    # choices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    v = tk.IntVar()     # holds the value of the currently selected choice panel
#    v.set(1)            # default to choice 1

    # get example choice panel
    #img_url2 = 'https://www.researchgate.net/profile/Steven-Thorne/publication/222416020/figure/fig4/AS:767804156411904@1560070184385/Problem-illustrating-the-Ravens-Progressive-Matrices-Test.png'
    #img2 = Image.open(requests.get(img_url2, stream=True).raw)
    #img2 = img2.resize((80, 80), Image.ANTIALIAS)
    #img2 = ImageTk.PhotoImage(img2)

    for i in range(8):
        img = Image.fromarray(npz_subfile_image[img_position])
        img.save(str(img_position)+".jpg")
        img= Image.open(os.path.join(os.getcwd(),str(img_position)+".jpg"))
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        images[img_position] = img
        radio_button = tk.Radiobutton(choice_panel_frame, 
                    #    text=choices[i],
                       padx = 20, 
                       variable = v, 
                       value = i+1,
                       command = lambda: update_choice(v),
                       bg = '#e9f5ff')
        radio_button.config(image = img)
        radio_button.pack(side = tk.LEFT, padx = (4, 10))
        if img_position != 16:
            img_position = img_position+1


    window.mainloop()
    
    
    
#main
#--------------------------------------
window = tk.Tk()

window.title('Visual IQ Test')
window.geometry('1000x600')
window.configure(bg='#e9f5ff')
window.resizable(False, False)      # make window non-resizable

human_total = 0
human_correct = 0
ai_total = 0
ai_correct = 0

choice = 0

# ----------- CONTEXT PANEL FRAME -----------
context_panel_frame = tk.LabelFrame(window, font=('Helvatical bold', 24), text='Context Panels', height=400, width=580, bg='#e9f5ff', padx=10, pady=10)
context_panel_frame.grid(row=0, column=0, padx=10, pady=10)

#------------ OBTAIN RAVEN PROBLEM ----------
# to be picked by GUI in the future
#RAVEN_folder = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\RAVEN-10000-release\\RAVEN-10000\\center_single"
RAVEN_folder = filedialog.askdirectory()
# to parse through one by one in the future
file_number = 0
file = "RAVEN_"+str(file_number)+"_train.npz"
file_number = file_number+1
npz_file = np.load(os.path.join(RAVEN_folder,file))
npz_subfile_image = npz_file["image"]
npz_subfile_target = npz_file["target"]
target = npz_subfile_target+1
print(target)

#---------------------------------------------


# the positions to place the 8 context panels
x_positions = [0, 180, 360]
y_positions = [0, 120, 240]


img_position = 0
images = np.empty(16, dtype=object)
for y in range(3):
    for x in range(3):
        img = Image.fromarray(npz_subfile_image[img_position])
        img.save(str(img_position)+".jpg")
        img = Image.open(os.path.join(os.getcwd(),str(img_position)+".jpg"))
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        images[img_position] = img
        # if last position (bottom-right), put question mark
        if(x == 2 and y == 2):
            label = tk.Label(context_panel_frame, text='?', font=('Helvatical bold', 24), bg='#e9f5ff')
            label.place(x = 400, y = 270)
        else:
            img_panel = tk.Label(context_panel_frame, image=images[img_position])
            img_panel.place(x = x_positions[x], y = y_positions[y])
        img_position = img_position+1
img_position = img_position-1            

# ----------- SCORE FRAME ----------- 
if human_total!=0:
    human_score = human_correct/human_total
else:
    human_score = ""
if ai_total!=0:
    ai_score = ai_correct/ai_total
else:
    ai_score = ""


score_frame = tk.LabelFrame(window, text='Scores', font=('Helvatical bold', 24), height=400, width=380, bg='#e9f5ff', padx=10, pady=10)
score_frame.grid(row=0, column=1, padx=10, pady=10)




score1 = tk.Label(score_frame, text = 'Human: '+str(human_score), bg='#e9f5ff', font=('Helvatical bold', 20))
score1.place(x = 0, y = 5)

score2 = tk.Label(score_frame, text = 'AI: '+str(ai_score), bg='#e9f5ff', font=('Helvatical bold', 20))
score2.place(x = 0, y = 35)

button = tk.Button(score_frame, text='Next question ->', font=('Helvatical bold', 18), highlightbackground='#E9F5FF', padx=5, pady=5, command=lambda: get_next_question(window, human_correct, human_total, target, choice, RAVEN_folder, file, file_number))
button.place(x=0, y=300)


# ----------- CHOICE PANEL FRAME -----------
choice_panel_frame = tk.LabelFrame(window, text='Choice Panels', font=('Helvatical bold', 24), height=260, width=980, bg='#e9f5ff', padx=10, pady=10)
choice_panel_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# choices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

v = tk.IntVar()     # holds the value of the currently selected choice panel
#v.set(1)            # default to choice 1



for i in range(8):
    img = Image.fromarray(npz_subfile_image[img_position])
    img.save(str(img_position)+".jpg")
    img= Image.open(os.path.join(os.getcwd(),str(img_position)+".jpg"))
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    images[img_position] = img
    radio_button = tk.Radiobutton(choice_panel_frame, 
                #    text=choices[i],
                   padx = 20, 
                   variable = v, 
                   value = i+1,
                   command =lambda: update_choice(v),
                   bg = '#e9f5ff')
    radio_button.config(image = img)
    radio_button.pack(side = tk.LEFT, padx = (4, 10))
    if img_position != 16:
        img_position = img_position+1


window.mainloop()