#https://stackoverflow.com/questions/27863832/calling-python-2-script-from-python-3
#import Tkinter

import concurrent.futures as cf

import tkinter as tk
#import tkFileDialog as filedialog
from tkinter import filedialog
import numpy as np
from PIL import ImageTk, Image
import requests
import os
#import torchvision.models as models
#from model.model_b3_p import Reab3p16
import model_adjusted.model_b3_p as m
#import model.model_b3_p as m
import torch 
import os
import tensorflow as tf 
import argparse
import zipfile
from data.load_data import load_data,Dataset
from data import preprocess 
import config
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict

from inference_snippet_test import run_inference


# reference: https://stackoverflow.com/questions/41989813/is-it-possible-to-use-concurrent-futures-to-execute-a-function-method-inside-a-t
# reference: https://stackoverflow.com/questions/62593706/how-to-fix-tkinter-gui-freezing-during-long-calculation
class Inference(): 
    def __init__(self, RAVEN_folder, file_type, file):
        self.RAVEN_folder = RAVEN_folder
        self.file_type = file_type
        self.file = file
        
    def inference_instance(self): 
        with cf.ProcessPoolExecutor() as executor:
            r = executor.submit(run_inference, self.RAVEN_folder, self.file_type, self.file)
            ai_prediction = r.result()
        return ai_prediction
    


class GUI:
    def __init__(self):
        pass

    def update_choice(self, v):
        global choice
        choice= v.get()
        #print(v.get())

    def get_next_question(self, window, human_correct, human_total, target, v, RAVEN_folder, file, file_number, ai_correct, ai_total, data_split, next_button_text):
        
        
        next_button_text.set("Please wait for AI to decide...")
        window.update()
        print("File: ", file)
        
        # ----------- CONTEXT PANEL FRAME -----------
        context_panel_frame = tk.LabelFrame(window, font=('Helvatical bold', 24), text='Context Panels', height=400, width=580, bg='#e9f5ff', padx=10, pady=10)
        context_panel_frame.grid(row=0, column=0, padx=10, pady=10)
        
        inference_instance = Inference(RAVEN_folder, data_split, file)
        ai_prediction = inference_instance.inference_instance()
        ai_choice = ai_prediction+1
        #print("ai_choice:", ai_choice)

        npz_file = np.load(os.path.join(RAVEN_folder,file))
        npz_subfile_image = npz_file["image"]
        npz_subfile_target = npz_file["predict"]
        target = npz_subfile_target +1
        #print("correct answer:", target)

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

        
        
        #i_instance = Inference(RAVEN_folder, "train", file)    
        #ai_prediction = i_instance.inference_instance()
        #ai_choice = ai_prediction+1
        #print("ai_choice:", ai_choice)
        
        window.update()
        
        file_type = ["train", "test", "val"]
        file = "RAVEN_"+str(file_number)+"_"+data_split+".npz"
        
        #file_number = file_number+1
        #exists = False
        #for i in range(3):
        #    file = "RAVEN_"+str(file_number)+"_"+file_type[i]+".npz"
        #   file_folder = "RAVEN_"+str(file_number)+"_"+file_type[i]
        #    if os.path.exists(os.path.join(RAVEN_folder, file)):
        #        break

        file_number = file_number+1
        file = "RAVEN_"+str(file_number)+"_"+data_split+".npz"
        exists = os.path.exists(os.path.join(RAVEN_folder,file))
        temp_count = 0
        while not exists:
            temp_count = temp_count + 1
            if temp_count>1000:
                window.destroy()
            file_number = file_number+1
            file = "RAVEN_"+str(file_number)+"_"+data_split+".npz"
            exists = os.path.isfile(os.path.join(RAVEN_folder,file))
        
        #print("file:", file)    
        
        #file_number = file_number+1
        global choice
        choice = v
        #print("human choice:", choice)
        #print()
        
        #if target==choice:
        #   human_correct = human_correct + 1
        #human_total = human_total + 1
        
        #window.destroy()
        #window = tk.Tk()

        window.title('Visual IQ Test')
        window.geometry('1200x600')
        window.configure(bg='#e9f5ff')
        window.resizable(False, False)      # make window non-resizable



        
        
        # -model----
        ##--model testing


        
        

        # ----------- SCORE FRAME -----------

        ai_total=ai_total+1
        human_total = human_total+1
        if human_total!=0:
            if (choice==target):
                human_correct = human_correct+1
            human_score = human_correct/human_total
        else:
            human_score = ""
        if ai_total!=0:
            if (ai_choice == target):
                ai_correct=ai_correct+1
            ai_score = ai_correct/ai_total
        else:
            ai_score = ""
        # ----------- SCORE FRAME ----------- 



        score_frame = tk.LabelFrame(window, text='Scores', font=('Helvatical bold', 24), height=400, width=380, bg='#e9f5ff', padx=10, pady=10)
        score_frame.grid(row=0, column=1, padx=10, pady=10)


        human_score_f = human_score*100
        ai_score_f = ai_score*100

        score1 = tk.Label(score_frame, text = 'Human Score: '+str(f"{human_score_f:.2f}")+"%", bg='#e9f5ff', font=('Helvatical bold', 20))
        score1.place(x = 0, y = 5)

        score2 = tk.Label(score_frame, text = 'AI Score: '+str(f"{ai_score_f:.2f}")+"%", bg='#e9f5ff', font=('Helvatical bold', 20))
        score2.place(x = 0, y = 115)
        
        score_total_human = tk.Label(score_frame, text = 'Total Correct: '+str(human_correct), bg='#e9f5ff', font=('Helvatical bold', 20))
        score_total_human.place(x = 0, y = 35)
    
        score_total_ai = tk.Label(score_frame, text = 'Total Correct: '+str(ai_correct), bg='#e9f5ff', font=('Helvatical bold', 20))
        score_total_ai.place(x = 0, y = 145)
    
        score_total_done_human = tk.Label(score_frame, text = 'Total Problems: '+str(human_total), bg='#e9f5ff', font=('Helvatical bold', 20))
        score_total_done_human.place(x = 0, y = 65)
    
        score_total_done_ai = tk.Label(score_frame, text = 'Total Problems: '+str(ai_total), bg='#e9f5ff', font=('Helvatical bold', 20))
        score_total_done_ai.place(x = 0, y = 175)
        
        
        previous_question = tk.Label(score_frame, text = 'Previous Question:', bg='#e9f5ff', font=('Helvatical bold', 10))
        previous_question.place(x = 0, y = 225)
    
    
        score_human_choice = tk.Label(score_frame, text = 'Human Choice: '+str(v), bg='#e9f5ff', font=('Helvatical bold', 12))
        score_human_choice.place(x = 30, y = 245)
    
        score_ai_choice = tk.Label(score_frame, text = 'AI Choice: '+str(ai_choice.item()), bg='#e9f5ff', font=('Helvatical bold', 12))
        score_ai_choice.place(x = 230, y = 245)
        
        score_correct = tk.Label(score_frame, text = 'Correct Choice: '+str(target), bg='#e9f5ff', font=('Helvatical bold', 12))
        score_correct.place(x = 130, y = 265)

        #button = tk.Button(score_frame, text='Next question ->', font=('Helvatical bold', 18), highlightbackground='#E9F5FF', padx=5, pady=5, command=lambda: self.get_next_question(window, human_correct, human_total, target, choice, RAVEN_folder, file, file_number, ai_correct, ai_total, data_split))
        #button.place(x=0, y=300)
        
        print("Human Score: ", human_score*100)
        print("Ai Score: ", ai_score*100)
        print("Total Correct for human: ", human_correct)
        print("Total Correct for ai: ", ai_correct)
        print("Total for Human: ", human_total)
        print("Total for ai: ", ai_total)
        print("Human Choice: ", choice)
        print("Ai Choice: ", ai_choice.item())
        print("Correct Answer: ", target)

        print() 
        
        
        next_button_text = tk.StringVar()
        next_button_text.set("Next question ->")
        button = tk.Button(score_frame, textvariable=next_button_text, font=('Helvatical bold', 18), highlightbackground='#E9F5FF', padx=5, pady=5, command=lambda: gui.get_next_question(window, human_correct, human_total, target, choice, RAVEN_folder, file, file_number, ai_correct, ai_total, data_split, next_button_text))
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
                           command = lambda: self.update_choice(v),
                           bg = '#e9f5ff')
            radio_button.config(image = img)
            radio_button.pack(side = tk.LEFT, padx = (4, 10))
            number = tk.Label(choice_panel_frame, text=i+1, bg='#e9f5ff', font=('Helvatical bold', 9))
            number.place(x=70+(i*140), y=-17)
            if img_position != 16:
                img_position = img_position+1
                

        
        window.mainloop()
    
    
if __name__ == '__main__':
        
    
    #main
    #--------------------------------------
    window = tk.Tk()

    window.title('Visual IQ Test')
    window.geometry('1200x600')
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
    #file_number = 1368
    file_number = 0
    data_split = "val"
    file = "RAVEN_"+str(file_number)+"_"+data_split+".npz"

    exists = os.path.exists(os.path.join(RAVEN_folder,file))
    while not exists:
        file_number = file_number+1
        file = "RAVEN_"+str(file_number)+"_"+data_split+".npz"
        exists = os.path.isfile(os.path.join(RAVEN_folder,file))
        
    #file = "RAVEN_"+str(file_number)+"_test.npz"

    #file_number = file_number+1
    npz_file = np.load(os.path.join(RAVEN_folder,file))
    npz_subfile_image = npz_file["image"]
    npz_subfile_target = npz_file["target"]
    target = npz_subfile_target+1
    #target = npz_subfile_target
    #print("correct answer:", target)

    #https://bugs.python.org/issue11077
    #https://stackoverflow.com/questions/41989813/is-it-possible-to-use-concurrent-futures-to-execute-a-function-method-inside-a-t
    
    #i_instance = Inference(RAVEN_folder, "train", file)    
    #ai_prediction = i_instance.inference_instance()
    #ai_choice = ai_prediction+1
    #print("ai_choice:", ai_choice)

    # ai_prediction = run_inference(RAVEN_folder, "train",  file)
    # ai_prediction =1
    #print("file:", file)

    with zipfile.ZipFile(os.path.join(RAVEN_folder, file), 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

    file_folder = "RAVEN_"+str(file_number)+"_"+data_split
    file_path = os.path.join(os.getcwd(), file_folder)
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


    ##--model testing


    model_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epoch100"

    #image_path = "C:\\Users\\sonam\\Desktop\\MS_Project\\test_model\\RAVEN_1368_test\\image.npy"
    image_path = file_folder


    args, unparsed = config.get_args()





    # ----------- SCORE FRAME ----------- 

    #ai_total=ai_total+1
    #if human_total!=0:
    #   if (choice==target):
    #       human_correct = human_correct+1
    #   human_score = human_correct/human_total
    #else:
    #   human_score = ""
    #if ai_total!=0:
    #   if (ai_choice == target):
    #       ai_correct=ai_correct+1
    #   ai_score = ai_correct/ai_total
    #else:
    #   ai_score = ""

    human_score=0
    ai_score = 0
    
    score_frame = tk.LabelFrame(window, text='Scores', font=('Helvatical bold', 24), height=400, width=380, bg='#e9f5ff', padx=10, pady=10)
    score_frame.grid(row=0, column=1, padx=10, pady=10)




    score1 = tk.Label(score_frame, text = 'Human Score: '+str(human_score*100)+"%", bg='#e9f5ff', font=('Helvatical bold', 20))
    score1.place(x = 0, y = 5)

    score2 = tk.Label(score_frame, text = 'AI Score: '+str(ai_score*100)+"%", bg='#e9f5ff', font=('Helvatical bold', 20))
    score2.place(x = 0, y = 115)
    
    score_total_human = tk.Label(score_frame, text = 'Total Correct: '+str(human_correct), bg='#e9f5ff', font=('Helvatical bold', 20))
    score_total_human.place(x = 0, y = 35)
    
    score_total_ai = tk.Label(score_frame, text = 'Total Correct: '+str(ai_correct), bg='#e9f5ff', font=('Helvatical bold', 20))
    score_total_ai.place(x = 0, y = 145)
    
    score_total_done_human = tk.Label(score_frame, text = 'Total Problems: '+str(human_total), bg='#e9f5ff', font=('Helvatical bold', 20))
    score_total_done_human.place(x = 0, y = 65)
    
    score_total_done_ai = tk.Label(score_frame, text = 'Total Problems: '+str(ai_total), bg='#e9f5ff', font=('Helvatical bold', 20))
    score_total_done_ai.place(x = 0, y = 175)
    
    

    gui = GUI()
    
    next_button_text = tk.StringVar()
    next_button_text.set("Next question ->")
    button = tk.Button(score_frame, textvariable=next_button_text, font=('Helvatical bold', 18), highlightbackground='#E9F5FF', padx=5, pady=5, command=lambda: gui.get_next_question(window, human_correct, human_total, target, choice, RAVEN_folder, file, file_number, ai_correct, ai_total, data_split, next_button_text))
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
                       command =lambda: gui.update_choice(v),
                       bg = '#e9f5ff')
        radio_button.config(image = img)
        radio_button.pack(side = tk.LEFT, padx = (4, 10))
        number = tk.Label(choice_panel_frame, text=i+1, bg='#e9f5ff', font=('Helvatical bold', 9))
        number.place(x=70+(i*140), y=-17)
        if img_position != 16:
            img_position = img_position+1
            
    #i = IntVar() #Basically Links Any Radiobutton With The Variable=i.
    
    print()
    print()

    # window.mainloop()
    window.mainloop()
