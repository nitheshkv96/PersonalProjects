# -*- coding: utf-8 -*-
"""
CIS 553 (FALL 2023) PROJECT

GROUP: CODEFORMERS
-----------------------
| AI PAIR PROGRAMMING |
-----------------------

@author: Nithesh Veerappa

"""
import os
# import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from aiPairProgramming import AI_PAIR_PROGRAMMING


key= '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'

class GUI:
    def __init__(self):
        self.aiPairProgObj = AI_PAIR_PROGRAMMING()
        
        #Configuring GUI
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        
        # Initializing the Main GUI wondow
        self.root = ctk.CTk()
        self.root.title("AI Pair Programming Assistant")
        self.root.configure(bg='grey23')
        self.root.geometry("1280x720")
        self.root.resizable(False, False)
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        background_image_path = os.path.join(self.script_dir, "icons", "bg.jpg")
        background_image = ImageTk.PhotoImage(Image.open(background_image_path))
        
        
        background_label = ctk.CTkLabel(self.root, image=self.background_image,text="")
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        ## GUI: API key Generation
        btn_image_path = os.path.join(self.script_dir, "icons", "key.png")

        genkey_btn_image = ImageTk.PhotoImage(Image.open(btn_image_path).resize((30,30)))
        
        button = ctk.CTkButton(self.root, image = genkey_btn_image,
                               text=" Get API Key",
                               command = lambda: self.generateapikey(),
                               compound = 'left',
                               fg_color='#D35B58', hover_color = "#C77C78")
        button.pack(pady = 5, padx = 5)
        
        
        ## Set API Key
        setkey_btn_image_path = os.path.join(self.script_dir, "icons", "set_key.png")
        setkey_btn_image = ImageTk.PhotoImage(Image.open(setkey_btn_image_path).resize((30,30)))
        popup_button = ctk.CTkButton(self.root, text="Set API Key",
                                     image = setkey_btn_image,
                                     command= lambda: self.open_apikey_popup(),
                                     compound = 'left',
                                     )
        popup_button.pack(pady = 0)

                
        self.taskdropmenu()
        
        self.createframe()
        
    ## Functions related to API get-set   
    def generateapikey(self):
        self.aiPairProgObj.configureAPI(apiExists=False)
        
    def open_apikey_popup(self):
        self.create_apikey_popup()
        self.popup.grab_set()
        self.popup.deiconify()  

    def submit_button_popup_callback(self):
        self.aiPairProgObj.api_key = self.entry.get()
        
        if not self.aiPairProgObj.api_key:
            self.submit_button_popup.configure(text="Enter Key, then click here!")
            return
        
        self.aiPairProgObj.configureAPI(apiExists=True)          
        self.taskcombo.configure(state='normal')
        self.popup.withdraw()  
        self.root.focus_set() 
        self.popup.grab_release()
    
    def disable_event(self):
        return
        
    def create_apikey_popup(self):
        self.popup = ctk.CTkToplevel(self.root)
        
        self.popup.title("Set your API-Key")
        self.popup.configure(bg="midnight blue")
        self.popup.geometry("500x300")
        
        label = ctk.CTkLabel(self.popup, text="Enter Your API Key")
        label.pack(pady=10)
    
        self.entry = ctk.CTkEntry(self.popup)                                  
        self.entry.pack(pady=2)

        self.submit_button_popup = ctk.CTkButton(self.popup, text="Submit", command= self.submit_button_popup_callback)
        self.submit_button_popup.pack(pady=2)    
        
        self.popup.withdraw()
    
    ## Functions related to task selection
    def taskdropmenu(self):
        tasks = ["<Select Task>","Code Simplification","Code Completion",
                 "Code Improvisation","Code Debugging"]
        self.taskcombo = ctk.CTkComboBox(self.root, values = tasks, width = 160,hover = True,
                                         font = ('Italica', 13),
                                         dropdown_font= ('Italica', 13),
                                         command=lambda value: self.picktask(value),
                                         corner_radius = 10, justify='left',
                                         border_color = 'SteelBlue3',
                                         button_color = 'SteelBlue3')
        self.taskcombo.place(y = 90, x = 559)
        self.taskcombo.configure(state='disabled')

        
    ## Task Selection
    def picktask(self, value):
        if value == "<Select Task>":
            self.genout_btn.configure(state='disabled')
            return
        self.intext.configure(state='normal')
        self.genout_btn.configure(state='normal')
        self.task = value



    ## Creating Input frame
    def createframe(self):
        self.intext = ctk.CTkTextbox(self.root, width=600, height=500,
                                bg_color="transparent",
                                activate_scrollbars = True,
                                wrap = 'none', 
                                )
        self.intext.insert("0.0","***Please place your code here for the selected task***")
        # self.intext.pack(padx = 10, pady = 70,side = 'left')
        self.intext.place(x = 10, y = 160)
        self.intext.configure(state='disabled')
        
        self.genout_btn = ctk.CTkButton(self.root,text="Generate", command= self.getinputllm, 
                                   fg_color='#D35B58', hover_color = "#C77C78") 
        self.genout_btn.place(x = 210, y = 680)
        self.genout_btn.configure(state='disabled')

        
        # self.text_input_label = ctk.CTkLabel(self.root, text="*** Place your code snippet here ***",
        #                                justify='center')  
        # self.text_input_label.place(x = 150, y = 120)
        

        self.outtext = ctk.CTkTextbox(self.root, width=600, height=500,
                                 bg_color="transparent",
                                 activate_scrollbars = True,
                                 wrap = 'none')
        self.outtext.insert("0.0", "***Please wait for the Output***")
        # self.outtext.pack(padx = 10, pady = 70, side = 'right')
        self.outtext.place(x = 667, y = 160)
        self.outtext.configure(state='disabled')
        
        self.copy_btn = ctk.CTkButton(self.root,text="Copy to Clipboard", command = self.copy_to_clipboard, fg_color='#D35B58', hover_color = "#C77C78") 
        self.copy_btn.place(x = 900, y = 680)
        self.copy_btn.configure(state='disabled')

        
    def getinputllm(self):
        self.outtext.configure(state='normal')
        self.outtext.delete("0.0", "end")

        self.llminput = self.intext.get("0.0", "end")
        self.aiPairProgObj.promptEngg(task=self.task, code=self.llminput, provideDetails=True)
        self.aiPairProgObj.predict()
        if self.aiPairProgObj.generated_text:
            self.copy_btn.configure(state='normal')
        self.outtext.insert('0.0', self.aiPairProgObj.generated_text)

    
    def copy_to_clipboard(self):
        self.root.clipboard_clear()
        
        # Use the clipboard_append method to set the clipboard contents
        self.root.clipboard_append(self.aiPairProgObj.generated_text)
    
        # Update the clipboard
        self.root.update()
        
    

#%%
# if __name__ == "__main__":
gui = GUI()
gui.root.mainloop()

