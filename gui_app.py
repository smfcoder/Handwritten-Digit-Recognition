"""
Created on Mon Aug 24 17:31:46 2020

@author: Marc
"""
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import Image, ImageGrab
import numpy as np

model = load_model('final_classifier.h5')

def predict_digit(img):
    # Resizing image to 28x28 pixels
    img = img.resize((28, 28))
    # Converting RGB to GRAYSCALE
    img = img.convert('L')
    img.save('image.png')
    img = np.array(img)
    # Reshaping to support our model input and Normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # Predicting the digit
    result = model.predict([img])[0]
    return np.argmax(result), max(result)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating the elements
        self.canvas = tk.Canvas(self, width = 300, height = 300, bg = 'black', cursor = 'cross')
        self.label = tk.Label(self, text = 'Thinking...', font = ('Helvetica', 48))
        self.classify_btn = tk.Button(self, text = 'Recognise', command = self.classify_handwritting)
        self.clear_btn = tk.Button(self, text = 'Clear', command = self.clear_all)
        # Creating the grid structure
        self.canvas.grid(row = 0, column = 0, pady = 2, sticky = W)
        self.label.grid(row = 0, column = 1, pady = 2, padx = 2)
        self.classify_btn.grid(row = 1, column = 1, pady = 2, padx = 2)
        self.clear_btn.grid(row = 1, column = 0, pady = 2)
        self.canvas.bind('<B1-Motion>', self.draw_lines)
    
    def clear_all(self):
        self.canvas.delete('all')
    
    def classify_handwritting(self):
        handle = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(handle)
        img = ImageGrab.grab(rect)
        
        digit, acc = predict_digit(img)
        self.label.configure(text = str(digit) + ', ' + str(int(acc * 100)) + '%')
        
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill = 'white', outline = 'white')
    
app = App()
mainloop()
