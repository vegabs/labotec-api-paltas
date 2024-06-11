import tkinter as tk
from tkinter import ttk

import sys
import os
from tkinter import *
from tkinter import messagebox,filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
from model_eval_la import *
from model_eval_la import predict_status

def assertlen(st, n):
    while(len(st)<n):
        st+=" "
    return st

class Application(tk.Frame):
    def __init__(self,master):
        super().__init__(master)
        self.pack()
        self.color_dict = {
            0:"red",
            1:"yellow",
            2:"green"
        }
        self.type_dict = {
            0:"Deficiencia",
            1:"Control",
            2:"Exceso"
        }
        self.master.geometry("1020x600")
        self.master.title("SENeHSoft")

        self.create_widgets()
        
    def create_widgets(self):
        #Canvas
        self.canvas1 = tk.Canvas(self)
        self.canvas1.configure(width=640, height=480, bg='white', selectforeground='black')
        #self.canvas1.create_rectangle(0,0,120, 70, fill='green')
        self.canvas1.grid(column=0, row=0, rowspan=2)
        self.canvas1.grid(padx=20, pady=20)
        self.canvas1.create_rectangle(2,2,640, 480, outline='gray')

        #Frame
        self.frame_button = ttk.LabelFrame(self)
        self.frame_button.configure(text='COMANDOS')
        self.frame_button.grid(column=0,row=2, columnspan=4)
        self.frame_button.grid(padx=20, pady=20)

        #Frame
        self.frame_pred = ttk.LabelFrame(self)
        self.frame_pred.configure(text='Confiabilidad de Análisis')
        self.frame_pred.grid(column=1,row=0, rowspan=1, columnspan=1)
        self.frame_pred.grid(padx=20, pady=10)
    
        self.frame_legend = ttk.LabelFrame(self)
        self.frame_legend.configure(text='Leyenda')
        self.frame_legend.grid(column=1,row=1, rowspan=1, columnspan=1)
        self.frame_legend.grid(padx=20, pady=10)


        #File open and Load Image
        self.button_open = ttk.Button(self.frame_button)
        self.button_open.configure(text = 'Cargar Imagen')
        self.button_open.grid(column=0, row=1)
        self.button_open.configure(command=self.loadImage)

        #Predict
        self.button_predict = ttk.Button(self.frame_button)
        self.button_predict.configure(text = 'Analizar Imagen')
        self.button_predict.grid(column=1, row=1)
        self.button_predict.configure(command=self.predictImage)

        # Clear Button
        self.button_clear = ttk.Button( self.frame_button )
        self.button_clear.configure( text='Limpiar Campos' )
        self.button_clear.grid( column=2, row=1 )
        self.button_clear.configure(command=self.clearImage)

        # Quit Button
        self.button_quit = ttk.Button( self.frame_button )
        self.button_quit.config( text='Salir' )
        self.button_quit.grid( column=3, row=1 )
        self.button_quit.configure(command = self.quit_app)

        #File open and Load Image
        self.button_n = Label(self.frame_pred)
        self.button_n.configure(text = 'Nitrógeno    ', relief='ridge')
        self.button_n.grid(column=0, row=0, sticky='w')

        self.button_n_ = Label(self.frame_pred)
        self.button_n_.configure(text = ' '*24, bg='gray', relief='ridge')
        self.button_n_.grid(column=1, row=0, sticky='w')
        #self.button_n.configure(command=self.loadImage)

        # Clear Button
        self.button_p = Label( self.frame_pred )
        self.button_p.configure( text='Fósforo        ', relief='ridge')
        self.button_p.grid(column=0, row=1, sticky='w')

        self.button_p_ = Label( self.frame_pred )
        self.button_p_.configure(text= ' '*24, bg='gray', relief='ridge')
        self.button_p_.grid(column=1, row=1, sticky='w')
        #self.button_p.configure(command=self.clearImage)

        # Quit Button
        self.button_k = Label( self.frame_pred )
        self.button_k.configure( text='Potasio         ', relief='ridge')
        self.button_k.grid(column=0, row=2, sticky='w')

        self.button_k_ = Label( self.frame_pred )
        self.button_k_.configure( text= ' '*24, bg='gray', relief='ridge')
        self.button_k_.grid(column=1, row=2, sticky='w')
        #self.button_k.configure(command = self.quit_app)
        #File open and Load Image
        self.button_h = Label(self.frame_pred)
        self.button_h.configure(text = 'Hídrico         ', relief='ridge')
        self.button_h.grid(column=0, row=3, sticky='w')

        self.button_h_ = Label(self.frame_pred)
        self.button_h_.configure(text = ' '*24, bg='gray', relief='ridge')
        self.button_h_.grid(column=1, row=3, sticky='w')
        #self.button_h.configure(command=self.loadImage)








        self.legend_ax_0 = Entry(self.frame_legend, width=18, font=('Arial',7,'bold'))
        self.legend_ax_0.insert(END, 'Código de colores')
        self.legend_ax_0.grid(column=0, row=0, sticky='w')

        self.legend_n_0 = Entry(self.frame_legend, width=16)
        self.legend_n_0.insert(END, 'Nitrógeno')
        self.legend_n_0.grid(column=0, row=1, sticky='w')

        self.legend_p_0 = Entry(self.frame_legend, width=16)
        self.legend_p_0.insert(END, 'Fósforo')
        self.legend_p_0.grid(column=0, row=2, sticky='w')

        self.legend_k_0 = Entry(self.frame_legend, width=16)
        self.legend_k_0.insert(END, 'Potasio')
        self.legend_k_0.grid(column=0, row=2, sticky='w')

        self.legend_h_0 = Entry(self.frame_legend, width=16)
        self.legend_h_0.insert(END, 'Hídrico')
        self.legend_h_0.grid(column=0, row=4, sticky='w')







        self.legend_ax_1 = Entry(self.frame_legend, width=10)
        self.legend_ax_1.insert(END, 'Deficiencia')
        self.legend_ax_1.grid(column=1, row=0, sticky='w')

        self.legend_n_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='red')
        self.legend_n_1.insert(END, 'Rojo')
        self.legend_n_1.grid(column=1, row=1, sticky='w')

        self.legend_p_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='red')
        self.legend_p_1.insert(END, 'Rojo')
        self.legend_p_1.grid(column=1, row=2, sticky='w')

        self.legend_k_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='red')
        self.legend_k_1.insert(END, 'Rojo')
        self.legend_k_1.grid(column=1, row=2, sticky='w')

        self.legend_h_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='red')
        self.legend_h_1.insert(END, 'Rojo')
        self.legend_h_1.grid(column=1, row=4, sticky='w')



        self.legend_ax_1 = Entry(self.frame_legend, width=10)
        self.legend_ax_1.insert(END, 'Control')
        self.legend_ax_1.grid(column=3, row=0, sticky='w')

        self.legend_n_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='yellow')
        self.legend_n_1.insert(END, 'Amarillo')
        self.legend_n_1.grid(column=3, row=1, sticky='w')

        self.legend_p_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='yellow')
        self.legend_p_1.insert(END, 'Amarillo')
        self.legend_p_1.grid(column=3, row=2, sticky='w')

        self.legend_k_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='yellow')
        self.legend_k_1.insert(END, 'Amarillo')
        self.legend_k_1.grid(column=3, row=2, sticky='w')

        self.legend_h_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='green')
        self.legend_h_1.insert(END, 'Verde')
        self.legend_h_1.grid(column=3, row=4, sticky='w')



        self.legend_ax_1 = Entry(self.frame_legend, width=10)
        self.legend_ax_1.insert(END, 'Exceso')
        self.legend_ax_1.grid(column=4, row=0, sticky='w')

        self.legend_n_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='green')
        self.legend_n_1.insert(END, 'Verde')
        self.legend_n_1.grid(column=4, row=1, sticky='w')

        self.legend_p_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='green')
        self.legend_p_1.insert(END, 'Verde')
        self.legend_p_1.grid(column=4, row=2, sticky='w')

        self.legend_k_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER, bg='green')
        self.legend_k_1.insert(END, 'Verde')
        self.legend_k_1.grid(column=4, row=2, sticky='w')

        self.legend_h_1 = Entry(self.frame_legend, width=10, justify=tk.CENTER)
        self.legend_h_1.insert(END, '-')
        self.legend_h_1.grid(column=4, row=4, sticky='w')

    # Event Call Back
    def loadImage(self):

        #self.folder_name = filedialog.askdirectory()
        self.filename = filedialog.askopenfilename()
        #print(self.folder_name)
        print(self.filename)

        self.image_bgr = cv2.imread(self.filename)
        self.height, self.width = self.image_bgr.shape[:2]
        #print(self.height, self.width)
        if self.width > self.height:
            self.new_size = (900,600)
        else:
            self.new_size = (600,900)

        self.image_bgr_resize = cv2.resize(self.image_bgr, self.new_size, interpolation=cv2.INTER_AREA)
        self.image_rgb = cv2.cvtColor( self.image_bgr_resize, cv2.COLOR_BGR2RGB )  #Since imread is BGR, it is converted to RGB

        #self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB) #Since imread is BGR, it is converted to RGB
        self.image_PIL = Image.fromarray(self.image_rgb) #Convert from RGB to PIL format
        self.image_tk = ImageTk.PhotoImage(self.image_PIL) #Convert to ImageTk format
        self.canvas1.create_image(375,250, image=self.image_tk)
        self.canvas1.create_rectangle(3,3,640, 480, outline='black', width=2)

    def clearImage(self):
        self.canvas1.delete("all")
        self.canvas1.create_rectangle(2,2,640, 480, outline='gray')
        self.button_n_.configure(text=" "*24, bg='gray')
        self.button_p_.configure(text=" "*24, bg='gray')
        self.button_k_.configure(text=" "*24, bg='gray')
        self.button_h_.configure(text=" "*24, bg='gray')
    
    def predictImage(self):
        print("Procesando")
        if self.filename:
            self.modelout = predict_status(self.filename)
            #print(self.modelout)
            n_tex = self.type_dict[int(list(self.modelout.keys())[0])] + " en "+str(round(float(list(self.modelout.values())[0]), 2))
            p_tex = self.type_dict[int(list(self.modelout.keys())[1])] + " en "+str(round(float(list(self.modelout.values())[1]), 2))
            k_tex = self.type_dict[int(list(self.modelout.keys())[2])] + " en "+str(round(float(list(self.modelout.values())[2]), 2))
            p_tex = assertlen(p_tex, 24)
            n_tex = assertlen(n_tex, 24)
            k_tex = assertlen(k_tex, 24)
            self.button_n_.configure(text=n_tex, bg=self.color_dict[int(list(self.modelout.keys())[0])], relief='sunken')
            self.button_p_.configure(text=p_tex, bg=self.color_dict[int(list(self.modelout.keys())[1])], relief='sunken')
            self.button_k_.configure(text=k_tex, bg=self.color_dict[int(list(self.modelout.keys())[2])], relief='sunken')
            haux = list(self.modelout.values())[-1]
            if haux>0.5:
                taux = "Control en " +str(haux)
                baux = "green"
            else:
                taux = "Deficiencia en " +str(round(1-haux, 2))
                baux = "red"
            taux = assertlen(taux, 24)
            self.button_h_.configure(text = taux, bg=baux, relief='sunken')

            print("-----")
            print(taux)
            print(baux)
            print("-----")
            print(n_tex)
            print(p_tex)
            print(k_tex)

    def quit_app(self):
        self.Msgbox = tk.messagebox.askquestion("Cerrar la aplicación", "¿Estas seguro?", icon="warning")
        if self.Msgbox == "yes":
            self.master.destroy()
        else:
            tk.messagebox.showinfo("Retorno", "Regresando a la pantalla principal")

def main():
    root = tk.Tk()
    app = Application(master=root)#Inherit
    app.mainloop()

if __name__ == "__main__":
    main()
