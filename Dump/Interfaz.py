import tkinter  
from tkinter.ttk import *  
import tkinter.messagebox 
from PIL import Image, ImageTk   
import time   
import random
import multiprocessing
from datetime import datetime

class adquisicion():
    
    def __init__(self):
        self.marcador = 0
        
    def set_marcador(self, val):
        self.marcador = val
        
    def get_marcador(self):
        print(str(self.marcador))
        
class BrainInterface():
        
    def __init__(self, ventana):  
        self.marcadores = []  
        self.ventana = ventana
        self.ventana.title("Interfaz de adquisición cerebral")
        self.ventana.configure(background="black")
        self.width = self.ventana.winfo_screenwidth()               
        self.height = self.ventana.winfo_screenheight()
        self.ventana.geometry('%dx%d+%d+%d' % (self.width,self.height,-10,0))
        self.s = Style()
        self.s.configure('Principal.TFrame', background='black')
        self.frame = Frame(self.ventana,style='Principal.TFrame')
        self.frame.pack()
        self.frame.place(height=self.height, width=self.width,x=-10,y=0)

        self.pie_der = ImageTk.PhotoImage(Image.open(r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Imágenes\F_PD.png"))
        self.pie_izq = ImageTk.PhotoImage(Image.open(r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Imágenes\F_PI.png"))
        self.brazo_der = ImageTk.PhotoImage(Image.open(r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Imágenes\F_BD.png"))
        self.brazo_izq = ImageTk.PhotoImage(Image.open(r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Imágenes\F_BI.png"))
        self.cruz = ImageTk.PhotoImage(Image.open(r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Imágenes\cruz.PNG"))
        self.black = ImageTk.PhotoImage(Image.open(r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Imágenes\Fondo.PNG"))

        self.imagen = tkinter.Label(self.frame)
        self.imagen.pack()
        self.imagen.place(relx = 0.5, rely = 0.5, anchor = 'center')   
        self.imagen.config(fg="white", bg="black")         
        
        self.var = tkinter.IntVar()
        self.button = tkinter.Button(self.ventana, text="Iniciar prueba", command=lambda: self.var.set(1)) 
        self.button.config(height=3, width=20, font=("bookman",20), fg="black", bg="white")
        self.button.place(relx=.5, rely=.5, anchor="center")          
        
        self.tiempo_inicial = 0
        self.tiempo = 0

    def finish_program(self):
        tkinter.messagebox.showinfo("Finalización","Recolección finalizada, muchas gracias por su colaboración")
        with open(r'Registros\1.txt', 'w') as writer:
            for marcador in self.marcadores:
                writer.write("%s\n" % marcador)     
        self.ventana.destroy()        

    def clear(self):
        print("Clear")
        self.imagen.config(image = self.black)
        self.imagen.update()

    def cross(self, callback):
        print("Cross")
        self.imagen.config(image = self.cruz)     
        self.imagen.update()
        self.ventana.after(random.randint(250, 450), callback)            
            
    def training(self):               
        self.tiempo_inicial = datetime.now()        
        self.button.place_forget()       
        print("Training")      
        self.ventana.after(3000, lambda:self.clear())       
        lista = [1,2,3,4]       
        self.ventana.after(1000, lambda:self.new_movement(lista, 0, 0))        
        
    def register(self): 
        self.imagen.config(text="Inicio del registro", font = ("bookman",20), fg="white", bg="black")
        self.imagen.update() 
        print("Register") 
        self.ventana.after(1000, lambda:self.clear())        
        lista = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
        random.shuffle(lista)
        self.ventana.after(1000, lambda:self.new_movement(lista, 0, 1))       
            
    def new_movement(self, lista, pos, fin):
        if pos < len(lista):
            i = pos    
            print("new_movement" + "(" + str(i) + ")")                          
            if lista[i] == 1:
                self.imagen.config(image=self.pie_der)    
                self.imagen.update()    
                self.tiempo = datetime.now()-self.tiempo_inicial
                self.marcadores.append(str(lista[i]) + " - " + str(self.tiempo))
            elif lista[i] == 2:
                self.imagen.config(image=self.pie_izq) 
                self.imagen.update()  
                self.tiempo = datetime.now()-self.tiempo_inicial     
                self.marcadores.append(str(lista[i]) + " - " + str(self.tiempo))
            elif lista[i] == 3:
                self.imagen.config(image=self.brazo_izq)  
                self.imagen.update()      
                self.tiempo = datetime.now()-self.tiempo_inicial
                self.marcadores.append(str(lista[i]) + " - " + str(self.tiempo))
            else:
                self.imagen.config(image=self.brazo_der)
                self.imagen.update()
                self.tiempo = datetime.now()-self.tiempo_inicial
                self.marcadores.append(str(lista[i]) + " - " + str(self.tiempo))
                
            self.ventana.after(200, self.clear)
            self.ventana.after(1800, lambda: self.cross(lambda:self.new_movement(lista, i+1, fin))) 
        else:
            if fin == 0:
                self.ventana.after(3000, lambda:self.register())
            else:
                self.ventana.after(1000, lambda:self.finish_program())
                
if __name__ == "__main__": 
    ventana = tkinter.Tk()     
    BI = BrainInterface(ventana)  
    ad = adquisicion()      
    BI.button.wait_variable(BI.var)
    p1 = multiprocessing.Process(target=BI.training(), args=(None,))    
    p2 = multiprocessing.Process(target=ad.get_marcador(), args=(None,))
    p2.start()        
    p1.start()    
    p1.join() 
    p2.join()     
    ventana.mainloop()  