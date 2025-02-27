from PIL import Image, ImageTk  
from datetime import datetime  
import tkinter.messagebox 
from tkinter.ttk import * 
import multiprocessing
import subprocess
import threading
import tkinter  
import random
import time
import csv
import sys
import os

class adquisicion():
    
    def __init__(self):
        self.marcador = 0
        
    def set_marcador(self, val):
        self.marcador = val
        
    def get_marcador(self):
        #print(str(self.marcador))
        self.marcador = self.marcador
        
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

        self.tiempo = 0
        self.formatted_time = 0
        
        self.datos = [];        
        
        self.proceso_adqui = 0
        
        self.ruta_grafica = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\FFT_micro.py"                
        self.ruta_datos = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Datos EEG"
        self.ruta_estimulos = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Aparición imagenes"
        self.ruta_SVM = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM characteristics"
            
    def monitor_child(self):
        exitcode = self.proceso_adqui.wait()
        if exitcode == 0:
            self.ventana.destroy()
    
    
    def merge_csv_files(self, file1, file2, output_file, time_format='%Y-%m-%d %H:%M:%S.%f'):        
        
        stimuli = []
        with open(file1, newline='', encoding='utf-8') as f1:
            reader1 = csv.reader(f1, delimiter=';')
            header1 = next(reader1, None) 
            for row in reader1:
                if len(row) < 2:
                    continue
                time_str = row[0].strip()
                stim_type = row[1].strip()
                try:
                    stim_time = datetime.strptime(time_str, time_format)
                    stimuli.append((stim_time, stim_type))
                except Exception as e:
                    print(f"Error al parsear tiempo del estímulo: '{time_str}'. Error: {e}")

        stimuli.sort(key=lambda x: x[0])
        
        with open(file2, newline='', encoding='utf-8') as f2, \
            open(output_file, mode='w', newline='', encoding='utf-8') as fout:
            reader2 = csv.reader(f2, delimiter=';')
            header2 = next(reader2, None)
            writer = csv.writer(fout, delimiter=';')
            
            output_header = ["Estimulo", "C1", "C2", "C3", "C4", "C5"]
            writer.writerow(output_header)
            
            for row in reader2:
                if len(row) < 7:
                    continue
                start_str = row[0].strip()
                end_str = row[1].strip()
                try:
                    start_time = datetime.strptime(start_str, time_format)
                    end_time = datetime.strptime(end_str, time_format)
                except Exception as e:
                    print(f"Error al parsear tiempos en archivo2: '{start_str}' o '{end_str}'. Error: {e}")
                    continue
                
                found_stimulus = "0"
                for stim_time, stim_type in stimuli:
                    if start_time <= stim_time <= end_time:
                        found_stimulus = stim_type
                        break
                
                channel_features = row[2:7]
                writer.writerow([found_stimulus] + channel_features)
            
    def generar_nombre_autoincremental(self, directorio, base_nombre, creacion):        
        if not os.path.exists(directorio):
            print("No directory")
            os.makedirs(directorio)
            
        archivos = [os.path.splitext(f)[0] for f in os.listdir(directorio) if f.startswith(base_nombre) and os.path.splitext(f)[0][len(base_nombre):].lstrip("_").isdigit()]         
        if archivos:
            numeros_existentes = [int(f[len(base_nombre):].lstrip("_")) for f in archivos]
            if creacion:
                nuevo_numero = max(numeros_existentes) + 1
            else:
                nuevo_numero = max(numeros_existentes)
        else:
            nuevo_numero = 1

        return os.path.join(directorio, f"{base_nombre}_{nuevo_numero}.csv")
                
    def finish_program(self):
        self.proceso_adqui.terminate()
        tkinter.messagebox.showinfo("Finalización","Recolección finalizada, muchas gracias por su colaboración")        
        #Guarda el archivo con las instrucciones y los tiempos de aparición 
        with open(self.generar_nombre_autoincremental(self.ruta_estimulos, "Instrucs", True), mode='w', newline='') as file:            
            writer = csv.writer(file, delimiter=';')            
            writer.writerow(['Time', 'Instruction'])
            writer.writerows(self.datos) 
            
        self.merge_csv_files(self.generar_nombre_autoincremental(self.ruta_estimulos, "Instrucs", False), self.generar_nombre_autoincremental(self.ruta_datos, "Caracs", False), self.generar_nombre_autoincremental(self.ruta_SVM, "SVM", True))
        print("Files created and storaged, register done succesfully!")  
        self.ventana.destroy()

    def clear(self):        
        self.imagen.config(image = self.black)
        self.imagen.update()

    def cross(self, callback):        
        self.imagen.config(image = self.cruz)     
        self.imagen.update()
        self.ventana.after(random.randint(250, 450), callback)            
            
    def training(self):           
        #Correr el código de python de la graficación 
        self.proceso_adqui = subprocess.Popen(["python", self.ruta_grafica])
        monitor_thread = threading.Thread(target=self.monitor_child, daemon=True)
        monitor_thread.start()
              
        print("Reading training data...")       
        #Iniciar el proceso de envío de instrucciones            
        self.button.place_forget()                     
        self.ventana.after(3000, lambda:self.clear())       
        lista = [1,2,3,4]       
        self.ventana.after(1000, lambda:self.new_movement(lista, 0, 0))        
        
    def register(self): 
        self.imagen.config(text="Inicio del registro", font = ("bookman",20), fg="white", bg="black")
        self.imagen.update()         
        self.ventana.after(1000, lambda:self.clear())        
        lista = [1,1]
        random.shuffle(lista)
        print("Reading test data...")        
        self.ventana.after(1000, lambda:self.new_movement(lista, 0, 1))       
            
    def new_movement(self, lista, pos, fin):
        if pos < len(lista):
            i = pos           
            self.tiempo = datetime.now()                          
            if lista[i] == 1:
                self.imagen.config(image=self.pie_der)    
                self.imagen.update()                                                
                self.formatted_time = self.tiempo.strftime("%Y-%m-%d %H:%M:%S.%f")
                self.datos.append((str(self.formatted_time),str(lista[i])))
            elif lista[i] == 2:
                self.imagen.config(image=self.pie_izq) 
                self.imagen.update()  
                self.formatted_time = self.tiempo.strftime("%Y-%m-%d %H:%M:%S.%f")     
                self.datos.append((str(self.formatted_time),str(lista[i])))
            elif lista[i] == 3:
                self.imagen.config(image=self.brazo_izq)  
                self.imagen.update()      
                self.formatted_time = self.tiempo.strftime("%Y-%m-%d %H:%M:%S.%f")
                self.datos.append((str(self.formatted_time),str(lista[i])))
            else:
                self.imagen.config(image=self.brazo_der)
                self.imagen.update()
                self.formatted_time = self.tiempo.strftime("%Y-%m-%d %H:%M:%S.%f")
                self.datos.append((str(self.formatted_time),str(lista[i])))
                
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
    