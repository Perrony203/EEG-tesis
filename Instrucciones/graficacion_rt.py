import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket
import os
import subprocess
import signal
import sys
import threading
import time

# Parámetros de configuración
N = 750
max_rows = 5000
rows_to_delete = 2000
host = 'localhost'  # Dirección IP del servidor C#
port = 5000         # Puerto del servidor
closing = False
# Ruta a la carpeta del archivo C# de adquisición de datos
ruta_adqui = "D:\\Universidad\\Trabajo de grado\\Desarrollo prototipo\\Código\\Instrucciones\\Adquisición pura"       
comando_adqui = ["dotnet", "run"]

#Inicializar el proceso de adquisición
try:
    proceso_adqui = subprocess.Popen(comando_adqui, cwd=ruta_adqui)
    time.sleep(2)
except Exception as e:
    print(f"Error starting aquisition: {e}")
    sys.exit(1)

# Función para manejar la señal de Ctrl+C
def handle_exit(sig=None, frame=None):
    global closing
    if closing:
        return   
    closing = True      
    if proceso_adqui:
        proceso_adqui.terminate()  # Finalizar el proceso C#
        proceso_adqui.wait()       # Esperar a que el proceso termine        
        print("Aquisition proccess ended")
    sys.exit(0)

# Asigna la función handle_exit para manejar Ctrl+C
signal.signal(signal.SIGINT, handle_exit)

# Lock para manejar el acceso concurrente a los datos
data_lock = threading.Lock()

# Listas para almacenar datos de los canales
data_channels = {i: [] for i in range(5)}

def receive_data():
    global data_channels
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((host, port))
        except Exception as e:
            print(f"Server connection error: {e}")
            return  # Salir de la función en caso de error
            
        while True:
            try:
                received = client_socket.recv(4096).decode('utf-8').strip()
                if received:
                    lines = received.split('\n')
                    with data_lock:
                        for line in lines:
                            parts = line.split(';')
                            if len(parts) == 6:
                                try:
                                    valores = [float(v.replace(',', '.')) for v in parts[1:]]
                                    for i in range(5):
                                        data_channels[i].append(valores[i])
                                    for i in range(5):
                                        if len(data_channels[i]) > max_rows:
                                            data_channels[i] = data_channels[i][-max_rows:]
                                            del data_channels[i][:rows_to_delete]
                                except ValueError:
                                    continue
            except ConnectionResetError:
                print("Connection closed by server")
                break  # Salir del bucle cuando se pierde la conexión
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

# Inicia el hilo para recibir datos de manera continua
data_thread = threading.Thread(target=receive_data, daemon=True)
data_thread.start()

fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
x_data = []

def update(frame):
    global x_data
    with data_lock:
        if len(data_channels[0]) > 0:
            for i in range(5):
                if len(data_channels[i]) > N:
                    data_channels[i] = data_channels[i][-N:]
            x_data = list(range(len(data_channels[0])))
            colors = ['black', 'orange', 'blue', 'yellow', 'green']
            channels = ['fcc5h', 'c1', 'cz', 'c2', 'fcc6h']  
            for ax in axes:
                ax.clear()
            for i in range(5):
                axes[i].plot(x_data, data_channels[i], color=colors[i], label=channels[i], linewidth=0.5)
                axes[i].legend(loc='upper right')
                axes[i].set_ylabel('Valor (mV)')
            axes[-1].set_xlabel('Número de Lecturas (250 SPS)')
            fig.suptitle('Gráfica en Tiempo Real')
            all_y_values = [val for values in data_channels.values() for val in values]
            if all_y_values:
                for ax in axes:
                    #ax.set_ylim(-0.6,0.6)
                    ax.relim()
                    ax.autoscale()
                axes[-1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.6f}'))

# Función para cerrar la ventana y terminar el programa    
fig.canvas.mpl_connect('close_event', handle_exit)  # Vincular el evento de cierre de la ventana

# Configura la animación
ani = animation.FuncAnimation(fig, update, interval=1, cache_frame_data=False)

plt.show()