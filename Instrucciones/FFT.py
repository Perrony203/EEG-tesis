import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import socket
import os
import subprocess
import signal
import sys
import threading
import time

# Parámetros de configuración
N = 250  # Ventana de datos para la FFT
max_rows = 5000
rows_to_delete = 2000
host = 'localhost'  # Dirección IP del servidor C#
port = 5000         # Puerto del servidor
closing = False

# Ruta a la carpeta del archivo C# de adquisición de datos
ruta_adqui = "D:\\Universidad\\Trabajo de grado\\Desarrollo prototipo\\Código\\Instrucciones\\Adquisición pura"       
comando_adqui = ["dotnet", "run"]

# Inicializar el proceso de adquisición
try:
    proceso_adqui = subprocess.Popen(comando_adqui, cwd=ruta_adqui)
    time.sleep(2)
except Exception as e:
    print(f"Error starting acquisition: {e}")
    sys.exit(1)

# Manejo de Ctrl+C para cerrar todo correctamente
def handle_exit(sig=None, frame=None):
    global closing
    if closing:
        return   
    closing = True      
    if proceso_adqui:
        proceso_adqui.terminate()
        proceso_adqui.wait()        
        print("Acquisition process ended")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# Lock para manejar acceso concurrente a los datos
data_lock = threading.Lock()
data_channels = {i: [] for i in range(5)}

def receive_data():
    """Función que recibe datos desde el socket TCP."""
    global data_channels
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((host, port))
        except Exception as e:
            print(f"Server connection error: {e}")
            return
            
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
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

# Iniciar hilo de recepción de datos
data_thread = threading.Thread(target=receive_data, daemon=True)
data_thread.start()

# Crear figura con 5 filas y 2 columnas (izq: señal, der: FFT)
fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex='col')

# Configuración de etiquetas
channel_names = ['fcc5h', 'c1', 'cz', 'c2', 'fcc6h']
colors = ['black', 'orange', 'blue', 'yellow', 'green']

# Escala fija para la FFT
fft_ylim = (0, 20)  # Ajustar según el rango que quieras
fft_xlim = (0, 60)   # Limitar a 125 Hz (Nyquist)

def update(frame):
    """Función que actualiza las gráficas."""
    with data_lock:
        if len(data_channels[0]) >= N:
            x_time = np.arange(N)  # Eje X para la señal en el tiempo
            freqs = np.fft.rfftfreq(N, d=1/250)  # Eje X para la FFT (250 Hz de muestreo)

            for i in range(5):
                # Seleccionar las últimas N muestras
                signal_data = np.array(data_channels[i][-N:])

                # Calcular FFT
                fft_values = np.abs(np.fft.rfft(signal_data))+1e-6  

                # Limpiar gráficos
                axes[i, 0].clear()
                axes[i, 1].clear()

                # Graficar señal original (columna izquierda)
                axes[i, 0].plot(x_time, signal_data, color=colors[i], linewidth=0.5)
                axes[i, 0].set_ylabel(channel_names[i])
                axes[i, 0].set_xlim([0, N])
                axes[i, 0].grid(True)

                # Graficar FFT (columna derecha)
                axes[i, 1].plot(freqs, fft_values, color=colors[i], linewidth=0.5)
                axes[i, 1].set_xlim(fft_xlim)
                axes[i, 1].set_ylim(fft_ylim)# Nyquist = 250 Hz / 2
                # axes[i, 1].relim()
                # axes[i, 1].autoscale()
                #axes[i, 1].set_yscale("log")                
                axes[i, 1].grid(True)

            # Etiquetas de la gráfica
            axes[0, 0].set_title("Señal en el dominio del tiempo")
            axes[0, 1].set_title("Transformada de Fourier (FFT)")
            axes[-1, 0].set_xlabel("Muestras (250 SPS)")
            axes[-1, 1].set_xlabel("Frecuencia (Hz)")

# Configurar animación
ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)

# Mostrar la figura
plt.show()
