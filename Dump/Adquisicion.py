import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket
import os
import subprocess
import signal
import sys
import threading

# Parámetros de configuración
N = 2500
max_rows = 5000
rows_to_delete = 2000
host = 'localhost'  # Dirección IP del servidor C#
port = 5000         # Puerto del servidor

# Ruta a la carpeta del archivo C# de adquisición de datos
ruta_adqui = "D:\\Universidad\\Trabajo de grado\\Desarrollo prototipo\\Código\\EEG-tesis\\Instrucciones\\Adquisición pura"       
comando_adqui = ["dotnet", "run"]

# Inicializar el proceso de adquisición
try:
    proceso_adqui = subprocess.Popen(comando_adqui, cwd=ruta_adqui)
except Exception as e:
    print(f"Error al iniciar el proceso de adquisición: {e}")
    sys.exit(1)

# Función para manejar la señal de Ctrl+C
def handle_exit(sig, frame):
    print("\nKilling acquisition program...")
    if proceso_adqui:
        proceso_adqui.terminate()  # Finalizar el proceso C#
        proceso_adqui.wait()       # Esperar a que el proceso termine
        print("Acquisition process ended successfully!")
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
            print(f"Error al conectar al servidor: {e}")
            return  # Salir de la función en caso de error
            
        while True:
            received = client_socket.recv(1024).decode('utf-8').strip()
            if received:
                try:
                    valores = [float(v.replace(',', '.')) for v in received.split(';')[1:]]
                    with data_lock:
                        for i in range(min(len(valores), 5)):  # Aseguramos que solo se procesen hasta 5 canales
                            data_channels[i].append(valores[i])
                    
                    # Mantener solo los últimos max_rows valores
                    for i in range(5):
                        if len(data_channels[i]) > max_rows:
                            data_channels[i] = data_channels[i][-max_rows:]
                            if len(data_channels[i]) > max_rows:
                                del data_channels[i][:rows_to_delete]

                except ValueError as e:
                    data_channels = data_channels

# Inicia el hilo para recibir datos de manera continua
data_thread = threading.Thread(target=receive_data, daemon=True)
data_thread.start()

fig, ax = plt.subplots()
x_data = []

def update(frame):
    global x_data
    with data_lock:  # Lock para asegurar acceso exclusivo
        if len(data_channels[0]) > 0:  # Asegúrate de que hay datos
            # Tomar los últimos N datos para graficar
            for i in range(5):
                if len(data_channels[i]) > N:
                    data_channels[i] = data_channels[i][-N:]

            x_data = list(range(len(data_channels[0])))  # Número de lecturas
            
            ax.clear()

            # Graficar los datos de cada canal con un color diferente
            colors = ['purple', 'blue', 'green','yellow','orange']
            for i in range(1):
                ax.plot(x_data, data_channels[i], color=colors[i], label=f'Canal {i+1}', linewidth=0.25)

            ax.set_title('Gráfica en Tiempo Real')
            ax.set_xlabel('Número de Lecturas')
            ax.set_ylabel('Valor Leído')
            ax.legend(loc='upper right')  # Agrega una leyenda para cada canal
            
            # Ajuste de los límites del eje y para mostrar todos los datos correctamente
            #all_y_values = [val for values in data_channels.values() for val in values]
            all_y_values = data_channels[0]
            if all_y_values:
                ax.relim()
                ax.autoscale_view()

            # Ajuste de los ticks en el eje y para una mejor separación
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.6f}'))

# Configura la animación
ani = animation.FuncAnimation(fig, update, interval=1, cache_frame_data=False)  # Actualiza cada 100 ms

plt.show()
