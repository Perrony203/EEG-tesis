import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket
import subprocess
import signal
import sys
import threading
import time
import numpy as np
from collections import deque

# =====================================================================
# PARÁMETROS DE CONFIGURACIÓN
# =====================================================================
N_TIME_SAMPLES = 750        # Muestras visibles en tiempo
MAX_BUFFER_SIZE = 800       # Máximo histórico en tiempo (deque)
FFT_WINDOW = 256            # Tamaño de ventana para FFT (debe coincidir con el microcontrolador)
SAMPLE_RATE = 250           # Tasa de muestreo (Hz)
HOST = 'localhost'
PORT = 5000

# =====================================================================
# CONFIGURACIÓN DE ADQUISICIÓN
# =====================================================================
ruta_adqui = "D:\\Universidad\\Trabajo de grado\\Desarrollo prototipo\\Código\\Instrucciones\\AdquisicionFFT"              
comando_adqui = ["dotnet", "run"]

# Inicializar proceso de adquisición
try:
    proceso_adqui = subprocess.Popen(comando_adqui, cwd=ruta_adqui)
    time.sleep(2)  # Esperar inicialización
except Exception as e:
    print(f"Error al iniciar adquisición: {e}")
    sys.exit(1)

# =====================================================================
# MANEJO DE SEÑALES Y BLOQUEOS
# =====================================================================
closing = False
data_lock = threading.Lock()

# Señal para terminar correctamente
def handle_exit(sig=None, frame=None):
    global closing
    closing = True
    if proceso_adqui:
        proceso_adqui.terminate()
        proceso_adqui.wait()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# =====================================================================
# ESTRUCTURAS DE DATOS
# =====================================================================
# Buffers circulares para tiempo (optimizados con deque)
data_channels = {i: deque(maxlen=MAX_BUFFER_SIZE) for i in range(5)}

# Datos FFT (se actualizan solo cuando hay nuevos cálculos)
fft_matrix = np.zeros((5, 128))
prev_fft = np.zeros((5, 128))  # Para detectar cambios
fft_bins = np.linspace(0, SAMPLE_RATE//2, 128)  # Eje de frecuencias

# =====================================================================
# HILO DE RECEPCIÓN DE DATOS (CORREGIDO)
# =====================================================================
def receive_data():
    global data_channels, fft_matrix, prev_fft
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
        except Exception as e:
            print(f"Error de conexión: {e}")
            return
            
        while not closing:
            try:
                data = s.recv(65535).decode('utf-8').strip()
                if not data:
                    continue
                    
                # Procesar todas las líneas recibidas
                for line in data.split('\n'):
                    parts = line.split(';')                    
                    if len(parts) == 646:  # Validar estructura  
                        try:
                            # Convertir valores a floats
                            valores = [float(v.replace(',', '.')) for v in parts[1:]]
                            
                            # Extraer componentes
                            time_data = valores[:5]
                            fft_matrix = np.array(valores[5:645]).reshape(5, 128)    
                            
                            # Actualizar datos de tiempo (siempre)
                            with data_lock:
                                for i in range(5):
                                    data_channels[i].append(time_data[i])  
                                    
                        except (ValueError, IndexError):
                            continue
                    
                    elif len(parts) == 7 and parts[-1] == ("OLD"):
                        try:
                            # Convertir valores a floats
                            valores = [float(v.replace(',', '.')) for v in parts[1:-1]]
                            
                            # Extraer componentes
                            time_data = valores     
                            
                            # Actualizar datos de tiempo (siempre)
                            with data_lock:
                                for i in range(5):
                                    data_channels[i].append(time_data[i])                   

                        except (ValueError, IndexError):
                            continue
                        
                    else:
                        continue
                        
            except ConnectionResetError:
                print("Conexión cerrada por el servidor")
                break
            except Exception as e:
                if not closing:
                    print(f"Error en recepción: {e}")
                break

# Iniciar hilo de recepción
data_thread = threading.Thread(target=receive_data, daemon=True)
data_thread.start()

# =====================================================================
# CONFIGURACIÓN DE GRÁFICAS (OPTIMIZADA)
# =====================================================================
fig, axes = plt.subplots(5, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [2, 1]})
plt.subplots_adjust(hspace=0.6, wspace=0.4)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
canales = ['FCC5H', 'C1', 'CZ', 'C2', 'FCC6H']

# Inicializar líneas
time_lines = []
fft_lines = []
for i in range(5):
    # Gráfica de tiempo
    axes[i][0].set_ylabel('Amplitud (mV)', fontsize=8)
    axes[i][0].relim()
    axes[i][0].autoscale()
    time_lines.append(axes[i][0].plot([], [], color=colors[i], linewidth=0.8, label=canales[i])[0])
    axes[i][0].legend(loc='upper right', fontsize=7)
    axes[i][0].grid(True, alpha=0.3)
    
    # Gráfica FFT
    fft_lines.append(axes[i][1].plot(fft_bins, fft_matrix[i], color=colors[i], linewidth=0.8)[0])
    axes[i][1].set_xlim(0, SAMPLE_RATE//2)
    axes[i][1].set_ylabel('Potencia (dB)', fontsize=8)
    axes[i][1].grid(True, alpha=0.3)

axes[-1][0].set_xlabel('Muestras', fontsize=8)
axes[-1][1].set_xlabel('Frecuencia (Hz)', fontsize=8)
fig.suptitle('Monitorización EEG en Tiempo Real', fontsize=12, y=0.95)

# =====================================================================
# FUNCIÓN DE ACTUALIZACIÓN (OPTIMIZADA CON BLITTING)
# =====================================================================
def update(frame):
    # Obtener copia local de los datos
    with data_lock:
        x_time = np.arange(len(data_channels[0]))
        y_time = [np.array(ch) for ch in data_channels.values()]
        current_fft = fft_matrix.copy()
    
    # Actualizar gráficos de tiempo
    for i in range(5):
        time_lines[i].set_data(x_time, y_time[i])        
        
        # Ajuste dinámico del eje Y        
        axes[i][0].autoscale()
        axes[i][0].relim()
    
    # Actualizar FFT solo si hay cambios
    for i in range(5):
        fft_lines[i].set_ydata(current_fft[i])
        axes[i][1].autoscale()
        axes[i][1].relim()
    
    return time_lines + fft_lines

# =====================================================================
# INICIAR ANIMACIÓN
# =====================================================================
ani = animation.FuncAnimation(
    fig,
    update,
    interval=40,       # ~25 FPS para tiempo
    blit=False,         # Blitting activado
    cache_frame_data=False
)

fig.canvas.mpl_connect('close_event', handle_exit)
plt.show()