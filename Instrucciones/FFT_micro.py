import matplotlib.animation as animation
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import numpy as np
import threading
import random
import serial
import signal
import time
import csv
import sys
import re
import os

# =====================================================================
# PARÁMETROS DE CONFIGURACIÓN
# =====================================================================
N_TIME_SAMPLES = 750        # Muestras visibles en tiempo
MAX_BUFFER_SIZE = 800       # Máximo histórico en tiempo (deque)
FFT_WINDOW = 256            # Tamaño de ventana para FFT (debe coincidir con el microcontrolador)
SAMPLE_RATE = 250           # Tasa de muestreo (Hz)
PORT = 'COM12'          
BAUD_RATE = 4000000           
TIMEOUT = 1

full_operation = False   

# =====================================================================
# MANEJO DE SEÑALES Y BLOQUEOS
# =====================================================================
closing = False
data_lock = threading.Lock()

# Señal para terminar correctamente
def handle_exit(sig=None, frame=None, err=False):
    global closing
    closing = True   
    if not err: 
        print("Ending operation...")
    plt.close('all')
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)
    print("Serial connection established successfully")
except:
    print("No Serial port available...")
    handle_exit()

# =====================================================================
# ESTRUCTURAS DE DATOS
# =====================================================================
# Buffers circulares para tiempo (optimizados con deque)
data_channels = {i: deque(maxlen=MAX_BUFFER_SIZE) for i in range(5)}

# Datos FFT (se actualizan solo cuando hay nuevos cálculos)
fft_matrix = np.zeros((5, 128))
prev_fft = np.zeros((5, 128))  # Para detectar cambios
fft_bins = np.linspace(0, SAMPLE_RATE//2, 128)  # Eje de frecuencias
valoresString = []
start = True
separador = ","

def generar_nombre_autoincremental(directorio=r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Datos EEG", base_nombre="Caracs"):        
    if not os.path.exists(directorio):
        print("No directory")
        os.makedirs(directorio)
        
    archivos = [os.path.splitext(f)[0] for f in os.listdir(directorio) if f.startswith(base_nombre) and os.path.splitext(f)[0][len(base_nombre):].lstrip("_").isdigit()]         
    if archivos:
        numeros_existentes = [int(f[len(base_nombre):].lstrip("_")) for f in archivos]
        nuevo_numero = max(numeros_existentes) + 1
    else:
        nuevo_numero = 1

    return os.path.join(directorio, f"{base_nombre}_{nuevo_numero}.csv")

path = generar_nombre_autoincremental()

# Escribir encabezado en los archivos
with open(path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')            
    writer.writerow(['Start_time', 'End_time', 'C1', 'C2', 'C3', 'C4', 'C5'])

def procesar_arreglo(arr, total_elementos):    
    
    if total_elementos < 5:
        raise ValueError("El número total de elementos debe ser al menos 5.")    
    
    primeros_5 = arr[:5] 
    patron = re.compile(r'^\d+\.\d{6}$')
    validos = [x for x in arr[5:] if patron.match(x)] 
    base = primeros_5 + validos
    
    if len(base) > total_elementos:
        base = primeros_5 + validos[:total_elementos - 5]
    
    if len(base) == total_elementos:
        return base

    faltantes = total_elementos - len(base)
    
    while faltantes > 0:
        longitud = len(base)
        indices_validos = list(range(5, longitud - 1)) 
        
        if not indices_validos:
            if longitud >= 4:
                vecinos = base[-4:]
                promedio = sum(float(v) for v in vecinos) / 4.0
                nuevo_valor = f"{promedio:.6f}"
                base.append(nuevo_valor)
                faltantes -= 1
            else:
                raise ValueError("No se pueden obtener 4 vecinos para calcular el promedio.")
            continue
        
        pos = random.choice(indices_validos)
        
        try:
            vec_izq1 = float(base[pos - 2])
            vec_izq2 = float(base[pos - 1])
            vec_der1 = float(base[pos])
            vec_der2 = float(base[pos + 1])
            
        except (IndexError, ValueError):
            if longitud >= 4:
                vecinos = base[-4:]
                promedio = sum(float(v) for v in vecinos) / 4.0
                nuevo_valor = f"{promedio:.6f}"
                base.append(nuevo_valor)
                faltantes -= 1
                continue
            else:
                raise ValueError("Error al obtener vecinos para el promedio.")
        
        promedio = (vec_izq1 + vec_izq2 + vec_der1 + vec_der2) / 4.0
        nuevo_valor = f"{promedio:.6f}"
        base.insert(pos, nuevo_valor)
        faltantes -= 1
        
    return base

def receive_data():
    global data_channels, fft_matrix, prev_fft, full_operation, valoresString, start, separador
    while not closing:
        try:
            valoresString.clear() 
            line = ser.readline().decode('iso-8859-1').strip()
            if line == "I":
                if start:
                    ini_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    start = False
                    
                counter = 0    
                while line != "NEW" and line != "OLD":
                    line = ser.readline().decode('iso-8859-1').strip()
                    valoresString.append(line)
                    counter += 1
                    if counter >= 10:
                        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")                   
                
                if len(valoresString) == 6 and valoresString[-1] == "OLD":
                    valoresString = valoresString[:-1]
                    
                    try:                            
                        valores = [float(v.replace(',', '.')) for v in valoresString] 
                        time_data = valores                                 
                        
                        with data_lock:
                            for i in range(5):
                                data_channels[i].append(time_data[i])                   

                    except (ValueError, IndexError):
                        continue
                    
                elif valoresString[-1] == "NEW" and valoresString[5] == "C" and valoresString[26] == "F": 
                    start = True
                    tiempo = valoresString[0:5]
                    caracteristicas = valoresString[6:26]
                    valoresString = valoresString[27:-1]                       
                    
                    if len(valoresString) != 640:
                        print("BOBINO ", len(valoresString))
                        valoresString = procesar_arreglo(valoresString, 640)
                    
                    try:
                        time_data = [float(v.replace(',', '.')) for v in tiempo]
                        fft_matrix = np.array([float(v.replace(',', '.')) for v in valoresString]).reshape(5, 128)   
                        caracs_data = np.array([float(v.replace(',', '.')) for v in caracteristicas]).reshape(5, 4)
                        
                        caracs_str = [separador.join(map(str, fila)) for fila in caracs_data]
                        
                        with open(path, mode='a', newline='') as file:
                            writer = csv.writer(file, delimiter=';')
                            writer.writerow([str(ini_time), str(end_time)] + caracs_str)
                        
                        with data_lock:
                            for i in range(5):
                                data_channels[i].append(time_data[i])  
                                
                    except (ValueError, IndexError):
                        continue 
                        
                if(not full_operation):
                    full_operation = True
                    print("Data is beeing retreived successfully...." )
                    print("Starting plotting process...")
        except serial.SerialException as e:
            print("Serial port error")
            handle_exit(None,None,True)
            break   
        # except Exception as e:
        #     if not closing:
        #         print(f"Error: {e}")
        #     break
    
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
    blit=True,         # Blitting activado
    cache_frame_data=False
)

fig.canvas.mpl_connect('close_event', handle_exit)
plt.show()
