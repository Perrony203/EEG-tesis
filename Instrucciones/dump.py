
# UNIR ARCHIVOS DE CARACTERÍSTICAS E INTRUCCIONES (CAMBIAR EL NÚMERO EN LOS TRES PATHS) 
# from datetime import datetime
# import csv

# file1 = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Aparición imagenes\Sebastian\Instrucs_12.csv"
# file2 = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Datos EEG\Sebastian\Caracs_12.csv"
# time_format ='%Y-%m-%d %H:%M:%S.%f'
# output_file = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM characteristics\Sebastian\SVM_12.csv"

# stimuli = []
# with open(file1, newline='', encoding='utf-8') as f1:
#     reader1 = csv.reader(f1, delimiter=';')
#     header1 = next(reader1, None) 
#     for row in reader1:
#         if len(row) < 2:
#             continue
#         time_str = row[0].strip()
#         stim_type = row[1].strip()
#         try:
#             stim_time = datetime.strptime(time_str, time_format)
#             stimuli.append((stim_time, stim_type))
#         except Exception as e:
#             print(f"Error al parsear tiempo del estímulo: '{time_str}'. Error: {e}")

# stimuli.sort(key=lambda x: x[0])  # Ordenar por tiempo

# first_occurrences = {'1': False, '2': False, '3': False, '4': False}  # Para rastrear las primeras ocurrencias

# with open(file2, newline='', encoding='utf-8') as f2, \
#     open(output_file, mode='w', newline='', encoding='utf-8') as fout:
    
#     reader2 = csv.reader(f2, delimiter=';')
#     header2 = next(reader2, None)
#     writer = csv.writer(fout, delimiter=';')
    
#     output_header = ["Estimulo", "C1", "C2", "C3", "C4", "C5"]
#     writer.writerow(output_header)
    
#     for row in reader2:
#         if len(row) < 7:
#             continue
#         start_str = row[0].strip()
#         end_str = row[1].strip()
#         try:
#             start_time = datetime.strptime(start_str, time_format)
#             end_time = datetime.strptime(end_str, time_format)
#         except Exception as e:
#             print(f"Error al parsear tiempos en archivo2: '{start_str}' o '{end_str}'. Error: {e}")
#             continue
        
#         found_stimulus = "0"
#         for stim_time, stim_type in stimuli:
#             if start_time <= stim_time <= end_time:
#                 found_stimulus = stim_type
#                 break
        
#         # Omitir la primera aparición de los estímulos 1, 2, 3 y 4
#         if found_stimulus in first_occurrences and not first_occurrences[found_stimulus]:
#             first_occurrences[found_stimulus] = True
#             continue  # Saltar esta fila

#         channel_features = row[2:7]
#         writer.writerow([found_stimulus] + channel_features)

#BUSCAR LA CANTIDAD DE ESTÍMULOS POR CADA TIPO
import pandas as pd
from pathlib import Path

# Ruta a la carpeta donde están los CSV
carpeta = Path(r'D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM_combined\Complete_data\Sebastian')  # <-- cambia esto

# Iterar sobre todos los archivos .csv en la carpeta
for archivo_csv in carpeta.glob("*.csv"):
    print(f"\nAnalizando archivo: {archivo_csv.name}")
    try:
        # Cargar el archivo con separador ';'
        df = pd.read_csv(archivo_csv, sep=';')

        # Verificar si la columna 'Estimulo' existe
        if 'Estimulo' in df.columns:
            conteo = df['Estimulo'].value_counts()
            print("Conteo de estímulos:")
            print(conteo)
        else:
            print("La columna 'Estimulo' no se encuentra en el archivo.")
    except Exception as e:
        print(f"Error al procesar {archivo_csv.name}: {e}")
