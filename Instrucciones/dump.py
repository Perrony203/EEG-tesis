from datetime import datetime
import csv

file1 = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Aparición imagenes\Sebastian\Instrucs_10.csv"
file2 = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Datos EEG\Sebastian\Caracs_10.csv"
time_format ='%Y-%m-%d %H:%M:%S.%f'
output_file = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM characteristics\Sebastian\SVM_10.csv"

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

stimuli.sort(key=lambda x: x[0])  # Ordenar por tiempo

first_occurrences = {'1': False, '2': False, '3': False, '4': False}  # Para rastrear las primeras ocurrencias

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
        
        # Omitir la primera aparición de los estímulos 1, 2, 3 y 4
        if found_stimulus in first_occurrences and not first_occurrences[found_stimulus]:
            first_occurrences[found_stimulus] = True
            continue  # Saltar esta fila

        channel_features = row[2:7]
        writer.writerow([found_stimulus] + channel_features)