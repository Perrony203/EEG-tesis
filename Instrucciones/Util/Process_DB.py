import glob
import csv
import os
import pandas as pd

def generar_nombre_autoincremental(directorio, base_nombre="total_SVM"):
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    archivos = [os.path.splitext(f)[0] for f in os.listdir(directorio) 
                if f.startswith(base_nombre) and os.path.splitext(f)[0][len(base_nombre):].lstrip("_").isdigit()]
    
    if archivos:
        numeros_existentes = [int(f[len(base_nombre):].lstrip("_")) for f in archivos]
        nuevo_numero = max(numeros_existentes) + 1
    else:
        nuevo_numero = 1

    return os.path.join(directorio, f"{base_nombre}_{nuevo_numero}.csv")

def combinar_csv(archivos, archivo_salida):
    if not archivos:
        print("No hay archivos CSV para combinar.")
        return

    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter=';')  # Mantener el delimitador correcto
        
        for i, archivo in enumerate(archivos):
            with open(archivo, 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.reader(f_in, delimiter=';')  
                
                if i == 0:
                    # Escribir la cabecera del primer archivo
                    writer.writerow(next(reader, None))  
                else:
                    # Omitir la cabecera en los siguientes archivos
                    next(reader, None)
                
                # Escribir las filas sin modificar el formato de las columnas
                for row in reader:
                    writer.writerow(row)

def modificar_estimulos(df, columna="Estimulo", regla=1):   
    df_modificado = df.copy()
    
    if regla == 1:
        df_modificado[columna] = df_modificado[columna].replace({1: 1, 2: 1, 3: 2, 4: 2})
    elif regla == 2:
        df_modificado[columna] = df_modificado[columna].replace({2: 1, 3: 1, 1: 2, 4: 2})
    
    return df_modificado

# Definir rutas
ruta_base = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados"
ruta_media = os.path.join(ruta_base, "SVM_combined")
sujetos_validos = ["Sebastian", "Nicolas"]

while(True):
    name = input("Ingrese sujeto de prueba: ")
    if name in sujetos_validos:
        print("Procesando archivos...")
        break
    else:
        print("Sujeto no válido")
        
ruta_archivos = os.path.join(os.path.join(ruta_base, "SVM characteristics"),name)
ruta_salida = os.path.join(os.path.join(ruta_media,"Complete_data"),name)
ruta_regla1 = os.path.join(os.path.join(ruta_media, "Legs-arms"),name)
ruta_regla2 = os.path.join(os.path.join(ruta_media, "Right-left"),name) 

# Crear lista de archivos CSV a combinar
lista_archivos = glob.glob(os.path.join(ruta_archivos, "*.csv"))

# Generar nombre dinámico para el archivo combinado
archivo_combinado = generar_nombre_autoincremental(ruta_salida, "total_SVM")

# Llamar a la función para combinar archivos
combinar_csv(lista_archivos, archivo_combinado)
print(f"Se han combinado {len(lista_archivos)} archivos.")

# Leer el CSV combinado
df = pd.read_csv(archivo_combinado, sep=";")
        
df_modificado_regla1 = modificar_estimulos(df, regla=1)
archivo_modificado_regla1 = generar_nombre_autoincremental(ruta_regla1, "Leg-arms_SVM")
df_modificado_regla1.to_csv(archivo_modificado_regla1, index=False, sep=";")

df_modificado_regla2 = modificar_estimulos(df, regla=2)
archivo_modificado_regla2 = generar_nombre_autoincremental(ruta_regla2, "Right-left_SVM")
df_modificado_regla2.to_csv(archivo_modificado_regla2, index=False, sep=";")

print(f"Se ha generado el archivo separando miembros superiores e inferiores.")
print(f"Se ha generado el archivo separando miembros izquierdos y derechos.")
