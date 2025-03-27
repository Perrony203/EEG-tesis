import glob
import csv
import os

def generar_nombre_autoincremental(directorio, base_nombre="total_SVM"):
    """
    Genera un nombre de archivo autoincremental en un directorio dado.

    :param directorio: Carpeta donde se guardará el archivo.
    :param base_nombre: Prefijo del archivo (por defecto "total_SVM").
    :return: Ruta del nuevo archivo CSV con número autoincremental.
    """
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
    """
    Combina varios archivos CSV en uno solo. Mantiene la cabecera del primer archivo y omite las de los demás.
    
    :param archivos: Lista de rutas de archivos CSV.
    :param archivo_salida: Ruta del CSV resultante.
    """
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

# Definir rutas
ruta_base = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados"
ruta_archivos = os.path.join(ruta_base, "SVM characteristics")
ruta_salida = os.path.join(ruta_base, "SVM_combined")

# Crear lista de archivos CSV a combinar
lista_archivos = glob.glob(os.path.join(ruta_archivos, "*.csv"))

# Generar nombre dinámico para el archivo combinado
archivo_combinado = generar_nombre_autoincremental(ruta_salida, "total_SVM")

# Llamar a la función para combinar archivos
combinar_csv(lista_archivos, archivo_combinado)

print(f"Se han combinado {len(lista_archivos)} archivos en {archivo_combinado}.")
