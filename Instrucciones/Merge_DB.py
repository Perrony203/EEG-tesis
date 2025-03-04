import glob
import csv

def combinar_csv(archivos, archivo_salida):
    """
    Combina varios archivos CSV en uno solo, omitiendo la primera línea (header) de cada archivo.
    
    :param archivos: Lista de rutas de archivos CSV.
    :param archivo_salida: Ruta del CSV resultante.
    """
    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        for archivo in archivos:
            with open(archivo, 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.reader(f_in, delimiter=';')  # Ajusta el delimitador si es necesario
                # Saltar la cabecera del archivo
                next(reader, None)
                for row in reader:
                    writer.writerow(row)

# Obtener la lista de archivos CSV a combinar (por ejemplo, todos en la carpeta "csv_files")
ruta_archivos = "csv_files/*.csv"  # Ajusta esta ruta a donde tengas tus CSV
lista_archivos = glob.glob(ruta_archivos)

# Archivo CSV combinado (sin header)
archivo_combinado = "archivo_combinado.csv"
combinar_csv(lista_archivos, archivo_combinado)

print(f"Se han combinado {len(lista_archivos)} archivos en {archivo_combinado}.")
