import csv
 
def leer_csv_a_arreglo(ruta_archivo):
    datos = []
    with open(ruta_archivo, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo, delimiter=';')
        for fila in lector:
            # Para cada columna se divide por ',' y se eliminan espacios extra
            fila_completa = []
            for columna in fila:
                # Se separa la cadena por ',' y se limpian los espacios
                valores = [valor.strip() for valor in columna.split(',')]
                fila_completa.extend(valores)
            datos.append(fila_completa)
    return datos

def lista_a_diccionario(lista):

    diccionario = {
        'target': [fila[0] for fila in lista[1:] if fila],
        'data': [fila[1:] for fila in lista[1:] if fila]
    }
    return diccionario

# Ejemplo de uso:
ruta = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM characteristics\SVM_1.csv"
arreglo_filas = leer_csv_a_arreglo(ruta)
diccionario = lista_a_diccionario(arreglo_filas)
print(diccionario)
