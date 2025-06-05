
import time
from micromlgen import port
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os
import csv
import warnings

from collections import Counter, defaultdict

import seaborn as sns

from sklearn.svm import SVC
from sklearn.calibration import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split         
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import make_scorer, roc_auc_score
from mlxtend.plotting import plot_decision_regions
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def leer_csv_a_arreglo(ruta_archivo):
    posiciones_a_eliminar = []  # Índices que deseas eliminar (basados en 0 restar 1 al de la grafica)

    datos = []
    with open(ruta_archivo, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo, delimiter=';')
        for fila in lector:
            fila_completa = []
            for i, columna in enumerate(fila):
                if i == 0:
                    fila_completa.append(columna.strip())
                else:
                    valores = [valor.strip() for valor in columna.split(',')]
                    fila_completa.extend(valores)

            if len(fila_completa) == 41: #Se ponen 17 porque son 2 canales y 1 estimulo. Si fueran los datos completos serían 41 y para 3 canales 25
                # Eliminar las posiciones indicadas
                fila_filtrada = [valor for i, valor in enumerate(fila_completa) if i not in posiciones_a_eliminar]
                datos.append(fila_filtrada)
            else:
                continue
    return datos

def lista_a_diccionario(lista):

    diccionario = {
        'target': [fila[0] for fila in lista if fila],
        'data': [fila[1:] for fila in lista if fila]
    }
    return diccionario

ruta_csv = r'Instrucciones\Registros almacenados\SVM_combined\Idle-movement\Sebastian\movement_SVM_1.csv'

lista_filas = leer_csv_a_arreglo(ruta_csv)
data_dict = lista_a_diccionario(lista_filas)


data_dict['target'] = [int(x) for x in data_dict['target']]
data_dict['data'] = [[float(val) for val in fila] for fila in data_dict['data']]


feature_columns = [f"f{i+1}" for i in range(40)]
df = pd.DataFrame(data_dict['data'], columns=feature_columns)
df['target'] = data_dict['target']

columnas_a_eliminar = ['f35','f19','f33','f36','f28','f6','f26','f3','f18','f12', 'f24','f9','f10','f8','f22','f17','f2','f7','f30','f1','f29','f32','f40','f37','f38','f31','f39','f23','f15','f16','f20','f13','f14','f4']
df = df.drop(columns=columnas_a_eliminar)

#df = df[df["target"] != 0].reset_index(drop=True)
#df = df[df["target"] != 1].reset_index(drop=True)
#df = df[df["target"] != 2].reset_index(drop=True)

# Actualizar lista de características
feature_columns = [col for col in df.columns if col.startswith('f')]

#print(df)

# Matriz de correlación
corr_matrix = df[feature_columns].corr()



# Extrae los pares de características (solo una vez cada par, usando la parte superior de la matriz)
correlaciones = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        correlaciones.append((col1, col2, corr_val))

# Ordena por valor absoluto de la correlación
correlaciones_ordenadas = sorted(correlaciones, key=lambda x: abs(x[2]), reverse=True)

# Filtra pares con correlación fuerte (por ejemplo, > 0.9 o < -0.9)
umbral = 0.9
correlaciones_fuertes = [(c1, c2, round(val, 3)) for c1, c2, val in correlaciones_ordenadas if abs(val) > umbral]

# Imprime
print("Pares de características con alta correlación (> 0.9 o < -0.9):")
for c1, c2, val in correlaciones_fuertes:
    print(f"{c1} y {c2}: {val}")


# Contador y acumulador de correlación ponderada
conteo_pares = defaultdict(int)
suma_ponderada= defaultdict(float)

# Recorre la matriz de correlación
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        if i != j:
            f1 = corr_matrix.columns[i]
            f2 = corr_matrix.columns[j]
            valor = corr_matrix.iloc[i, j]
            abs_val = abs(valor)

            # Contar solo correlaciones fuertes para el número de pares
            if abs_val > umbral:
                conteo_pares[f1] += 1

            # Siempre sumar la magnitud de la correlación para la ponderación
            suma_ponderada[f1] += abs_val

# Imprime resumen por característica
print("Resumen por característica (número de correlaciones, suma ponderada de correlaciones):")
for f in sorted(feature_columns, key=lambda x: suma_ponderada[x], reverse=True):
    print(f"{f}: {conteo_pares[f]} pares, suma ponderada = {round(suma_ponderada[f], 3)}")
# Calcular la matriz de correlación
corr_matrix = df[feature_columns].corr().abs()

# Seleccionar la parte superior del triángulo de la matriz
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Encuentra las columnas con una correlación mayor a 0.9
columnas_correlacionadas = [column for column in upper.columns if any(upper[column] > 0.9)]

print (columnas_correlacionadas)

# Elimina columnas altamente correlacionadas
df = df.drop(columns=columnas_correlacionadas)

# Actualiza las columnas de características
feature_columns = [col for col in df.columns if col.startswith('f')]




# Visualización con mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación entre características")
plt.show()

print (df)