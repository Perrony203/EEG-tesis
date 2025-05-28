
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

            if len(fila_completa) == 17: #Se ponen 17 porque son 2 canales y 1 estimulo. Si fueran los datos completos serían 41 y para 3 canales 25
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

ruta_csv = r'Instrucciones\Registros almacenados\SVM_combined\Arms\Sebastian\Arms_SVM_1.csv'

lista_filas = leer_csv_a_arreglo(ruta_csv)
data_dict = lista_a_diccionario(lista_filas)


data_dict['target'] = [int(x) for x in data_dict['target']]
data_dict['data'] = [[float(val) for val in fila] for fila in data_dict['data']]


feature_columns = [f"f{i+1}" for i in range(16)]
df = pd.DataFrame(data_dict['data'], columns=feature_columns)
df['target'] = data_dict['target']

columnas_a_eliminar = ['f12']
df = df.drop(columns=columnas_a_eliminar)


print(df)