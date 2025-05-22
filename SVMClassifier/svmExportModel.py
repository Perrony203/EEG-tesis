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
from sklearn.pipeline import make_pipeline

def leer_csv_a_arreglo(ruta_archivo):

    datos = []
    with open(ruta_archivo, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo, delimiter=';')
        for fila in lector:
            fila_completa = []
            for i, columna in enumerate(fila):
                if i == 0:
                    # Primer valor: target (se toma tal cual)
                    fila_completa.append(columna.strip())
                else:
                    # Las columnas restantes se dividen por coma
                    valores = [valor.strip() for valor in columna.split(',')]
                    fila_completa.extend(valores)
            if len(fila_completa) == 25: #Se ponen 17 porque son 2 canales y 1 estimulo. Si fueran los datos completos serían 41 y para 3 canales 25
                datos.append(fila_completa)
            else:
                continue
    return datos

def lista_a_diccionario(lista):

    diccionario = {
        'target': [fila[0] for fila in lista if fila],
        'data': [fila[1:] for fila in lista if fila]
    }
    return diccionario

ruta_csv = r'Instrucciones\Registros almacenados\SVM_combined\Legs\Sebastian\Legs_SVM_1.csv'

lista_filas = leer_csv_a_arreglo(ruta_csv)
data_dict = lista_a_diccionario(lista_filas)


data_dict['target'] = [int(x) for x in data_dict['target']]
data_dict['data'] = [[float(val) for val in fila] for fila in data_dict['data']]


feature_columns = [f"f{i+1}" for i in range(24)]
df = pd.DataFrame(data_dict['data'], columns=feature_columns)
df['target'] = data_dict['target']

df = df[df["target"] != 0].reset_index(drop=True)
#df = df[df["target"] != 3].reset_index(drop=True)
#df = df[df["target"] != 2].reset_index(drop=True)

# print("DataFrame head:")
#print(df.head())
print("Cantidad de muestras luego del filtrado:", df.shape[0])
scaler = StandardScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])
# print(df.head())

X = df[feature_columns].to_numpy()
y = df['target'].to_numpy()

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(C=np.float64(359381.36638046405), gamma=np.float64(2.7825594022071143), kernel='rbf', decision_function_shape='ovr')
model.fit(X_train, y_train)

model_code = port(model)

training_predict = model.predict(X_train)

print("Reporte de clasificación (entrenamiento):")
print(metrics.classification_report(y_train, training_predict, digits = 3, zero_division=0))

print("Matriz de confusión (entrenamiento):")
print(metrics.confusion_matrix(y_train, training_predict))

print(f'Model accuracy: {round(metrics.accuracy_score(y_train, training_predict)*100,2)}%')

test_predict = model.predict(X_test)

print("Reporte de clasificación de prueba:")
print(metrics.classification_report(y_test, test_predict, digits = 3, zero_division=0))

print("Matriz de confusión (prueba):")
print(metrics.confusion_matrix(y_test, test_predict))

print(f'Model accuracy: {round(metrics.accuracy_score(y_test, test_predict)*100,2)}%')

plt.show()

# Guardar en un archivo .h o .cpp
ruta_salida = r'D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\SVMClassifier\PSoC\Complete\svmComplete.h'

with open(ruta_salida, 'w') as f:
    f.write(model_code)

print(f"Modelo exportado exitosamente a: {ruta_salida}")