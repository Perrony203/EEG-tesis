
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

columnas_a_eliminar = ['f35','f19','f33','f36','f28','f6','f26','f3','f18','f12', 'f24','f9','f10','f8','f22','f17','f2','f7','f30','f1','f29','f32','f40','f37','f38','f31','f39','f23','f15','f16','f20','f13','f14','f4']    #Verdadera manera de quitar columnas
df = df.drop(columns=columnas_a_eliminar)

# Actualizar lista de características
feature_columns = [col for col in df.columns if col.startswith('f')]

#df = df[df["target"] != 0].reset_index(drop=True)
#df = df[df["target"] != 1].reset_index(drop=True)
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

# print (y)

# print("Valores únicos en y antes de LabelEncoder:", np.unique(y))

# print('Input shape: ', X.shape)
# print('Target variable shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Búsqueda en hiperparámetros
param_grid = {
    'kernel': ['rbf'],
    'C': np.logspace(-10, 10, 80),
    #'C': np.linspace(1, 10000, 1),
    #'C': np.logspace(-4, 4, 100),
    
    'gamma': np.logspace(-5, 4, 80),
    #'gamma': np.linspace(1, 10000, 1),
    #'gamma':[np.float64(2.7825594022071143)], 
    
    #'decision_function_shape': ['ovr']
}

print("Iniciando entrenamiento")
inicio = time.time()

grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Mejores parámetros:", grid_search.best_params_)
print("Mejor score en validación cruzada:", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

train_sizes, train_scores, test_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=3    ,
    scoring='accuracy',
    n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.title("Curva de Aprendizaje del SVM")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precisión (accuracy)")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Precisión en entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Precisión en validación cruzada")

plt.legend(loc="best")
plt.tight_layout()

best_params = grid_search.best_params_
model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])#, decision_function_shape=best_params['decision_function_shape'])

model.fit(X_train, y_train)

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

fin = time.time()

print("Tiempo de entrenamiento: ", fin-inicio)

plt.show()