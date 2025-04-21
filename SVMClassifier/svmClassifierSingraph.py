
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
            if len(fila_completa) == 41:
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

ruta_csv = r'D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM_combined\Legs-arms\Sebastian\Legs-arms_SVM_1.csv'

lista_filas = leer_csv_a_arreglo(ruta_csv)
data_dict = lista_a_diccionario(lista_filas)


data_dict['target'] = [int(x) for x in data_dict['target']]
data_dict['data'] = [[float(val) for val in fila] for fila in data_dict['data']]


feature_columns = [f"f{i+1}" for i in range(40)]
df = pd.DataFrame(data_dict['data'], columns=feature_columns)
df['target'] = data_dict['target']

df = df[df["target"] != 0].reset_index(drop=True)
##df = df[df["target"] != 3].reset_index(drop=True)
##df = df[df["target"] != 2].reset_index(drop=True)

# print("DataFrame head:")
#print(df.head())

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Definición de parámetros
Const = 20.0
Gamma = 0.6
Kernel = 'linear'


#PROBANDO


# Escalado + SVM
pipeline = make_pipeline(StandardScaler(), SVC())

# Búsqueda en hiperparámetros
param_grid = {
    'svc__kernel': ['rbf'],
    'svc__C': np.logspace(-10, 10, 100),
    'svc__gamma': np.logspace(-12, 4, 100)
}
print("Iniciando entrenamiento")
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Mejores parámetros:", grid_search.best_params_)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

train_sizes, train_scores, test_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 30),
    cv=10,
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



#PROBANDO


best_params = grid_search.best_params_
model = SVC(C=best_params['svc__C'], gamma=best_params['svc__gamma'], kernel=best_params['svc__kernel'])

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

# Optimización
# params = {
#     "C": np.arange(2, 100, 2),
#     "gamma": np.arange(0.1, 1, 0.01),
#     "kernel": ['linear']
# }

# auc = make_scorer(roc_auc_score)

# best_model = RandomizedSearchCV(model, param_distributions=params, random_state=42,
#                                 n_iter=2, cv=3, verbose=0, n_jobs=1,
#                                 return_train_score=True, scoring = auc)

# best_model.fit(X_train, y_train)

# def report_best_scores(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                 results['mean_test_score'][candidate],
#                 results['std_test_score'][candidate]))
#             best_params = results['params'][candidate]
#             print("Best parameters found:")
#             for param, value in best_params.items():
#                 print("  {0}: {1}".format(param, value))
#             print("")

# report_best_scores(best_model.cv_results_, 1)


plt.show()