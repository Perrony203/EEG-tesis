
# TinyML - Support Vector Machine (Classifier)
"""

!pip install micromlgen

"""

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

"""## 2. Load Dataset"""

#########################################
# Funciones para leer y procesar el CSV
#########################################
def leer_csv_a_arreglo(ruta_archivo):
    """
    Lee un archivo CSV en el que:
      - Las columnas están separadas por ';'
      - La primera columna es el target.
      - Las siguientes 5 columnas contienen 4 valores cada una, separados por ','.
    Cada fila resultante tendrá 21 elementos: 1 target y 20 features.
    """
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
    """
    Transforma una lista de listas en un diccionario con dos llaves:
      - 'target': array con el primer elemento de cada fila.
      - 'data': array de arrays con el resto de los elementos de cada fila.
    """
    diccionario = {
        'target': [fila[0] for fila in lista if fila],
        'data': [fila[1:] for fila in lista if fila]
    }
    return diccionario

#########################################
# Cargar el CSV y preparar el DataFrame
#########################################
# Cambia 'tu_archivo.csv' por la ruta real de tu archivo CSV.
ruta_csv = r'D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM_combined\Legs-arms\Sebastian\Legs-arms_SVM_1.csv'

lista_filas = leer_csv_a_arreglo(ruta_csv)
data_dict = lista_a_diccionario(lista_filas)

# Convertir los tipos:
# - target a entero
# - data a flotantes (se esperan 20 features por fila)
data_dict['target'] = [int(x) for x in data_dict['target']]
data_dict['data'] = [[float(val) for val in fila] for fila in data_dict['data']]

# Create a DataFrame
# Generar nombres para las 20 características: f1, f2, ..., f20.
feature_columns = [f"f{i+1}" for i in range(40)]
df = pd.DataFrame(data_dict['data'], columns=feature_columns)
df['target'] = data_dict['target']

df = df[df["target"] != 0].reset_index(drop=True)
##df = df[df["target"] != 3].reset_index(drop=True)
##df = df[df["target"] != 2].reset_index(drop=True)

print("DataFrame head:")
print(df.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])
print(df.head())

#########################################
# Preparar datos para el modelo
#########################################
X = df[feature_columns].to_numpy()
y = df['target'].to_numpy()

# Si fuera necesario (por ejemplo, si los targets fueran cadenas), se puede usar LabelEncoder.
y = LabelEncoder().fit_transform(y)

print (y)

print("Valores únicos en y antes de LabelEncoder:", np.unique(y))

# """## 3. Dataset Visualization"""

# #########################################
# # Visualización 3D del dataset (usando f1, f2 y f3)
# #########################################
# fig = go.Figure()
# fig.add_trace(go.Scatter3d(
#     x=df['f1'],
#     y=df['f2'],
#     z=df['f3'],
#     mode='markers',
#     marker=dict(color='blue')
# ))
# fig.update_layout(
#     scene=dict(
#         xaxis_title='Feature 1',
#         yaxis_title='Feature 2',
#         zaxis_title='Feature 3'
#     ),
#     scene_camera=dict(eye=dict(x=1.87, y=0.88, z=-0.64)),
#     width=1000,
#     height=600
# )
# # Descomenta la siguiente línea si deseas visualizar la gráfica.
# fig.show()

print('Input shape: ', X.shape)
print('Target variable shape: ', y.shape)

"""## 4. Split into training and test data"""

#########################################
# División en entrenamiento y prueba
#########################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train

y_test

"""## 5. Create the classification model"""

#########################################
# Creación y entrenamiento del modelo SVM
#########################################
model = SVC(C = 20.0, gamma= 0.5999999999999998, kernel='poly')

"""## 6. Train the model"""

model.fit(X_train, y_train)


"""## 6. Evaluating the model with the training data"""

# Evaluación con datos de entrenamiento
training_predict = model.predict(X_train)

print("Reporte de clasificación (entrenamiento):")
print(metrics.classification_report(y_train, training_predict, digits = 3))

print("Matriz de confusión (entrenamiento):")
print(metrics.confusion_matrix(y_train, training_predict))

print(f'Model accuracy: {round(metrics.accuracy_score(y_train, training_predict)*100,2)}%')

"""7. Hyperlane Train Data Visualization"""

# x_grid, y_grid = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
#                              np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))
# z_grid = np.zeros_like(x_grid)


# num_features = 40  # Ensure this matches the number of features used during training

# for i in range(len(x_grid)):
#     for j in range(len(y_grid)):
#         feature_vector = np.zeros(num_features)  # Initialize with zeros
#         feature_vector[0] = x_grid[i, j]  # Set the first feature
#         feature_vector[1] = y_grid[i, j]  # Set the second feature
#         # Leave other features as 0 or assign their mean value if available
#         z_grid[i, j] = model.decision_function([feature_vector])

# fig = go.Figure()

# fig.add_trace(go.Scatter3d(x=X_train[:, 0], y=X_train[:, 1], z=X_train[:, 2], mode='markers',
#                            marker=dict(size=5, color=y_train, opacity=0.7), name='Dados de Treinamento'))

# fig.add_trace(go.Surface(z=z_grid, x=x_grid, y=y_grid, opacity=0.5, colorscale='Bluered_r'))


# fig.update_layout(scene=dict(xaxis_title='Sepal Width (cm)',
#                              yaxis_title='Petal Length (cm)',
#                              zaxis_title='Petal Width (cm)'))

# fig.update_layout(width=1000, height=600)

# fig.show()

"""## 8. Evaluating the model with test data"""

# Evaluación con datos de entrenamiento
test_predict = model.predict(X_test)

print("Reporte de clasificación de prueba:")
print(metrics.classification_report(y_test, test_predict, digits = 3))

print("Matriz de confusión (prueba):")
print(metrics.confusion_matrix(y_test, test_predict))

print(f'Model accuracy: {round(metrics.accuracy_score(y_test, test_predict)*100,2)}%')

input("Para continuar con las gráficas ingrese cualquier tecla")

"""9. Hyperplane Test Data Visualization"""

# x_grid, y_grid = np.meshgrid(np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100),
#                              np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100))
# z_grid = np.zeros_like(x_grid)

# num_features = 40  # Ensure this matches the number of features used during training

# for i in range(len(x_grid)):
#     for j in range(len(y_grid)):
#         feature_vector = np.zeros(num_features)  # Initialize with zeros
#         feature_vector[0] = x_grid[i, j]  # Set the first feature
#         feature_vector[1] = y_grid[i, j]  # Set the second feature
#         # Leave other features as 0 or assign their mean value if available
#         z_grid[i, j] = model.decision_function([feature_vector])

# fig = go.Figure()


# fig.add_trace(go.Scatter3d(x=X_test[:, 0], y=X_test[:, 1], z=X_test[:, 2], mode='markers',
#                            marker=dict(size=5, color=y_test), name='Dados de Treinamento'))


# fig.add_trace(go.Surface(z=z_grid, x=x_grid, y=y_grid, opacity=0.5, colorscale='Bluered_r'))

# fig.update_layout(scene=dict(xaxis_title='Sepal Width (cm)',
#                              yaxis_title='Petal Length (cm)',
#                              zaxis_title='Petal Width (cm)'))

# fig.update_layout(width=1000, height=600)

# fig.show()

"""## 10. Obtaining the model to be implemented in the microcontroller"""

# print(port(model))

# """## 11. Saves the template in a .h file"""

# with open('./SVMClassifier/SVMClassifier.h', 'w') as file:
#     file.write(port(model))

# """## (BONUS) Hyperparameter tuning

# RandomizedSearchCV is a function provided by the scikit-learn library in Python, commonly used for hyperparameter tuning in machine learning models through cross-validation. This technique proves beneficial when dealing with an extensive search space for hyperparameters and aims to identify the most effective combination of values.

# Step-by-Step Explanation
# 1. Definition of Parameter Space:
# Before utilizing RandomizedSearchCV, one needs to specify a search space for the model's hyperparameters. Rather than providing a specific grid of values, distributions are defined for each hyperparameter.

# 2. Random Sampling:
# Instead of evaluating all conceivable combinations of hyperparameters (as in the case of GridSearchCV), RandomizedSearchCV randomly selects a fixed set of combinations for evaluation. This proves advantageous when dealing with a large search space.

# 3. Model Training:
# For each randomly selected set of hyperparameters, RandomizedSearchCV trains the model using cross-validation. The data is divided into folds, with the model being trained on some folds and evaluated on the remaining folds.

# 4. Performance Evaluation:
# Performance is measured using a specified metric (e.g., accuracy, F1-score). The objective is to find hyperparameters that maximize or minimize this metric, depending on the problem at hand (e.g., maximizing accuracy in a classification problem).

# 5. Selection of the Best Model:
# Upon completion of the random search, RandomizedSearchCV returns the set of hyperparameters that led to the best average performance during cross-validation.

# By employing RandomizedSearchCV, computational time can be saved compared to an exhaustive grid search (GridSearchCV), especially when dealing with a large search space. This efficiency stems from exploring a random sample of the hyperparameter space rather than evaluating all possible combinations.

# ### 2. Set Grid search for Combinations of Parameters
# """

params = {
    "C": np.arange(2, 100, 2), #0.01 A 1000 TIPICAMENTE VALORES LOGARITMICOS
    "gamma": np.arange(0.1, 1, 0.01),
    "kernel": ['linear', 'poly', 'rbf']
}

# """### 3. Define Performance Measure"""

auc = make_scorer(roc_auc_score)

# from sklearn.metrics import make_scorer, f1_score

# f1 = make_scorer(f1_score, average="weighted")
# best_model = RandomizedSearchCV(
#     model,
#     param_distributions=params,
#     random_state=42,
#     n_iter=2,
#     cv=5,
#     verbose=2,
#     n_jobs=-1,
#     return_train_score=True,
#     scoring=f1
# )

# from sklearn.metrics import make_scorer, average_precision_score

# pr_auc = make_scorer(average_precision_score, needs_proba=True)
# best_model = RandomizedSearchCV(
#     model,
#     param_distributions=params,
#     random_state=42,
#     n_iter=2,
#     cv=5,
#     verbose=2,
#     n_jobs=-1,
#     return_train_score=True,
#     scoring=pr_auc
# )

# """### 4. Runs the search for the best model

# STANDARD SCALER
# """
print("Entrando al abismo")
best_model = RandomizedSearchCV(model, param_distributions=params, random_state=42,
                                n_iter=2, cv=3, verbose=0, n_jobs=1,
                                return_train_score=True, scoring = auc)
print("Se logró")
best_model.fit(X_train, y_train)
print("Se relogró")
# """### 5. Report the best model"""

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            best_params = results['params'][candidate]
            print("Best parameters found:")
            for param, value in best_params.items():
                print("  {0}: {1}".format(param, value))
            print("")

report_best_scores(best_model.cv_results_, 1)

# from sklearn.datasets import load_digits
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# X, y = load_digits(return_X_y=True)
# naive_bayes = GaussianNB()
# svc = SVC(kernel="rbf", gamma=0.001)

# import matplotlib.pyplot as plt
# import numpy as np

# from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

# common_params = {
#     "X": X,
#     "y": y,
#     "train_sizes": np.linspace(0.1, 1.0, 5),
#     "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
#     "score_type": "both",
#     "n_jobs": 4,
#     "line_kw": {"marker": "o"},
#     "std_display_style": "fill_between",
#     "score_name": "Accuracy",
# }

# for ax_idx, estimator in enumerate([naive_bayes, svc]):
#     LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
#     handles, label = ax[ax_idx].get_legend_handles_labels()
#     ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
#     ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")