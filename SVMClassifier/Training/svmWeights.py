import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder

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


# 1. Load the dataset
ruta_csv = r'Instrucciones\Registros almacenados\SVM_combined\Arms\Sebastian\Arms_SVM_1.csv'

lista_filas = leer_csv_a_arreglo(ruta_csv)
data_dict = lista_a_diccionario(lista_filas)


data_dict['target'] = [int(x) for x in data_dict['target']]
data_dict['data'] = [[float(val) for val in fila] for fila in data_dict['data']]


feature_columns = [f"f{i+1}" for i in range(16)]
df = pd.DataFrame(data_dict['data'], columns=feature_columns)
df['target'] = data_dict['target']

columnas_a_eliminar = ['f2', 'f5', 'f10']       #Verdadera manera de quitar columnas
df = df.drop(columns=columnas_a_eliminar)

df = df[df["target"] != 0].reset_index(drop=True)           #idle
#df = df[df["target"] != 1].reset_index(drop=True)           #pierna derecha
#df = df[df["target"] != 2].reset_index(drop=True)           #pierna derecha


scaler = StandardScaler()

df[feature_columns] = scaler.fit_transform(df[feature_columns])
# Encode labels and standardize features
X = df[feature_columns].to_numpy()
y = df['target'].to_numpy()

y = LabelEncoder().fit_transform(y)

# 3. Train/test split (for stability in permutation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Define and train SVM model with RBF kernel
model = make_pipeline(
    StandardScaler(),
    SVC(C=37275.93720314938, gamma=232.99518105153624, kernel='rbf')
)
model.fit(X_train, y_train)

# 5. Compute permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# 6. Create importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': result.importances_mean,
    'Std': result.importances_std
}).sort_values('Importance', ascending=False)

# 7. Show top features
print("Feature importance (sorted):\n", feature_importance)

# 8. Plot importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Mean Importance')
plt.title('Permutation Feature Importance (RBF SVM)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
