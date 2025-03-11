import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from micromlgen import port
import joblib

# 1) Función para leer y procesar el CSV
def cargar_datos(ruta_csv):
    """
    - Lee el CSV con encabezado y separador ';'.
    - 'Estímulo' es la primera columna (etiqueta).
    - C1, C2, C3, C4, C5 son columnas que tienen 4 valores cada una (separados por coma).
    """
    df = pd.read_csv(ruta_csv, sep=';', header=0)
    etiquetas = df['Estimulo'].values
    df.drop(columns=['Estimulo'], inplace=True)
    datos = []
    for _, row in df.iterrows():
        fila = []
        for col in ['C1', 'C2', 'C3', 'C4', 'C5']:
            # Ejemplo: "4103097.967085,164502337.041529,28.679824,86.163472"
            valores_str = row[col].split(',')
            fila.extend([float(v.strip()) for v in valores_str])
        datos.append(fila)
    return etiquetas, np.array(datos)

# 2) Construcción del DataFrame
def preparar_dataframe(etiquetas, datos):
    num_features = datos.shape[1]  # Debería ser 20 (5 columnas x 4 valores)
    columnas = [f"f{i+1}" for i in range(num_features)]
    df = pd.DataFrame(datos, columns=columnas)
    df['target'] = etiquetas
    return df

# 3) Visualización 3D (usando f1, f2 y f3)
def visualizar_datos(df):
    fig = px.scatter_3d(
        df, x='f1', y='f2', z='f3',
        color=df['target'].astype(str),
        title="Visualización 3D de los datos"
    )
    fig.show()

# 4) División de datos
def dividir_datos(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5) Entrenamiento y evaluación del modelo SVM
def entrenar_svm(X_train, X_test, y_train, y_test):
    modelo = SVC(probability=True, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    print("Reporte de Clasificación (Entrenamiento):")
    print(classification_report(y_train, y_pred_train))
    print("Reporte de Clasificación (Prueba):")
    print(classification_report(y_test, y_pred_test))
    print("Matriz de Confusión (Prueba):")
    print(confusion_matrix(y_test, y_pred_test))
    return modelo

# 6) Exportación del modelo para microcontroladores
def exportar_modelo(modelo, nombre_archivo="modelo.h"):
    with open(nombre_archivo, "w") as f:
        f.write(port(modelo))
    print(f"Modelo exportado a {nombre_archivo}")

# --- Función de scoring personalizada ---
def multiclass_roc_auc_score(y_true, y_proba, **kwargs):
    """
    Calcula el ROC AUC.
      - En clasificación binaria, si y_proba es 2D se utiliza la probabilidad de la clase positiva (columna 1); 
        si es 1D, se usa directamente.
      - En clasificación multiclase, se espera que y_proba sea 2D; si no lo es, se devuelve 0.
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        # Caso binario
        if y_proba.ndim == 2:
            # Usamos la probabilidad de la clase positiva (columna 1)
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            return roc_auc_score(y_true, y_proba)
    else:
        # Caso multiclase: se espera un arreglo 2D con probabilidades para cada clase.
        if y_proba.ndim != 2:
            return 0  # O se podría lanzar una excepción o manejarlo de otra forma
        return roc_auc_score(y_true, y_proba, multi_class='ovr')

# 7) Optimización de hiperparámetros con RandomizedSearchCV
def optimizar_svm(X_train, y_train):
    roc_auc_scorer = make_scorer(multiclass_roc_auc_score, needs_proba=True)
    param_distributions = {
        'C': np.logspace(-3, 3, 10),
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 10)),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    random_search = RandomizedSearchCV(
        estimator=SVC(probability=True),
        param_distributions=param_distributions,
        scoring=roc_auc_scorer,
        n_iter=20,
        cv=5,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print("Mejores parámetros encontrados:", random_search.best_params_)
    return random_search.best_estimator_

# --- Ejecución completa ---
if __name__ == "__main__":
    ruta_csv = r"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\SVM characteristics\SVM_1.csv"  # Ajusta la ruta a tu CSV

    # Cargar y procesar datos
    etiquetas, datos = cargar_datos(ruta_csv)
    df = preparar_dataframe(etiquetas, datos)
    visualizar_datos(df)
    X_train, X_test, y_train, y_test = dividir_datos(df)
    
    # Entrenar y evaluar modelo base
    modelo = entrenar_svm(X_train, X_test, y_train, y_test)
    
    # Optimizar hiperparámetros
    mejor_modelo = optimizar_svm(X_train, y_train)
    
    # Evaluación con el modelo optimizado
    print("\nEvaluación con el modelo optimizado:")
    y_pred_test_opt = mejor_modelo.predict(X_test)
    print(classification_report(y_test, y_pred_test_opt))
    print("Matriz de Confusión (modelo optimizado):")
    print(confusion_matrix(y_test, y_pred_test_opt))
    
    # Exportar modelo optimizado
    exportar_modelo(mejor_modelo, "modelo_optimizado.h")
