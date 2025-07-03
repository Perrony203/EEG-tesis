import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # # Datos
# conf_train = np.array([
#      [145, 139],
#      [0, 291]
   
#  ])
# acc_train = 75.83

# conf_test = np.array([
#      [71, 56],
#      [0, 120]
#  ])
# acc_test = 77.33

 # Datos
conf_train = np.array([
    [160, 2, 1, 8, 24],
    [19, 100, 29, 9, 38],
    [18, 31, 127, 8, 11],
    [18, 7, 2, 56, 112],
    [27, 6, 4, 21, 137]

])
acc_train = 59.49

conf_test = np.array([
    [13, 8, 19, 20],
    [13, 31, 9, 12],
    [11, 4, 30, 14],
    [6, 3, 18, 23]
])
acc_test = 41.45

# Etiquetas de clases (ajusta según tu caso)
labels = ['Estado Basal', 'PD', 'PI', 'BI', 'BD']
# Función para graficar matrices de confusión
def plot_conf_matrix(conf_matrix, accuracy, dataset_name, save_path, figsize=(7,6)):
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False, square=True,
                annot_kws={"size": 10})
  
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor real', fontsize=12)
    plt.title(f'Matriz de confusión ({dataset_name})\nExactitud: {accuracy:.2f}%', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

 # Guardar imágenes
plot_conf_matrix(conf_train, acc_train, "Pruebas Offline", "conf_matrix_train_multiclase.png", figsize=(7,6))
plot_conf_matrix(conf_test, acc_test, "Prueba", "conf_matrix_test_multiclase.png", figsize=(7,6))



# # # Labels
# labels = ['Movimiento', 'Estado Basal']

# # # Function to plot confusion matrix
# def plot_conf_matrix(conf_matrix, accuracy, dataset_name, save_path):
#      plt.figure(figsize=(5, 4))
#      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#                  xticklabels=labels, yticklabels=labels, cbar=False)
    
#      plt.xlabel('Predicción')
#      plt.ylabel('Valor real')
#      plt.title(f'Matriz de confusión ({dataset_name})\nPrecisión: {accuracy:.2f}%', fontsize=14)
   
#      plt.tight_layout()
#      plt.savefig(save_path, dpi=300)
#      plt.close()

#  # Plot and save both matrices
# plot_conf_matrix(conf_train, acc_train, "Entrenamiento", "conf_matrix_train.png")
# plot_conf_matrix(conf_test, acc_test, "Prueba", "conf_matrix_test.png")
