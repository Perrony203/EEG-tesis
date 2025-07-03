import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # Datos
conf_train = np.array([
     [232, 48],
     [105, 161]
   
 ])
acc_train =71.98

conf_test = np.array([
     [81, 29],
     [58, 66]
 ])
acc_test = 62.82


# # Datos
# conf_train = np.array([
#     [115, 5, 15, 21],
#     [12, 99, 20, 26],
#     [16, 17, 104, 29],
#     [12, 19, 22, 98]

# ])
# acc_train = 66.03

# conf_test = np.array([
#     [28, 9, 19, 15],
#     [7, 13, 24, 24],
#     [12, 11, 17, 19],
#     [17, 13, 18, 26]
# ])
# acc_test = 31.11

# # Etiquetas de clases (ajusta según tu caso)
# labels = ['PD', 'PI', 'BI', 'BD']

# # Función para graficar matrices de confusión
# def plot_conf_matrix(conf_matrix, accuracy, dataset_name, save_path, figsize=(7,6)):
#     plt.figure(figsize=figsize)
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=labels, yticklabels=labels, cbar=False, square=True,
#                 annot_kws={"size": 10})
    
#     plt.xlabel('Predicción', fontsize=12)
#     plt.ylabel('Valor real', fontsize=12)
#     plt.title(f'Matriz de confusión ({dataset_name})\nPrecisión: {accuracy:.2f}%', fontsize=14)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()

# # Guardar imágenes
# plot_conf_matrix(conf_train, acc_train, "Entrenamiento", "conf_matrix_train_multiclase.png", figsize=(7,6))
# plot_conf_matrix(conf_test, acc_test, "Prueba", "conf_matrix_test_multiclase.png", figsize=(7,6))



# # Labels
labels = ['LD', 'LI']

# # Function to plot confusion matrix
def plot_conf_matrix(conf_matrix, accuracy, dataset_name, save_path):
     plt.figure(figsize=(5, 4))
     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=labels, yticklabels=labels, cbar=False)
    
     plt.xlabel('Predicción')
     plt.ylabel('Valor real')
     plt.title(f'Matriz de confusión ({dataset_name})\nExactitud: {accuracy:.2f}%', fontsize=14)
   
     plt.tight_layout()
     plt.savefig(save_path, dpi=300)
     plt.close()

 # Plot and save both matrices
plot_conf_matrix(conf_train, acc_train, "Entrenamiento", "conf_matrix_train.png")
plot_conf_matrix(conf_test, acc_test, "Prueba", "conf_matrix_test.png")
