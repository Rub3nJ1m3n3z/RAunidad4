import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Datos de ejemplo (edad, ingresos)
X = np.array([
    [25, 2000],
    [30, 2500],
    [35, 3000],
    [40, 3500],
    [20, 1500],
    [50, 4000],
    [55, 4500],
    [60, 5000]
])

# Crear el modelo K-Means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)

# Entrenar el modelo
kmeans.fit(X)

# Obtener etiquetas (clusters)
labels = kmeans.labels_

# Obtener centroides
centroids = kmeans.cluster_centers_

# Graficar resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
plt.xlabel('Edad')
plt.ylabel('Ingresos')
plt.title('Clustering con K-Means')
plt.show()