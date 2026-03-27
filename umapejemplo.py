import umap
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

# Crear modelo UMAP
modelo = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)

# Ajustar y transformar
X_reducido = modelo.fit_transform(X)

# Graficar resultados
plt.scatter(X_reducido[:, 0], X_reducido[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.title("UMAP - Reducción de Dimensionalidad (Iris)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()