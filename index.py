import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- 0. Configuración Inicial y Stopwords ---

# Descargar las stopwords si aún no están presentes en NLTK
try:
    stopwords.words('spanish')
except LookupError:
    print("Descargando el paquete 'stopwords' de NLTK...")
    nltk.download('stopwords')


# --- 1. Datos de Ejemplo (Titulares) ---

titulares = [
    "La bolsa de valores sube por buenos resultados financieros",
    "El Banco Central subió las tasas de interés para controlar la inflación",
    "Analistas esperan una recesión económica mundial",
    "El actor principal de la película ganó un Oscar a mejor director",
    "Críticas positivas para el nuevo documental de ciencia ficción",
    "El nuevo videojuego espacial rompe récords de venta en la semana",
    "El FMI revisa la proyección de crecimiento del PIB global",
    "Estrellas de cine asisten a la gala de premios en Los Ángeles",
]
print("Total de titulares cargados:", len(titulares))


# --- 2. Preprocesamiento del Texto ---

stop_words_es = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar puntuación y caracteres especiales
    texto = re.sub(r'[^\w\s]', '', texto)
    # Tokenizar y eliminar stopwords
    palabras = texto.split()
    palabras = [palabra for palabra in palabras if palabra not in stop_words_es]
    return " ".join(palabras)

# Aplicar la limpieza
titulares_limpios = [limpiar_texto(t) for t in titulares]


# --- 3. Vectorización con TF-IDF ---

# TF-IDF convierte el texto en una matriz numérica
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(titulares_limpios)

print("Matriz TF-IDF creada. Dimensiones:", X.shape)


# --- 4. Aplicación de K-Means ---

K = 3 # Número de clusters a buscar

# Entrenar el modelo K-Means
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X)

# Asignar las etiquetas de cluster a cada titular
etiquetas_cluster = kmeans.labels_

# Unir resultados al DataFrame original
df_titulares = pd.DataFrame({'Titular': titulares, 'Cluster': etiquetas_cluster})

print("\n--- Resultados del Clustering K-Means ---")
print(df_titulares.sort_values(by='Cluster'))


# --- 5. Visualización con PCA (Reducción a 2D) ---

# a) Aplicar PCA para reducir las dimensiones
# SOLUCIÓN AL ERROR: Usamos .toarray() para obtener un numpy.ndarray,
# compatible con las últimas versiones de scikit-learn y numpy.
X_array = X.toarray() 
pca = PCA(n_components=2)
componentes_principales = pca.fit_transform(X_array)

# b) Preparar los datos para Matplotlib
pca_df = pd.DataFrame(data=componentes_principales, columns=['PCA_1', 'PCA_2'])
pca_df['Cluster'] = etiquetas_cluster

# c) Graficar los Clusters
plt.figure(figsize=(10, 8))
colores = ['#FF5733', '#335EFF', '#33FF57'] # Rojo, Azul, Verde

for i in range(K):
    cluster_data = pca_df[pca_df['Cluster'] == i]
    
    plt.scatter(cluster_data['PCA_1'], cluster_data['PCA_2'], 
                label=f'Cluster {i}', 
                color=colores[i], 
                alpha=0.8,
                s=100)

plt.title(f'Visualización de Clusters de Titulares (PCA a K={K})', fontsize=14)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Grupo Temático', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()