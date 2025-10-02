import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# --- CONFIGURACIÓN DE CLASIFICACIÓN (CORREGIDA) ---
# ¡BASE_DIR ahora está sin espacios! Debe coincidir con el nombre de tu carpeta.
BASE_DIR = 'Imagenes' 
TARGET_SIZE = (100, 100) 
# Clases con la capitalización exacta de tus carpetas
IMAGE_CLASSES = ['AVE', 'PERRO', 'Pez'] 
TEST_SIZE_RATIO = 0.2  
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

# ----------------------------------------------------------------------
# --- 1. FUNCIÓN DE EXTRACCIÓN Y CARGA ---
# ----------------------------------------------------------------------

def load_and_preprocess_images(base_directory, classes, size, extensions):
    """Carga imágenes, las redimensiona y crea vectores de características con etiquetas."""
    data = []    
    labels = []  
    
    # DIAGNÓSTICO: Muestra la ruta de búsqueda
    current_working_dir = os.getcwd()
    expected_path = os.path.join(current_working_dir, base_directory)
    print(f"\n--- DIAGNÓSTICO DE RUTA FINAL ---")
    print(f"Buscando carpeta principal de datos en: {expected_path}")
    print(f"Subcarpetas esperadas: {classes}")
    print("----------------------------------\n")

    print(f"Iniciando carga de imágenes desde '{base_directory}'...")

    for class_name in classes:
        # Se construye la ruta: ImagenesClasificacion/AVE
        class_path = os.path.join(base_directory, class_name)
        
        if not os.path.isdir(class_path):
            print(f"ADVERTENCIA: La carpeta de clase '{class_path}' NO existe. Saltando.")
            continue
            
        print(f"-> Procesando clase: {class_name}...")
        
        for file_name in os.listdir(class_path):
            file_extension = os.path.splitext(file_name)[1].lower()
            
            # 1. Filtrar solo archivos de imagen válidos
            if file_extension not in extensions:
                continue

            file_path = os.path.join(class_path, file_name)
            
            # 2. Cargar imagen en escala de grises
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue

            # 3. Redimensionar la imagen a 100x100
            img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            # 4. Aplanar la matriz 2D a un vector 1D
            feature_vector = img_resized.flatten()
            
            data.append(feature_vector)
            labels.append(class_name)
                
    return np.array(data), np.array(labels)

# ----------------------------------------------------------------------
# --- 2. EJECUCIÓN DEL FLUJO DE CLASIFICACIÓN ---
# ----------------------------------------------------------------------

# Cargar y preprocesar los datos
X, y = load_and_preprocess_images(BASE_DIR, IMAGE_CLASSES, TARGET_SIZE, VALID_EXTENSIONS)

if len(X) == 0:
    print("\n--- ERROR CRÍTICO ---")
    print("No se encontraron imágenes válidas. Las carpetas de clase están vacías o la ruta sigue siendo incorrecta.")
    exit()

print(f"\nTotal de imágenes cargadas: {len(X)}")
print(f"Dimensión del vector de características (Pixeles por imagen): {X.shape[1]}")
print("--- DATOS PREPARADOS ---")

# 3. Separar los datos para entrenamiento y prueba (80%/20%)
(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=TEST_SIZE_RATIO, random_state=42, stratify=y
)

print(f"Conjunto de Entrenamiento: {len(X_train)} muestras")
print(f"Conjunto de Prueba: {len(X_test)} muestras")


# 4. Entrenar el Clasificador (K-Nearest Neighbors)
print("\n--- ENTRENAMIENTO DEL MODELO (K-NN) ---")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 5. Evaluar el modelo
print("\n--- REPORTE DE EVALUACIÓN DE LA CLASIFICACIÓN ---")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=IMAGE_CLASSES))