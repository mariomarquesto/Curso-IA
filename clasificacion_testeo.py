import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# --- CONFIGURACIÓN DE RUTAS Y CLASES ---
# ¡IMPORTANTE! Nombre de la única carpeta de datos organizada
BASE_DIR = 'DATOS_TOTALES' 
# Parámetros
TARGET_SIZE = (100, 100) 
IMAGE_CLASSES = ['AVE', 'PERRO', 'Pez'] 
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
TEST_SIZE_RATIO = 0.2 

# ----------------------------------------------------------------------
# --- 1. FUNCIÓN DE CARGA Y PREPROCESAMIENTO ---
# ----------------------------------------------------------------------

def load_and_preprocess_images(base_directory, classes, size, extensions):
    """Carga TODAS las imágenes del único directorio de datos organizado."""
    data = []    
    labels = []  
    
    print(f"\n--- INICIANDO CARGA DE DATOS ---")
    print(f"Buscando clases en la carpeta base: {base_directory}")

    for class_name in classes:
        # Se verifica la ruta completa: DATOS_TOTALES/AVE
        class_path = os.path.join(base_directory, class_name)
        
        if not os.path.isdir(class_path):
            print(f"ERROR: Falta la subcarpeta de clase '{class_path}'.")
            # Devolvemos un array vacío si falta alguna carpeta de clase
            return np.array([]), np.array([])
            
        print(f"-> Procesando clase: {class_name}...")
        
        for file_name in os.listdir(class_path):
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension not in extensions:
                continue

            file_path = os.path.join(class_path, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue

            img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            feature_vector = img_resized.flatten()
            
            data.append(feature_vector)
            labels.append(class_name)
                
    return np.array(data), np.array(labels)

# ----------------------------------------------------------------------
# --- 2. FLUJO DE CLASIFICACIÓN (Entrenamiento y Testeo Interno) ---
# ----------------------------------------------------------------------

X, y = load_and_preprocess_images(BASE_DIR, IMAGE_CLASSES, TARGET_SIZE, VALID_EXTENSIONS)

if len(X) == 0:
    print(f"\nERROR CRÍTICO: No se encontraron imágenes válidas. La carpeta '{BASE_DIR}' debe contener subcarpetas con las imágenes.")
    exit()

print(f"\nTotal de imágenes cargadas: {len(X)}")

# 3. Separar los datos para entrenamiento (80%) y testeo (20%)
(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=TEST_SIZE_RATIO, random_state=42, stratify=y
)

print(f"Conjunto de Entrenamiento (80%): {len(X_train)} muestras")
print(f"Conjunto de Prueba (20%): {len(X_test)} muestras")

# 4. Entrenar el modelo K-NN
print("\n--- ENTRENAMIENTO DEL MODELO (K-NN) ---")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 5. Evaluar (Genera el reporte final de tu testeo)
print("\n--- REPORTE DE EVALUACIÓN FINAL ---")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=IMAGE_CLASSES, zero_division=0))