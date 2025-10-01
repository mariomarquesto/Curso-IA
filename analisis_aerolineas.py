import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN DE ARCHIVOS Y COLUMNAS ---
# !!! AJUSTA EL NOMBRE DEL ARCHIVO DE EXCEL AQUÍ SI ES NECESARIO !!!
# Si tu archivo se llama 'Comentarios Aerolinea.xlsx - Hoja1.csv' AÚN cuando es Excel, 
# debes ajustarlo aquí, pero usa pd.read_excel para leerlo.
FILE_NAME = 'Comentarios Aerolinea.xlsx' 
TEXT_COLUMN_NAME = 'Comentario' 
SHEET_NAME = 'Hoja1' 


# --- 1. LÉXICO DE POLARIDAD EN ESPAÑOL (Simulación) ---

LEXICON_ES = {
    'positive': {
        'excelente': 1.0, 'amo': 0.8, 'increíble': 0.9, 'feliz': 0.7, 
        'bueno': 0.5, 'mejor': 1.0, 'rápido': 0.6, 'cómodo': 0.7, 
        'amable': 0.8, 'fantástico': 0.9, 'gran': 0.6, 'limpio': 0.5,
        'maravilla': 0.9, 'servicial': 0.7, 'eficiente': 0.6, 'atento': 0.7,
        'suave': 0.6, 'seguro': 0.8, 'claro': 0.5 
    },
    'negative': {
        'terrible': -1.0, 'mal': -0.7, 'horrible': -0.9, 'decepcionado': -0.8,
        'pésimo': -0.6, 'odio': -0.9, 'problema': -0.5, 'retraso': -0.7,
        'sucio': -0.6, 'lento': -0.5, 'caro': -0.4, 'peor': -1.0,
        'falla': -0.8, 'ineficiente': -0.7, 'intransigente': -0.8, 'incómodo': -0.6,
        'descortés': -0.7, 'caótico': -0.5, 'obsoleto': -0.4, 'malo': -0.6
    }
}


# --- 2. FUNCIÓN DE ANÁLISIS DE SENTIMIENTO ---

def analyze_sentiment_lexicon_es(text):
    """Asigna un puntaje de sentimiento basado en el léxico definido."""
    if pd.isna(text):
        return 0, 'N/A'
    
    score = 0
    text = re.sub(r'[^a-zA-Záéíóúüñ\s]', '', str(text).lower())
    words = text.split()
    
    for word in words:
        if word in LEXICON_ES['positive']:
            score += LEXICON_ES['positive'][word]
        elif word in LEXICON_ES['negative']:
            score += LEXICON_ES['negative'][word]
            
    # Clasificación usando umbrales
    if score > 0.3:
        sentiment = 'Positivo'
    elif score < -0.3:
        sentiment = 'Negativo'
    else:
        sentiment = 'Neutral'
        
    return score, sentiment

# --- 3. CARGA DE DATOS Y APLICACIÓN DEL ANÁLISIS ---

# Construir la ruta absoluta del archivo Excel relativa al script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, FILE_NAME)


try:
    # Cargar el archivo EXCEL (¡CLAVE!)
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
    
    print(f"Total de comentarios cargados: {len(df)}")
    
    # Aplicar la función de análisis
    df[['Score', 'Sentimiento_Lexico']] = df[TEXT_COLUMN_NAME].apply(
        lambda x: analyze_sentiment_lexicon_es(x)
    ).apply(pd.Series)

except KeyError:
    print(f"\n--- ERROR DE COLUMNA: {TEXT_COLUMN_NAME} ---")
    print("El archivo se cargó, pero no se encontró la columna de texto. Verifica su nombre.")
    exit()
except Exception as e:
    print(f"\n--- ERROR CRÍTICO AL CARGAR DATOS ---")
    print(f"Error: {e}")
    print(f"\nSOLUCIÓN: El archivo '{FILE_NAME}' o la hoja '{SHEET_NAME}' no fueron encontrados.")
    print(f"          Ruta intentada: {file_path}")
    print("\nREVISA 1: ¿Instalaste 'pip install openpyxl'?")
    print("REVISA 2: El archivo Excel debe estar en el mismo lugar que el script.")
    print("REVISA 3: El nombre del archivo o de la hoja ('Hoja1') es INCORRECTO.")
    exit()

# --- 4. MOSTRAR RESULTADOS Y RESUMEN ---

print("\n--- Resultados del Análisis de Sentimiento Léxico ---")
print("Muestra de los 5 primeros comentarios con sentimiento asignado:")
# Se asume que la columna 'Clase' existe, si no, bórrala de la línea de abajo.
print(df[[TEXT_COLUMN_NAME, 'Clase', 'Score', 'Sentimiento_Lexico']].head())

print("\n--- Resumen de la Distribución de Sentimientos (Generado por Léxico) ---")
sentiment_counts = df['Sentimiento_Lexico'].value_counts()
print(sentiment_counts)


# --- 5. GENERACIÓN DEL GRÁFICO (Visualización) ---

colors = {'Positivo': 'green', 'Negativo': 'red', 'Neutral': 'blue', 'N/A': 'gray'}
sentiment_order = ['Positivo', 'Negativo', 'Neutral', 'N/A']

plt.figure(figsize=(9, 6))

sentiment_counts = sentiment_counts.reindex(sentiment_order, fill_value=0)

bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
               color=[colors.get(s, 'gray') for s in sentiment_counts.index], alpha=0.8)

plt.title('Distribución de Sentimientos en Comentarios de Aerolínea (Análisis Léxico)', fontsize=14)
plt.xlabel('Polaridad de Sentimiento', fontsize=12)
plt.ylabel('Número de Comentarios', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Añadir las etiquetas de conteo
for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.01, int(yval), 
                 ha='center', va='bottom', fontsize=10)

plt.show()