import re
import pandas as pd
import os
import matplotlib.pyplot as plt # Importamos Matplotlib para el gráfico

# --- 1. Cargar Stopwords y Definir Léxico de Polaridad ---

# Ruta al archivo de stopwords. Asume que está en el mismo directorio.
STOPWORDS_FILE = 'stopwords.txt'

def load_stopwords(filepath):
    """Carga la lista de stopwords desde un archivo."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Leemos cada línea, eliminamos espacios en blanco y convertimos a un set
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}. Asegúrate de que está en el mismo directorio.")
        return set()

# Cargar las stopwords del archivo
STOPWORDS_EN = load_stopwords(STOPWORDS_FILE)

# --- Léxico de Polaridad (Simulación) ---
# Necesitamos un diccionario para puntuar. Esto simula el "conocimiento" de emociones.
LEXICON = {
    'positive': {
        'amazing': 1.0, 'love': 0.8, 'excellent': 0.9, 'happy': 0.7, 
        'great': 0.6, 'good': 0.5, 'best': 1.0, 'winning': 0.8, 'better': 0.6 
    },
    'negative': {
        'terrible': -1.0, 'bad': -0.7, 'horrible': -0.9, 'disappointed': -0.8,
        'awful': -0.6, 'hate': -0.9, 'problem': -0.5, 'failure': -0.7
    }
}


# --- 2. Función de Limpieza y Puntuación ---

def analyze_sentiment_lexicon(text):
    """
    Realiza un análisis de sentimiento simple basado en léxico.
    """
    score = 0
    
    # 1. Limpieza y Tokenización
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    
    # 2. Eliminar Stopwords
    meaningful_words = [word for word in words if word not in STOPWORDS_EN]
    
    # 3. Puntuación (Scoring)
    for word in meaningful_words:
        # Buscar en léxicos positivos
        if word in LEXICON['positive']:
            score += LEXICON['positive'][word]
        # Buscar en léxicos negativos
        elif word in LEXICON['negative']:
            score += LEXICON['negative'][word]
            
    # 4. Clasificación
    # Usamos un umbral (threshold) simple para clasificar
    if score > 0.2:
        sentiment = 'Positive'
    elif score < -0.2:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return score, sentiment

# --- 3. Ejecución del Análisis ---

reviews = [
    "The new service is amazing and much better than the old one.",
    "It was a terrible experience, the service was awful and a complete failure.",
    "I have a problem with the delay, but the product itself is excellent.",
    "The company's performance was great this year.",
]

results = []
print("--- Análisis de Sentimiento Lexicon-Based ---")

for review in reviews:
    sentiment_score, final_sentiment = analyze_sentiment_lexicon(review)
    
    results.append({
        'Review': review,
        'Score': round(sentiment_score, 2),
        'Sentiment': final_sentiment
    })
    
# Crear DataFrame de resultados
df_results = pd.DataFrame(results)

print("\nResultados del Análisis:")
print(df_results)

# --- 4. Generación del Gráfico ---

# Contar la frecuencia de cada sentimiento
sentiment_counts = df_results['Sentiment'].value_counts()

# Colores para el gráfico
colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
# Asegurarse de que el orden de las barras sea consistente
sentiment_order = ['Positive', 'Negative', 'Neutral']

# Graficar
plt.figure(figsize=(8, 5))

# Usamos reindex para asegurar que el orden de las barras sea el deseado,
# incluso si falta alguna categoría (ej. si no hay reseñas 'Neutral').
sentiment_counts = sentiment_counts.reindex(sentiment_order, fill_value=0)

bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
               color=[colors[s] for s in sentiment_counts.index], alpha=0.8)

plt.title('Distribución de Sentimientos en las Reseñas', fontsize=14)
plt.xlabel('Polaridad de Sentimiento', fontsize=12)
plt.ylabel('Número de Reseñas', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Añadir las etiquetas de conteo encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom', fontsize=10)

plt.show()