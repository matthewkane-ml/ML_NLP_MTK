# Clasificador de Spam con NLP

> Un detector de spam en URLs construido con vectorización TF-IDF y una Máquina de Vectores de Soporte ajustada — clasifica si una URL es spam o legítima.

---

## Problema

Las URLs de spam son uno de los vectores más comunes de phishing, malware y contenido no deseado. Revisar URLs manualmente a escala es impracticable, por lo que este proyecto construye un clasificador binario basado en NLP que analiza el contenido textual de una URL y predice si es spam. El pipeline cubre el flujo completo de NLP: limpieza de texto, lematización, vectorización y ajuste del modelo.

## Conjunto de datos

- **Fuente:** [4GeeksAcademy NLP Project Tutorial](https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv)
- **Tamaño:** CSV público de URLs etiquetadas
- **Variable objetivo:** `is_spam` (0 = Legítima, 1 = Spam)
- **Características clave:** Texto crudo de la URL, preprocesado y vectorizado con TF-IDF (top 5.000 características)

## Metodología

1. **Limpieza de texto:** Eliminación de caracteres no alfabéticos, tokens de un solo carácter y espacios en blanco múltiples con regex. Todo en minúsculas.
2. **Lematización y eliminación de stopwords:** Aplicación de `WordNetLemmatizer` de NLTK, eliminación de stopwords en inglés y descarte de tokens de menos de 3 caracteres.
3. **Vectorización:** `TfidfVectorizer` con `max_features=5000`, `max_df=0.8` y `min_df=5` para convertir los tokens de URL en una matriz de características dispersa.
4. **Modelo base:** SVC lineal para establecer un rendimiento de referencia.
5. **Ajuste de hiperparámetros:** `GridSearchCV` (5 pliegues) sobre tipo de kernel, `C`, `degree` y `gamma`. Mejor configuración: `kernel=poly`, `C=1000`, `degree=1`, `gamma=auto`.
6. **Exportación del modelo:** Modelo optimizado guardado con `pickle` para despliegue futuro.

## Resultados

El SVM optimizado (kernel polinomial, C=1000) superó al modelo base lineal. La precisión en el conjunto de prueba (20% de separación) fue evaluada con `accuracy_score`. SVM con TF-IDF es una combinación probada para clasificación de texto — el kernel polinomial con grado bajo captura eficazmente patrones de co-ocurrencia de tokens sin sobreajuste.

## Tecnologías utilizadas

`Python` · `pandas` · `scikit-learn` · `NLTK` · `regex` · `TF-IDF` · `pickle`

## Ejecución local

```bash
git clone https://github.com/matthewkane-ml/ML_NLP_MTK.git
cd ML_NLP_MTK
pip install -r requirements.txt
jupyter notebook src/explore.ipynb
```

## Próximos pasos

- Evaluar métricas adicionales más allá de la exactitud — precisión, recall y F1 son más importantes para datasets de spam desbalanceados
- Probar Naive Bayes como referencia, que frecuentemente da resultados sorprendentemente buenos en tareas de texto tipo bag-of-words
- Experimentar con n-gramas a nivel de carácter o tokenización específica del dominio para capturar mejor la estructura de la URL (subdominios, TLDs, segmentos de ruta)
- Añadir una capa de despliegue — una interfaz Flask o Streamlit para verificación de URLs en tiempo real

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-kane-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
