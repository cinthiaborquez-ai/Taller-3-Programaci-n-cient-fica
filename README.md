Taller 3 Programación Científica
Integrantes: Cinthia Bórquez y Martina Torres
Fecha de entrega: 7 de Julio de 2026
Universidad Católica del norte

Descripción
Sistema cliente-servidor para el análisis interactivo del corpus bíblico (King James Version).  
La API desarrollada en FastAPI realiza todo el procesamiento textual, mientras que la aplicación  
Streamlit actúa como interfaz de usuario consumiendo los endpoints de la API.

Estructura del proyecto
├── api/
│   ├── routers/
│   │   ├── dashboard.py       # Endpoints del dashboard principal
│   │   ├── search.py          # Motor de búsqueda semántico (TF-IDF)
│   │   ├── generator.py       # Generador de texto con n-gramas
│   │   └── vectorization.py   # PCA sobre TF-IDF y Word2Vec
│   ├── data_loader.py         # Carga y filtrado centralizado del corpus
│   └── main.py                # Punto de entrada de la API
├── streamlit_app/
│   └── app.py                 # Interfaz de usuario con 4 páginas
├── data/
│   ├── t_kjv.csv              # Corpus bíblico KJV original
│   ├── key_english.csv        # Nombres de libros y testamentos
│   ├── bible_kjv_clean.csv    # Dataset limpio generado automáticamente
│   └── prepare_data.py        # Script para generar el dataset limpio
└── README.md

Requisitos
- Python 3.10 o superior
- Las siguientes librerías:
---
Intalación y ejecución
1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd "Taller 3 progra"
```
2. Craer y activar entorno vitual
```bash
python -m venv venv --without-pip
python -m ensurepip --upgrade
.\venv\Scripts\Activate.ps1
```
3. Instalar dependencias
```bash
pip install fastapi uvicorn pandas scikit-learn wordcloud streamlit requests plotly numpy nltk
```
4. Preparar el dataset
```bash
cd data
python prepare_data.py
cd ..
```
5. Levantar la API (Terminal 1)
```bash
cd api
uvicorn main:app --reload --port 8000
```
- API disponible en: http://localhost:8000  
- Documentación interactiva en: http://localhost:8000/docs
6. Levantar la app Streamlit (Terminal 2)
```bash
cd streamlit_app
streamlit run app.py
```
- App disponible en: http://localhost:8501

Funcionalidades
Dashboard Principal
- Cantidad de versículos por libro
- Longitud promedio de versículos por libro
- Top N palabras más frecuentes (configurable con slider)
- Nube de palabras
- Filtros por testamento, libro y capítulo (todo procesado en la API)

Buscador Semántico
- TF-IDF implementado manualmente + similitud coseno
- Retorna tabla ordenada por similitud descendente
- Índice construido una sola vez en memoria (caché automático)

Visualizador PCA y Word2Vec 
- PCA aplicado sobre representación TF-IDF
- Word2Vec implementado desde cero con numpy
- Visualización interactiva en 2D y 3D con Plotly
- Coloreado por testamento (OT/NT)

Generador de Versículos
- Modelos: unigram, bigram, trigram, 4-gram y 5-gram
- Parámetros configurables: palabra inicial y largo máximo
- Comparación automática entre los 3 modelos principales

Arquitectura cliente-servidor
La app Streamlit **no almacena el corpus completo**. Cada operación
(filtrado, búsqueda, visualización, generación) se delega completamente
a la API, que retorna solo los datos necesarios para mostrar en pantalla.

Nota sobre dependencias
Este proyecto usa Python 3.14. La librería `gensim` no es compatible
con Python 3.14, por lo que Word2Vec fue implementado desde cero usando `numpy`.
