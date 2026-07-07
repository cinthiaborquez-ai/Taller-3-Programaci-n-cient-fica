import base64
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Dashboard Bíblico", layout="wide")

pagina = st.sidebar.radio("Navegación", [
    " Dashboard",
    " Buscador Semántico",
    " Generador N-gramas",
    " PCA y Word2Vec"
])

# ══════════════════════════════════════════════════════════════════════
# PÁGINA 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════
if pagina == " Dashboard":
    st.title(" Dashboard del Corpus Bíblico")
    st.caption("Análisis interactivo del corpus bíblico (KJV) vía API")
    st.sidebar.header("Filtros")

    testament_label = st.sidebar.selectbox(
        "Testamento", ["Todos", "Antiguo Testamento (OT)", "Nuevo Testamento (NT)"]
    )
    testament = None
    if testament_label.startswith("Antiguo"):
        testament = "OT"
    elif testament_label.startswith("Nuevo"):
        testament = "NT"

    books_resp = requests.get(f"{API_URL}/dashboard/books", params={"testament": testament}).json()
    book_options = ["Todos"] + books_resp["books"]
    book_label = st.sidebar.selectbox("Libro", book_options)
    book = None if book_label == "Todos" else book_label

    chapter = None
    if book:
        chapters_resp = requests.get(f"{API_URL}/dashboard/chapters", params={"book": book}).json()
        chapter_options = ["Todos"] + chapters_resp["chapters"]
        chapter_label = st.sidebar.selectbox("Capítulo", chapter_options)
        chapter = None if chapter_label == "Todos" else chapter_label

    params = {"testament": testament, "book": book, "chapter": chapter}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cantidad de versículos por libro")
        data = requests.get(f"{API_URL}/dashboard/verse-counts", params=params).json()["data"]
        if data:
            st.bar_chart(data, x="book_name", y="verse_count", height=400)
        else:
            st.info("Sin datos para los filtros seleccionados.")

    with col2:
        st.subheader("Longitud promedio de versículos por libro")
        data = requests.get(f"{API_URL}/dashboard/avg-length", params=params).json()["data"]
        if data:
            st.bar_chart(data, x="book_name", y="avg_word_length", height=400)
        else:
            st.info("Sin datos para los filtros seleccionados.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top palabras más frecuentes")
        top_n = st.slider("Cantidad de palabras a mostrar", 5, 50, 20)
        data = requests.get(
            f"{API_URL}/dashboard/top-words", params={**params, "n": top_n}
        ).json()["data"]
        if data:
            st.bar_chart(data, x="word", y="count", height=400)
        else:
            st.info("Sin datos para los filtros seleccionados.")

    with col4:
        st.subheader("Nube de palabras")
        wc_resp = requests.get(f"{API_URL}/dashboard/wordcloud", params=params).json()
        img_bytes = base64.b64decode(wc_resp["image_base64"])
        st.image(img_bytes, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PÁGINA 2: BUSCADOR SEMÁNTICO
# ══════════════════════════════════════════════════════════════════════
elif pagina == " Buscador Semántico":
    st.title(" Buscador Semántico de Versículos")
    st.caption("Encuentra versículos similares usando TF-IDF + similitud coseno (del Taller 2)")
    st.info(" La primera búsqueda puede tardar 1-2 minutos mientras se construye el índice TF-IDF.")

    query = st.text_input("Escribe una frase o palabras a buscar:", placeholder="Ej: God created the heaven and the earth")
    top_k = st.slider("Cantidad de resultados", 5, 30, 10)

    if st.button(" Buscar") and query:
        with st.spinner("Buscando versículos similares..."):
            resp = requests.get(
                f"{API_URL}/search/query",
                params={"q": query, "top_k": top_k}
            ).json()

        results = resp.get("results", [])
        if results:
            st.success(f"Se encontraron {len(results)} versículos similares a: *\"{query}\"*")
            df = pd.DataFrame(results)
            df = df[["rank", "book", "chapter", "verse", "similarity", "text"]]
            df.columns = ["#", "Libro", "Capítulo", "Versículo", "Similitud", "Texto"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No se encontraron resultados.")

# ══════════════════════════════════════════════════════════════════════
# PÁGINA 3: GENERADOR DE VERSÍCULOS
# ══════════════════════════════════════════════════════════════════════
elif pagina == " Generador N-gramas":
    st.title(" Generador de Versículos Bíblicos")
    st.caption("Genera texto bíblico usando modelos de n-gramas entrenados sobre el corpus KJV")
    st.info(" La primera generación con cada modelo puede tardar unos segundos.")

    col1, col2, col3 = st.columns(3)
    with col1:
        modelo = st.selectbox(
            "Modelo", [1, 2, 3, 4, 5], index=1,
            format_func=lambda x: {1:"Unigram (n=1)", 2:"Bigram (n=2)",
                                    3:"Trigram (n=3)", 4:"4-gram (n=4)", 5:"5-gram (n=5)"}[x]
        )
    with col2:
        seed = st.text_input("Palabra inicial", value="god")
    with col3:
        max_len = st.slider("Largo máximo", 5, 100, 20)

    if st.button(" Generar"):
        with st.spinner("Generando texto..."):
            resp = requests.get(
                f"{API_URL}/generator/generate",
                params={"n": modelo, "seed": seed, "max_len": max_len}
            ).json()

        st.success("Texto generado:")
        st.markdown(f"###  {resp['generated_text']}")
        st.caption(f"Modelo: {resp['model']} | Semilla: '{resp['seed']}' | Largo máx: {resp['max_len']}")

        st.subheader("Comparación entre modelos")
        cols = st.columns(3)
        for i, (n_val, nombre) in enumerate([(1,"Unigram"), (2,"Bigram"), (3,"Trigram")]):
            with cols[i]:
                r = requests.get(
                    f"{API_URL}/generator/generate",
                    params={"n": n_val, "seed": seed, "max_len": max_len}
                ).json()
                st.markdown(f"**{nombre}**")
                st.info(r["generated_text"])

# ══════════════════════════════════════════════════════════════════════
# PÁGINA 4: PCA Y WORD2VEC
# ══════════════════════════════════════════════════════════════════════
elif pagina == " PCA y Word2Vec":
    st.title(" Visualizador PCA y Word2Vec")
    st.caption("Representación vectorial de versículos reducida a 2D o 3D")
    st.info(" El primer cálculo puede tardar 1-2 minutos. Los resultados quedan en caché.")

    col1, col2 = st.columns(2)
    with col1:
        metodo = st.selectbox("Método", ["PCA sobre TF-IDF", "Word2Vec + PCA"])
    with col2:
        dims = st.selectbox("Dimensiones", [2, 3], format_func=lambda x: f"{x}D")

    if st.button(" Visualizar"):
        endpoint = "pca" if metodo == "PCA sobre TF-IDF" else "word2vec"
        with st.spinner(f"Calculando {metodo} en {dims}D... (puede tardar la primera vez)"):
            resp = requests.get(
                f"{API_URL}/vectorization/{endpoint}",
                params={"dims": dims}
            ).json()

        points = resp["points"]
        variance = resp["variance_explained"]
        df = pd.DataFrame(points)

        st.caption(f"Varianza explicada: PC1={variance[0]:.2%}, PC2={variance[1]:.2%}" +
                   (f", PC3={variance[2]:.2%}" if dims == 3 else ""))

        if dims == 2:
            fig = px.scatter(
                df, x="x", y="y",
                color="testament",
                hover_data=["book"],
                title=f"{metodo} — 2D",
                labels={"x": "PC1", "y": "PC2", "testament": "Testamento"},
                color_discrete_map={"OT": "#2196F3", "NT": "#FF9800"},
                opacity=0.5,
                height=600,
            )
        else:
            fig = px.scatter_3d(
                df, x="x", y="y", z="z",
                color="testament",
                hover_data=["book"],
                title=f"{metodo} — 3D",
                labels={"x": "PC1", "y": "PC2", "z": "PC3", "testament": "Testamento"},
                color_discrete_map={"OT": "#2196F3", "NT": "#FF9800"},
                opacity=0.5,
                height=650,
            )

        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(fig, use_container_width=True)
