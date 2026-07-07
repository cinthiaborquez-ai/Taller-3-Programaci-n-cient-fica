"""
routers/search.py
------------------
Buscador semántico (20% de la rúbrica).
Reutiliza TFIDFVectorizer y SemanticSearchEngine del Taller 2.
"""

from collections import Counter, defaultdict
import re, string
import numpy as np
from fastapi import APIRouter, Query
from data_loader import load_data
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

router = APIRouter(prefix="/search", tags=["Buscador Semántico"])

# ── Clases reutilizadas del Taller 2 ─────────────────────────────────

class BiblePreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text, remove_sw=True):
        tokens = self._clean_text(text).split()
        if remove_sw:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens


class TFIDFVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf_values = {}
        self.N = 0

    def _compute_tf(self, tokens):
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {w: c/total for w, c in tf.items()}

    def fit_transform(self, token_lists):
        self.N = len(token_lists)
        all_words = set(w for tokens in token_lists for w in tokens)
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        df_count = Counter(w for tokens in token_lists for w in set(tokens))
        self.idf_values = {w: np.log(self.N / (df_count[w] + 1)) for w in self.vocab}
        matrix = np.zeros((self.N, len(self.vocab)))
        for i, tokens in enumerate(token_lists):
            tf = self._compute_tf(tokens)
            for word, tf_val in tf.items():
                if word in self.vocab:
                    matrix[i, self.vocab[word]] = tf_val * self.idf_values[word]
        return matrix


class SemanticSearchEngine:
    def __init__(self, preprocessor, tfidf):
        self.preprocessor = preprocessor
        self.tfidf = tfidf
        self.tfidf_matrix = None
        self.df_index = None

    def fit(self, df):
        self.df_index = df.reset_index(drop=True)
        tokens_list = [self.preprocessor.preprocess(t) for t in self.df_index['text']]
        self.tfidf_matrix = self.tfidf.fit_transform(tokens_list)
        return self

    def _cosine_sim(self, query_vec, matrix):
        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0:
            return np.zeros(matrix.shape[0])
        query_vec = query_vec / qnorm
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1
        return (matrix / norms[:, np.newaxis]) @ query_vec

    def search(self, query_text, k=10):
        tokens = self.preprocessor.preprocess(query_text)
        query_vec = np.zeros(len(self.tfidf.vocab))
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        for word, count in tf.items():
            if word in self.tfidf.vocab:
                j = self.tfidf.vocab[word]
                query_vec[j] = (count/total) * self.tfidf.idf_values.get(word, 0)
        sims = self._cosine_sim(query_vec, self.tfidf_matrix)
        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_idx:
            row = self.df_index.iloc[idx]
            results.append({
                'rank': len(results) + 1,
                'book': row['book_name'],
                'testament': row['testament'],
                'chapter': int(row['chapter']),
                'verse': int(row['verse']),
                'text': row['text'],
                'similarity': round(float(sims[idx]), 4),
            })
        return results


# ── Cache: se construye solo la primera vez ───────────────────────────
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        print("Construyendo índice de búsqueda (solo una vez)...")
        df = load_data()
        prep = BiblePreprocessor()
        tfidf = TFIDFVectorizer()
        _engine = SemanticSearchEngine(prep, tfidf).fit(df)
        print("Índice listo.")
    return _engine


# ── Endpoint ──────────────────────────────────────────────────────────
@router.get("/query")
def search_verses(
    q: str = Query(..., description="Frase o palabras a buscar"),
    top_k: int = Query(10, ge=1, le=50, description="Cantidad de resultados"),
):
    """Retorna los versículos más similares semánticamente a la consulta."""
    engine = _get_engine()
    results = engine.search(q, k=top_k)
    return {"query": q, "results": results}
