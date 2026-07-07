"""
routers/vectorization.py
-------------------------
Visualizador PCA + Word2Vec (25% de la rúbrica).
Reutiliza TFIDFVectorizer del Taller 2 para PCA.
Word2Vec implementado con numpy (sin gensim).
"""

from collections import Counter, defaultdict
import re, string
import numpy as np
from fastapi import APIRouter, Query
from sklearn.decomposition import PCA
from data_loader import load_data
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

router = APIRouter(prefix="/vectorization", tags=["PCA y Word2Vec"])

# ── Preprocesador (mismo del Taller 2) ───────────────────────────────

class BiblePreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return [t for t in text.split() if t not in self.stop_words and len(t) > 2]


# ── TF-IDF manual (mismo del Taller 2) ───────────────────────────────

class TFIDFVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf_values = {}
        self.N = 0

    def fit_transform(self, token_lists):
        self.N = len(token_lists)
        all_words = set(w for tokens in token_lists for w in tokens)
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        df_count = Counter(w for tokens in token_lists for w in set(tokens))
        self.idf_values = {w: np.log(self.N / (df_count[w] + 1)) for w in self.vocab}
        matrix = np.zeros((self.N, len(self.vocab)))
        for i, tokens in enumerate(token_lists):
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            for word, count in tf.items():
                if word in self.vocab:
                    matrix[i, self.vocab[word]] = (count/total) * self.idf_values[word]
        return matrix


# ── Word2Vec simplificado con numpy (sin gensim) ─────────────────────

class SimpleWord2Vec:
    """
    Implementación simplificada de Word2Vec (CBOW) usando numpy.
    Reemplaza gensim que no es compatible con Python 3.14.
    """
    def __init__(self, vector_size=50, window=2, min_count=5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.word_vectors = {}
        self.vocab = []

    def fit(self, token_lists):
        # Construir vocabulario
        word_counts = Counter(w for tokens in token_lists for w in tokens)
        self.vocab = [w for w, c in word_counts.items() if c >= self.min_count]
        word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)

        # Inicializar matrices de embeddings aleatoriamente
        np.random.seed(42)
        W1 = np.random.randn(V, self.vector_size) * 0.01  # embeddings entrada
        W2 = np.random.randn(self.vector_size, V) * 0.01  # embeddings salida

        # Entrenar con pares contexto-objetivo (ventana deslizante)
        lr = 0.01
        for tokens in token_lists:
            indices = [word_to_idx[w] for w in tokens if w in word_to_idx]
            for i, target_idx in enumerate(indices):
                # Contexto: palabras en la ventana alrededor del objetivo
                start = max(0, i - self.window)
                end = min(len(indices), i + self.window + 1)
                context_indices = [indices[j] for j in range(start, end) if j != i]
                if not context_indices:
                    continue
                # Vector contexto promedio
                h = np.mean(W1[context_indices], axis=0)
                # Forward pass
                u = W2.T @ h
                # Softmax aproximado (solo target vs muestra negativa pequeña)
                exp_u = np.exp(u - u.max())
                y_hat = exp_u / exp_u.sum()
                # Error
                e = y_hat.copy()
                e[target_idx] -= 1
                # Backward pass
                dW2 = np.outer(h, e)
                dh = W2 @ e
                for ctx_idx in context_indices:
                    W1[ctx_idx] -= lr * dh / len(context_indices)
                W2 -= lr * dW2

        # Guardar vectores finales
        self.word_vectors = {w: W1[i] for i, w in enumerate(self.vocab)}
        return self

    def get_vectors(self, words):
        """Retorna matriz de vectores para las palabras dadas."""
        filtered = [w for w in words if w in self.word_vectors]
        if not filtered:
            return np.array([]), []
        matrix = np.array([self.word_vectors[w] for w in filtered])
        return matrix, filtered


# ── Cache ─────────────────────────────────────────────────────────────
_cache = {}

def _get_tokens(sample_size=5000):
    if 'tokens' not in _cache:
        print("Preprocesando corpus para PCA/Word2Vec...")
        df = load_data().sample(n=sample_size, random_state=42).reset_index(drop=True)
        prep = BiblePreprocessor()
        df['tokens'] = df['text'].apply(prep.preprocess)
        _cache['df'] = df
        _cache['tokens'] = df['tokens'].tolist()
        print("Preprocesamiento listo.")
    return _cache['df'], _cache['tokens']


def _get_pca_coords(dims=2):
    key = f'pca_{dims}d'
    if key not in _cache:
        print(f"Calculando TF-IDF + PCA {dims}D...")
        df, token_lists = _get_tokens()
        tfidf = TFIDFVectorizer()
        matrix = tfidf.fit_transform(token_lists)
        pca = PCA(n_components=dims, random_state=42)
        coords = pca.fit_transform(matrix)
        _cache[key] = {
            'coords': coords,
            'df': df,
            'variance': pca.explained_variance_ratio_.tolist()
        }
        print(f"PCA {dims}D listo.")
    return _cache[key]


def _get_w2v_coords(dims=2):
    key = f'w2v_{dims}d'
    if key not in _cache:
        print(f"Entrenando Word2Vec + PCA {dims}D...")
        df, token_lists = _get_tokens()
        w2v = SimpleWord2Vec(vector_size=50, window=2, min_count=5)
        w2v.fit(token_lists)
        # Vector de cada versículo = promedio de sus palabras
        verse_vectors = []
        for tokens in token_lists:
            vecs = [w2v.word_vectors[t] for t in tokens if t in w2v.word_vectors]
            if vecs:
                verse_vectors.append(np.mean(vecs, axis=0))
            else:
                verse_vectors.append(np.zeros(w2v.vector_size))
        matrix = np.array(verse_vectors)
        pca = PCA(n_components=dims, random_state=42)
        coords = pca.fit_transform(matrix)
        _cache[key] = {
            'coords': coords,
            'df': df,
            'variance': pca.explained_variance_ratio_.tolist()
        }
        print(f"Word2Vec + PCA {dims}D listo.")
    return _cache[key]


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/pca")
def get_pca(dims: int = Query(2, ge=2, le=3)):
    """PCA sobre TF-IDF de versículos. dims=2 o dims=3."""
    result = _get_pca_coords(dims)
    coords = result['coords']
    df = result['df']
    points = []
    for i in range(len(df)):
        p = {
            'book': df.iloc[i]['book_name'],
            'testament': df.iloc[i]['testament'],
            'x': round(float(coords[i, 0]), 6),
            'y': round(float(coords[i, 1]), 6),
        }
        if dims == 3:
            p['z'] = round(float(coords[i, 2]), 6)
        points.append(p)
    return {
        'dims': dims,
        'method': 'TF-IDF + PCA',
        'variance_explained': result['variance'],
        'points': points
    }


@router.get("/word2vec")
def get_word2vec(dims: int = Query(2, ge=2, le=3)):
    """Word2Vec + PCA sobre versículos. dims=2 o dims=3."""
    result = _get_w2v_coords(dims)
    coords = result['coords']
    df = result['df']
    points = []
    for i in range(len(df)):
        p = {
            'book': df.iloc[i]['book_name'],
            'testament': df.iloc[i]['testament'],
            'x': round(float(coords[i, 0]), 6),
            'y': round(float(coords[i, 1]), 6),
        }
        if dims == 3:
            p['z'] = round(float(coords[i, 2]), 6)
        points.append(p)
    return {
        'dims': dims,
        'method': 'Word2Vec + PCA',
        'variance_explained': result['variance'],
        'points': points
    }
