"""
routers/generator.py
---------------------
Generador de versículos con n-gramas (25% de la rúbrica).
Reutiliza NgramLanguageModel del Taller 2.
"""

from collections import Counter, defaultdict
import re, string
import numpy as np
from fastapi import APIRouter, Query
from data_loader import load_data
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

router = APIRouter(prefix="/generator", tags=["Generador N-gramas"])

# ── Clase reutilizada del Taller 2 ───────────────────────────────────

class BiblePreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text, remove_sw=False):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        if remove_sw:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens


class NgramLanguageModel:
    """
    Modelo de lenguaje estadístico basado en n-gramas.
    Reutilizado directamente del Taller 2.
    """
    def __init__(self, n=1):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.vocab = []
        self.unigram_counts = Counter()

    def fit(self, token_lists):
        for tokens in token_lists:
            padded = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            self.unigram_counts.update(tokens + ['<END>'])
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                next_word = padded[i + self.n - 1]
                self.ngram_counts[context][next_word] += 1
        self.vocab = list(self.unigram_counts.keys())
        return self

    def _next_word(self, context):
        context = tuple(context)
        if context in self.ngram_counts and self.ngram_counts[context]:
            candidates = self.ngram_counts[context]
            words  = list(candidates.keys())
            counts = np.array(list(candidates.values()), dtype=float)
            probs  = counts / counts.sum()
            return np.random.choice(words, p=probs)
        words  = list(self.unigram_counts.keys())
        counts = np.array(list(self.unigram_counts.values()), dtype=float)
        probs  = counts / counts.sum()
        return np.random.choice(words, p=probs)

    def generate(self, seed_word=None, max_len=20):
        if seed_word and seed_word.lower() in self.vocab:
            generated = [seed_word.lower()]
        else:
            top_words = [w for w, _ in self.unigram_counts.most_common(100)
                        if w not in ['<END>', '<START>']]
            generated = [np.random.choice(top_words)]

        for _ in range(max_len - 1):
            context = () if self.n == 1 else tuple(generated[-(self.n - 1):])
            while len(context) < self.n - 1:
                context = ('<START>',) + context
            next_w = self._next_word(context)
            if next_w == '<END>':
                break
            generated.append(next_w)

        return ' '.join(generated)


# ── Cache: modelos entrenados una sola vez ────────────────────────────
_models: dict[int, NgramLanguageModel] = {}
_preprocessor = None

def _get_model(n: int) -> NgramLanguageModel:
    global _models, _preprocessor
    if _preprocessor is None:
        _preprocessor = BiblePreprocessor()
    if n not in _models:
        print(f"Entrenando modelo {n}-gram (solo una vez)...")
        df = load_data()
        token_lists = [_preprocessor.preprocess(t) for t in df["text"]]
        _models[n] = NgramLanguageModel(n=n).fit(token_lists)
        print(f"Modelo {n}-gram listo.")
    return _models[n]


# ── Endpoint ──────────────────────────────────────────────────────────
@router.get("/generate")
def generate_text(
    n: int = Query(2, ge=1, le=5, description="Orden del modelo: 1=unigram, 2=bigram, 3=trigram..."),
    seed: str = Query("god", description="Palabra inicial para la generación"),
    max_len: int = Query(20, ge=5, le=100, description="Largo máximo del texto generado"),
):
    """
    Genera texto bíblico usando un modelo de n-gramas entrenado sobre el corpus KJV.
    El modelo se entrena la primera vez que se usa y queda en caché.
    """
    model = _get_model(n)
    generated = model.generate(seed_word=seed, max_len=max_len)
    return {
        "model": f"{n}-gram",
        "seed": seed,
        "max_len": max_len,
        "generated_text": generated,
    }