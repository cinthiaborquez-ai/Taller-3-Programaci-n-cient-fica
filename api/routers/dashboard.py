"""
routers/dashboard.py
---------------------
Endpoints del Dashboard principal (30% de la rúbrica):
- /dashboard/books            -> lista de libros
- /dashboard/chapters         -> lista de capítulos de un libro
- /dashboard/verse-counts     -> cantidad de versículos por libro
- /dashboard/avg-length       -> longitud promedio de versículos por libro
- /dashboard/top-words        -> top N palabras más frecuentes
- /dashboard/wordcloud        -> imagen PNG (base64) de nube de palabras

Todo el filtrado (testamento / libro / capítulo) se hace aquí, en la API.
Streamlit solo recibe los resultados ya calculados.
"""

import base64
import io
import re
from collections import Counter

from fastapi import APIRouter, Query
from wordcloud import WordCloud

from data_loader import filter_verses, get_books, get_chapters

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "that", "he", "shall", "for", "unto",
    "i", "his", "they", "be", "is", "him", "not", "them", "it", "with", "all",
    "thou", "thy", "thee", "ye", "have", "was", "which", "but", "as", "their",
    "will", "you", "said", "from", "this", "her", "she", "we", "so", "by",
    "are", "were", "had", "me", "my", "if", "when", "shalt", "upon",
    "out", "into", "your", "also", "there", "at", "one", "us", "an", "or",
    "on", "do", "did", "let", "then", "even", "what", "who", "made", "came",
    "go", "up", "down", "before", "after", "hath", "yet", "more", "now",
}

_word_re = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _word_re.findall(text)]


@router.get("/books")
def list_books(testament: str | None = Query(None, description="OT o NT")):
    return {"books": get_books(testament)}


@router.get("/chapters")
def list_chapters(book: str = Query(..., description="Nombre del libro")):
    return {"chapters": get_chapters(book)}


@router.get("/verse-counts")
def verse_counts(
    testament: str | None = Query(None),
    book: str | None = Query(None),
    chapter: int | None = Query(None),
):
    df = filter_verses(testament=testament, book=book, chapter=chapter)
    counts = (
        df.groupby(["book_num", "book_name"], as_index=False)
        .size()
        .sort_values("book_num")
        .rename(columns={"size": "verse_count"})
    )
    return {"data": counts[["book_name", "verse_count"]].to_dict(orient="records")}


@router.get("/avg-length")
def avg_length(
    testament: str | None = Query(None),
    book: str | None = Query(None),
    chapter: int | None = Query(None),
):
    df = filter_verses(testament=testament, book=book, chapter=chapter).copy()
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].apply(lambda t: len(_tokenize(t)))
    agg = (
        df.groupby(["book_num", "book_name"], as_index=False)
        .agg(avg_char_length=("char_len", "mean"), avg_word_length=("word_len", "mean"))
        .sort_values("book_num")
    )
    agg["avg_char_length"] = agg["avg_char_length"].round(2)
    agg["avg_word_length"] = agg["avg_word_length"].round(2)
    return {"data": agg[["book_name", "avg_char_length", "avg_word_length"]].to_dict(orient="records")}


@router.get("/top-words")
def top_words(
    testament: str | None = Query(None),
    book: str | None = Query(None),
    chapter: int | None = Query(None),
    n: int = Query(20, ge=1, le=200),
):
    df = filter_verses(testament=testament, book=book, chapter=chapter)
    counter = Counter()
    for text in df["text"]:
        tokens = [w for w in _tokenize(text) if w not in STOPWORDS and len(w) > 2]
        counter.update(tokens)
    top = counter.most_common(n)
    return {"data": [{"word": w, "count": c} for w, c in top]}


@router.get("/wordcloud")
def wordcloud(
    testament: str | None = Query(None),
    book: str | None = Query(None),
    chapter: int | None = Query(None),
):
    df = filter_verses(testament=testament, book=book, chapter=chapter)
    counter = Counter()
    for text in df["text"]:
        tokens = [w for w in _tokenize(text) if w not in STOPWORDS and len(w) > 2]
        counter.update(tokens)

    if not counter:
        counter = Counter({"sin_datos": 1})

    wc = WordCloud(width=900, height=500, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(counter)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image_base64": img_base64}