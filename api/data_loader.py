"""
data_loader.py
----------------
Carga el corpus bíblico (KJV) una sola vez al iniciar la API y expone
funciones de acceso/filtrado. Mantener esto centralizado evita que cada
endpoint reimplemente la lectura del CSV.
"""

import os
import pandas as pd

# Ruta al CSV limpio generado a partir de bible_databases (KJV)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bible_kjv_clean.csv")

# DataFrame global cargado una sola vez en memoria del proceso de la API.
# Columnas: id, book_num, book_name, testament, chapter, verse, text
_df: pd.DataFrame | None = None


def load_data() -> pd.DataFrame:
    """Carga (o retorna cache) del corpus completo como DataFrame."""
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        _df["chapter"] = _df["chapter"].astype(int)
        _df["verse"] = _df["verse"].astype(int)
        _df["text"] = _df["text"].astype(str)
    return _df


def get_books(testament: str | None = None) -> list[str]:
    """Lista de nombres de libros, opcionalmente filtrados por testamento."""
    df = load_data()
    if testament:
        df = df[df["testament"] == testament]
    return (
        df[["book_num", "book_name"]]
        .drop_duplicates()
        .sort_values("book_num")["book_name"]
        .tolist()
    )


def get_chapters(book_name: str) -> list[int]:
    """Lista de capítulos disponibles para un libro dado."""
    df = load_data()
    sub = df[df["book_name"] == book_name]
    return sorted(sub["chapter"].unique().tolist())


def filter_verses(
    testament: str | None = None,
    book: str | None = None,
    chapter: int | None = None,
) -> pd.DataFrame:
    """Filtra el corpus según testamento, libro y/o capítulo."""
    df = load_data()
    if testament:
        df = df[df["testament"] == testament]
    if book:
        df = df[df["book_name"] == book]
    if chapter is not None:
        df = df[df["chapter"] == chapter]
    return df