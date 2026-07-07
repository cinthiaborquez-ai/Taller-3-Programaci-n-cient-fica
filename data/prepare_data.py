"""
prepare_data.py
-----------------
Combina t_kjv.csv (versículos) con key_english.csv (nombres de libros y
testamento) en un único CSV limpio: bible_kjv_clean.csv

Ejecutar UNA SOLA VEZ desde la carpeta data:
    python prepare_data.py
"""

import pandas as pd

kjv = pd.read_csv("t_kjv.csv")
key = pd.read_csv("key_english.csv")

# Renombramos columnas a nombres claros
kjv.columns = ["id", "book_num", "chapter", "verse", "text"]
key.columns = ["book_num", "book_name", "testament", "genre"]

# Unimos por book_num para traer el nombre del libro y el testamento (OT/NT)
df = kjv.merge(key[["book_num", "book_name", "testament"]], on="book_num", how="left")
df = df[["id", "book_num", "book_name", "testament", "chapter", "verse", "text"]]

df.to_csv("bible_kjv_clean.csv", index=False)

print("Listo. Filas:", len(df))
print("Libros:", df["book_name"].nunique())
print("Testamentos:", df["testament"].unique())