from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import dashboard
from routers import search
from routers import generator
from routers import vectorization

app = FastAPI(
    title="Bible Corpus API",
    description="API para análisis y exploración del corpus bíblico (Laboratorio 3)",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(search.router)
app.include_router(generator.router)
app.include_router(vectorization.router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Bible Corpus API funcionando"}

@app.get("/health")
def health():
    return {"status": "healthy"}