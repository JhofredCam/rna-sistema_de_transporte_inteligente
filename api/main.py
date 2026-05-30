from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import demand, distraction

app = FastAPI(
    title="Sistema Inteligente de Transporte",
    description="API de Predicción, Seguridad Vial y Recomendación",
    version="1.0.0"
)

# URLs permitidas
origins = [
    "https://sistema-transporte-inteligente-rna.netlify.app",
    "http://localhost:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(demand.router)
app.include_router(distraction.router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Sistema Integrado de Transporte activo",
        "endpoints": ["/demand", "/distraction", "/recommender"]
    }
