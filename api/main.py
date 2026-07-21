import sys
from pathlib import Path

# Bootstrap de sys.path: permite arrancar tanto con `uvicorn api.main:app`
# (desde la raíz del repo) como con `uvicorn main:app` (comando autodetectado
# por Render con el directorio api/ como raíz de trabajo).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


class StripDoubleSlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.scope.get("path", "")
        if "//" in path:
            request.scope["path"] = path.replace("//", "/")
        return await call_next(request)

from api.routers import demand, distraction, recommender

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
app.add_middleware(StripDoubleSlashMiddleware)

# Routers
app.include_router(demand.router)
app.include_router(distraction.router)
app.include_router(recommender.router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Sistema Integrado de Transporte activo",
        "endpoints": ["/demand", "/distraction", "/recommender"]
    }
