from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import face_to_sketch

app = FastAPI(title="Generative AI Assignment App")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

app.include_router(face_to_sketch.router)
