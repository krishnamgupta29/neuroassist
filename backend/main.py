import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from database import init_db
from routes import auth_routes, patient_routes, scan_routes, admin_routes

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure upload directories exist
for d in ["uploads/mri_scans", "uploads/gradcam", "uploads/reports"]:
    os.makedirs(d, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle hook."""
    logger.info("NeuroAssist API starting — initialising MongoDB collections & indexes …")
    try:
        await init_db()
        logger.info("Database ready.")
    except Exception as e:
        logger.warning(f"Database startup notice: {e}")
    yield
    logger.info("NeuroAssist API shutting down.")


app = FastAPI(
    title="NeuroAssist API",
    description="Enterprise AI Healthcare Platform — Neurological Screening & MRI Intelligence",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving (GradCAM / uploaded scans) ───────────────────────────
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_routes.router)
app.include_router(patient_routes.router)
app.include_router(scan_routes.router)
app.include_router(admin_routes.router)


@app.get("/api")
async def read_root():
    return {
        "status": "online",
        "app": "NeuroAssist API",
        "version": "3.0.0",
        "database": "MongoDB Atlas",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health")
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "NeuroAssist AI Backend",
        "version": "3.0.0",
        "platform": "Hugging Face Cloud"
    }
