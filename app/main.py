"""
KYC-OCR — FastAPI application entry point.

Endpoints:
  GET  /          — Serve the upload UI
  POST /api/ocr   — Accept an image, run the OCR pipeline, return extracted fields
  GET  /api/health — Health check
"""

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

from app.services.image_service import preprocess_image
from app.services.ocr_service import process_document
from app.services.rag_service import get_rag_service
from app.models.schemas import OCRResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
MAX_FILE_SIZE_MB = 10

app = FastAPI(
    title="KYC-OCR",
    description="AI-powered Indian KYC document scanner using GPT-4o + RAG",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def startup_event():
    """Pre-warm the RAG vector store on startup to avoid cold-start latency."""
    logger.info("Initialising RAG knowledge base...")
    get_rag_service()
    logger.info("KYC-OCR is ready.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "KYC-OCR"}


@app.post("/api/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Main OCR endpoint.

    1. Validate the uploaded file.
    2. Preprocess the image (OpenCV enhancement).
    3. Run the two-pass GPT-4o pipeline (detect → RAG → extract).
    4. Return structured JSON with document type + field values.
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload JPEG, PNG, or WebP.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Save original (optional — for audit / debugging) ─────────────────────
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    save_path.write_bytes(raw_bytes)
    logger.info("Saved upload: %s (%d KB)", save_path.name, len(raw_bytes) // 1024)

    # ── Preprocess ────────────────────────────────────────────────────────────
    logger.info("Preprocessing image...")
    image_b64_uri = preprocess_image(raw_bytes)

    # ── OCR Pipeline ──────────────────────────────────────────────────────────
    logger.info("Running OCR pipeline...")
    result = process_document(image_b64_uri)

    return JSONResponse(content=result)
