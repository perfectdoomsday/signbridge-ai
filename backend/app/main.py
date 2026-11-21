from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.app.core.config import settings
from backend.app.inference.predictor import Predictor

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="Image-only static sign alphabet recognition API.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
predictor = Predictor(settings.model_path, settings.confidence_threshold, settings.image_size)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def root():
    index = Path("frontend/dist/index.html")
    if index.exists():
        from fastapi.responses import FileResponse
        return FileResponse(index)
    return {"name": settings.app_name, "docs": "/docs", "frontend": "Run npm run dev in frontend/"}


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "model_ready": predictor.ready,
        "language": settings.sign_language,
        "runtime": "ONNX Runtime / CPU",
    }


@app.get("/api/v1/models")
def models():
    return {
        "loaded": predictor.ready,
        "architecture": predictor.architecture,
        "classes": len(predictor.classes),
        "labels": predictor.classes,
        "checkpoint": str(predictor.model_path),
    }


@app.get("/api/v1/languages")
def languages():
    return {
        "languages": [{"code": "ISL", "name": "Indian Sign Language", "configured": settings.sign_language.upper() == "ISL"}],
        "note": "This starter recognizes static ISL alphabet images (A-Z), not continuous signing or sentence grammar.",
    }


@app.post("/api/v1/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB.")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        source = Image.open(BytesIO(raw))
        if max(source.size) > 8192:
            raise ValueError("Image dimensions must not exceed 8192 pixels on either side")
        source.verify()
        image = Image.open(BytesIO(raw)).convert("RGB")
        result = predictor.predict(image)
        return {**result, "language": settings.sign_language}
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


# Retain a clear 404 response for clients using the retired video API.
@app.post("/api/v1/predict/video")
def retired_video_endpoint():
    raise HTTPException(status_code=410, detail="Video recognition was removed. Use POST /api/v1/predict/image with a JPEG/PNG/WebP image.")


# Serve the compiled React app when deployed as one container.
from fastapi.staticfiles import StaticFiles

_dist = Path("frontend/dist")
if _dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")
