"""FastAPI backend for PruneVision."""

from __future__ import annotations

import io
import json
import logging
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from backend.database import get_upload_record, init_database, insert_upload_record, list_upload_records
from backend.model import PrunableCNN
from backend.preprocessor import ConfidenceResult, IntelligentPreprocessor


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"
DEFAULT_CHECKPOINT_PATH = BACKEND_DIR / "model_checkpoints" / "best_model.pt"
DEFAULT_RESULTS_PATH = BASE_DIR / "results.json"
DEFAULT_HISTORY_PATH = BACKEND_DIR / "training_history.json"
DEFAULT_DATABASE_PATH = BACKEND_DIR / "prunevision.db"
DEFAULT_UPLOADS_DIR = BASE_DIR / "uploads"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CLASSES = ["Healthy", "Disease A", "Disease B"]
DEFAULT_LAMBDA = 0.001
REQUEST_LOG_TEMPLATE = "Request %s completed in %.2f ms"
DEFAULT_UPLOAD_LIMIT = 200


DEFAULT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class PredictResponse(BaseModel):
    """Schema for prediction responses."""

    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    confidence_status: str = Field(..., description="Confidence quality bucket")
    recommendation: str = Field(..., description="Recommended next action")
    preprocessing_report: dict[str, Any] = Field(..., description="Preprocessing diagnostics")
    model_stats: dict[str, Any] = Field(..., description="Model pruning statistics")
    upload_id: int = Field(..., description="Identifier for this upload in the portal history")
    image_url: str = Field(..., description="Public URL of stored uploaded image")
    warning: Optional[str] = Field(default=None, description="Optional confidence warning")


class UploadPortalRecord(BaseModel):
    """Schema for portal upload history entries."""

    id: int
    original_filename: str
    image_url: str
    label: str
    confidence: float
    confidence_status: str
    recommendation: str
    warning: Optional[str]
    quality_score: float
    leaf_detected: bool
    enhancements_applied: list[str]
    processing_time_ms: float
    created_at: str
    pdf_url: str


class UploadFolderResponse(BaseModel):
    """Schema for upload folder metadata responses."""

    folder_path: str
    file_count: int


class ModelStatsResponse(BaseModel):
    """Schema for model statistics responses."""

    sparsity_percent: float
    test_accuracy: float
    model_size_mb: float
    lambda_used: float
    total_parameters: int
    pruned_parameters: int
    active_parameters: int


class HealthResponse(BaseModel):
    """Schema for health responses."""

    status: str
    model_loaded: bool
    device: str


@dataclass(slots=True)
class AppState:
    """Mutable application state stored on the FastAPI app instance."""

    model: Optional[PrunableCNN]
    class_names: list[str]
    lambda_used: float
    test_accuracy: float
    model_loaded: bool
    device: torch.device
    preprocessor: IntelligentPreprocessor
    checkpoint_path: Path
    history_path: Path
    results_path: Path
    database_path: Path
    uploads_dir: Path


def _configure_logging() -> None:
    """Initialize logging if the application has not configured it yet."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _load_results(results_path: Path) -> dict[str, Any]:
    """Load persisted comparison results if available."""

    if results_path.exists():
        try:
            return json.loads(results_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to load results.json: %s", exc)
    return {}


def _load_history(history_path: Path) -> Any:
    """Load training history from disk if present."""

    if history_path.exists():
        try:
            return json.loads(history_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to load training history: %s", exc)
    return {}


def _load_model_state(device: torch.device, checkpoint_path: Path) -> tuple[Optional[PrunableCNN], list[str], float, float, bool]:
    """Load a checkpointed model or return a default model fallback."""

    if not checkpoint_path.exists():
        LOGGER.warning("Checkpoint not found at %s; starting with default model", checkpoint_path)
        model = PrunableCNN(num_classes=len(DEFAULT_CLASSES)).to(device)
        return model, DEFAULT_CLASSES, DEFAULT_LAMBDA, 0.0, False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
    lambda_used = float(checkpoint.get("lambda_sparse", DEFAULT_LAMBDA))
    test_accuracy = float(checkpoint.get("test_accuracy", 0.0))
    model = PrunableCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    LOGGER.info("Loaded model checkpoint from %s", checkpoint_path)
    return model, class_names, lambda_used, test_accuracy, True


def _image_to_tensor(processed_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a processed BGR image into a normalized batch tensor."""

    rgb_image = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    array = rgb_image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device)


def _read_upload_file(upload_file: UploadFile) -> bytes:
    """Read and validate an uploaded file."""

    try:
        return upload_file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to read uploaded file") from exc


def _serialize_preprocessing_report(report: Any) -> dict[str, Any]:
    """Serialize a preprocessing report for the API response."""

    return {
        "quality_score": report.quality_score,
        "enhancements_applied": report.enhancements_applied,
        "leaf_detected": report.leaf_detected,
        "processing_time_ms": report.processing_time_ms,
    }


def _build_model_stats(model: PrunableCNN, lambda_used: float, test_accuracy: float) -> dict[str, Any]:
    """Compute model statistics for the API."""

    total_parameters = model.get_total_parameters()
    pruned_parameters = model.get_pruned_parameters()
    active_parameters = model.get_active_parameters()
    sparsity_percent = model.get_total_sparsity()
    model_size_mb = model.get_model_size_mb()
    return {
        "sparsity_percent": sparsity_percent,
        "test_accuracy": test_accuracy,
        "model_size_mb": model_size_mb,
        "lambda_used": lambda_used,
        "total_parameters": total_parameters,
        "pruned_parameters": pruned_parameters,
        "active_parameters": active_parameters,
    }


def _safe_filename(original_name: str) -> str:
    """Create a safe stored filename preserving extension where possible."""

    source = Path(original_name)
    extension = source.suffix.lower() if source.suffix else ".jpg"
    return f"{uuid.uuid4().hex}{extension}"


def _store_uploaded_image(image_bytes: bytes, original_name: str, uploads_dir: Path) -> str:
    """Persist uploaded bytes to the uploads folder and return the stored filename."""

    stored_name = _safe_filename(original_name)
    file_path = uploads_dir / stored_name
    file_path.write_bytes(image_bytes)
    return stored_name


def _render_prediction_pdf(record: UploadPortalRecord) -> bytes:
    """Generate a compact PDF report for one upload record."""

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 48

    lines = [
        "PruneVision Prediction Report",
        "",
        f"Upload ID: {record.id}",
        f"Filename: {record.original_filename}",
        f"Created At: {record.created_at}",
        "",
        f"Label: {record.label}",
        f"Confidence: {record.confidence:.4f}",
        f"Confidence Status: {record.confidence_status}",
        f"Recommendation: {record.recommendation}",
        f"Warning: {record.warning or 'None'}",
        "",
        f"Quality Score: {record.quality_score:.2f}",
        f"Leaf Detected: {'Yes' if record.leaf_detected else 'No'}",
        f"Enhancements Applied: {', '.join(record.enhancements_applied) or 'None'}",
        f"Processing Time (ms): {record.processing_time_ms:.2f}",
    ]

    pdf.setTitle(f"PruneVision Upload {record.id}")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, lines[0])
    y -= 28
    pdf.setFont("Helvetica", 11)

    for line in lines[1:]:
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 48
        pdf.drawString(40, y, line)
        y -= 18

    pdf.save()
    buffer.seek(0)
    return buffer.read()


def _to_portal_record(record: Any) -> UploadPortalRecord:
    """Convert persistence model into API portal response schema."""

    return UploadPortalRecord(
        id=record.id,
        original_filename=record.original_filename,
        image_url=f"/uploads/files/{record.stored_filename}",
        label=record.label,
        confidence=record.confidence,
        confidence_status=record.confidence_status,
        recommendation=record.recommendation,
        warning=record.warning,
        quality_score=record.quality_score,
        leaf_detected=record.leaf_detected,
        enhancements_applied=record.enhancements_applied,
        processing_time_ms=record.processing_time_ms,
        created_at=record.created_at,
        pdf_url=f"/uploads/{record.id}/pdf",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load application resources before serving requests."""

    _configure_logging()
    device = torch.device(DEFAULT_DEVICE)
    state = AppState(
        model=None,
        class_names=DEFAULT_CLASSES,
        lambda_used=DEFAULT_LAMBDA,
        test_accuracy=0.0,
        model_loaded=False,
        device=device,
        preprocessor=IntelligentPreprocessor(),
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        history_path=DEFAULT_HISTORY_PATH,
        results_path=DEFAULT_RESULTS_PATH,
        database_path=DEFAULT_DATABASE_PATH,
        uploads_dir=DEFAULT_UPLOADS_DIR,
    )
    init_database(state.database_path)
    state.uploads_dir.mkdir(parents=True, exist_ok=True)
    model, class_names, lambda_used, test_accuracy, loaded = _load_model_state(device, state.checkpoint_path)
    state.model = model
    state.class_names = class_names
    state.lambda_used = lambda_used
    state.test_accuracy = test_accuracy
    state.model_loaded = loaded
    app.state.prunevision = state
    LOGGER.info("Application ready on device=%s", device)
    yield


app = FastAPI(title="PruneVision", version="1.0.0", lifespan=lifespan)
app.mount("/uploads/files", StaticFiles(directory=DEFAULT_UPLOADS_DIR), name="uploads-files")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_time(request, call_next):
    """Measure and log processing time for every request."""

    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.info(REQUEST_LOG_TEMPLATE, request.url.path, duration_ms)


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the single-page frontend."""

    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(FRONTEND_INDEX)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Run preprocessing, inference, and confidence gating on an uploaded image."""

    app_state: AppState = app.state.prunevision
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    request_start = time.perf_counter()
    image_bytes = _read_upload_file(file)
    stored_filename = _store_uploaded_image(image_bytes, file.filename or "upload.jpg", app_state.uploads_dir)

    try:
        processed_image, report = app_state.preprocessor.process(image_bytes)
    except Exception as exc:
        LOGGER.exception("Preprocessing failed")
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}") from exc

    tensor = _image_to_tensor(processed_image, app_state.device)
    app_state.model.eval()
    with torch.no_grad():
        logits = app_state.model(tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    label = app_state.class_names[int(predicted_index.item())]
    confidence_value = float(confidence.item())
    confidence_result: ConfidenceResult = app_state.preprocessor.gate_confidence(label=label, confidence=confidence_value)

    upload_id = insert_upload_record(
        app_state.database_path,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        label=confidence_result.label,
        confidence=confidence_result.confidence,
        confidence_status=confidence_result.status,
        recommendation=confidence_result.recommendation,
        warning=confidence_result.warning,
        quality_score=report.quality_score,
        leaf_detected=report.leaf_detected,
        enhancements_applied=report.enhancements_applied,
        processing_time_ms=report.processing_time_ms,
    )

    response = PredictResponse(
        label=confidence_result.label,
        confidence=confidence_result.confidence,
        confidence_status=confidence_result.status,
        recommendation=confidence_result.recommendation,
        preprocessing_report=_serialize_preprocessing_report(report),
        model_stats=_build_model_stats(app_state.model, app_state.lambda_used, app_state.test_accuracy),
        upload_id=upload_id,
        image_url=f"/uploads/files/{stored_filename}",
        warning=confidence_result.warning,
    )

    total_ms = (time.perf_counter() - request_start) * 1000.0
    LOGGER.info("Predict request processed in %.2f ms", total_ms)
    return response


@app.get("/model-stats", response_model=ModelStatsResponse)
async def model_stats() -> ModelStatsResponse:
    """Return model pruning and evaluation statistics."""

    app_state: AppState = app.state.prunevision
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    stats = _build_model_stats(app_state.model, app_state.lambda_used, app_state.test_accuracy)
    return ModelStatsResponse(**stats)


@app.get("/training-history")
async def training_history() -> Any:
    """Return the saved training history JSON."""

    app_state: AppState = app.state.prunevision
    return _load_history(app_state.history_path)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a health summary for the API."""

    app_state: AppState = app.state.prunevision
    return HealthResponse(status="ok", model_loaded=app_state.model_loaded, device=str(app_state.device))


@app.get("/uploads", response_model=list[UploadPortalRecord])
async def uploads(limit: int = DEFAULT_UPLOAD_LIMIT) -> list[UploadPortalRecord]:
    """Return recent uploaded image history for the portal."""

    app_state: AppState = app.state.prunevision
    records = list_upload_records(app_state.database_path, limit=limit)
    return [_to_portal_record(record) for record in records]


@app.get("/uploads/{upload_id}/pdf")
async def upload_pdf(upload_id: int) -> Response:
    """Generate and return a PDF report for a specific upload record."""

    app_state: AppState = app.state.prunevision
    record = get_upload_record(app_state.database_path, upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Upload record not found")

    portal_record = _to_portal_record(record)
    pdf_bytes = _render_prediction_pdf(portal_record)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=prunevision_upload_{upload_id}.pdf",
        },
    )


@app.get("/uploads-folder", response_model=UploadFolderResponse)
async def uploads_folder() -> UploadFolderResponse:
    """Return uploads folder metadata for portal display."""

    app_state: AppState = app.state.prunevision
    file_count = sum(1 for path in app_state.uploads_dir.iterdir() if path.is_file())
    return UploadFolderResponse(folder_path=str(app_state.uploads_dir), file_count=file_count)


@app.post("/open-uploads-folder")
async def open_uploads_folder() -> dict[str, Any]:
    """Open the uploads directory in the host operating system explorer."""

    app_state: AppState = app.state.prunevision
    folder_path = app_state.uploads_dir
    try:
        subprocess.Popen(["explorer", str(folder_path)], shell=False)
        return {"opened": True, "folder_path": str(folder_path)}
    except Exception as exc:
        LOGGER.warning("Failed to open uploads folder: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not open uploads folder: {exc}") from exc
