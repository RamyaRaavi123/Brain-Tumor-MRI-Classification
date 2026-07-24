"""FastAPI inference service for brain tumor MRI classification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference import BrainTumorPredictor
from src.report import ReportStore, generate_pdf
from src.utils.config import load_config

app = FastAPI(
    title="Brain Tumor MRI Classification API",
    description="Predict glioma, meningioma, pituitary, or no tumor from MRI images.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
reports_dir = Path(__file__).parent / "reports"
report_store = ReportStore(reports_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

_predictor: BrainTumorPredictor | None = None


class PredictionResponse(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class MedicalReportData(BaseModel):
    report_id: str
    patient_id: str
    timestamp: str
    predicted_class: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    urgency: str
    mri_analysis: str
    tumor_description: str
    model_observation: str
    recommendation: str


class AnalyzeResponse(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]
    gradcam_image: str
    original_image: str
    report: MedicalReportData
    pdf_url: str


class ReportSummary(BaseModel):
    report_id: str
    patient_id: str
    timestamp: str
    predicted_class: str
    predicted_label: str
    confidence: float
    urgency: str


def get_predictor() -> BrainTumorPredictor:
    """Lazy-load predictor from default checkpoint path."""
    global _predictor
    if _predictor is None:
        config = load_config("configs/base.yaml")
        checkpoint = Path(config.inference.default_checkpoint)
        if not checkpoint.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Checkpoint not found at {checkpoint}. Train a model first.",
            )
        _predictor = BrainTumorPredictor(checkpoint)
    return _predictor


def _decode_upload(contents: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR array."""
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return image


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def index():
    """Serve the web UI."""
    return FileResponse(static_dir / "index.html")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict tumor class from an uploaded MRI image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    contents = await file.read()
    image = _decode_upload(contents)
    predictor = get_predictor()
    result = predictor.predict_from_array(image)
    return PredictionResponse(**result)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    patient_id: str | None = Form(default=None),
) -> AnalyzeResponse:
    """Full MRI analysis with Grad-CAM visualization and automated medical report."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    contents = await file.read()
    image = _decode_upload(contents)
    predictor = get_predictor()
    result = predictor.analyze_from_array(image, patient_id=patient_id)

    report_store.save(
        result["report"],
        gradcam_b64=result["gradcam_image"],
        original_b64=result["original_image"],
    )

    report_id = result["report"]["report_id"]
    return AnalyzeResponse(**result, pdf_url=f"/reports/{report_id}/pdf")


@app.get("/reports", response_model=list[ReportSummary])
def list_reports() -> list[ReportSummary]:
    """List all stored medical reports for the doctor dashboard."""
    return [ReportSummary(**r) for r in report_store.list_reports()]


@app.get("/reports/{report_id}")
def get_report(report_id: str) -> dict:
    """Get full report details by ID."""
    report = report_store.get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/reports/{report_id}/pdf")
def download_report_pdf(report_id: str) -> Response:
    """Generate and download PDF for a stored report."""
    report = report_store.get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    gradcam_image = None
    if report.get("gradcam_image"):
        import base64

        b64_data = report["gradcam_image"].split(",", 1)[-1]
        img_bytes = base64.b64decode(b64_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        gradcam_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    pdf_bytes = generate_pdf(report, gradcam_image)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id}.pdf"'},
    )
