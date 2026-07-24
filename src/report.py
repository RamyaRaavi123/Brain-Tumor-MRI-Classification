"""Automated medical report generation and PDF export."""

from __future__ import annotations

import base64
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

CLASS_INFO: dict[str, dict[str, str]] = {
    "glioma": {
        "label": "Glioma Tumor",
        "description": (
            "Gliomas are primary brain tumors arising from glial cells. They may appear "
            "as irregular, heterogeneous masses with variable contrast enhancement on MRI."
        ),
        "observation": (
            "The model detected features consistent with glioma pathology, including "
            "abnormal tissue density and irregular mass boundaries in the highlighted regions."
        ),
        "recommendation": (
            "Urgent referral to neuro-oncology is recommended. Further evaluation with "
            "contrast-enhanced MRI, MR spectroscopy, and possible biopsy should be considered."
        ),
        "urgency": "High",
    },
    "meningioma": {
        "label": "Meningioma Tumor",
        "description": (
            "Meningiomas are typically extra-axial tumors arising from the meninges. "
            "They often present as well-circumscribed, dural-based masses on MRI."
        ),
        "observation": (
            "The model identified patterns suggestive of meningioma, with focal hyperintense "
            "regions indicating a mass effect on adjacent brain tissue."
        ),
        "recommendation": (
            "Neurosurgical consultation is advised for treatment planning. Serial MRI follow-up "
            "or surgical resection may be indicated depending on size, location, and symptoms."
        ),
        "urgency": "Moderate",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "description": (
            "Pituitary adenomas are tumors of the pituitary gland, often visible in the "
            "sellar/suprasellar region on MRI scans."
        ),
        "observation": (
            "The model detected features aligned with pituitary pathology in the central "
            "region of the scan, consistent with sellar region abnormality."
        ),
        "recommendation": (
            "Endocrinology and neurosurgical evaluation are recommended. Hormonal panel "
            "testing and dedicated pituitary MRI sequences should be obtained."
        ),
        "urgency": "Moderate",
    },
    "notumor": {
        "label": "No Tumor Detected",
        "description": (
            "No significant intracranial mass lesion was identified. Brain parenchyma "
            "appears within normal limits for the analyzed slice."
        ),
        "observation": (
            "The model did not detect abnormal mass-like features. Attention regions "
            "correspond primarily to normal anatomical structures and background tissue."
        ),
        "recommendation": (
            "Routine clinical follow-up as indicated. If symptoms persist, consider "
            "full multi-sequence MRI protocol or specialist referral for comprehensive evaluation."
        ),
        "urgency": "Low",
    },
}


def _image_quality_notes(image: NDArray[np.uint8]) -> str:
    """Generate basic MRI image quality assessment text."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    h, w = gray.shape[:2]

    quality = "acceptable"
    if std_intensity < 15:
        quality = "low contrast"
    elif std_intensity > 80:
        quality = "high contrast"

    return (
        f"Image dimensions: {w}×{h} pixels. Mean intensity: {mean_intensity:.1f}, "
        f"standard deviation: {std_intensity:.1f}. Overall signal quality assessed as {quality}. "
        f"Preprocessing pipeline applied: grayscale conversion, denoising, ROI extraction, "
        f"and normalization to 128×128 for model input."
    )


def _gradcam_observation(
    class_name: str,
    confidence: float,
    probabilities: dict[str, float],
) -> str:
    """Build model observation text from prediction outputs."""
    info = CLASS_INFO.get(class_name, CLASS_INFO["notumor"])
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    secondary = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0.0)

    lines = [
        info["observation"],
        f"Primary classification: {info['label']} with {confidence * 100:.1f}% confidence.",
    ]
    if secondary[1] > 0.05:
        sec_label = CLASS_INFO.get(secondary[0], {}).get("label", secondary[0])
        lines.append(
            f"Secondary differential: {sec_label} at {secondary[1] * 100:.1f}% probability."
        )
    if confidence < 0.75:
        lines.append(
            "Note: Moderate confidence level — clinical correlation and additional imaging "
            "are strongly recommended before definitive diagnosis."
        )
    return " ".join(lines)


def build_medical_report(
    class_name: str,
    confidence: float,
    probabilities: dict[str, float],
    original_image: NDArray[np.uint8],
    patient_id: str | None = None,
) -> dict[str, Any]:
    """Create structured medical report data."""
    info = CLASS_INFO.get(class_name, CLASS_INFO["notumor"])
    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "report_id": report_id,
        "patient_id": patient_id or f"PAT-{report_id}",
        "timestamp": timestamp,
        "predicted_class": class_name,
        "predicted_label": info["label"],
        "confidence": confidence,
        "probabilities": probabilities,
        "urgency": info["urgency"],
        "mri_analysis": _image_quality_notes(original_image),
        "tumor_description": info["description"],
        "model_observation": _gradcam_observation(class_name, confidence, probabilities),
        "recommendation": info["recommendation"],
    }


def encode_image_base64(image: NDArray[np.uint8], fmt: str = ".jpg") -> str:
    """Encode a BGR/RGB image array as base64 data URI."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError("Failed to encode image.")
    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def generate_pdf(report: dict[str, Any], gradcam_image: NDArray[np.uint8] | None = None) -> bytes:
    """Generate a PDF medical report and return raw bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#0d9488"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#134e4a"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.gray,
        leading=10,
    )

    story: list[Any] = []

    story.append(Paragraph("NeuroScan AI — MRI Analysis Report", title_style))
    story.append(Spacer(1, 6))

    meta_data = [
        ["Report ID:", report["report_id"]],
        ["Patient ID:", report["patient_id"]],
        ["Date:", report["timestamp"][:19].replace("T", " UTC ")],
        ["Urgency:", report["urgency"]],
    ]
    meta_table = Table(meta_data, colWidths=[1.2 * inch, 4.5 * inch])
    meta_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Diagnostic Classification", heading_style))
    result_data = [
        ["Predicted Class", report["predicted_label"]],
        ["Confidence", f"{report['confidence'] * 100:.2f}%"],
    ]
    for cls, prob in report["probabilities"].items():
        label = CLASS_INFO.get(cls, {}).get("label", cls)
        result_data.append([label, f"{prob * 100:.2f}%"])

    result_table = Table(result_data, colWidths=[2.5 * inch, 3.2 * inch])
    result_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ccfbf1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(result_table)

    story.append(Paragraph("Patient MRI Analysis", heading_style))
    story.append(Paragraph(report["mri_analysis"], body_style))
    story.append(Paragraph(report["tumor_description"], body_style))

    story.append(Paragraph("Model Observation (Grad-CAM)", heading_style))
    story.append(Paragraph(report["model_observation"], body_style))

    if gradcam_image is not None:
        img_buffer = io.BytesIO()
        cv2.imencode(".jpg", gradcam_image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(img_buffer)
        img_buffer.seek(0)
        rl_img = RLImage(img_buffer, width=3.5 * inch, height=3.5 * inch)
        story.append(rl_img)
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Figure: Grad-CAM heatmap overlay showing brain regions that most influenced "
                "the model's prediction (red = high influence, blue = low influence).",
                disclaimer_style,
            )
        )

    story.append(Paragraph("Clinical Recommendation", heading_style))
    story.append(Paragraph(report["recommendation"], body_style))

    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "DISCLAIMER: This report is generated by an AI-assisted diagnostic support system "
            "and is intended for research and clinical decision support only. It does not "
            "constitute a definitive medical diagnosis. All findings must be reviewed and "
            "validated by a qualified radiologist or neurologist before clinical action.",
            disclaimer_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()


class ReportStore:
    """Simple JSON file store for medical reports."""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        report: dict[str, Any],
        gradcam_b64: str | None = None,
        original_b64: str | None = None,
    ) -> Path:
        """Persist report metadata and images."""
        report_path = self.store_dir / f"{report['report_id']}.json"
        payload = {**report, "gradcam_image": gradcam_b64, "original_image": original_b64}
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report_path

    def list_reports(self) -> list[dict[str, Any]]:
        """Return all stored reports, newest first."""
        reports: list[dict[str, Any]] = []
        for path in sorted(self.store_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summary = {
                    "report_id": data["report_id"],
                    "patient_id": data["patient_id"],
                    "timestamp": data["timestamp"],
                    "predicted_class": data["predicted_class"],
                    "predicted_label": data["predicted_label"],
                    "confidence": data["confidence"],
                    "urgency": data["urgency"],
                }
                reports.append(summary)
            except (json.JSONDecodeError, KeyError):
                continue
        return reports

    def get(self, report_id: str) -> dict[str, Any] | None:
        """Load a single report by ID."""
        path = self.store_dir / f"{report_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
