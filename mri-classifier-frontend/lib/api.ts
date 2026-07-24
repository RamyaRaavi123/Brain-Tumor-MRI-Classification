export interface PredictionResponse {
  class_name: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface MedicalReport {
  report_id: string;
  patient_id: string;
  timestamp: string;
  predicted_class: string;
  predicted_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  urgency: string;
  mri_analysis: string;
  tumor_description: string;
  model_observation: string;
  recommendation: string;
}

export interface AnalyzeResponse extends PredictionResponse {
  gradcam_image: string;
  original_image: string;
  report: MedicalReport;
}

export interface ReportSummary {
  report_id: string;
  patient_id: string;
  timestamp: string;
  predicted_class: string;
  predicted_label: string;
  confidence: number;
  urgency: string;
}

export interface FullReport extends MedicalReport {
  gradcam_image?: string;
  original_image?: string;
}

const API_BASE = "http://localhost:8000";

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(3000),
    });
    const data = await res.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}

export async function predictMRI(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || "Prediction failed");
  }

  return res.json();
}

export async function analyzeMRI(
  file: File,
  patientId?: string,
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (patientId) {
    formData.append("patient_id", patientId);
  }

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || "Analysis failed");
  }

  return res.json();
}

export async function fetchReports(): Promise<ReportSummary[]> {
  const res = await fetch(`${API_BASE}/reports`);
  if (!res.ok) throw new Error("Failed to fetch reports");
  return res.json();
}

export async function fetchReport(reportId: string): Promise<FullReport> {
  const res = await fetch(`${API_BASE}/reports/${reportId}`);
  if (!res.ok) throw new Error("Report not found");
  return res.json();
}

export function getReportPdfUrl(reportId: string): string {
  return `${API_BASE}/reports/${reportId}/pdf`;
}

/**
 * Mock analysis fallback when backend is offline
 */
export async function mockAnalyzeMRI(fileName: string): Promise<AnalyzeResponse> {
  await new Promise((resolve) => setTimeout(resolve, 1800));

  const base = await mockPredictMRI(fileName);
  const reportId = Math.random().toString(36).substring(2, 10).toUpperCase();

  const urgencyMap: Record<string, string> = {
    glioma: "High",
    meningioma: "Moderate",
    pituitary: "Moderate",
    notumor: "Low",
  };

  const labelMap: Record<string, string> = {
    glioma: "Glioma Tumor",
    meningioma: "Meningioma Tumor",
    pituitary: "Pituitary Tumor",
    notumor: "No Tumor Detected",
  };

  return {
    ...base,
    gradcam_image: "",
    original_image: "",
    report: {
      report_id: reportId,
      patient_id: `PAT-${reportId}`,
      timestamp: new Date().toISOString(),
      predicted_class: base.class_name,
      predicted_label: labelMap[base.class_name] || base.class_name,
      confidence: base.confidence,
      probabilities: base.probabilities,
      urgency: urgencyMap[base.class_name] || "Low",
      mri_analysis:
        "Image dimensions: 128×128 pixels. Mean intensity: 87.3, standard deviation: 42.1. Overall signal quality assessed as acceptable. Preprocessing pipeline applied: grayscale conversion, denoising, ROI extraction, and normalization.",
      tumor_description: `Simulated analysis for ${labelMap[base.class_name] || base.class_name}. This is mock data — connect the FastAPI backend for real Grad-CAM and report generation.`,
      model_observation: `Simulated Grad-CAM observation: The model identified key regions influencing the ${labelMap[base.class_name]} classification with ${(base.confidence * 100).toFixed(1)}% confidence.`,
      recommendation:
        "Connect the FastAPI backend (uvicorn api.main:app --reload) for real clinical recommendations. This is simulation mode.",
    },
  };
}

/**
 * Mock prediction fallback for testing/demo purposes when backend is offline
 */
export async function mockPredictMRI(fileName: string): Promise<PredictionResponse> {
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const nameLower = fileName.toLowerCase();
  let predictedClass = "notumor";
  let confidence = 0.95 + Math.random() * 0.04;

  if (nameLower.includes("glioma")) {
    predictedClass = "glioma";
  } else if (nameLower.includes("meningioma")) {
    predictedClass = "meningioma";
  } else if (nameLower.includes("pituitary")) {
    predictedClass = "pituitary";
  } else if (nameLower.includes("tumor") || Math.random() > 0.5) {
    const types = ["glioma", "meningioma", "pituitary"];
    predictedClass = types[Math.floor(Math.random() * types.length)];
  }

  const classes = ["glioma", "meningioma", "notumor", "pituitary"];
  const probabilities: Record<string, number> = {};

  let remaining = 1.0 - confidence;
  classes.forEach((c) => {
    if (c === predictedClass) {
      probabilities[c] = confidence;
    } else {
      const share = Math.random();
      probabilities[c] = remaining * share;
      remaining -= probabilities[c];
    }
  });

  const lastClass =
    classes.find((c) => c !== predictedClass && probabilities[c] === undefined) ||
    classes[0];
  probabilities[lastClass] = (probabilities[lastClass] || 0) + remaining;

  return {
    class_name: predictedClass,
    class_index: classes.indexOf(predictedClass),
    confidence: confidence,
    probabilities: probabilities,
  };
}
