"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  Upload,
  FileImage,
  AlertCircle,
  RefreshCw,
  Info,
  Server,
  Play,
  FileText,
  Download,
  Eye,
  Stethoscope,
} from "lucide-react";
import {
  checkBackendHealth,
  analyzeMRI,
  mockAnalyzeMRI,
  AnalyzeResponse,
  getReportPdfUrl,
} from "@/lib/api";
import { classLabels } from "@/lib/modelResults";

const sampleImages = [
  { name: "Glioma Sample", file: "glioma.jpg", label: "glioma" },
  { name: "Meningioma Sample", file: "meningioma.jpg", label: "meningioma" },
  { name: "No Tumor Sample", file: "notumor.jpg", label: "notumor" },
  { name: "Pituitary Sample", file: "pituitary.jpg", label: "pituitary" },
];

const urgencyColors: Record<string, string> = {
  High: "text-alert-red bg-alert-red/10 border-alert-red/25",
  Moderate: "text-amber-400 bg-amber-500/10 border-amber-500/25",
  Low: "text-success-green bg-success-green/10 border-success-green/25",
};

export default function DemoPage() {
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSampleUrl, setSelectedSampleUrl] = useState<string | null>(null);
  const [patientId, setPatientId] = useState("");
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showGradcam, setShowGradcam] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    verifyBackend();
  }, []);

  const verifyBackend = async () => {
    setIsCheckingHealth(true);
    const online = await checkBackendHealth();
    setIsBackendOnline(online);
    setIsCheckingHealth(false);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        processNewFile(file);
      } else {
        setError("Only image files (JPEG, PNG) are supported.");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processNewFile(e.target.files[0]);
    }
  };

  const processNewFile = (file: File) => {
    setSelectedFile(file);
    setSelectedSampleUrl(null);
    setAnalysis(null);
    setError(null);
  };

  const selectSample = async (sample: (typeof sampleImages)[0]) => {
    setError(null);
    setAnalysis(null);
    setSelectedFile(null);
    setSelectedSampleUrl(`/mri-samples/${sample.file}`);
  };

  const handleInference = async () => {
    setIsLoading(true);
    setError(null);

    try {
      let fileToAnalyze: File;

      if (selectedFile) {
        fileToAnalyze = selectedFile;
      } else if (selectedSampleUrl) {
        const response = await fetch(selectedSampleUrl);
        const blob = await response.blob();
        fileToAnalyze = new File(
          [blob],
          selectedSampleUrl.split("/").pop() || "sample.jpg",
          { type: "image/jpeg" },
        );
      } else {
        throw new Error("No image selected");
      }

      let res: AnalyzeResponse;
      if (isBackendOnline) {
        res = await analyzeMRI(fileToAnalyze, patientId || undefined);
      } else {
        res = await mockAnalyzeMRI(fileToAnalyze.name);
      }

      setAnalysis(res);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "An error occurred during analysis.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const report = analysis?.report;

  return (
    <div className="flex-1 bg-bg-deep pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto space-y-10">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-border-slate/50">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-text-primary font-heading">
              MRI Tumor Classifier
            </h1>
            <p className="text-sm text-text-muted mt-1 max-w-xl">
              Upload a brain MRI scan for classification with Grad-CAM explainability and automated medical report generation.
            </p>
          </div>

          <div className="flex items-center gap-3 bg-surface border border-border-slate rounded-2xl p-4">
            <div className="flex items-center gap-2">
              <Server className="w-4.5 h-4.5 text-text-muted" />
              <div className="text-left">
                <span className="text-[9px] uppercase tracking-wider text-text-dim block font-bold">
                  FastAPI Server
                </span>
                <span
                  className={`text-xs font-bold font-heading ${
                    isBackendOnline === true
                      ? "text-success-green"
                      : isBackendOnline === false
                        ? "text-amber-400"
                        : "text-text-muted"
                  }`}
                >
                  {isBackendOnline === true
                    ? "Online (Full Analysis)"
                    : isBackendOnline === false
                      ? "Offline (Simulation Mode)"
                      : "Checking Status..."}
                </span>
              </div>
            </div>
            <button
              onClick={verifyBackend}
              disabled={isCheckingHealth}
              className="p-1.5 rounded-lg hover:bg-surface-light border border-border-slate/85 transition-colors disabled:opacity-50"
              aria-label="Refresh backend status"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 text-text-muted ${isCheckingHealth ? "animate-spin" : ""}`}
              />
            </button>
          </div>
        </div>

        {isBackendOnline === false && (
          <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 text-amber-300 rounded-xl p-4 text-xs">
            <Info className="w-5 h-5 flex-shrink-0" />
            <p>
              The FastAPI server is offline. Running in <strong>Simulation Mode</strong> without
              real Grad-CAM. Start the API via{" "}
              <code>uvicorn api.main:app --reload</code> for full analysis and PDF reports.
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-surface rounded-2xl border border-border-slate p-4">
              <label className="text-[10px] uppercase font-bold text-text-muted tracking-wider block mb-2">
                Patient ID (optional)
              </label>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g. PAT-2024-001"
                className="w-full px-3 py-2 bg-bg-deep border border-border-slate rounded-lg text-sm text-text-primary placeholder:text-text-dim focus:outline-none focus:border-teal-accent/50"
              />
            </div>

            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
                dragActive
                  ? "border-teal-accent bg-teal-accent/5 scale-[0.99]"
                  : selectedFile || selectedSampleUrl
                    ? "border-border-slate bg-surface/30"
                    : "border-border-slate hover:border-teal-accent/40 bg-surface/10"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />

              {selectedFile || selectedSampleUrl ? (
                <div className="space-y-6">
                  <div className="relative w-full max-w-[280px] aspect-square mx-auto rounded-xl overflow-hidden border border-border-slate">
                    {analysis && showGradcam && analysis.gradcam_image ? (
                      <img
                        src={analysis.gradcam_image}
                        alt="Grad-CAM heatmap overlay on MRI scan"
                        className="w-full h-full object-cover"
                      />
                    ) : selectedFile ? (
                      <img
                        src={URL.createObjectURL(selectedFile)}
                        alt="Uploaded MRI scan preview"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <img
                        src={selectedSampleUrl || ""}
                        alt="Selected MRI sample preview"
                        className="w-full h-full object-cover"
                      />
                    )}
                  </div>

                  {analysis?.gradcam_image && (
                    <div className="flex justify-center gap-2">
                      <button
                        onClick={() => setShowGradcam(true)}
                        className={`px-3 py-1.5 text-[10px] font-bold uppercase rounded-lg border transition-colors ${
                          showGradcam
                            ? "bg-teal-accent text-bg-deep border-teal-accent"
                            : "border-border-slate text-text-muted hover:border-teal-accent/30"
                        }`}
                      >
                        <Eye className="w-3 h-3 inline mr-1" />
                        Grad-CAM
                      </button>
                      <button
                        onClick={() => setShowGradcam(false)}
                        className={`px-3 py-1.5 text-[10px] font-bold uppercase rounded-lg border transition-colors ${
                          !showGradcam
                            ? "bg-teal-accent text-bg-deep border-teal-accent"
                            : "border-border-slate text-text-muted hover:border-teal-accent/30"
                        }`}
                      >
                        Original
                      </button>
                    </div>
                  )}

                  <div>
                    <h3 className="text-sm font-bold text-text-primary">
                      {selectedFile ? selectedFile.name : selectedSampleUrl?.split("/").pop()}
                    </h3>
                    <p className="text-xs text-text-dim mt-1">
                      {selectedFile
                        ? `${(selectedFile.size / 1024).toFixed(1)} KB`
                        : "Dataset Sample Scan"}
                    </p>
                  </div>
                  <div className="flex justify-center gap-3">
                    <button
                      onClick={() => {
                        setSelectedFile(null);
                        setSelectedSampleUrl(null);
                        setAnalysis(null);
                        setError(null);
                      }}
                      className="px-4 py-2 border border-border-slate text-text-muted text-xs font-semibold rounded-lg hover:bg-surface-light transition-colors"
                    >
                      Clear Selection
                    </button>
                    <button
                      onClick={handleInference}
                      disabled={isLoading}
                      className="inline-flex items-center gap-1.5 px-5 py-2 bg-teal-accent text-bg-deep text-xs font-bold uppercase rounded-lg hover:bg-teal-light transition-all disabled:opacity-50"
                    >
                      {isLoading ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5 fill-current" />
                          Run Full Analysis
                        </>
                      )}
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  className="cursor-pointer py-10"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="w-16 h-16 rounded-full bg-surface-light border border-border-slate flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-6 h-6 text-teal-accent" />
                  </div>
                  <h3 className="text-base font-bold text-text-primary">
                    Drag and drop your MRI file here
                  </h3>
                  <p className="text-xs text-text-muted mt-2">
                    or click to browse (JPEG, PNG) — includes Grad-CAM + medical report
                  </p>
                </div>
              )}
            </div>

            <div className="bg-surface rounded-2xl border border-border-slate p-6">
              <h3 className="text-xs font-bold uppercase tracking-wider text-text-primary mb-4 font-heading">
                Quick Test Samples
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {sampleImages.map((sample) => (
                  <button
                    key={sample.file}
                    onClick={() => selectSample(sample)}
                    className={`group text-left border rounded-xl overflow-hidden bg-bg-deep transition-all duration-300 ${
                      selectedSampleUrl === `/mri-samples/${sample.file}`
                        ? "border-teal-accent ring-2 ring-teal-accent/20"
                        : "border-border-slate hover:border-teal-accent/30"
                    }`}
                  >
                    <div className="relative aspect-square w-full bg-bg-deep border-b border-border-slate">
                      <Image
                        src={`/mri-samples/${sample.file}`}
                        alt={sample.name}
                        fill
                        className="object-cover group-hover:scale-105 transition-transform duration-300"
                        sizes="(max-width: 640px) 50vw, 25vw"
                      />
                    </div>
                    <div className="p-3">
                      <p className="text-[10px] font-bold text-text-primary leading-tight">
                        {sample.name}
                      </p>
                      <p className="text-[9px] text-teal-accent mt-0.5 font-semibold uppercase">
                        {sample.label}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 space-y-6">
            {error && (
              <div className="bg-alert-red/10 border border-alert-red/25 text-alert-red rounded-2xl p-5 flex items-start gap-3 text-xs leading-relaxed">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <div>
                  <h4 className="font-bold">Execution Error</h4>
                  <p className="mt-1">{error}</p>
                </div>
              </div>
            )}

            {analysis && report ? (
              <div className="space-y-4">
                <div className="bg-surface border border-border-slate rounded-2xl p-6 space-y-5">
                  <div className="flex justify-between items-start pb-4 border-b border-border-slate/40">
                    <div>
                      <span className="text-[10px] text-text-dim font-bold uppercase tracking-widest block">
                        Diagnostic Classification
                      </span>
                      <h3 className="text-2xl font-bold font-heading mt-1 text-text-primary">
                        {classLabels[analysis.class_name] || analysis.class_name}
                      </h3>
                    </div>
                    <div className="text-right">
                      <span className="text-[9px] text-text-dim font-bold uppercase tracking-widest block">
                        Confidence
                      </span>
                      <span className="text-xl font-bold font-heading text-teal-accent">
                        {(analysis.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[10px] font-bold uppercase px-2.5 py-1 rounded-full border ${urgencyColors[report.urgency] || urgencyColors.Low}`}
                    >
                      {report.urgency} Priority
                    </span>
                    <span className="text-[10px] text-text-dim">
                      Report #{report.report_id}
                    </span>
                  </div>

                  <div className="space-y-3">
                    {Object.entries(analysis.probabilities).map(([className, probability]) => {
                      const isWinner = className === analysis.class_name;
                      const pct = (probability * 100).toFixed(1);
                      return (
                        <div key={className} className="space-y-1">
                          <div className="flex justify-between text-xs font-semibold">
                            <span
                              className={
                                isWinner ? "text-teal-accent font-bold" : "text-text-muted"
                              }
                            >
                              {classLabels[className] || className}
                            </span>
                            <span className={isWinner ? "text-teal-accent" : "text-text-muted"}>
                              {pct}%
                            </span>
                          </div>
                          <div className="w-full h-2 rounded bg-bg-deep border border-border-slate/40 overflow-hidden">
                            <div
                              style={{ width: `${pct}%` }}
                              className={`h-full rounded transition-all duration-1000 ${
                                isWinner ? "bg-teal-accent" : "bg-surface-light"
                              }`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="bg-surface border border-border-slate rounded-2xl p-6 space-y-4">
                  <div className="flex items-center gap-2 pb-3 border-b border-border-slate/40">
                    <FileText className="w-4 h-4 text-teal-accent" />
                    <h3 className="text-sm font-bold font-heading text-text-primary">
                      Automated Medical Report
                    </h3>
                  </div>

                  <ReportSection title="MRI Analysis" content={report.mri_analysis} />
                  <ReportSection title="Tumor Description" content={report.tumor_description} />
                  <ReportSection title="Model Observation" content={report.model_observation} />
                  <ReportSection
                    title="Recommendation"
                    content={report.recommendation}
                    highlight
                  />

                  <div className="flex gap-3 pt-2">
                    {isBackendOnline && (
                      <a
                        href={getReportPdfUrl(report.report_id)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 px-4 py-2 bg-teal-accent text-bg-deep text-xs font-bold uppercase rounded-lg hover:bg-teal-light transition-all"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download PDF
                      </a>
                    )}
                    <Link
                      href="/dashboard"
                      className="inline-flex items-center gap-1.5 px-4 py-2 border border-border-slate text-text-primary text-xs font-bold uppercase rounded-lg hover:bg-surface-light transition-all"
                    >
                      <Stethoscope className="w-3.5 h-3.5" />
                      Doctor Dashboard
                    </Link>
                  </div>
                </div>
              </div>
            ) : (
              <div className="border border-border-slate bg-surface/10 rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[300px]">
                <FileImage className="w-12 h-12 text-text-dim/40 mb-3" />
                <h3 className="text-sm font-semibold text-text-muted">No analysis available</h3>
                <p className="text-xs text-text-dim mt-2 max-w-xs leading-relaxed">
                  Select an MRI scan and click &ldquo;Run Full Analysis&rdquo; for classification,
                  Grad-CAM heatmap, and automated medical report.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ReportSection({
  title,
  content,
  highlight = false,
}: {
  title: string;
  content: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <h4 className="text-[10px] uppercase font-bold text-text-muted tracking-wider mb-1.5">
        {title}
      </h4>
      <p
        className={`text-xs leading-relaxed ${
          highlight ? "text-teal-accent font-medium" : "text-text-muted"
        }`}
      >
        {content}
      </p>
    </div>
  );
}
