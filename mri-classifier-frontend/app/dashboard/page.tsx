"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  Stethoscope,
  RefreshCw,
  Download,
  Eye,
  AlertCircle,
  FileText,
  Users,
  Activity,
  ChevronRight,
  Server,
} from "lucide-react";
import {
  checkBackendHealth,
  fetchReports,
  fetchReport,
  getReportPdfUrl,
  ReportSummary,
  FullReport,
} from "@/lib/api";
import { classLabels } from "@/lib/modelResults";

const urgencyColors: Record<string, string> = {
  High: "text-alert-red bg-alert-red/10 border-alert-red/25",
  Moderate: "text-amber-400 bg-amber-500/10 border-amber-500/25",
  Low: "text-success-green bg-success-green/10 border-success-green/25",
};

export default function DashboardPage() {
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [selectedReport, setSelectedReport] = useState<FullReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterUrgency, setFilterUrgency] = useState<string>("all");

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const online = await checkBackendHealth();
      setIsBackendOnline(online);
      if (online) {
        const data = await fetchReports();
        setReports(data);
      }
    } catch {
      setError("Failed to connect to the API. Ensure the FastAPI server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const viewReport = async (reportId: string) => {
    try {
      const full = await fetchReport(reportId);
      setSelectedReport(full);
    } catch {
      setError("Failed to load report details.");
    }
  };

  const filteredReports =
    filterUrgency === "all"
      ? reports
      : reports.filter((r) => r.urgency === filterUrgency);

  const stats = {
    total: reports.length,
    high: reports.filter((r) => r.urgency === "High").length,
    moderate: reports.filter((r) => r.urgency === "Moderate").length,
    low: reports.filter((r) => r.urgency === "Low").length,
  };

  return (
    <div className="flex-1 bg-bg-deep pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-border-slate/50">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Stethoscope className="w-5 h-5 text-teal-accent" />
              <span className="text-[10px] uppercase tracking-widest text-teal-accent font-bold">
                Clinical Portal
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-text-primary font-heading">
              Doctor Dashboard
            </h1>
            <p className="text-sm text-text-muted mt-1">
              Review AI-generated MRI analysis reports, Grad-CAM visualizations, and patient classifications.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-surface border border-border-slate rounded-xl px-4 py-2.5">
              <Server className="w-4 h-4 text-text-muted" />
              <span
                className={`text-xs font-bold ${
                  isBackendOnline ? "text-success-green" : "text-amber-400"
                }`}
              >
                {isBackendOnline ? "API Online" : "API Offline"}
              </span>
            </div>
            <button
              onClick={loadDashboard}
              disabled={isLoading}
              className="p-2.5 rounded-xl border border-border-slate hover:bg-surface-light transition-colors disabled:opacity-50"
              aria-label="Refresh dashboard"
            >
              <RefreshCw className={`w-4 h-4 text-text-muted ${isLoading ? "animate-spin" : ""}`} />
            </button>
            <Link
              href="/demo"
              className="inline-flex items-center gap-1.5 px-4 py-2.5 bg-teal-accent text-bg-deep text-xs font-bold uppercase rounded-xl hover:bg-teal-light transition-all"
            >
              New Analysis
              <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        {!isBackendOnline && !isLoading && (
          <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 text-amber-300 rounded-xl p-4 text-xs">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>
              The FastAPI server is offline. Start it with{" "}
              <code>uvicorn api.main:app --reload</code> and run analyses from the demo page to
              populate this dashboard.
            </p>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-3 bg-alert-red/10 border border-alert-red/25 text-alert-red rounded-xl p-4 text-xs">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard icon={FileText} label="Total Reports" value={stats.total} />
          <StatCard icon={Activity} label="High Priority" value={stats.high} color="text-alert-red" />
          <StatCard icon={Users} label="Moderate" value={stats.moderate} color="text-amber-400" />
          <StatCard icon={Users} label="Low Priority" value={stats.low} color="text-success-green" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold font-heading text-text-primary uppercase tracking-wider">
                Patient Reports
              </h2>
              <select
                value={filterUrgency}
                onChange={(e) => setFilterUrgency(e.target.value)}
                className="text-xs bg-surface border border-border-slate rounded-lg px-2 py-1.5 text-text-muted focus:outline-none focus:border-teal-accent/50"
              >
                <option value="all">All Urgency</option>
                <option value="High">High</option>
                <option value="Moderate">Moderate</option>
                <option value="Low">Low</option>
              </select>
            </div>

            {isLoading ? (
              <div className="text-center py-12 text-text-dim text-sm">Loading reports...</div>
            ) : filteredReports.length === 0 ? (
              <div className="border border-border-slate bg-surface/10 rounded-2xl p-10 text-center">
                <FileText className="w-10 h-10 text-text-dim/40 mx-auto mb-3" />
                <p className="text-sm text-text-muted">No reports yet</p>
                <p className="text-xs text-text-dim mt-1">
                  Run an MRI analysis from the demo page to generate reports.
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
                {filteredReports.map((r) => (
                  <button
                    key={r.report_id}
                    onClick={() => viewReport(r.report_id)}
                    className={`w-full text-left p-4 rounded-xl border transition-all ${
                      selectedReport?.report_id === r.report_id
                        ? "border-teal-accent bg-teal-accent/5"
                        : "border-border-slate bg-surface hover:border-teal-accent/30"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-xs font-bold text-text-primary">
                          {r.patient_id}
                        </p>
                        <p className="text-[10px] text-text-dim mt-0.5">
                          {classLabels[r.predicted_class] || r.predicted_label}
                        </p>
                      </div>
                      <span
                        className={`text-[9px] font-bold uppercase px-2 py-0.5 rounded-full border shrink-0 ${urgencyColors[r.urgency] || urgencyColors.Low}`}
                      >
                        {r.urgency}
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-[10px] text-text-dim">
                        #{r.report_id} · {new Date(r.timestamp).toLocaleDateString()}
                      </span>
                      <span className="text-[10px] font-bold text-teal-accent">
                        {(r.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="lg:col-span-7">
            {selectedReport ? (
              <div className="bg-surface border border-border-slate rounded-2xl p-6 space-y-6">
                <div className="flex items-start justify-between pb-4 border-b border-border-slate/40">
                  <div>
                    <span className="text-[10px] text-text-dim font-bold uppercase tracking-widest">
                      Report #{selectedReport.report_id}
                    </span>
                    <h3 className="text-xl font-bold font-heading text-text-primary mt-1">
                      {selectedReport.patient_id}
                    </h3>
                    <p className="text-xs text-text-dim mt-1">
                      {new Date(selectedReport.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <a
                      href={getReportPdfUrl(selectedReport.report_id)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 px-3 py-1.5 bg-teal-accent text-bg-deep text-[10px] font-bold uppercase rounded-lg hover:bg-teal-light transition-all"
                    >
                      <Download className="w-3 h-3" />
                      PDF
                    </a>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {selectedReport.gradcam_image && (
                    <div>
                      <h4 className="text-[10px] uppercase font-bold text-text-muted tracking-wider mb-2 flex items-center gap-1">
                        <Eye className="w-3 h-3" /> Grad-CAM Visualization
                      </h4>
                      <div className="relative aspect-square rounded-xl overflow-hidden border border-border-slate">
                        <img
                          src={selectedReport.gradcam_image}
                          alt="Grad-CAM heatmap"
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>
                  )}

                  <div className="space-y-4">
                    <div>
                      <span className="text-[10px] text-text-dim uppercase font-bold">
                        Classification
                      </span>
                      <p className="text-lg font-bold text-text-primary font-heading">
                        {selectedReport.predicted_label}
                      </p>
                      <p className="text-sm text-teal-accent font-bold">
                        {(selectedReport.confidence * 100).toFixed(2)}% confidence
                      </p>
                    </div>

                    <ReportField title="MRI Analysis" content={selectedReport.mri_analysis} />
                    <ReportField
                      title="Model Observation"
                      content={selectedReport.model_observation}
                    />
                    <ReportField
                      title="Recommendation"
                      content={selectedReport.recommendation}
                      highlight
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="border border-border-slate bg-surface/10 rounded-2xl p-16 text-center flex flex-col items-center justify-center min-h-[400px]">
                <Stethoscope className="w-12 h-12 text-text-dim/40 mb-3" />
                <h3 className="text-sm font-semibold text-text-muted">Select a report</h3>
                <p className="text-xs text-text-dim mt-2 max-w-sm">
                  Choose a patient report from the list to view full analysis details, Grad-CAM
                  visualization, and clinical recommendations.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  color = "text-teal-accent",
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
  color?: string;
}) {
  return (
    <div className="bg-surface border border-border-slate rounded-2xl p-5">
      <Icon className={`w-5 h-5 ${color} mb-2`} />
      <p className="text-2xl font-bold font-heading text-text-primary">{value}</p>
      <p className="text-[10px] uppercase tracking-wider text-text-dim font-bold mt-1">{label}</p>
    </div>
  );
}

function ReportField({
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
      <h4 className="text-[10px] uppercase font-bold text-text-muted tracking-wider mb-1">
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
