import { useState } from "react";
import { Icon } from "./ui";

export function Dashboard({ history, go }) {
  const risk = history.filter((h) => h.type === "risk").length, histo = history.filter((h) => h.type === "image").length, mri = history.filter((h) => h.type === "mri").length, ct = history.filter((h) => h.type === "ct").length, unk = history.filter((h) => h.unknown).length;
  const stats = [["Total analyses", history.length], ["Risk assessments", risk], ["Histopathology", histo], ["MRI analyses", mri], ["CT analyses", ct]];
  const max = Math.max(1, risk, histo, mri, ct, unk);
  const activity = [
    ["Risk assessments", risk, "var(--blue)"],
    ["Histopathology", histo, "var(--teal)"],
    ["MRI analyses", mri, "var(--red)"],
    ["CT analyses", ct, "var(--amber)"],
    ["Unknown images", unk, "#7c5cd6"],
  ];
  const chartWidth = 820, chartHeight = 300, chartLeft = 58, chartRight = 24, chartTop = 22, chartBottom = 54;
  const plotWidth = chartWidth - chartLeft - chartRight, plotHeight = chartHeight - chartTop - chartBottom;
  const points = activity.map(([label, value, color], index) => ({
    label,
    value,
    color,
    x: chartLeft + (plotWidth / Math.max(1, activity.length - 1)) * index,
    y: chartTop + plotHeight - (value / max) * plotHeight,
  }));
  const pointString = points.map(({ x, y }) => `${x},${y}`).join(" ");
  return (
    <div className="stack fade">
      <div><h1>Welcome to LungDetect AI</h1><p className="muted">AI-assisted lung health analysis for educational and research purposes.</p></div>
      <div className="grid2">
        <div className="panel card"><Icon n="activity" size={26} /><h3>Risk Assessment</h3><p className="muted">Evaluate risk using health and symptom indicators.</p><button className="btn" onClick={() => go("risk")}>Start Risk Assessment</button></div>
        <div className="panel card"><Icon n="image" size={26} /><h3>Histopathology Analysis</h3><p className="muted">Analyze lung tissue images using the ResNet18 image classification model.</p><button className="btn" onClick={() => go("histo")}>Analyze Image</button></div>
        <div className="panel card"><Icon n="scan" size={26} /><h3>MRI Analysis</h3><p className="muted">Classify lung MRI images using the ConvMixer model.</p><button className="btn" onClick={() => go("mri")}>Analyze MRI</button></div>
        <div className="panel card"><Icon n="scan" size={26} /><h3>CT Analysis</h3><p className="muted">Classify lung CT scans using the ResNet18 model.</p><button className="btn" onClick={() => go("ct")}>Analyze CT</button></div>
      </div>
      <div className="stats">{stats.map(([l, v]) => <div key={l}><b>{v}</b><span>{l}</span></div>)}</div>
      <div className="panel"><h3>Activity overview</h3>
        <div className="activity-line-chart" role="img" aria-label="Activity overview line graph showing saved analyses by section">
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} preserveAspectRatio="none">
            {[0, 25, 50, 75, 100].map((level) => {
              const y = chartTop + plotHeight - (level / 100) * plotHeight;
              return <g key={level}><line className="chart-grid" x1={chartLeft} x2={chartWidth - chartRight} y1={y} y2={y} /><text className="chart-axis" x={chartLeft - 12} y={y + 4} textAnchor="end">{level}%</text></g>;
            })}
            <line className="chart-axis-line" x1={chartLeft} x2={chartLeft} y1={chartTop} y2={chartTop + plotHeight} />
            <line className="chart-axis-line" x1={chartLeft} x2={chartWidth - chartRight} y1={chartTop + plotHeight} y2={chartTop + plotHeight} />
            <polyline className="chart-edge" points={pointString} />
            {points.map(({ label, value, color, x, y }) => <g key={label}>
              <circle className="chart-node-halo" cx={x} cy={y} r="9" style={{ stroke: color }} />
              <circle className="chart-node" cx={x} cy={y} r="5" style={{ fill: color }} />
              <text className="chart-value" x={x} y={y - 14} textAnchor="middle">{((value / max) * 100).toFixed(1)}%</text>
              <text className="chart-label" x={x} y={chartHeight - 18} textAnchor="middle">{label.replace(" assessments", "").replace(" analyses", "")}</text>
            </g>)}
          </svg>
        </div>
        <div className="activity-legend" aria-label="Chart legend">
          {activity.map(([label, , color]) => <span key={label}><i style={{ background: color }} />{label}</span>)}
        </div>
        <p className="tiny">Counts from your saved Supabase account records.</p></div>
    </div>
  );
}

export function History({ history }) {
  const [flt, setFlt] = useState("all");
  const rows = history.filter((h) => flt === "all" || h.type === flt);
  return (
    <div className="stack fade"><h1>History</h1>
      <div className="seg">{[["all", "All"], ["risk", "Risk Assessments"], ["image", "Histopathology"], ["mri", "MRI"], ["ct", "CT"]].map(([v, l]) => <button key={v} className={flt === v ? "on" : ""} onClick={() => setFlt(v)}>{l}</button>)}</div>
      <div className="panel tablewrap"><table><thead><tr><th>Type</th><th>Result</th><th>Confidence</th><th>Date</th></tr></thead>
        <tbody>{rows.length ? rows.map((h, i) => <tr key={i}><td>{h.type === "risk" ? "Risk" : h.type === "mri" ? "MRI" : h.type === "ct" ? "CT" : "Histopathology"}</td><td>{h.result}</td><td>{Number(h.confidence).toFixed(1)}%</td><td>{new Date(h.date).toLocaleString()}</td></tr>)
          : <tr><td colSpan="4" className="muted">No analyses yet.</td></tr>}</tbody></table></div>
      <p className="tiny">Showing analyses saved to your private Supabase account.</p></div>
  );
}

export function About() {
  return (
    <div className="stack fade"><h1>About LungDetect AI</h1>
      <p>LungDetect AI is an educational/research prototype exploring four separate analysis methods. Each result is an experimental model output, not a medical diagnosis.</p>
      <div className="grid2">
        <div className="panel"><h3>Symptom risk assessment</h3><p className="muted">Uses health and symptom indicators with Gaussian Naive Bayes to estimate a screening risk pattern. It does not confirm or rule out cancer.</p></div>
        <div className="panel"><h3>Histopathology</h3><p className="muted">Examines tissue images from a biopsy. A pathologist uses cellular patterns, staining, specimen quality, and clinical context to make a diagnosis; this prototype only provides experimental pattern scores.</p></div>
        <div className="panel"><h3>MRI scan</h3><p className="muted">Uses magnetic fields and radio waves to create detailed chest images without ionizing radiation. MRI findings must be interpreted by a radiologist or clinician.</p></div>
        <div className="panel"><h3>CT scan</h3><p className="muted">Uses X-rays and computer processing to create cross-sectional images of the lungs and chest. CT classification is an experimental ResNet18 output, not a clinical conclusion.</p></div></div>
      <div className="panel"><h3>Histopathology classes</h3><ul><li>lung_n - Normal lung tissue</li><li>lung_aca - Lung adenocarcinoma pattern</li><li>lung_scc - Lung squamous cell carcinoma pattern</li></ul>
        <p className="tiny">Images below the confidence threshold (default 70%) are marked "unknown" for manual review and retained in the uploads folder, separate from training classes.</p></div>
      <div className="panel warn"><h3>Important Disclaimer</h3><p>This application is an educational/research prototype and is not a medical diagnostic tool. Results should not be used as a substitute for professional medical evaluation.</p></div></div>
  );
}
