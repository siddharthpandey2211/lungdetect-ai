import { useState } from "react";
import { Icon, Bar } from "./ui";

export function Dashboard({ history, go }) {
  const risk = history.filter((h) => h.type === "risk").length, img = history.filter((h) => h.type === "image").length, unk = history.filter((h) => h.unknown).length;
  const stats = [["Total analyses", history.length], ["Risk assessments", risk], ["Images analyzed", img], ["Unknown images", unk]];
  const max = Math.max(1, risk, img, unk);
  return (
    <div className="stack fade">
      <div><h1>Welcome to LungDetect AI</h1><p className="muted">AI-assisted lung health analysis for educational and research purposes.</p></div>
      <div className="grid2">
        <div className="panel card"><Icon n="activity" size={26} /><h3>Risk Assessment</h3><p className="muted">Evaluate risk using health and symptom indicators.</p><button className="btn" onClick={() => go("risk")}>Start Risk Assessment</button></div>
        <div className="panel card"><Icon n="image" size={26} /><h3>Histopathology Analysis</h3><p className="muted">Analyze lung tissue images using the ResNet18 image classification model.</p><button className="btn" onClick={() => go("histo")}>Analyze Image</button></div>
      </div>
      <div className="stats">{stats.map(([l, v]) => <div key={l}><b>{v}</b><span>{l}</span></div>)}</div>
      <div className="panel"><h3>Activity overview</h3>
        <Bar label="Risk assessments" value={(risk / max) * 100} color="var(--blue)" /><Bar label="Images analyzed" value={(img / max) * 100} color="var(--teal)" /><Bar label="Unknown images" value={(unk / max) * 100} color="var(--amber)" />
        <p className="tiny">Counts from your saved Supabase account records.</p></div>
    </div>
  );
}

export function History({ history }) {
  const [flt, setFlt] = useState("all");
  const rows = history.filter((h) => flt === "all" || h.type === flt);
  return (
    <div className="stack fade"><h1>History</h1>
      <div className="seg">{[["all", "All"], ["risk", "Risk Assessments"], ["image", "Histopathology"]].map(([v, l]) => <button key={v} className={flt === v ? "on" : ""} onClick={() => setFlt(v)}>{l}</button>)}</div>
      <div className="panel tablewrap"><table><thead><tr><th>Type</th><th>Result</th><th>Confidence</th><th>Date</th></tr></thead>
        <tbody>{rows.length ? rows.map((h, i) => <tr key={i}><td>{h.type === "risk" ? "Risk" : "Histopathology"}</td><td>{h.result}</td><td>{Number(h.confidence).toFixed(1)}%</td><td>{new Date(h.date).toLocaleString()}</td></tr>)
          : <tr><td colSpan="4" className="muted">No analyses yet.</td></tr>}</tbody></table></div>
      <p className="tiny">Showing analyses saved to your private Supabase account.</p></div>
  );
}

export function About() {
  return (
    <div className="stack fade"><h1>About LungDetect AI</h1>
      <p>LungDetect AI is an educational/research prototype exploring AI-assisted lung health analysis with two independent models.</p>
      <div className="grid2">
        <div className="panel"><h3>Gaussian Naive Bayes</h3><p className="muted">Used for symptom/health-indicator based risk analysis.</p></div>
        <div className="panel"><h3>ResNet18</h3><p className="muted">Used for lung histopathology image classification.</p></div></div>
      <div className="panel"><h3>Histopathology classes</h3><ul><li>lung_n - Normal lung tissue</li><li>lung_aca - Lung adenocarcinoma pattern</li><li>lung_scc - Lung squamous cell carcinoma pattern</li></ul>
        <p className="tiny">Images below the confidence threshold (default 70%) are classed "unknown" and saved for manual review.</p></div>
      <div className="panel warn"><h3>Important Disclaimer</h3><p>This application is an educational/research prototype and is not a medical diagnostic tool. Results should not be used as a substitute for professional medical evaluation.</p></div></div>
  );
}
