import { useState, useRef } from "react";
import { predictImage, get, num } from "../services/api";
import Assistant from "./Assistant";
import { Icon, Bar } from "./ui";

const NAMES = { lung_n: "Normal lung tissue", lung_aca: "Lung adenocarcinoma pattern", lung_scc: "Lung squamous cell carcinoma pattern", unknown: "Unknown / Needs Manual Verification" };
const COL = { lung_n: "var(--teal)", lung_aca: "var(--blue)", lung_scc: "#7c5cd6" };

export function ImageResult({ r }) {
  const cls = String(get(r, "class", "storage class") ?? ""), unknown = cls === "unknown" || get(r, "manual verification required") === true;
  const probs = get(r, "class probabilities (%)", "class probabilities") || {};
  return (
    <div className="stack fade">
      {unknown && <div className="panel warn"><h3>Manual Verification Required</h3><p>The model confidence is below the configured threshold. This image has been placed in the unknown category for manual review.</p></div>}
      <div className="panel"><small>Histopathology result</small><h2>{unknown ? NAMES.unknown : NAMES[cls] || String(get(r, "prediction"))}</h2>
        <p className="muted">Model: {String(get(r, "model") ?? "ResNet18")} - not a medical diagnosis</p>
        <div className="meta"><div><span>Confidence</span><b>{num(get(r, "confidence (%)", "confidence")).toFixed(1)}%</b></div>
          <div><span>Threshold</span><b>{num(get(r, "threshold (%)", "threshold")).toFixed(0)}%</b></div>
          <div><span>Storage class</span><b>{String(get(r, "storage class") ?? "-")}</b></div>
          <div><span>Manual verification</span><b>{unknown ? "Required" : "Not required"}</b></div></div></div>
      <div className="panel"><h3>Model class scores</h3>
        {Object.entries(probs).sort((a, b) => num(b[1]) - num(a[1])).map(([k, v]) => <Bar key={k} label={`${k} - ${NAMES[k] || ""}`} value={num(v)} color={COL[k]} />)}
        <p className="tiny">These are uncalibrated model scores, not probabilities of having or not having cancer.</p>
        <p className="tiny">{String(get(r, "disclaimer") ?? "Educational/research prototype. Not a diagnostic tool.")}</p></div>
          {r._persistenceError && <div className="err" role="alert">Image analyzed, but saving failed: {r._persistenceError}</div>}
          <Assistant assessment={r} contextType="histopathology" analysisId={r._analysisId} persistenceError={r._persistenceError} />
    </div>
  );
}

export default function Histo({ onDone, guard }) {
  const [file, setFile] = useState(null), [busy, setBusy] = useState(false), [err, setErr] = useState(""), [res, setRes] = useState(null), [drag, setDrag] = useState(false);
  const ref = useRef();
  const pick = (fl) => {
    if (!fl) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(fl.type)) return setErr("Please upload a JPG, PNG, or WEBP image.");
    setErr(""); setRes(null); setFile(fl);
  };
  const run = async () => {
    if (!file) return setErr("Please select an image first.");
    setBusy(true); setErr("");
    try {
      const result = await predictImage(file);
      let analysisId = null, persistenceError = "";
      try {
        analysisId = await onDone({ type: "image", result, imageFile: file });
      } catch (error) {
        persistenceError = error.message || "Could not save this image analysis.";
      }
      setRes({ ...result, _analysisId: analysisId, _persistenceError: persistenceError });
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };
  return (
    <div className="stack fade">
      <div className={"drop" + (drag ? " over" : "")} onClick={() => ref.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}>
        <Icon n="upload" size={34} /><h3>Upload a lung histopathology image</h3><p className="muted">Drag & drop or click - JPG, PNG, WEBP</p>
        <input ref={ref} type="file" hidden accept=".jpg,.jpeg,.png,.webp" onChange={(e) => pick(e.target.files[0])} />
      </div>
      {file && <div className="panel filecard"><img src={URL.createObjectURL(file)} alt="preview" />
        <div><b>{file.name}</b><p className="muted">{(file.size / 1024).toFixed(0)} KB</p>
          <button className="btn" disabled={busy} onClick={() => guard(run)}>{busy ? "Analyzing histopathology image..." : "Analyze Image"}</button>{" "}
          <button className="btn ghost" disabled={busy} onClick={() => { setFile(null); setRes(null); }}>Remove</button></div></div>}
      {!file && <button className="btn" disabled={busy} onClick={() => guard(run)}>{busy ? "Analyzing histopathology image..." : "Analyze Image"}</button>}
      {err && <div className="err">{err}</div>}
      {res && <ImageResult r={res} />}
    </div>
  );
}
