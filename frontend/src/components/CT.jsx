import { useRef, useState } from "react";
import { predictCT, get, num } from "../services/api";
import { Icon } from "./ui";
import Assistant from "./Assistant";

export function CTResult({ result }) {
  const predictedClass = String(get(result, "class") || "");
  const probabilities = get(result, "class probabilities (%)", "class probabilities") || {};
  const winningClass = Object.entries(probabilities)
    .sort((a, b) => num(b[1]) - num(a[1]))[0]?.[0] || predictedClass;
  const unknown = predictedClass === "unknown" || get(result, "manual verification required") === true;
  const status = unknown
    ? "Unknown / Needs Manual Verification"
    : winningClass === "normal"
      ? "No Cancer"
      : `Cancer (${winningClass === "malignant" ? "Malignant" : "Benign"})`;
  const winningPercentage = unknown
    ? 0
    : num(probabilities[winningClass] ?? get(result, "confidence (%)", "confidence"));

  return (
    <div className="stack fade">
      {unknown && <div className="panel warn"><h3>Unknown image - manual verification required</h3><p>{String(get(result, "unknown reason") || "This image could not be validated as a lung CT scan.")}</p><p className="tiny">The model was not used to assign a cancer class.</p></div>}
      <div className="panel">
        <small>Model Classification: Lung CT Scan</small>
        <h2>{status}</h2>
        <p className="muted">Model: {String(get(result, "model") || "ResNet18")} - not a medical diagnosis</p>
        <div className="meta">
          <div><span>Highest score</span><b>{winningPercentage.toFixed(1)}%</b></div>
          <div><span>Result</span><b>{status}</b></div>
          <div><span>Modality</span><b>{String(get(result, "modality") || "Lung CT Scan")}</b></div>
          <div><span>Device</span><b>{String(get(result, "device") || "-")}</b></div>
        </div>
      </div>
      <div className="panel">
        <p className="tiny">These are uncalibrated model scores, not probabilities of having or not having cancer.</p>
        <p className="tiny">Educational/research prototype. This result is not a clinical diagnosis. Discuss concerns and original scans with a qualified clinician.</p>
      </div>
      {result._persistenceError && <div className="err" role="alert">CT analyzed, but saving failed: {result._persistenceError}</div>}
      <Assistant assessment={result} contextType="ct" analysisId={result._analysisId} persistenceError={result._persistenceError} />
    </div>
  );
}

export default function CT({ onDone, guard }) {
  const [file, setFile] = useState(null), [busy, setBusy] = useState(false), [err, setErr] = useState(""), [result, setResult] = useState(null), [drag, setDrag] = useState(false);
  const ref = useRef();

  const pick = (selected) => {
    if (!selected) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(selected.type)) return setErr("Please upload a JPG, PNG, or WEBP CT image.");
    setErr(""); setResult(null); setFile(selected);
  };

  const run = async () => {
    if (!file) return setErr("Please select a CT image first.");
    setBusy(true); setErr("");
    try {
      const prediction = await predictCT(file);
      let analysisId = null, persistenceError = "";
      try {
        analysisId = await onDone({ type: "ct", result: prediction, imageFile: file });
      } catch (error) {
        persistenceError = error.message || "Could not save this CT analysis.";
      }
      setResult({ ...prediction, _analysisId: analysisId, _persistenceError: persistenceError });
    } catch (error) { setErr(error.message); }
    setBusy(false);
  };

  return (
    <div className="stack fade">
      <div className="panel intro-panel"><small>Before you upload</small><h3>What is a lung CT scan?</h3><p>A lung CT scan uses X-rays and computer processing to create cross-sectional images of the lungs and chest. Clinicians use CT imaging to examine structures and findings that may need further evaluation. This prototype classifies an uploaded image as <b>benign</b>, <b>malignant</b>, or <b>normal</b> using an experimental ResNet18 model.</p><p className="tiny">The result is not a diagnosis and the model may be wrong, especially for poor-quality images or images that are not lung CT scans. A radiologist or clinician must interpret the original scan and clinical context.</p></div>
      <div className={"drop" + (drag ? " over" : "")} onClick={() => ref.current.click()}
        onDragOver={(event) => { event.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(event) => { event.preventDefault(); setDrag(false); pick(event.dataTransfer.files[0]); }}>
        <Icon n="scan" size={34} /><h3>Upload a lung CT scan</h3><p className="muted">Drag & drop or click - JPG, PNG, WEBP</p>
        <input ref={ref} type="file" hidden accept=".jpg,.jpeg,.png,.webp" onChange={(event) => pick(event.target.files[0])} />
      </div>
      {file && <div className="panel filecard"><img src={URL.createObjectURL(file)} alt="CT scan preview" />
        <div><b>{file.name}</b><p className="muted">{(file.size / 1024).toFixed(0)} KB</p>
          <button className="btn" disabled={busy} onClick={() => guard(run)}>{busy ? "Analyzing CT scan..." : "Analyze CT"}</button>{" "}
          <button className="btn ghost" disabled={busy} onClick={() => { setFile(null); setResult(null); }}>Remove</button></div></div>}
      {!file && <button className="btn" disabled={busy} onClick={() => guard(run)}>{busy ? "Analyzing CT scan..." : "Analyze CT"}</button>}
      {err && <div className="err" role="alert">{err}</div>}
      {result && <CTResult result={result} />}
    </div>
  );
}