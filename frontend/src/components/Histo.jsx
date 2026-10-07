import { useState, useRef } from "react";
import { predictImage, get, num } from "../services/api";
import Assistant from "./Assistant";
import { Icon } from "./ui";

export function ImageResult({ r }) {
  const cls = String(get(r, "class", "storage class", "model prediction") ?? "");
  const unknown = cls === "unknown" || get(r, "manual verification required") === true;
  const status = unknown ? "Unknown / Needs Manual Verification" : cls === "lung_aca" || cls === "lung_scc" ? "Cancer" : "No Cancer";
  const definition = String(get(r, "definition") ?? (
    unknown
      ? get(r, "unknown reason") ?? "The image needs manual verification."
      : cls === "lung_aca"
        ? "Adenocarcinoma is defined as a malignant cancer that starts in gland cells."
        : cls === "lung_scc"
          ? "Squamous cell carcinoma is a malignant tumor arising from squamous cells."
          : "The image is classified as normal lung tissue, with no cancer pattern detected."
  ));
  return (
    <div className="stack fade">
      {unknown && <div className="panel warn"><h3>Unknown image - manual verification required</h3><p>{definition}</p></div>}
      <div className="panel"><small>Histopathology result</small><h2>{status}</h2>
        <p className="muted">Model: {String(get(r, "model") ?? "ResNet18")} - not a medical diagnosis</p>
        <div className="meta"><div><span>Confidence</span><b>{num(get(r, "confidence (%)", "confidence")).toFixed(1)}%</b></div>
          <div><span>Classification</span><b>{status}</b></div></div>
        <p>{definition}</p>
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
      <div className="panel intro-panel"><small>Before you upload</small><h3>What is histopathology?</h3><p>Histopathology examines tissue under a microscope. In lung cancer care, a pathologist studies a biopsy to look for cellular patterns and helps establish a diagnosis. This prototype classifies a tissue image into experimental pattern classes such as normal tissue, adenocarcinoma, or squamous cell carcinoma.</p><p className="tiny">A model score is not a pathology diagnosis. Only a qualified pathologist can interpret the original tissue, staining, specimen quality, and clinical context.</p></div>
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
