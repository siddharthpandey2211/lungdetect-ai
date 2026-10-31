import { useState } from "react";
import { FIELDS, buildPayload, predictRisk, get, num } from "../services/api";
import Assistant from "./Assistant";
import { Ring, Bar, Seg, Toggle } from "./ui";

export function RiskResult({ r, patient, back }) {
  if (r.available === false) {
    return (
      <div className="stack fade">
        <div className="panel hero-result">
          <small>Risk Assessment Result</small>
          <h2>No data available</h2>
          <p className="muted">{String(r.message)}</p>
          <button className="btn" onClick={back}>New assessment</button>
        </div>
      </div>
    );
  }
  const percent = (...keys) => {
    const raw = get(r, ...keys);
    return raw === undefined || raw === null || raw === "" ? null : num(raw);
  };
  const p = percent("cancer probability (%)", "cancer probability");
  const c = percent("confidence (%)", "confidence");
  const prediction = String(get(r, "prediction") ?? "-").toUpperCase();
  const displayPrediction = prediction === "YES"
    ? "Cancer"
    : prediction === "NO"
      ? "No cancer"
      : String(get(r, "prediction") ?? "-");
  return (
    <div className="stack fade">
      <div className="panel hero-result">
        <small>Risk Assessment Result</small>
        <h2>{displayPrediction}</h2>
        <p className="muted">Model: {String(get(r, "model") ?? "Gaussian Naive Bayes")}</p>
        <div className="rings"><Ring value={p} label="Cancer probability" color="var(--red)" /><Ring value={c} label="Confidence" /></div>
      </div>
      <div className="panel"><h3>Comparison</h3><Bar label="Cancer probability" value={p} color="var(--red)" /><Bar label="Confidence" value={c} color="var(--teal)" /></div>
      <Assistant assessment={r} patient={patient} analysisId={r._analysisId} persistenceError={r._persistenceError} />
      <div className="panel"><h3>Understanding your result</h3>
        <p>This experimental AI screening result is not a diagnosis. Persistent or worsening symptoms need medical review, whatever the score.</p>
        <button className="btn" onClick={back}>New assessment</button></div>
    </div>
  );
}

export default function Risk({ guard, onDone }) {
  const [f, setF] = useState(Object.fromEntries(FIELDS.map((x) => [x.key, x.def])));
  const [busy, setBusy] = useState(false), [err, setErr] = useState(""), [res, setRes] = useState(null);
  const set = (k, v) => setF({ ...f, [k]: v });
  const run = async () => {
    setBusy(true); setErr("");
    try {
      const patientAnswers = buildPayload(f);
      const result = await predictRisk(patientAnswers);
      let analysisId = null, persistenceError = "";
      try {
        analysisId = await onDone({ type: "risk", result, patientAnswers });
      } catch (error) {
        persistenceError = error.message || "Could not save this assessment.";
      }
      setRes({ ...result, _analysisId: analysisId, _persistenceError: persistenceError });
      if (persistenceError) setErr(`Prediction completed, but saving failed: ${persistenceError}`);
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  };
  if (res) return <RiskResult r={res} patient={buildPayload(f)} back={() => setRes(null)} />;
  return (
    <div className="stack fade">
      {["Basic Information", "Lifestyle", "Symptoms"].map((sec) => (
        <div className="panel" key={sec}><h3>{sec}</h3>
          <div className="grid">{FIELDS.filter((x) => x.section === sec).map((x) => (
            <div key={x.key}>
              {x.type === "seg" && <><label className="lbl">{x.label}</label><Seg opts={x.opts} value={f[x.key]} onChange={(v) => set(x.key, v)} /></>}
              {x.type === "num" && <label className="lbl">{x.label}<input type="number" value={f[x.key]} onChange={(e) => set(x.key, e.target.value)} /></label>}
              {x.type === "bool" && <Toggle label={x.label} on={f[x.key]} onChange={(v) => set(x.key, v)} />}
            </div>))}</div></div>
      ))}
      {err && <div className="err">{err}</div>}
      <button className="btn" disabled={busy} onClick={() => guard(run)}>{busy ? "Calculating risk..." : "Calculate Risk"}</button>
    </div>
  );
}
