// All backend calls live here. Backend URL unchanged.
const BASE = "http://127.0.0.1:5050";
const YES = 2, NO = 1;

// !! Keys/labels below must match your Patient schema. Copy the exact keys from your old App.jsx if different.
export const FIELDS = [
  { key: "gender", label: "Gender", section: "Basic Information", type: "seg", opts: [["M", "Male"], ["F", "Female"]], def: "M" },
  { key: "age", label: "Age", section: "Basic Information", type: "num", def: 50 },
  { key: "smoking", label: "Smoking", section: "Lifestyle", type: "bool", def: false },
  { key: "alcohol_consuming", label: "Alcohol consumption", section: "Lifestyle", type: "bool", def: false },
  { key: "peer_pressure", label: "Peer pressure", section: "Lifestyle", type: "bool", def: false },
  ...[["yellow_fingers", "Yellow fingers"], ["anxiety", "Anxiety"], ["chronic_disease", "Chronic disease"], ["fatigue", "Fatigue"],
    ["allergy", "Allergy"], ["wheezing", "Wheezing"], ["coughing", "Coughing"], ["shortness_of_breath", "Shortness of breath"],
    ["swallowing_difficulty", "Swallowing difficulty"], ["chest_pain", "Chest pain"]]
    .map(([key, label]) => ({ key, label, section: "Symptoms", type: "bool", def: false })),
];

export const buildPayload = (f) =>
  Object.fromEntries(FIELDS.map((x) => [x.key, x.type === "bool" ? (f[x.key] ? YES : NO) : x.type === "num" ? Number(f[x.key]) : f[x.key]]));

// Case/spacing-insensitive lookup so "Cancer Probability (%)" or cancer_probability both work
const n = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
export const get = (o, ...ks) => { for (const k of Object.keys(o || {})) if (ks.some((x) => n(x) === n(k))) return o[k]; };
export const num = (v) => { const x = parseFloat(String(v ?? "").replace("%", "")); return isNaN(x) ? 0 : x; };

async function post(path, body, form) {
  let r;
  try {
    r = await fetch(BASE + path, form ? { method: "POST", body } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch { throw new Error("Unable to connect to LungDetect AI backend. Please make sure the FastAPI server is running."); }
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || "Analysis failed. Please try again.");
  return data;
}
export const predictRisk = async (payload) => {
  const data = await post("/predict", payload);
  return data.prediction || data;
};
export const predictImage = (file) => { const fd = new FormData(); fd.append("file", file); return post("/predict-image", fd, true); };
export const chatWithAssistant = (payload) => post("/assistant/chat", payload);
