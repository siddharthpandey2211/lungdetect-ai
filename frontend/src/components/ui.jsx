const P = {
  logo: "M12 3v8M12 11c-2 0-6 1-6 6s2 4 4 3 2-4 2-9M12 11c2 0 6 1 6 6s-2 4-4 3-2-4-2-9",
  home: "M3 11l9-8 9 8v10h-6v-6H9v6H3z", activity: "M3 12h4l3-8 4 16 3-8h4",
  image: "M4 4h16v16H4zM4 15l5-5 5 5 3-3 3 3", clock: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 7v5l3 2",
  info: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 11v6M12 7.5v.5", user: "M12 12a4 4 0 100-8 4 4 0 000 8zM4 21a8 8 0 0116 0",
  gear: "M12 15a3 3 0 100-6 3 3 0 000 6zM19 12h2M3 12h2M12 3v2M12 19v2", menu: "M4 6h16M4 12h16M4 18h16",
  upload: "M12 16V4M7 9l5-5 5 5M4 20h16", scan: "M4 7V5a1 1 0 0 1 1-1h2M17 4h2a1 1 0 0 1 1 1v2M20 17v2a1 1 0 0 1-1 1h-2M7 20H5a1 1 0 0 1-1-1v-2M8 12h8M12 8v8", alert: "M12 3l10 18H2zM12 10v5M12 18v.5",
};
export const Icon = ({ n, size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={P[n]} /></svg>
);

// Animated percentage ring (value comes straight from backend)
export const Ring = ({ value, label, color = "var(--teal)" }) => {
  const v = Math.max(0, Math.min(100, value)), r = 44, c = 2 * Math.PI * r;
  return (
    <div className="ring">
      <svg viewBox="0 0 100 100"><circle cx="50" cy="50" r={r} className="ring-bg" />
        <circle cx="50" cy="50" r={r} className="ring-fg" stroke={color} strokeDasharray={c} strokeDashoffset={c * (1 - v / 100)} transform="rotate(-90 50 50)" /></svg>
      <b>{v.toFixed(1)}%</b><span>{label}</span>
    </div>
  );
};

export const Bar = ({ label, value, color = "var(--blue)" }) => (
  <div className="bar"><div className="bar-top"><span>{label}</span><b>{value.toFixed(1)}%</b></div>
    <div className="bar-track"><div className="bar-fill" style={{ width: Math.min(100, value) + "%", background: color }} /></div></div>
);

export const Seg = ({ opts, value, onChange }) => (
  <div className="seg">{opts.map(([v, l]) => <button type="button" key={v} className={v === value ? "on" : ""} onClick={() => onChange(v)}>{l}</button>)}</div>
);

export const Toggle = ({ label, on, onChange }) => (
  <button type="button" className={"tog" + (on ? " on" : "")} onClick={() => onChange(!on)}><span>{label}</span><i /></button>
);
