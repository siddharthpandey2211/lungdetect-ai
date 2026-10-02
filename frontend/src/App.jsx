import { useEffect, useState } from "react";
import { Icon } from "./components/ui";
import Auth from "./components/Auth";
import Risk from "./components/Risk";
import Histo from "./components/Histo";
import MRI from "./components/MRI";
import CT from "./components/CT";
import { Dashboard, History, About } from "./components/Pages";
import { deleteMyData, loadAnalyses, saveAnalysis } from "./services/persistence";
import { supabase, supabaseConfigured } from "./services/supabase";

const NAV = [["dash", "Dashboard", "home"], ["risk", "Risk Assessment", "activity"], ["histo", "Histopathology", "image"], ["mri", "MRI Analysis", "scan"], ["ct", "CT Analysis", "scan"], ["history", "History", "clock"], ["about", "About", "info"]];

function describeHistoryError(error) {
  const message = String(error?.message || "");
  if (error?.code === "PGRST205" || message.includes("public.analyses") && message.includes("schema cache")) {
    return "Supabase tables are missing or not refreshed. In Supabase SQL Editor, run the project's supabase/schema.sql file, then refresh this page.";
  }
  return `Could not load saved history: ${message}`;
}

export default function App() {
  const [page, setPage] = useState("dash"), [open, setOpen] = useState(false);
  const [user, setUser] = useState(null);
  const [history, setHistory] = useState([]);
  const [pending, setPending] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [appError, setAppError] = useState("");
  const go = (p) => { setPage(p); setOpen(false); };

  useEffect(() => {
    if (!supabaseConfigured || !supabase) {
      setAuthReady(true);
      return;
    }

    let active = true;
    supabase.auth.getSession().then(({ data, error }) => {
      if (!active) return;
      if (error) setAppError("Could not restore your Supabase session. Please sign in again.");
      setUser(data.session?.user ?? null);
      setAuthReady(true);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      setAuthReady(true);
    });

    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (!user) {
      setHistory([]);
      setHistoryLoading(false);
      return;
    }

    let active = true;
    setHistoryLoading(true);
    loadAnalyses()
      .then((rows) => { if (active) setHistory(rows); })
      .catch((error) => { if (active) setAppError(describeHistoryError(error)); })
      .finally(() => { if (active) setHistoryLoading(false); });

    return () => { active = false; };
  }, [user?.id]);

  const addHistory = async (entry) => {
    const saved = await saveAnalysis(entry);
    const row = {
      id: saved.id,
      type: entry.type,
      result: entry.type === "image" || entry.type === "mri" || entry.type === "ct"
        ? entry.result.Class || entry.result.Prediction
        : entry.result.Prediction,
      confidence: Number(entry.result["Confidence (%)"] || 0),
      unknown: entry.result.Class === "unknown",
      date: saved.date,
    };
    setHistory((current) => [row, ...current]);
    return saved.id;
  };

  const guard = (run) => {
    if (!supabaseConfigured) {
      setAppError("Connect Supabase before analyzing so health data can be saved to a protected account.");
      go("login");
    } else if (!user) {
      setPending(() => run);
    } else {
      run();
    }
  };

  const login = (nextUser) => {
    setUser(nextUser);
    setAppError("");
    const run = pending;
    setPending(null);
    if (!run) go("dash");
    if (run) run();
  };

  const logout = async () => {
    if (!supabase) return;
    const { error } = await supabase.auth.signOut();
    if (error) setAppError("Could not sign out. Please try again.");
    else {
      setUser(null);
      setHistory([]);
      go("dash");
    }
  };

  const reset = async () => {
    if (!user || !confirm("Delete your Supabase analyses, chat messages, and pathology images? The backend's separate local review archive is not affected.")) return;
    setAppError("");
    try {
      await deleteMyData();
      setHistory([]);
    } catch (error) {
      setAppError(`Could not delete saved data: ${error.message}`);
    }
  };

  if (!authReady) return <main><p className="muted">Checking secure account...</p></main>;

  return (
    <div className="app">
      <header className="topbar"><button onClick={() => setOpen(!open)}><Icon n="menu" /></button><b>LungDetect AI</b></header>
      <aside className={"side" + (open ? " open" : "")}>
        <div className="brand"><Icon n="logo" size={24} /> LungDetect AI</div>
        <nav>{NAV.map(([k, l, i]) => <button key={k} className={page === k ? "on" : ""} onClick={() => go(k)}><Icon n={i} size={18} />{l}</button>)}</nav>
        <div className="sidebot">
          {user ? <button onClick={logout}><Icon n="user" size={18} />{user.user_metadata?.full_name || user.email} - Logout</button> : <button onClick={() => go("login")}><Icon n="user" size={18} />Login</button>}
        </div>
      </aside>
      <main key={page}>
        {appError && <div className="err" role="alert">{appError}</div>}
        {historyLoading && <p className="tiny muted">Loading your saved history...</p>}
        {page === "dash" && <Dashboard history={history} go={go} />}
        {page === "risk" && <><h1>Risk Assessment</h1><Risk guard={guard} onDone={addHistory} /></>}
        {page === "histo" && <><h1>Histopathology Analysis</h1><Histo guard={guard} onDone={addHistory} /></>}
        {page === "mri" && <><h1>MRI Analysis</h1><MRI guard={guard} onDone={addHistory} /></>}
        {page === "ct" && <><h1>CT Analysis</h1><CT guard={guard} onDone={addHistory} /></>}
        {page === "history" && <History history={history} />}
        {page === "about" && <About />}
        {page === "login" && <Auth gated={Boolean(pending)} onLogin={login} onCancel={() => { setPending(null); go("dash"); }} />}
      </main>
      {pending && <div className="modal"><Auth gated onLogin={login} onCancel={() => setPending(null)} /></div>}
    </div>
  );
}
