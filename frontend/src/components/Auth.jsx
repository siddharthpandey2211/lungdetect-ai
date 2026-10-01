import { useState } from "react";
import { Icon } from "./ui";
import { supabase, supabaseConfigured } from "../services/supabase";

export default function Auth({ onLogin, onCancel, gated }) {
  const [mode, setMode] = useState("login");
  const [f, setF] = useState({ name: "", email: "", password: "", confirm: "" });
  const [err, setErr] = useState(supabaseConfigured ? "" : "Supabase is not configured. Add its URL and anon key to frontend/.env.local, then restart Vite.");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    setNotice("");
    if (!supabase || !supabaseConfigured) return setErr("Supabase is not configured. Add its URL and anon key to frontend/.env.local, then restart Vite.");
    if (!f.email.includes("@") || f.password.length < 8) return setErr("Enter a valid email and a password with at least 8 characters.");
    if (mode === "register" && (!f.name.trim() || f.password !== f.confirm)) return setErr("Enter your name and make sure the passwords match.");

    setBusy(true);
    try {
      if (mode === "login") {
        const { data, error } = await supabase.auth.signInWithPassword({ email: f.email.trim(), password: f.password });
        if (error) throw error;
        onLogin(data.user);
      } else {
        const { data, error } = await supabase.auth.signUp({
          email: f.email.trim(),
          password: f.password,
          options: { data: { full_name: f.name.trim() } },
        });
        if (error) throw error;
        if (data.session && data.user) onLogin(data.user);
        else setNotice("Account created. Check your email to confirm the account, then sign in.");
      }
    } catch (authError) {
      setErr(mode === "login" ? "Could not sign in. Check your email and password." : authError.message || "Could not create the account.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <div className="brand big"><Icon n="logo" size={28} /> LungDetect AI</div>
      <h2>{gated ? "Sign in required" : mode === "login" ? "Welcome back" : "Create your account"}</h2>
      {gated && <p className="muted">Sign in to save assessments, pathology images, and chat history to your private account.</p>}
      <form onSubmit={submit}>
        {mode === "register" && <label>Name<input autoComplete="name" value={f.name} onChange={set("name")} /></label>}
        <label>Email<input type="email" autoComplete="email" value={f.email} onChange={set("email")} /></label>
        <label>Password<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={f.password} onChange={set("password")} /></label>
        {mode === "register" && <label>Confirm password<input type="password" autoComplete="new-password" value={f.confirm} onChange={set("confirm")} /></label>}
        {err && <div className="err">{err}</div>}
        {notice && <div className="notice">{notice}</div>}
        <button className="btn" disabled={busy || !supabaseConfigured}>{busy ? "Please wait..." : mode === "login" ? "Login" : "Create Account"}</button>
        <button type="button" className="btn ghost" disabled={busy || !supabaseConfigured} onClick={() => { setMode(mode === "login" ? "register" : "login"); setErr(""); setNotice(""); }}>
          {mode === "login" ? "Create Account" : "Back to Login"}</button>
        <button type="button" className="btn ghost" onClick={onCancel}>Back</button>
      </form>
      <p className="tiny">Supabase authentication secures your saved records. This educational prototype is not a medical diagnostic tool.</p>
    </div>
  );
}
