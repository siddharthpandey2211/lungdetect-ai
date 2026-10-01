import { useEffect, useRef, useState } from "react";
import { chatWithAssistant } from "../services/api";
import { saveChatExchange } from "../services/persistence";

const RISK_QUESTIONS = [
  "Suggest general food ideas, daily routines, things to limit, and what to ask my doctor.",
  "What does this screening result mean?",
  "How should I discuss medication options with my doctor?",
];

const HISTOPATHOLOGY_QUESTIONS = [
  "Explain these class scores in plain language.",
  "Why might this image need manual verification?",
  "What should I ask my pathologist or doctor about this result?",
];

export default function Assistant({ assessment, patient, contextType = "risk", analysisId, persistenceError = "" }) {
  const isHistopathology = contextType === "histopathology";
  const quickQuestions = isHistopathology ? HISTOPATHOLOGY_QUESTIONS : RISK_QUESTIONS;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  const send = async (text) => {
    const content = text.trim();
    if (!content || busy) return;
    if (!analysisId) {
      setError(persistenceError || "Save the analysis to your account before starting a chat.");
      return;
    }

    const previousMessages = messages;
    setMessages([...previousMessages, { role: "user", content }]);
    setInput("");
    setError("");
    setBusy(true);

    try {
      const result = await chatWithAssistant({
        message: content,
        history: previousMessages.slice(-12),
        context_type: contextType,
        assessment,
        patient,
      });
      setMessages((current) => [...current, { role: "assistant", content: result.reply }]);
      try {
        await saveChatExchange(analysisId, content, result.reply);
      } catch (saveError) {
        setError(`Reply received, but chat history could not be saved: ${saveError.message}`);
      }
    } catch (requestError) {
      setError(requestError.message?.includes("OPENAI_API_KEY")
        ? "AI chat needs an OpenAI API key. Set OPENAI_API_KEY in the backend environment, then restart the backend."
        : requestError.message || "The AI assistant could not respond.");
    } finally {
      setBusy(false);
    }
  };

  const submit = (event) => {
    event.preventDefault();
    send(input);
  };

  return (
    <section className="panel assistant" aria-label={`${isHistopathology ? "Histopathology" : "Risk assessment"} chat`}>
      <div className="assistant-head">
        <h3>{isHistopathology ? "Ask about this image result" : "Ask LungDetect AI"}</h3>
        <p className="muted">{isHistopathology
          ? "Ask about the classifier output, manual verification, or what to discuss with a pathologist."
          : "Ask about this screening result, daily routines, food, medication questions, or what to discuss with a clinician."}</p>
      </div>
      <div className="chat-log" aria-live="polite">
        {!messages.length && <p className="tiny">{isHistopathology
          ? "The original image is stored in your private account. Chat uses only model-result details, not the image."
          : "Your assessment and chat are stored in your private account. Local Ollama handles chat by default; OpenAI mode sends assessment details to OpenAI."}</p>}
        {messages.map((message, index) => (
          <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
            {message.content.split("\n").filter(Boolean).map((paragraph, paragraphIndex) => <p key={paragraphIndex}>{paragraph}</p>)}
          </div>
        ))}
        {busy && <div className="chat-message assistant" role="status">Thinking...</div>}
        <div ref={bottomRef} />
      </div>
      {!messages.length && <div className="assistant-questions">
        {quickQuestions.map((question) => <button className="btn ghost" key={question} disabled={busy} onClick={() => send(question)}>{question}</button>)}
      </div>}
      {error && <div className="err" role="alert">{error}</div>}
      <form className="chat-compose" onSubmit={submit}>
        <textarea aria-label="Message LungDetect AI" maxLength={2000} placeholder="Ask a question..." value={input} onChange={(event) => setInput(event.target.value)} disabled={!analysisId || busy} />
        <button className="btn" type="submit" disabled={!analysisId || busy || !input.trim()}>{busy ? "Sending..." : "Send"}</button>
      </form>
      <p className="tiny chat-note">Educational information only. The assistant cannot diagnose or recommend starting, stopping, or changing medication.</p>
    </section>
  );
}