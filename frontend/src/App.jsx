import { useEffect, useRef, useState } from "react";
import { analyzeJob, analyzeChat, followUp, SAMPLE_JOB, FOLLOW_UPS, USE_MOCK } from "./services/api.js";

const LEVEL = { "Lower risk": "low", "Moderate risk": "mid", "Higher risk": "high" };

function RiskBar({ score, category }) {
  const level = LEVEL[category];
  return (
    <div>
      <div className="riskhead">
        <span className={`chip ${level}`}>{category}</span>
        <strong>{score} / 100</strong>
      </div>
      <div className="track">
        <div className={`fill ${level}`} style={{ width: `${score}%` }} />
      </div>
      <p className="hint">Provisional score. It is not a probability.</p>
    </div>
  );
}

function Timeline({ items }) {
  const w = 320, h = 100, pad = 14;
  const step = items.length > 1 ? (w - pad * 2) / (items.length - 1) : 0;
  const pts = items.map((m, i) => [pad + i * step, h - pad - (m.cumulative / 100) * (h - pad * 2)]);
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="timeline" role="img" aria-label="Risk over the conversation">
        <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} className="axis" />
        <polyline points={pts.map((p) => p.join(",")).join(" ")} className="line" fill="none" />
        {pts.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r={items[i].sign ? 5 : 3} className={items[i].sign ? "dot flag" : "dot"}>
            <title>{`${items[i].time}: ${items[i].sign || "no warning sign"}`}</title>
          </circle>
        ))}
      </svg>
      <ul className="flags">
        {items.filter((m) => m.sign).map((m, i) => (
          <li key={i}>
            <span className="time">{m.time}</span>
            <strong>{m.sign}</strong>
            <span className="quote">"{m.text}"</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ReportPanel({ report }) {
  if (!report) {
    return (
      <aside className="panel empty">
        <h2>Report</h2>
        <p>Run an analysis to see the risk score, warning signs, evidence, and timeline here.</p>
      </aside>
    );
  }
  return (
    <aside className="panel">
      <h2>{report.type === "chat" ? "Conversation report" : "Job posting report"}</h2>
      <RiskBar score={report.risk_score} category={report.risk_category} />
      <p className="pred"><span>Classifier:</span> {report.classifier_prediction}</p>

      <h3>Warning signs</h3>
      {report.warning_signs.length ? (
        <ul>{report.warning_signs.map((s) => <li key={s}>{s}</li>)}</ul>
      ) : (
        <p className="muted">None detected.</p>
      )}

      {report.timeline && (<><h3>Risk timeline</h3><Timeline items={report.timeline} /></>)}

      {report.supporting_evidence.length > 0 && (
        <>
          <h3>Supporting evidence</h3>
          {report.supporting_evidence.map((e) => (
            <details key={e.id} className="evidence">
              <summary>
                <span>{e.id}</span>
                <span className={`chip small ${e.label === "Fraudulent" ? "high" : "low"}`}>{e.label}</span>
                <span className="sim">sim {e.similarity}</span>
              </summary>
              <p>"{e.snippet}"</p>
              <p className="muted">Source: {e.source}</p>
            </details>
          ))}
        </>
      )}

      <h3>Uncertainty notes</h3>
      <ul className="notes">{report.uncertainty_notes.map((n) => <li key={n}>{n}</li>)}</ul>
    </aside>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [report, setReport] = useState(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);

  const add = (m) => setMessages((prev) => [...prev, m]);

  async function runJob(text) {
    add({ role: "user", text });
    setBusy(true);
    try {
      const r = await analyzeJob(text);
      setReport(r);
      add({ role: "assistant", text: r.explanation, report: r });
    } catch {
      add({ role: "assistant", text: "Something went wrong while analyzing. Please try again.", error: true });
    }
    setBusy(false);
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    if (text.length < 120) {
      add({ role: "user", text });
      add({ role: "assistant", text: "That looks short. Please paste the full job posting, or upload an exported chat as a .txt file, so I can check it." });
      return;
    }
    runJob(text);
  }

  async function upload(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file || busy) return;
    add({ role: "user", text: `Uploaded file: ${file.name}` });
    if (!file.name.toLowerCase().endsWith(".txt")) {
      add({ role: "assistant", text: "That file type is not supported yet. Please export the chat as plain text (.txt).", error: true });
      return;
    }
    setBusy(true);
    try {
      const r = await analyzeChat(file);
      setReport(r);
      add({ role: "assistant", text: r.explanation, report: r });
    } catch {
      add({ role: "assistant", text: "Something went wrong while reading the chat. Please try again.", error: true });
    }
    setBusy(false);
  }

  function ask(q) {
    if (busy || !report) return;
    add({ role: "user", text: q });
    add({ role: "assistant", text: followUp(q, report) });
  }

  const lastAssistantWithReport = [...messages].reverse().find((m) => m.report);

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Job Scam Assistant</h1>
          <p>Check a job posting or a recruiter chat for warning signs</p>
        </div>
      </header>

      <main className="layout">
        <section className="chat" aria-label="Chat">
          <div className="messages">
            {messages.length === 0 && (
              <div className="intro">
                <h2>Paste a job posting or upload a chat</h2>
                <p>I look for warning signs and show the evidence. I can say a posting is potentially fraudulent, but I cannot prove it.</p>
                <div className="samples">
                  <button onClick={() => setInput(SAMPLE_JOB)}>Use a sample job posting</button>
                  <button onClick={() => { add({ role: "user", text: "Uploaded file: sample_chat.txt" }); setBusy(true); analyzeChat(null).then((r) => { setReport(r); add({ role: "assistant", text: r.explanation, report: r }); setBusy(false); }); }}>Use a sample chat</button>
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`bubble ${m.role} ${m.error ? "error" : ""}`}>
                <p>{m.text}</p>
                {m.report && (
                  <p className="inline-risk">
                    <span className={`chip ${LEVEL[m.report.risk_category]}`}>{m.report.risk_category}</span>
                    <span>Score {m.report.risk_score}. Details are in the report panel.</span>
                  </p>
                )}
                {m === lastAssistantWithReport && !busy && (
                  <div className="followups">
                    {FOLLOW_UPS[m.report.type].map((q) => (
                      <button key={q} onClick={() => ask(q)}>{q}</button>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {busy && <div className="bubble assistant typing">Analyzing<span>.</span><span>.</span><span>.</span></div>}
            <div ref={endRef} />
          </div>

          <div className="composer">
            <input ref={fileRef} type="file" accept=".txt,text/plain" hidden onChange={upload} />
            <button className="ghost" onClick={() => fileRef.current?.click()} disabled={busy} aria-label="Upload chat file">Upload chat</button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Paste a job posting here..."
              rows={2}
            />
            <button className="primary" onClick={send} disabled={busy || !input.trim()}>Analyze</button>
          </div>
        </section>

        <ReportPanel report={report} />
      </main>
    </div>
  );
}

