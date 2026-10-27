
import { useEffect, useRef, useState } from "react";
import {
  analyzeJob,
  analyzeChat,
  followUp,
  SAMPLE_JOB,
  FOLLOW_UPS,
} from "./services/api.js";

const LEVEL = {
  "Lower risk": "low",
  "Moderate risk": "mid",
  "Higher risk": "high",
};

const CHECKS = [
  "Upfront payment or registration fees",
  "Unrealistic salary or no-interview offers",
  "Requests for ID, bank, or password details",
  "Urgent or pressuring language",
  "Moving the chat to another app",
  "Mismatched company and contact details",
];

const STEPS = [
  [
    "Paste or upload",
    "Add a job posting, or an exported WhatsApp or Telegram chat as a .txt file.",
  ],
  [
    "Warning signs are checked",
    "The text is compared with known warning signs and similar past cases.",
  ],
  [
    "Review the evidence",
    "See the risk score, the evidence behind it, and what to verify next.",
  ],
];

// Phrases highlighted in the submitted text.
// These highlights are illustrative and do not necessarily mean a phrase is fraudulent.
const HL =
  /(refundable registration fee|registration fee|earn rs [\d,]+ per week|bank details|aadhaar|guaranteed placement|guaranteed|no interview|urgent|telegram|deposit|\bfee\b)/gi;

function Logo({ size = 38 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#0e7490" />
          <stop offset="1" stopColor="#4f46e5" />
        </linearGradient>
      </defs>

      <path
        d="M16 2l11 4v8c0 7-4.6 12.4-11 15C9.6 26.4 5 21 5 14V6l11-4z"
        fill="url(#lg)"
      />

      <circle
        cx="15"
        cy="14"
        r="5"
        fill="none"
        stroke="#fff"
        strokeWidth="2"
      />

      <line
        x1="18.8"
        y1="17.8"
        x2="22"
        y2="21"
        stroke="#fff"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function Highlighted({ text }) {
  return (
    <>
      {text.split(HL).map((part, i) =>
        i % 2 ? (
          <mark key={i} title="Possible warning sign">
            {part}
          </mark>
        ) : (
          part
        )
      )}
    </>
  );
}

function Gauge({ score, category }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const level = LEVEL[category];

  return (
    <div className="gauge">
      <svg
        viewBox="0 0 120 120"
        role="img"
        aria-label={`Risk score ${score} out of 100`}
      >
        <circle cx="60" cy="60" r={r} className="g-track" />

        <circle
          cx="60"
          cy="60"
          r={r}
          className={`g-fill ${level}`}
          strokeDasharray={`${(score / 100) * c} ${c}`}
          transform="rotate(-90 60 60)"
        />

        <text
          x="60"
          y="62"
          textAnchor="middle"
          className="g-score"
        >
          {score}
        </text>

        <text
          x="60"
          y="80"
          textAnchor="middle"
          className="g-of"
        >
          out of 100
        </text>
      </svg>

      <div>
        <span className={`chip big ${level}`}>{category}</span>

        <p className="hint">
          Provisional score. It is not a probability.
        </p>
      </div>
    </div>
  );
}

function Timeline({ items }) {
  const w = 360;
  const h = 130;
  const px = 14;
  const py = 12;

  const x = (i) =>
    px +
    (items.length > 1
      ? (i * (w - px * 2)) / (items.length - 1)
      : 0);

  const y = (v) => h - py - (v / 100) * (h - py * 2);

  const pts = items.map((m, i) => [x(i), y(m.cumulative)]);
  const line = pts.map((p) => p.join(",")).join(" ");
  const area = `${px},${y(0)} ${line} ${x(items.length - 1)},${y(0)}`;

  return (
    <div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="timeline"
        role="img"
        aria-label="Risk over the conversation"
      >
        <rect
          x="0"
          y={y(100)}
          width={w}
          height={y(60) - y(100)}
          className="band high"
        />

        <rect
          x="0"
          y={y(60)}
          width={w}
          height={y(30) - y(60)}
          className="band mid"
        />

        <rect
          x="0"
          y={y(30)}
          width={w}
          height={y(0) - y(30)}
          className="band low"
        />

        <polygon points={area} className="area" />

        <polyline
          points={line}
          className="line"
          fill="none"
        />

        {pts.map((p, i) => (
          <circle
            key={i}
            cx={p[0]}
            cy={p[1]}
            r={items[i].sign ? 5 : 3}
            className={items[i].sign ? "dot flag" : "dot"}
          >
            <title>
              {`${items[i].time}: ${
                items[i].sign || "no warning sign"
              }`}
            </title>
          </circle>
        ))}

        <text x={px} y={h - 1} className="tl-label">
          {items[0].time}
        </text>

        <text
          x={w - px}
          y={h - 1}
          textAnchor="end"
          className="tl-label"
        >
          {items[items.length - 1].time}
        </text>
      </svg>

      <ul className="flags">
        {items
          .filter((m) => m.sign)
          .map((m, i) => (
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
      <aside className="panel">
        <h2>What gets checked</h2>

        <ul className="checks">
          {CHECKS.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>

        <p className="disclaimer">
          Results are warning signs, not proof that a company is fraudulent.
        </p>
      </aside>
    );
  }

  const evidence = report.supporting_evidence || [];
  const uncertaintyNotes = report.uncertainty_notes || [];
  const warningSigns = report.warning_signs || [];

  return (
    <aside className="panel">
      <h2>
        {report.type === "chat"
          ? "Conversation report"
          : "Job posting report"}
      </h2>

      <Gauge
        score={report.risk_score}
        category={report.risk_category}
      />

      {report.type !== "chat" &&
        report.classifier_risk_score !== undefined && (
          <p className="muted">
            Classifier score: {report.classifier_risk_score}/100
            {" · "}
            Warning adjustment: +{report.warning_adjustment ?? 0}
          </p>
        )}

      <p className="pred">
        <span>Classifier:</span> {report.classifier_prediction}
      </p>

      <h3>Warning signs</h3>

      {warningSigns.length > 0 ? (
        <ul className="signs">
          {warningSigns.map((s, index) => {
            const isRuleFinding = typeof s === "object" && s !== null;
            const label = isRuleFinding ? s.label : s;
            const matchedText = isRuleFinding ? s.matched_text : null;

            return (
              <li key={`${label}-${matchedText || index}`}>
                <strong>{label}</strong>

                {matchedText && (
                  <span className="muted">
                    {" "}— Matched text: "{matchedText}"
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="muted">None detected.</p>
      )}

      {report.source_text && (
        <>
          <h3>Your text, with warning phrases marked</h3>

          <div className="source">
            <Highlighted text={report.source_text} />
          </div>
        </>
      )}

      {report.timeline?.length > 0 && (
        <>
          <h3>Risk timeline</h3>
          <Timeline items={report.timeline} />
        </>
      )}

      {evidence.length > 0 && (
        <>
          <h3>Related Scam-Prevention Guidance</h3>

          <p className="muted">
            These sources describe common recruitment scams and ways to
            identify suspicious job offers. They provide general guidance
            related to this analysis; they do not independently verify
            whether this specific job posting is legitimate or fraudulent.
            Similarity scores indicate text relevance, not the probability
            that a scam has occurred.
          </p>

          {evidence.map((e, index) => (
            <details
              key={e.id || `${e.title || "evidence"}-${index}`}
              className="evidence"
            >
              <summary>
                <span className="eid">{e.title || e.id || "Source"}</span>

                <span className="sim" title="Text similarity">
                  <span className="simbar">
                    <span
                      style={{
                        width: `${
                          Math.max(
                            0,
                            Math.min(1, Number(e.similarity) || 0)
                          ) * 100
                        }%`,
                      }}
                    />
                  </span>

                  {typeof e.similarity === "number"
                    ? e.similarity.toFixed(2)
                    : "N/A"}
                </span>
              </summary>

              <p>
                {e.excerpt || "No evidence excerpt is available."}
              </p>

              <p className="muted">
                Source: {e.publisher || "Unknown publisher"}
              </p>

              {e.warning_sign_tags?.length > 0 && (
                <p className="muted">
                  Related warning signs: {e.warning_sign_tags.join(", ")}
                </p>
              )}

              {e.url && (
                <p>
                  <a
                    href={e.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Read original source ↗
                  </a>
                </p>
              )}
            </details>
          ))}
        </>
      )}

      <h3>Limitations and uncertainty</h3>

      {uncertaintyNotes.length > 0 ? (
        <ul className="notes">
          {uncertaintyNotes.map((n, i) => (
            <li key={`${i}-${n}`}>{n}</li>
          ))}
        </ul>
      ) : (
        <p className="muted">
          No additional uncertainty notes were returned by the analysis.
        </p>
      )}
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

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const add = (m) =>
    setMessages((prev) => [...prev, m]);

  const reset = () => {
    setMessages([]);
    setReport(null);
    setInput("");
  };

  async function runJob(text) {
    add({ role: "user", text });
    setBusy(true);

    try {
      const r = {
        ...(await analyzeJob(text)),
        source_text: text,
      };

      setReport(r);

      add({
        role: "assistant",
        text:
          r.analysis?.explanation ||
          r.explanation ||
          "Analysis completed. See the report for details.",
        report: r,
      });
    } catch (error) {
      add({
        role: "assistant",
        text:
          error instanceof Error
            ? error.message
            : "Something went wrong while analyzing. Please try again.",
        error: true,
      });
    } finally {
      setBusy(false);
    }
  }

  async function runChat(file, label) {
    add({ role: "user", text: `Uploaded file: ${label}` });
    setBusy(true);

    try {
      const r = await analyzeChat(file);

      setReport(r);

      add({
        role: "assistant",
        text: r.explanation,
        report: r,
      });
    } catch (error) {
      add({
        role: "assistant",
        text:
          error instanceof Error
            ? error.message
            : "Something went wrong while reading the chat. Please try again.",
        error: true,
      });
    } finally {
      setBusy(false);
    }
  }

  function send() {
    const text = input.trim();

    if (!text || busy) return;

    setInput("");

    if (text.length < 30) {
      add({ role: "user", text });

      add({
        role: "assistant",
        text: "That looks short. Please paste the full job posting, or upload an exported chat as a .txt file, so I can check it.",
      });

      return;
    }

    runJob(text);
  }

  function upload(e) {
    const file = e.target.files?.[0];
    e.target.value = "";

    if (!file || busy) return;

    if (!file.name.toLowerCase().endsWith(".txt")) {
      add({ role: "user", text: `Uploaded file: ${file.name}` });

      add({
        role: "assistant",
        text: "That file type is not supported yet. Please export the chat as plain text (.txt).",
        error: true,
      });

      return;
    }

    runChat(file, file.name);
  }

  function ask(q) {
    if (busy || !report) return;

    add({ role: "user", text: q });

    add({
      role: "assistant",
      text: followUp(q, report),
    });
  }

  const lastWithReport = [...messages]
    .reverse()
    .find((m) => m.report);

  return (
    <>
      <header className="top">
        <div className="top-inner">
          <div className="brand">
            <Logo />

            <div>
              <h1>Job Scam Assistant</h1>
              <p>
                Check a job posting or a recruiter chat for warning signs
              </p>
            </div>
          </div>

          {messages.length > 0 && (
            <button className="ghost" onClick={reset}>
              New analysis
            </button>
          )}
        </div>
      </header>

      <div className="app">
        <main className="layout">
          <section className="chat" aria-label="Chat">
            <div className="messages">
              {messages.length === 0 && (
                <div className="intro">
                  <Logo size={56} />

                  <h2>Is this job offer safe?</h2>

                  <p>
                    Paste a job posting or upload a recruiter chat. I point
                    out warning signs and show the evidence. I can say
                    something is potentially fraudulent, but I cannot prove it.
                  </p>

                  <div className="steps">
                    {STEPS.map(([t, d], i) => (
                      <div key={t} className="step">
                        <span className="num">{i + 1}</span>
                        <strong>{t}</strong>
                        <span>{d}</span>
                      </div>
                    ))}
                  </div>

                  <div className="samples">
                    <button onClick={() => setInput(SAMPLE_JOB)}>
                      Try a sample job posting
                    </button>

                    <button
                      onClick={() => runChat(null, "sample_chat.txt")}
                    >
                      Try a sample chat
                    </button>
                  </div>
                </div>
              )}

              {messages.map((m, i) => (
                <div key={i} className={`row ${m.role}`}>
                  {m.role === "assistant" && (
                    <span className="avatar">
                      <Logo size={22} />
                    </span>
                  )}

                  <div
                    className={`bubble ${m.role} ${
                      m.error ? "error" : ""
                    }`}
                  >
                    <p>{m.text}</p>

                    {m.report && (
                      <p className="inline-risk">
                        <span
                          className={`chip ${
                            LEVEL[m.report.risk_category]
                          }`}
                        >
                          {m.report.risk_category}
                        </span>

                        <span>
                          Score {m.report.risk_score}. Details are in the
                          report.
                        </span>
                      </p>
                    )}

                    {m === lastWithReport && !busy && (
                      <div className="followups">
                        {(FOLLOW_UPS[m.report.type] || []).map((q) => (
                          <button
                            key={q}
                            onClick={() => ask(q)}
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {busy && (
                <div className="row assistant">
                  <span className="avatar">
                    <Logo size={22} />
                  </span>

                  <div className="bubble assistant typing">
                    Analyzing
                    <span>.</span>
                    <span>.</span>
                    <span>.</span>
                  </div>
                </div>
              )}

              <div ref={endRef} />
            </div>

            <div className="composer">
              <input
                ref={fileRef}
                type="file"
                accept=".txt,text/plain"
                hidden
                onChange={upload}
              />

              <button
                className="ghost"
                onClick={() => fileRef.current?.click()}
                disabled={busy}
              >
                Upload chat
              </button>

              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder="Paste a job posting here..."
                rows={2}
                aria-label="Job posting text"
              />

              <button
                className="primary"
                onClick={send}
                disabled={busy || !input.trim()}
              >
                Analyze
              </button>
            </div>

            <p className="hintline">
              Enter to send, Shift+Enter for a new line. Results are warning
              signs, not proof.
            </p>
          </section>

          <ReportPanel report={report} />
        </main>
      </div>
    </>
  );
}
