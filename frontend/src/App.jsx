
import { useEffect, useRef, useState } from "react";
import {
  analyzeJob,
  analyzeJobFields,
  analyzeChat,
  fetchDemoExamples,
  followUp,
  signLabel,
  JOB_FIELDS,
  SAMPLE_JOB,
  FOLLOW_UPS,
} from "./services/api.js";
import GuidanceCards from "./components/GuidanceCards.jsx";
import JobReport from "./JobReport.jsx";
import { conversationFile, OVERRIDE_LABEL, ROUTED_NOTE, routePastedText } from "./services/conversation.js";
import {
  chatPrediction, chatScore, clampDisplayScore, NO_FLAGGED_MESSAGES, showChatGuidance, splitExplanationNote,
} from "./services/chatView.js";
import { evidenceView, nextStepsFor, plainSentence } from "./services/reportView.js";

const FIELD_LABELS = {
  title: "Title",
  company_profile: "Company profile",
  description: "Description",
  requirements: "Requirements",
  benefits: "Benefits",
};

const EMPTY_FIELDS = Object.fromEntries(JOB_FIELDS.map((name) => [name, ""]));

// The backend only uses the structured path when one of these is present;
// a company profile on its own is not enough.
const CONTENT_FIELDS = ["title", "description", "requirements", "benefits"];

const LEVEL = {
  "Lower risk": "low",
  "Moderate risk": "mid",
  "Higher risk": "high",
};

// Verdict styling. "Needs Verification" is a caution, never a reassurance.
const VERDICT = {
  "Likely Legitimate": {
    level: "low",
    icon: "✓",
    note: "Not flagged by the classifier. This is not a guarantee.",
  },
  "Needs Verification": {
    level: "mid",
    icon: "!",
    note: "Caution: verify the company independently before sharing money or personal details.",
  },
  "High Fraud Risk": {
    level: "high",
    icon: "!",
    note: "Do not pay any fee or share personal details until the employer is verified.",
  },
  "Insufficient Information": {
    level: "neutral",
    icon: "?",
    note: "Not enough text to assess. Add a description.",
  },
};

const INSUFFICIENT = "Insufficient Information";

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

function VerdictBadge({ verdict, showNote = true }) {
  const info = VERDICT[verdict];

  if (!info) return null;

  return (
    <div className={`verdict ${info.level}`} role="status">
      <span className="v-icon" aria-hidden="true">
        {info.icon}
      </span>

      <div>
        <strong>{verdict}</strong>
        {showNote && <p className="v-note">{info.note}</p>}
      </div>
    </div>
  );
}

function Gauge({ score, category, level: levelOverride, label, note }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const level = levelOverride || LEVEL[category];

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
        {label ? (
          <strong className="g-label">{label}</strong>
        ) : (
          <span className={`chip big ${level}`}>{category}</span>
        )}

        <p className="hint">Higher means more risk</p>

        {note && <p className="hint g-note">{note}</p>}
      </div>
    </div>
  );
}

// A pasted conversation has no timestamps; label its messages by position instead of leaving the time blank.
const timeLabel = (item, index) => item.time || `message ${index + 1}`;

function Timeline({ items, highlight }) {
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
              {`${timeLabel(items[i], i)}: ${
                items[i].sign || "no warning sign"
              }`}
            </title>
          </circle>
        ))}

        <text x={px} y={h - 1} className="tl-label">
          {timeLabel(items[0], 0)}
        </text>

        <text
          x={w - px}
          y={h - 1}
          textAnchor="end"
          className="tl-label"
        >
          {timeLabel(items[items.length - 1], items.length - 1)}
        </text>
      </svg>

      {!items.some((m) => m.sign) && <p className="muted">{NO_FLAGGED_MESSAGES}</p>}

      {items.some((m) => m.sign) && (
        <ul className="flags">
          {items
            .filter((m) => m.sign)
            .map((m, i) => (
              <li
                key={i}
                id={m.evidence_id ? `ev-${m.evidence_id}` : undefined}
                className={
                  m.evidence_id && highlight === m.evidence_id ? "highlighted" : undefined
                }
              >
                <span className="time">{timeLabel(m, items.indexOf(m))}</span>
                <strong>
                  {m.evidence_id && <span className="ev-id">[{m.evidence_id}] </span>}
                  {m.sign}
                </strong>
                <span className="quote">"{m.text}"</span>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}

// "[S1]" or grouped "[R1,R3]" / "[R1, R3]" (same pattern the backend validates).
const CITATION = /(\[\s*[A-Z]\d{1,3}(?:\s*,\s*[A-Z]\d{1,3})*\s*\])/;

// Explanation text with citations turned into one button per known evidence id.
function CitedText({ text, ids, onCite }) {
  return (
    <>
      {String(text || "")
        .split(CITATION)
        .map((part, i) => {
          if (!CITATION.test(part)) return part;
          const group = part.slice(1, -1).split(",").map((id) => id.trim());
          if (!group.some((id) => ids.has(id))) return part;
          // Known ids render as chips (no brackets); unknown ids keep their bracketed text.
          return (
            <span key={i} className="cite-group">
              {group.map((id, j) => (
                <span key={id + j}>
                  {ids.has(id) ? (
                    <button
                      type="button"
                      className="cite"
                      onClick={() => onCite(id)}
                      aria-label={`Show evidence ${id}`}
                    >
                      {id}
                    </button>
                  ) : (
                    `[${id}]`
                  )}
                </span>
              ))}
            </span>
          );
        })}
    </>
  );
}

function EvidenceSection({ report, highlight, onCite }) {
  const ev = report.evidence;
  const similar = ev.similar_postings || [];
  const graphItems = ev.graph?.items || [];
  const rules = ev.rule_signs || [];
  // T ids (chat messages) are shown in the timeline list, which carries their anchors.
  const ids = new Set(
    [...similar, ...graphItems, ...rules, ...(ev.guidance || []), ...(ev.messages || [])].map(
      (e) => e.id
    )
  );
  const cls = (id) => `ev-item${highlight === id ? " highlighted" : ""}`;
  // Without an AI explanation only the matched signs and similar postings are shown (or one "no evidence" line);
  // the template explanation would just repeat the headline sentence.
  const view = evidenceView(report);

  return (
    <section className="evidence-block" aria-label="Evidence">
      <h3>Evidence</h3>

      {view.showExplanation && (
        <p className="explanation">
          <CitedText text={splitExplanationNote(report.explanation).body} ids={ids} onCite={onCite} />
        </p>
      )}

      {view.emptyLine && <p className="muted">{view.emptyLine}</p>}

      {view.aiAvailable && similar.length === 0 && ev.similar_status === "ok" && (
        <>
          <h4>Similar known postings</h4>
          <p className="muted">No close matches in known postings.</p>
        </>
      )}

      {similar.length > 0 && (
        <>
          <h4>Similar known postings</h4>
          <ul className="ev-list">
            {similar.map((s) => (
              <li key={s.id} id={`ev-${s.id}`} className={cls(s.id)}>
                <span className="ev-id">[{s.id}]</span>{" "}
                <span className={`chip small ${s.label === 1 ? "high" : "low"}`}>
                  {s.label === 1 ? "Known fraud" : "Known legitimate"}
                </span>{" "}
                <span className="muted" title="Text similarity, not a probability">
                  similarity {Number(s.similarity).toFixed(2)}
                </span>
                {s.snippet && <p className="ev-snippet">{s.snippet}</p>}
              </li>
            ))}
          </ul>
        </>
      )}

      {view.showGraph && (graphItems.length > 0 || ev.graph?.note) && (
        <>
          <h4>Linked companies and contacts</h4>
          {graphItems.length === 0 && <p className="muted">{ev.graph.note}</p>}
          <ul className="ev-list">
            {graphItems.map((g) => (
              <li key={g.id} id={`ev-${g.id}`} className={cls(g.id)}>
                <span className="ev-id">[{g.id}]</span>{" "}
                {g.message ||
                  `This ${g.kind_name} appears in ${g.linked_count} known posting${
                    g.linked_count === 1 ? "" : "s"
                  } (${g.fraud_count} labelled fraud).`}
              </li>
            ))}
          </ul>
        </>
      )}

      {view.showRules && <h4>Warning signs</h4>}
      {view.showRules && rules.length === 0 && <p className="muted">None detected.</p>}

      {view.showRules && rules.length > 0 && (
        <>
          <ul className="ev-list">
            {rules.map((r) => (
              <li key={r.id} id={`ev-${r.id}`} className={cls(r.id)}>
                <span className="ev-id">[{r.id}]</span> <strong>{r.label}</strong>
                {r.matched_text && (
                  <span className="muted"> — "{r.matched_text}"</span>
                )}
              </li>
            ))}
          </ul>
        </>
      )}

    </section>
  );
}

function ReportPanel({ report, onChooseYou }) {
  const [highlight, setHighlight] = useState(null);

  useEffect(() => setHighlight(null), [report]);

  const cite = (id) => {
    setHighlight(id);
    const el = document.getElementById(`ev-${id}`);
    if (el) {
      if (el.tagName === "DETAILS") el.open = true;
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  };

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

  if (report.verdict === INSUFFICIENT) {
    return (
      <aside className="panel">
        <h2>Job posting report</h2>
        <VerdictBadge verdict={INSUFFICIENT} />
      </aside>
    );
  }

  // Job postings have their own plain-language screen (JobReport.jsx); everything below is the chat report.
  if (report.type !== "chat") return <JobReport report={report} />;

  const uncertaintyNotes = report.uncertainty_notes || [];
  const warningSigns = report.warning_signs || [];
  // With an evidence block the rule signs are listed there (citable as R1..), so not twice.
  const showWarningList = !report.evidence;
  const prediction = chatPrediction(report);
  const nextSteps = nextStepsFor(report);
  // The note about a reused explanation is shown in the collapsed section, not after the explanation.
  const explanationNote = splitExplanationNote(report.explanation).note;

  return (
    <aside className="panel">
      <h2>Conversation report</h2>

      <Gauge
        score={chatScore(report)}
        category={report.risk_category}
      />

      {report.participants?.length > 1 && onChooseYou && (
        <p className="you-picker">
          <label htmlFor="you-select">You in this chat:</label>{" "}
          <select
            id="you-select"
            value={report.you || ""}
            onChange={(e) => onChooseYou(e.target.value)}
          >
            {!report.you && <option value="">Not sure (score everyone)</option>}
            {report.participants.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>{" "}
          <span className="muted">
            {report.you
              ? "Your own messages are not scored."
              : "Pick your name so your own messages are not scored."}
          </span>
        </p>
      )}

      {report.evidence && (
        <EvidenceSection report={report} highlight={highlight} onCite={cite} />
      )}

      {prediction.text && (
        <p className="pred">
          {prediction.labelled && <span>Rule check:</span>} {prediction.text}
        </p>
      )}

      {nextSteps.length > 0 && (
        <>
          <h3>What to do next</h3>
          <ul className="next">
            {nextSteps.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </>
      )}

      {explanationNote && (
        <details className="how-decided">
          <summary>How this was decided</summary>
          <ul>
            <li>{explanationNote}</li>
          </ul>
        </details>
      )}

      {showWarningList && <h3>Warning signs</h3>}

      {!showWarningList ? null : warningSigns.length > 0 ? (
        <ul className="signs">
          {warningSigns.map((s, index) => {
            const isRuleFinding = typeof s === "object" && s !== null;
            const label = signLabel(s);
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

      {report.timeline?.length > 0 && (
        <>
          <h3>Risk timeline</h3>
          <Timeline items={report.timeline} highlight={highlight} />
        </>
      )}

      {showChatGuidance(report) && <GuidanceCards report={report} highlight={highlight} />}

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
  const [fields, setFields] = useState(EMPTY_FIELDS);
  const [pasteMode, setPasteMode] = useState(false);
  const [formError, setFormError] = useState("");
  const [busy, setBusy] = useState(false);
  const [examples, setExamples] = useState([]);
  const exampleIdx = useRef(0);
  const [exampleNote, setExampleNote] = useState("");

  useEffect(() => {
    fetchDemoExamples().then(setExamples);
  }, []);

  function loadExample() {
    if (!examples.length) return;
    const ex = examples[exampleIdx.current % examples.length];
    exampleIdx.current += 1;
    setFields(Object.fromEntries(JOB_FIELDS.map((name) => [name, ex[name] || ""])));
    setExampleNote(ex.note || "");
    setPasteMode(false);
    setFormError("");
  }

  const endRef = useRef(null);
  const fileRef = useRef(null);
  const lastChatRef = useRef(null); // the last chat file, to re-check with a chosen "you"

  const chooseYou = (name) => {
    const last = lastChatRef.current;
    if (!busy && last && name) {
      runChat(last.file, last.label, name, { untimestamped: last.untimestamped, routedFrom: last.routedFrom });
    }
  };

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const add = (m) =>
    setMessages((prev) => [...prev, m]);

  const reset = () => {
    setMessages([]);
    setReport(null);
    setInput("");
    setFields(EMPTY_FIELDS);
    setFormError("");
  };

  const setField = (name, value) => {
    setFields((prev) => ({ ...prev, [name]: value }));
    setFormError("");
  };

  // request is either { text, profile } (pasted posting) or { fields } (form).
  async function runJob(text, request) {
    add({ role: "user", text: request.userLabel || text });
    setBusy(true);

    try {
      const result = request.fields
        ? await analyzeJobFields(request.fields)
        : await analyzeJob(request.text, request.profile);
      const r = { ...result, source_text: text };

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

  // options: { untimestamped, routedFrom, userText } for a conversation pasted into the job box and routed here.
  async function runChat(file, label, you = "", options = {}) {
    const prev = lastChatRef.current;
    const sameChat = prev && prev.file === file;
    // Keep the first analysis's explanation; re-checks reuse it instead of calling the LLM again.
    const firstExplanation = you && sameChat ? prev.firstExplanation : "";
    const { untimestamped = false, routedFrom = null, userText = "" } = options;
    lastChatRef.current = { file, label, firstExplanation, untimestamped, routedFrom };
    add({
      role: "user",
      text: userText || (you ? `Re-check ${label} with "${you}" as me` : `Uploaded file: ${label}`),
    });
    setBusy(true);

    try {
      const r = await analyzeChat(file, you, firstExplanation, untimestamped);
      if (r.explanation_source !== "first_analysis") lastChatRef.current.firstExplanation = r.explanation;

      setReport(r);

      add({
        role: "assistant",
        text: r.explanation,
        report: r,
        routedFrom,
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
    if (busy) return;
    if (pasteMode) sendPasted();
    else sendFields();
  }

  function sendFields() {
    if (!CONTENT_FIELDS.some((name) => fields[name].trim())) {
      setFormError(
        "Add at least a title, description, requirements or benefits. " +
          "A company profile on its own is not enough to analyze a posting."
      );
      return;
    }

    // Shown in the chat and highlighted in the report; mirrors the field order.
    const summary = JOB_FIELDS.filter((name) => fields[name].trim())
      .map((name) => `${FIELD_LABELS[name]}: ${fields[name].trim()}`)
      .join("\n");

    runJob(summary, { fields });
    setFields(EMPTY_FIELDS);
    setFormError("");
  }

  function sendPasted() {
    const text = input.trim();

    if (!text) return;

    setInput("");

    if (text.length < 30) {
      add({ role: "user", text });

      add({
        role: "assistant",
        text: "That looks short. Please paste the full job posting, or upload an exported chat as a .txt file, so I can check it.",
      });

      return;
    }

    if (routePastedText(text) === "chat") {
      // Speaker-labelled lines from at least two people: check it as a chat. The user can switch back below.
      const profile = fields.company_profile;
      setField("company_profile", "");
      runChat(conversationFile(text), "pasted conversation", "", {
        untimestamped: true,
        routedFrom: { text, profile },
        userText: text,
      });
      return;
    }

    runJob(text, { text, profile: fields.company_profile });
    setField("company_profile", "");
  }

  // The override: run the pasted text as a job posting after all (routePastedText(text, { forceJob: true })).
  function checkAsJobPosting(routed) {
    if (busy) return;
    runJob(routed.text, { text: routed.text, profile: routed.profile, userLabel: OVERRIDE_LABEL });
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
                    <button
                      onClick={() => {
                        setPasteMode(true);
                        setInput(SAMPLE_JOB);
                      }}
                    >
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
                    {m.routedFrom && (
                      <p className="route-note">
                        {ROUTED_NOTE}{" "}
                        <button
                          type="button"
                          className="link-button"
                          onClick={() => checkAsJobPosting(m.routedFrom)}
                          disabled={busy}
                        >
                          {OVERRIDE_LABEL}
                        </button>
                      </p>
                    )}

                    <p>{(m.report && plainSentence(m.report)) || splitExplanationNote(m.text).body}</p>

                    {m.report?.verdict === INSUFFICIENT ? (
                      <p className="inline-risk">
                        <span className="chip neutral">{INSUFFICIENT}</span>
                        <span>No score: add a description.</span>
                      </p>
                    ) : m.report &&
                      (VERDICT[m.report.verdict] &&
                      typeof m.report.display_risk_score === "number" ? (
                        <p className="inline-risk">
                          <span
                            className={`chip ${
                              VERDICT[m.report.verdict].level
                            }`}
                          >
                            {m.report.verdict}
                          </span>

                          <span>
                            Risk score {clampDisplayScore(m.report.display_risk_score)} / 100.
                            Higher means more risk.
                          </span>
                        </p>
                      ) : VERDICT[m.report.verdict] ? (
                        <p className="inline-risk">
                          <span
                            className={`chip ${
                              VERDICT[m.report.verdict].level
                            }`}
                          >
                            {m.report.verdict}
                          </span>

                          <span>Details are in the report.</span>
                        </p>
                      ) : (
                        <p className="inline-risk">
                          <span
                            className={`chip ${
                              LEVEL[m.report.risk_category]
                            }`}
                          >
                            {m.report.risk_category}
                          </span>

                          <span>
                            Risk score {chatScore(m.report)} / 100. Higher
                            means more risk.
                          </span>
                        </p>
                      ))}

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

            <div className="job-form">
              <label className="mode-toggle">
                <input
                  type="checkbox"
                  checked={pasteMode}
                  onChange={(e) => {
                    setPasteMode(e.target.checked);
                    setFormError("");
                  }}
                  disabled={busy}
                />
                Paste full posting instead
              </label>

              {!pasteMode && (
                <div className="field">
                  <label htmlFor="job-title">
                    Title <span className="muted">(optional)</span>
                  </label>
                  <input
                    id="job-title"
                    type="text"
                    value={fields.title}
                    onChange={(e) => setField("title", e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        send();
                      }
                    }}
                    placeholder="e.g. Data Entry Clerk"
                    disabled={busy}
                  />
                </div>
              )}

              <div className="field">
                <label htmlFor="company-profile">
                  Company profile <span className="muted">(optional)</span>
                </label>
                <textarea
                  id="company-profile"
                  value={fields.company_profile}
                  onChange={(e) => setField("company_profile", e.target.value)}
                  placeholder="Paste the 'About the company' section, if the posting has one..."
                  rows={2}
                  disabled={busy}
                  aria-describedby="company-profile-note"
                />
                <p id="company-profile-note" className="field-note">
                  Leaving this empty is treated as "no company profile". In our
                  data, postings without one were about 6 times more often
                  fraudulent, so an empty profile raises the risk estimate.
                </p>
              </div>

              {!pasteMode &&
                ["description", "requirements", "benefits"].map((name) => (
                  <div className="field" key={name}>
                    <label htmlFor={`job-${name}`}>
                      {FIELD_LABELS[name]}{" "}
                      <span className="muted">(optional)</span>
                    </label>
                    <textarea
                      id={`job-${name}`}
                      value={fields[name]}
                      onChange={(e) => setField(name, e.target.value)}
                      rows={name === "description" ? 3 : 2}
                      disabled={busy}
                    />
                  </div>
                ))}

              {formError && (
                <p className="form-error" role="alert">
                  {formError}
                </p>
              )}
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

              {pasteMode ? (
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
              ) : (
                <span className="composer-spacer" />
              )}

              {!pasteMode && examples.length > 0 && (
                <p className="example-row">
                  <button type="button" onClick={loadExample} disabled={busy}>
                    Load an example
                  </button>
                  <span className="muted">
                    {" "}Examples come from the research dataset (contacts are hashed).
                  </span>
                  {exampleNote && (
                    <span className="muted example-note">
                      <br />
                      {exampleNote}
                    </span>
                  )}
                </p>
              )}

              <button
                className="primary"
                onClick={send}
                disabled={busy || (pasteMode && !input.trim())}
              >
                Analyze
              </button>
            </div>

            <p className="hintline">
              {pasteMode
                ? "Enter to send, Shift+Enter for a new line. "
                : "Fill in the fields you have; separate fields match how the model was trained. "}
              Results are warning signs, not proof.
            </p>
          </section>

          <ReportPanel report={report} onChooseYou={chooseYou} />
        </main>
      </div>
    </>
  );
}
