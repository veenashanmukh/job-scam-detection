import GuidanceCards from "./components/GuidanceCards.jsx";
import { highlightSegments } from "./services/highlight.js";
import { buildMainScreen, technicalDetails } from "./services/reportView.js";

// Marks only the phrases the report lists as the matched text of a warning sign (see services/highlight.js).
export function Highlighted({ text, signs }) {
  return (
    <>
      {highlightSegments(text, signs).map((segment, i) =>
        segment.label ? (
          <mark key={i} title={`Warning sign: ${segment.label}`}>
            {segment.text}
          </mark>
        ) : (
          segment.text
        )
      )}
    </>
  );
}

// Everything technical lives here, collapsed by default.
function HowDecided({ report }) {
  const tech = technicalDetails(report);

  return (
    <details className="how-decided">
      <summary>How this was decided</summary>

      {tech.rows.length > 0 && (
        <ul>
          {tech.rows.map((row) => (
            <li key={row.label}>
              <span>{row.label}:</span> {row.text}
            </li>
          ))}
        </ul>
      )}

      {tech.rules.length > 0 && (
        <>
          <h4>Matched rules</h4>
          <div className="table-scroll">
            <table className="rules-table">
              <thead>
                <tr>
                  <th>Id</th>
                  <th>Rule</th>
                  <th>Matched text</th>
                </tr>
              </thead>
              <tbody>
                {tech.rules.map((r) => (
                  <tr key={`${r.id}-${r.label}`}>
                    <td>{r.id}</td>
                    <td>{r.label}</td>
                    <td>{r.matched_text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {report.source_text && (
        <>
          <h4>Your text, with warning phrases marked</h4>
          <div className="source">
            <Highlighted text={report.source_text} signs={report.warning_signs} />
          </div>
        </>
      )}

      <GuidanceCards report={report} highlight={null} />

      {tech.notes.length > 0 && (
        <>
          <h4>Limitations and uncertainty</h4>
          <ul className="notes">
            {tech.notes.map((n, i) => (
              <li key={`${i}-${n}`}>{n}</li>
            ))}
          </ul>
        </>
      )}
    </details>
  );
}

// The main result screen for a job posting, in this order: verdict and plain sentence, score, what we checked,
// what to do next, why (only with an AI explanation), similar postings and linked contacts (only with content),
// the footer line, then the collapsed technical section. All text comes from buildMainScreen.
export default function JobReport({ report }) {
  const screen = buildMainScreen(report);

  return (
    <aside className="panel">
      <h2>Job posting report</h2>

      {screen.headline && (
        <div className={`verdict ${screen.level}`} role="status">
          <span className="v-icon" aria-hidden="true">
            {screen.icon}
          </span>
          <div>
            <strong>{screen.headline}</strong>
          </div>
        </div>
      )}

      {screen.sentence && (
        <p className={`plain-verdict ${screen.level}`}>{screen.sentence}</p>
      )}

      {screen.scoreText && (
        <p className="score-line">
          <strong>{screen.scoreText}</strong>
          <span>{screen.scoreNote}</span>
        </p>
      )}

      <h3>What we checked</h3>
      <ul className="checked">
        {screen.checked.map((item) => (
          <li key={`${item.kind}-${item.label}`} className={item.kind}>
            <span className="mark-icon" aria-hidden="true">
              {item.kind === "ok" ? "✓" : "!"}
            </span>{" "}
            {item.text}
            {item.phrase && (
              <>
                {": "}
                <mark title="Matched text">{item.phrase}</mark>
              </>
            )}
          </li>
        ))}
      </ul>

      {screen.next.length > 0 && (
        <>
          <h3>What to do next</h3>
          <ul className="next">
            {screen.next.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </>
      )}

      {screen.why && (
        <>
          <h3>Why</h3>
          <p>{screen.why.text}</p>
          {screen.why.sources.length > 0 && (
            <p className="why-sources">
              Sources:{" "}
              {screen.why.sources.map((s, i) => (
                <span key={s.url}>
                  {i > 0 && ", "}
                  <a href={s.url} target="_blank" rel="noopener noreferrer">
                    {s.title}
                  </a>
                </span>
              ))}
            </p>
          )}
        </>
      )}

      {screen.similar.length > 0 && (
        <>
          <h3>Similar known postings</h3>
          <ul className="ev-list">
            {screen.similar.map((s) => (
              <li key={s.id}>
                <span className={`chip small ${s.genuine ? "low" : "high"}`}>
                  {s.genuine ? "Known genuine" : "Known fraud"}
                </span>{" "}
                <span className="muted" title="Text similarity">
                  similarity {s.similarity.toFixed(2)}
                </span>
                {s.snippet && <p className="ev-snippet">{s.snippet}</p>}
              </li>
            ))}
          </ul>
        </>
      )}

      {screen.linked.length > 0 && (
        <>
          <h3>Linked companies and contacts</h3>
          <ul className="ev-list">
            {screen.linked.map((g) => (
              <li key={g.id}>{g.text}</li>
            ))}
          </ul>
        </>
      )}

      <p className="footer-line">{screen.footer}</p>

      <HowDecided report={report} />
    </aside>
  );
}
