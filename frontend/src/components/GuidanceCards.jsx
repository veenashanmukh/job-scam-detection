import { plainTags } from "../services/textDisplay.js";

// Related scam-prevention guidance cards (extracted from the report panel; same markup and behaviour).
export default function GuidanceCards({ report, highlight }) {
  const evidence = report.supporting_evidence || [];

  return (
    <>
      {evidence.length === 0 && report.evidence?.guidance_status === "ok" && (
        <>
          <h3>Related Scam-Prevention Guidance</h3>
          <p className="muted">No closely related guidance found.</p>
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
              id={/^K\d+$/.test(e.id || "") ? `ev-${e.id}` : undefined}
              className={`evidence${
                highlight && highlight === e.id ? " highlighted" : ""
              }`}
            >
              <summary>
                <span className="eid">
                  {/^K\d+$/.test(e.id || "") && `[${e.id}] `}
                  {e.title || e.id || "Source"}
                </span>

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

              {plainTags(e.warning_sign_tags).length > 0 && (
                <p className="muted">
                  Related warning signs: {plainTags(e.warning_sign_tags).join(", ")}
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
    </>
  );
}
