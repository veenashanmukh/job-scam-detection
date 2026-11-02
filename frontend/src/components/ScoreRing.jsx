// Circular risk score: the ring fills to the score and its colour follows the provisional score bands
// (0-29 green, 30-59 amber, 60-100 red), so a higher score reads as more red. The score sits in the centre.
export function scoreLevel(score) {
  if (score < 30) return "low";
  if (score < 60) return "mid";
  return "high";
}

export default function ScoreRing({ score, note }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const level = scoreLevel(score);

  return (
    <div className="score-ring">
      <svg viewBox="0 0 120 120" role="img" aria-label={`Risk score ${score} out of 100`}>
        <circle cx="60" cy="60" r="44" className={`g-inner ${level}`} />
        <circle cx="60" cy="60" r={r} className="g-track" />
        <circle
          cx="60"
          cy="60"
          r={r}
          className={`g-fill ${level}`}
          strokeDasharray={`${(score / 100) * c} ${c}`}
          transform="rotate(-90 60 60)"
        />
        <text x="60" y="66" textAnchor="middle" className={`g-score ${level}`}>
          {score}
        </text>
        <text x="60" y="84" textAnchor="middle" className="g-of">
          out of 100
        </text>
      </svg>
      {note && <span className="score-ring-note">{note}</span>}
    </div>
  );
}
