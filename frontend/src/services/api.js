
// Connect the frontend to the real Flask backend.
export const USE_MOCK = false;

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

export const SAMPLE_JOB =
  "Urgent hiring! Work from home data entry. Earn Rs 40,000 per week with no interview. " +
  "Pay a refundable registration fee of Rs 2,500 to confirm your seat. " +
  "Send your Aadhaar and bank details on Telegram to start today. Guaranteed placement.";

export const FOLLOW_UPS = {
  job: [
    "Which parts look suspicious?",
    "What evidence supports this result?",
    "What should I verify next?",
  ],
  chat: [
    "Summarize the warning signs",
    "When did the risk rise?",
    "What should I verify next?",
  ],
};

const RISKY = /(\bfee\b|deposit|bank details|aadhaar|guaranteed|no interview|urgent|telegram)/i;

const SAMPLE_REPORT_HIGH = {
  type: "job",
  risk_score: 72,
  risk_category: "Higher risk",
  classifier_prediction: "Potentially fraudulent",
  warning_signs: [
    "Upfront payment request (registration fee)",
    "Request for sensitive information (Aadhaar, bank details)",
    "Unrealistic pay with no interview",
    "Urgent language",
    "Asks to move to Telegram",
  ],
  supporting_evidence: [
    {
      id: "ref-1042",
      label: "Fraudulent",
      similarity: 0.87,
      source: "Sample reference record",
      snippet:
        "Pay a refundable security deposit to secure your position and start earning...",
    },
    {
      id: "ref-0318",
      label: "Fraudulent",
      similarity: 0.81,
      source: "Sample reference record",
      snippet:
        "Earn weekly income from home with no experience. Send your ID documents to begin...",
    },
    {
      id: "ref-5521",
      label: "Legitimate",
      similarity: 0.52,
      source: "Sample reference record",
      snippet:
        "Data entry associate, fixed monthly salary, two interview rounds required...",
    },
  ],
  explanation:
    "This posting has several warning signs that also appear in records labeled fraudulent. " +
    "It asks for money before work starts and for sensitive documents. These are warning signs, " +
    "not proof. Check the company on its official website before sharing anything.",
  uncertainty_notes: [
    "No company details were given, so no company check was possible.",
    "Prototype: this result is sample data, not output from a real model.",
  ],
};

const SAMPLE_REPORT_LOW = {
  type: "job",
  risk_score: 18,
  risk_category: "Lower risk",
  classifier_prediction: "Likely legitimate",
  warning_signs: [],
  supporting_evidence: [
    {
      id: "ref-5521",
      label: "Legitimate",
      similarity: 0.74,
      source: "Sample reference record",
      snippet:
        "Data entry associate, fixed monthly salary, two interview rounds required...",
    },
  ],
  explanation:
    "Few warning signs were found, and the closest reference record is labeled legitimate. " +
    "This does not guarantee the job is genuine. Verify the employer independently.",
  uncertainty_notes: [
    "Prototype: this result is sample data, not output from a real model.",
  ],
};

const SAMPLE_TIMELINE = [
  {
    time: "09:02",
    speaker: "Recruiter",
    text: "Hello, we found your profile and have a role for you.",
    sign: null,
    cumulative: 5,
  },
  {
    time: "09:03",
    speaker: "You",
    text: "Thanks, can you share the details?",
    sign: null,
    cumulative: 5,
  },
  {
    time: "09:05",
    speaker: "Recruiter",
    text: "Salary is Rs 60,000 per month, no experience needed.",
    sign: "Unrealistic pay",
    cumulative: 20,
  },
  {
    time: "09:06",
    speaker: "You",
    text: "Is there an interview?",
    sign: null,
    cumulative: 20,
  },
  {
    time: "09:08",
    speaker: "Recruiter",
    text: "No interview. Move to Telegram for onboarding.",
    sign: "Moves to another app",
    cumulative: 35,
  },
  {
    time: "09:15",
    speaker: "Recruiter",
    text: "Pay Rs 3,000 registration fee to confirm your seat.",
    sign: "Payment request",
    cumulative: 60,
  },
  {
    time: "09:16",
    speaker: "You",
    text: "Why do I need to pay?",
    sign: null,
    cumulative: 60,
  },
  {
    time: "09:18",
    speaker: "Recruiter",
    text: "Pay in the next 30 minutes or the seat goes to someone else.",
    sign: "Pressure tactic",
    cumulative: 75,
  },
  {
    time: "09:20",
    speaker: "Recruiter",
    text: "Also send your bank details and an Aadhaar photo.",
    sign: "Sensitive information request",
    cumulative: 90,
  },
];

const SAMPLE_REPORT_CHAT = {
  type: "chat",
  risk_score: 88,
  risk_category: "Higher risk",
  classifier_prediction: "Potentially fraudulent conversation",
  warning_signs: [
    "Unrealistic pay",
    "Moves to another app",
    "Payment request",
    "Pressure tactic",
    "Sensitive information request",
  ],
  supporting_evidence: [],
  timeline: SAMPLE_TIMELINE,
  explanation:
    "Warning signs start early and rise after the payment request. The recruiter then adds a deadline " +
    "and asks for bank and ID details. This pattern is common in recruitment scams, but it is a warning, not proof.",
  uncertainty_notes: [
    "Prototype: this is a built-in sample conversation, not your uploaded file.",
  ],
};

export async function analyzeJob(text) {
  if (!USE_MOCK) {
    const res = await fetch("/api/analyze-job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(
        data.error || data.message || `Job analysis failed (${res.status})`
      );
    }

    return { type: "job", ...data };

  }

  await delay(900);
  return RISKY.test(text) ? SAMPLE_REPORT_HIGH : SAMPLE_REPORT_LOW;
}

export async function analyzeChat(file) {
  if (!USE_MOCK) {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch("/api/analyze-chat", {
      method: "POST",
      body: form,
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(
        data.error || data.message || `Chat analysis failed (${res.status})`
      );
    }

    return { type: "chat", ...data };

  }

  await delay(1100);
  return SAMPLE_REPORT_CHAT;
}

export function followUp(question, report) {
  const signs = report.warning_signs || [];

  switch (question) {
    case "Which parts look suspicious?":
      return signs.length
        ? "These parts stood out: " +
            signs.join("; ") +
            ". Each one is a warning sign, not proof of fraud."
        : "No strong warning signs were found in this text.";

    case "What evidence supports this result?": {
      const evidence = report.supporting_evidence || [];

      if (!evidence.length) {
        return "No similar reference records were retrieved. Open the panel on the right to see what was found.";
      }

      const top = evidence[0];

      return `${evidence.length} supporting records were retrieved. The closest is ${
        top.title || top.id || "a reference record"
      }${
        top.similarity != null
          ? ` (similarity ${top.similarity.toFixed(3)})`
          : ""
      }. Open the evidence list on the right to read them. Similarity measures textual relevance, not whether a company is fraudulent.`;
    }

    case "Summarize the warning signs":
      return signs.length
        ? "In order: " +
            signs.join(", ") +
            ". Together they may indicate a suspicious recruitment pattern."
        : "No strong warning signs were returned.";

    case "When did the risk rise?":
      return report.timeline?.length
        ? "Review the timeline to see which messages introduced warning signs."
        : "A message-by-message timeline is not available for this job-posting analysis.";

    default:
      return (
        "Verify the employer through its official website and careers page, not through links in the message. " +
        "Contact the company using details you find yourself. Never pay to get a job. " +
        "Do not share bank or identity details before you have confirmed the offer."
      );
  }
}
