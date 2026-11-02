
// Connect the frontend to the real Flask backend.
export const USE_MOCK = false;

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

export const SAMPLE_JOB =
  "Urgent hiring! Work from home data entry. Earn Rs 40,000 per week with no interview. " +
  "Pay a refundable registration fee of Rs 2,500 to confirm your seat. " +
  "Send your Aadhaar and bank details on Telegram to start today. Guaranteed placement.";

export const SAMPLE_CHAT_TEXT = [
  "[24/10/26, 9:02 AM] Recruiter: Hello, we found your profile and have a role for you.",
  "[24/10/26, 9:03 AM] You: Thanks, can you share the details?",
  "[24/10/26, 9:05 AM] Recruiter: Salary is Rs 60,000 per month, no experience needed.",
  "[24/10/26, 9:06 AM] You: Is there an interview?",
  "[24/10/26, 9:08 AM] Recruiter: No interview. Move to Telegram for onboarding.",
  "[24/10/26, 9:15 AM] Recruiter: Pay Rs 3,000 registration fee to confirm your seat.",
  "[24/10/26, 9:16 AM] You: Why do I need to pay?",
  "[24/10/26, 9:18 AM] Recruiter: Pay in the next 30 minutes or the seat goes to someone else.",
  "[24/10/26, 9:20 AM] Recruiter: Also send your bank details and an Aadhaar photo.",
].join("\n");

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

export const SERVER_UNREACHABLE = "Could not reach the analysis server. Please try again in a moment.";

// The server's own message when it sent one; otherwise (a bare 500 from a proxy or a crash) the plain message above.
export function requestErrorMessage(data) {
  return (data && typeof data === "object" && (data.error || data.message)) || SERVER_UNREACHABLE;
}

// fetch + JSON with one error path: a network failure or a failed response without a message gives SERVER_UNREACHABLE.
async function send(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error(SERVER_UNREACHABLE);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(requestErrorMessage(data));
  return data;
}

export async function analyzeJob(text, companyProfile = "") {
  if (!USE_MOCK) {
    const body = { text };

    // Omit the field when empty so the backend reports "company profile not provided".
    if (companyProfile && companyProfile.trim()) {
      body.company_profile = companyProfile;
    }

    const data = await send("/api/analyze-job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    return { type: "job", ...data };

  }

  await delay(900);
  return RISKY.test(text) ? SAMPLE_REPORT_HIGH : SAMPLE_REPORT_LOW;
}

// Posting fields in the order the classifier was trained on.
export const JOB_FIELDS = [
  "title",
  "company_profile",
  "description",
  "requirements",
  "benefits",
];

// Send the posting as separate fields so the backend builds the classifier input
// in the training format. `text` is deliberately not sent alongside them.
export async function analyzeJobFields(fields) {
  const body = {};
  for (const name of JOB_FIELDS) {
    body[name] = typeof fields[name] === "string" ? fields[name] : "";
  }

  if (!USE_MOCK) {
    const data = await send("/api/analyze-job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    return { type: "job", ...data };
  }

  await delay(900);
  return RISKY.test(JOB_FIELDS.map((name) => body[name]).join("\n"))
    ? SAMPLE_REPORT_HIGH
    : SAMPLE_REPORT_LOW;
}

// Warning signs arrive as strings (LLM, chat) or objects (rules: {label, matched_text}).
export function signLabel(sign) {
  if (typeof sign === "string") return sign;
  if (sign && typeof sign === "object") {
    return String(
      sign.label || sign.category || sign.name || sign.title || JSON.stringify(sign)
    );
  }
  return String(sign ?? "");
}

export function formatSign(sign) {
  const label = signLabel(sign);
  const matched = sign && typeof sign === "object" ? sign.matched_text : null;
  return matched ? `${label} ("${matched}")` : label;
}

// `you` is optional: the sender name to treat as the user (their messages are not scored).
// `firstExplanation` (re-checks only) is reused by the server instead of calling the LLM again.
// `untimestamped` (a pasted "Speaker: text" conversation) lets the server read lines that have no timestamp.
export async function analyzeChat(file, you = "", firstExplanation = "", untimestamped = false) {
  if (file === null) {
    const sampleFile = new File(
      [SAMPLE_CHAT_TEXT],
      "sample_chat.txt",
      { type: "text/plain" }
    );
    return analyzeChat(sampleFile, you, firstExplanation, untimestamped);
  }

  if (!file) {
    throw new Error("Please select a chat export to analyze.");
  }

  if (!USE_MOCK) {
    const form = new FormData();
    form.append("file", file);
    if (you) form.append("you", you);
    if (you && firstExplanation) form.append("first_explanation", firstExplanation);
    if (untimestamped) form.append("untimestamped", "1");

    const data = await send("/api/analyze-chat", {
      method: "POST",
      body: form,
    });

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
            signs.map(formatSign).join("; ") +
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
            signs.map(formatSign).join(", ") +
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

// Prepared example postings; an empty list if the server has none.
export async function fetchDemoExamples() {
  try {
    const res = await fetch("/api/demo-examples");
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}
