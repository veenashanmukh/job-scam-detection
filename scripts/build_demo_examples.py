"""Build backend/data/demo_examples.json: three hand-written fictional fraud postings plus one fictional legitimate one.

Offline. No dataset posting text is read or copied. The only thing taken from the research data is a hashed
contact token (#EMAIL_..#, #PHONE_..# or #URL_..#) per fraud example, chosen from the cached TRAIN-only
entity graph (loaded with the manifest's train hash; train.csv, val.csv and test.csv are never opened).
A token qualifies when it is linked to at least 2 fraud postings and 0 legitimate ones; the least hub-like
token (fewest linked postings, then smallest id) is used, one distinct token per example, so the result is
deterministic. The token is inserted as a "Contact:" line so the live graph lookup can link the example.

Usage: python scripts/build_demo_examples.py [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "backend" / "data" / "demo_examples.json"
CONTACT_KINDS = ("EMAIL", "PHONE", "URL")
MIN_FRAUD_LINKS = 2

FRAUD_NOTE = ("Constructed example: fictional text with a contact token taken from the research dataset "
              "to demonstrate linking.")
LEGIT_NOTE = "Constructed example: fictional text only; no dataset content."

# Invented text with no company names. "{contact}" is replaced by "Contact: <hashed token>".
FRAUD_EXAMPLES = [
    {
        "title": "Data Entry Clerk - Work From Home",
        "company_profile": "",
        "description": "Earn up to 900 per week typing simple forms from your own computer. No interview and "
                       "no experience needed. To secure your place you must pay a registration fee of 150 "
                       "today, and your first batch of forms will be sent right after payment. Only 5 places "
                       "are left, so act fast. {contact}",
        "requirements": "A computer and an internet connection. Must be able to start immediately.",
        "benefits": "Guaranteed weekly income. Flexible hours.",
    },
    {
        "title": "Remote Customer Support Assistant",
        "company_profile": "",
        "description": "Work from home answering messages for our clients. We will send you a cheque to buy "
                       "your own laptop and headset from our approved supplier. Deposit the cheque, keep your "
                       "wages, and transfer the remaining balance to the supplier within 24 hours. Please "
                       "send your bank login details so we can set up your payroll. {contact}",
        "requirements": "Good written English. A bank account. Must reply quickly.",
        "benefits": "Equipment provided. Paid training from day one.",
    },
    {
        "title": "Overseas Hospitality Worker - Visa Sponsored",
        "company_profile": "",
        "description": "Immediate openings abroad with free accommodation and a high salary. Selected "
                       "candidates must pay a visa processing fee of 600 and a refundable security deposit "
                       "before travel documents are issued. Send a photo of your passport and continue the "
                       "chat on Telegram to receive the payment details. {contact}",
        "requirements": "Valid passport. Able to travel within two weeks.",
        "benefits": "Free flights. Accommodation included. Guaranteed contract.",
    },
]

NORTHBRIDGE = {
    "title": "Junior Software Engineer",
    "company_profile": "Northbridge Systems builds scheduling software for community clinics. "
                       "We are a team of 40 based in Leeds, with a public careers page and named managers.",
    "description": "You will join the platform team to build and test features for our appointment "
                   "scheduling product. You will pair with a senior engineer, review code in pull requests "
                   "and take part in weekly planning. Interviews are held on video with the hiring manager "
                   "and two engineers. Salary is stated in the offer letter after the final interview.",
    "requirements": "A degree or equivalent experience in computer science. Familiarity with Python or "
                    "JavaScript. Good written communication.",
    "benefits": "25 days holiday, a learning budget, and flexible working hours.",
}


def pick_contact_tokens(graph, count: int) -> list[str]:
    """Up to `count` hashed contact tokens linked to >= 2 fraud and 0 legitimate train postings, least hub-like first."""
    candidates = []
    for node, data in graph.nodes(data=True):
        if data.get("bipartite") != 1 or data.get("kind") not in CONTACT_KINDS:
            continue
        labels = [graph.nodes[neighbour]["label"] for neighbour in graph.neighbors(node)]
        fraud = sum(labels)
        if fraud >= MIN_FRAUD_LINKS and fraud == len(labels):
            candidates.append((len(labels), node))
    candidates.sort()
    tokens = []
    for _, node in candidates[:count]:
        kind, digest = node.split(":", 1)
        tokens.append(f"#{kind}_{digest}#")
    return tokens


def build_examples(graph) -> list[dict]:
    tokens = pick_contact_tokens(graph, len(FRAUD_EXAMPLES))
    examples = []
    for template, token in zip(FRAUD_EXAMPLES, tokens):
        example = {key: value.replace("{contact}", f"Contact: {token}") for key, value in template.items()}
        examples.append({**example, "note": FRAUD_NOTE})
    return examples + [{**NORTHBRIDGE, "note": LEGIT_NOTE}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    from backend.services import entity_graph as eg
    from backend.services import posting_index
    try:
        graph = eg.load_graph(posting_index.manifest_train_sha256())
    except (OSError, ValueError, KeyError) as exc:
        print(f"Entity graph cache unavailable ({type(exc).__name__}); run scripts/build_evidence_indexes.py.",
              file=sys.stderr)
        return 3
    examples = build_examples(graph)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    fraud = len(examples) - 1
    print(f"Wrote {len(examples)} examples ({fraud} fraud, 1 fictional legitimate) to {args.out.name}")
    if fraud < len(FRAUD_EXAMPLES):
        print(f"Only {fraud} of {len(FRAUD_EXAMPLES)} fraud examples got a qualifying contact token.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
