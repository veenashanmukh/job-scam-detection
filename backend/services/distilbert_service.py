
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "backend" / "models_v2" / "distilbert_job_scam"

MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.291163


class DistilBERTJobScamService:
    """Load a trained DistilBERT model and predict job scam risk."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"DistilBERT model directory not found: {MODEL_DIR}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, job_description: str) -> dict:
        """Return the predicted class and estimated fraud probability."""

        if not isinstance(job_description, str) or not job_description.strip():
            raise ValueError("Job description must be a non-empty string.")

        inputs = self.tokenizer(
            job_description,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        fraud_probability = float(probabilities[1].item())
        legitimate_probability = float(probabilities[0].item())

        prediction = int(fraud_probability >= self.threshold)

        return {
            "model": "DistilBERT",
            "prediction": prediction,
            "label": "fraudulent" if prediction == 1 else "legitimate",
            "fraud_probability": round(fraud_probability, 6),
            "legitimate_probability": round(legitimate_probability, 6),
            "threshold": self.threshold,
            "device": str(self.device),
        }
