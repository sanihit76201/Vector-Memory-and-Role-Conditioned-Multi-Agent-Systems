"""HotpotQA evaluator — exact match and token-level F1."""

import re
import string
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class HotpotQAEvaluator:
    """
    Evaluate HotpotQA answers using exact match and token F1.

    Follows the official HotpotQA evaluation protocol:
    - Normalize both prediction and gold answer
    - Compute exact match (1 if identical after normalization, else 0)
    - Compute token-level F1 (overlap of word tokens)
    """

    # ── Normalization ──────────────────────────────────────────────

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize answer text.
        - Lowercase
        - Remove punctuation
        - Remove articles (a, an, the)
        - Collapse whitespace
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        return text

    @staticmethod
    def get_tokens(text: str) -> List[str]:
        """Tokenize normalized text into words."""
        return HotpotQAEvaluator.normalize(text).split()

    # ── Metrics ────────────────────────────────────────────────────

    @staticmethod
    def exact_match(prediction: str, gold: str) -> bool:
        """Return True if normalized prediction == normalized gold."""
        return (
            HotpotQAEvaluator.normalize(prediction)
            == HotpotQAEvaluator.normalize(gold)
        )

    @staticmethod
    def token_f1(prediction: str, gold: str) -> float:
        """
        Compute token-level F1 between prediction and gold answer.

        F1 = 2 * precision * recall / (precision + recall)
        where precision and recall are over shared tokens.
        """
        pred_tokens = HotpotQAEvaluator.get_tokens(prediction)
        gold_tokens = HotpotQAEvaluator.get_tokens(gold)

        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)

        pred_set = {}
        for t in pred_tokens:
            pred_set[t] = pred_set.get(t, 0) + 1

        gold_set = {}
        for t in gold_tokens:
            gold_set[t] = gold_set.get(t, 0) + 1

        common = sum(
            min(pred_set.get(t, 0), gold_set.get(t, 0))
            for t in gold_set
        )

        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall    = common / len(gold_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        return f1

    # ── Main evaluate ──────────────────────────────────────────────

    def evaluate(self, prediction: str, gold: str) -> Dict:
        """
        Evaluate a single prediction against the gold answer.

        Args:
            prediction : Model's answer string
            gold       : Gold answer string from dataset

        Returns:
            Dict with keys:
            - passed : bool  (True if exact match)
            - em     : float (1.0 or 0.0)
            - f1     : float (0.0 to 1.0)
            - error  : str or None (description if wrong)
        """
        # Extract answer from model output if it contains extra text
        prediction = self._extract_answer(prediction)

        em = self.exact_match(prediction, gold)
        f1 = self.token_f1(prediction, gold)

        if em:
            error = None
        else:
            error = (
                f"Predicted: '{prediction}' | "
                f"Gold: '{gold}' | "
                f"F1: {f1:.3f}"
            )

        return {
            "passed"    : em,
            "em"        : float(em),
            "f1"        : round(f1, 4),
            "prediction": prediction,
            "error"     : error,
        }

    @staticmethod
    def _extract_answer(text: str) -> str:
        """
        Extract the final answer from model output.

        Models sometimes output reasoning before the answer.
        We look for common answer patterns.
        """
        text = text.strip()

        # Pattern: "Answer: X" or "Final answer: X"
        patterns = [
            r"(?:final\s+)?answer\s*:\s*(.+?)(?:\n|$)",
            r"(?:the\s+)?answer\s+is\s*:\s*(.+?)(?:\n|$)",
            r"(?:the\s+)?answer\s+is\s+(.+?)(?:\.|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern found, take the last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[-1] if lines else text