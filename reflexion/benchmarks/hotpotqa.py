"""HotpotQA benchmark loader."""

import logging
from typing import List, Dict
from datasets import load_dataset

logger = logging.getLogger(__name__)


class HotpotQALoader:
    """Load tasks from HotpotQA dataset."""

    @staticmethod
    def load(num_samples: int = 100, split: str = "validation") -> List[Dict]:
        """
        Load HotpotQA tasks.

        Args:
            num_samples : Number of tasks to load
            split       : Dataset split ('validation' or 'train')

        Returns:
            List of task dictionaries with keys:
            - task_id         : Unique identifier
            - question        : Multi-hop question
            - answer          : Gold answer string
            - context         : List of (title, sentences) paragraph tuples
            - supporting_facts: List of (title, sent_idx) gold evidence pairs
            - type            : 'comparison' or 'bridge'
            - level           : 'easy', 'medium', or 'hard'
        """
        logger.info(f"Loading HotpotQA ({split}, {num_samples} samples)...")

        ds = load_dataset("hotpot_qa", "distractor", split=split)
        ds = ds.select(range(min(num_samples, len(ds))))

        tasks = []
        for i, item in enumerate(ds):
            # Build context as list of dicts for easy access
            context = []
            for title, sentences in zip(
                item["context"]["title"],
                item["context"]["sentences"]
            ):
                context.append({
                    "title"    : title,
                    "sentences": sentences,
                    "paragraph": " ".join(sentences)
                })

            tasks.append({
                "task_id"         : f"HotpotQA/{i}",
                "question"        : item["question"],
                "answer"          : item["answer"],
                "context"         : context,
                "supporting_facts": list(zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"]
                )),
                "type"            : item["type"],
                "level"           : item["level"],
            })

        logger.info(f"Loaded {len(tasks)} HotpotQA tasks")
        return tasks

    @staticmethod
    def format_context(context: List[Dict]) -> str:
        """
        Format context paragraphs into a single string for LLM prompt.

        Args:
            context: List of context dicts from load()

        Returns:
            Formatted string with titled paragraphs
        """
        lines = []
        for i, para in enumerate(context):
            lines.append(f"[{i+1}] {para['title']}: {para['paragraph']}")
        return "\n".join(lines)