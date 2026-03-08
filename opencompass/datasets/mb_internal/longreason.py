"""LongReason Dataset for OpenCompass.

LongReason is a long-context reasoning benchmark from ByteDance that tests
models' ability to perform complex reasoning over long documents
(8k-128k tokens).

Reference: https://huggingface.co/datasets/lz1bytedance/LongReason
"""

import re
from typing import Dict, List, Optional

from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

__all__ = [
    'LongReasonDataset',
    'LongReasonEvaluator',
    'longreason_postprocess',
]


@LOAD_DATASET.register_module()
class LongReasonDataset(BaseDataset):
    """LongReason dataset for long-context reasoning evaluation.

    The dataset includes multiple splits with different context lengths:
    - original: Original questions without expanded context
    - expanded: Questions with expanded context
    - 8k, 16k, 32k, 64k, 128k: Questions with specific context lengths
    """

    @staticmethod
    def load(
        path: str = 'lz1bytedance/LongReason',
        split: str = '128k',
        max_input_len: Optional[int] = 128000,
        num_samples: Optional[int] = None,
        seed: Optional[int] = 42,
        # cache_dir: Optional[str] = None,
        **kwargs
    ):
        """Load LongReason dataset from HuggingFace.

        Args:
            path: Deprecated, kept for compatibility
            split: Dataset split to load. Options: 'original', 'expanded',
                   '8k', '16k', '32k', '64k', '128k' (default: '128k')
            num_samples: Number of samples to load (None for all)
            seed: Random seed for sampling
            cache_dir: Directory to cache the dataset
            **kwargs: Additional arguments

        Returns:
            Dataset object
        """
        import random
        path = get_data_path(path)

        print(f'[LongReason] Loading dataset (split: {split})')

        dataset = load_dataset(
            path,
            split=split,
            # cache_dir=cache_dir,
            trust_remote_code=True,
            download_mode='reuse_cache_if_exists',  # Reuse partial downloads
        )

        print(f'[LongReason] Loaded {len(dataset)} samples')

        # Sample if needed
        if num_samples is not None and len(dataset) > num_samples:
            random.seed(seed)
            indices = random.sample(range(len(dataset)), num_samples)
            indices.sort()  # Keep original order
            dataset = dataset.select(indices)
            print(f'[LongReason] Sampled {num_samples} samples')

        # # Convert to expected format
        # result_data = []
        # for idx, sample in enumerate(dataset):
        #     # The dataset provides:
        #     # - prompt: Complete prompt with context and question
        #     # - question: The question text only
        #     # - answer: Ground truth answer (A-E)
        #     # - analysis: Optional analysis text
        #     # - example_idx: Sample index
        #
        #     result_data.append({
        #         'prompt': sample.get('prompt', ''),
        #         'question': sample.get('question', ''),
        #         'answer': sample.get('answer', ''),
        #         'analysis': sample.get('analysis', ''),
        #         'example_idx': sample.get('example_idx', idx),
        #     })

        # Print statistics
        # if result_data:
        avg_prompt_len = sum(len(d['prompt'])
                             for d in dataset) / len(dataset)
        print(f'[LongReason] Average prompt length: {avg_prompt_len:.0f} '
              f'chars (~{avg_prompt_len/4:.0f} tokens)')

        return dataset


@TEXT_POSTPROCESSORS.register_module('longreason')
def longreason_postprocess(text: str) -> str:
    """Extract answer choice (A-E) from model output.

    Supports multiple patterns:
    - "The answer is X"
    - "Answer: X"
    - Last occurrence of option letter

    Args:
        text: Model output text

    Returns:
        Extracted answer (A-E) or empty string if not found
    """
    if not text:
        return ''

    # Remove special tokens
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

    # Pattern 1: "The answer is X"
    pattern = r'[Tt]he answer is\s*([A-E])'
    match = re.search(pattern, text)
    if match:
        return match.group(1).upper()

    # Pattern 2: "Answer: X"
    pattern = r'[Aa]nswer\s*:\s*([A-E])'
    match = re.search(pattern, text)
    if match:
        return match.group(1).upper()

    # Pattern 3: Find last occurrence of option letter
    pattern = r'\b([A-E])\b'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].upper()

    return ''


class LongReasonEvaluator(BaseEvaluator):
    """Evaluator for LongReason dataset.

    Computes accuracy by comparing extracted answers with ground truth.
    """

    def score(self, predictions: List, references: List) -> Dict:
        """Calculate accuracy score.

        Args:
            predictions: List of model predictions
            references: List of ground truth answers

        Returns:
            Dictionary containing accuracy and other metrics
        """
        if len(predictions) != len(references):
            return {
                'error': 'Length mismatch',
                'accuracy': 0.0
            }

        correct = 0
        total = 0
        details = []

        for pred, ref in zip(predictions, references):
            total += 1

            # Extract answer from prediction
            if isinstance(pred, str):
                pred_answer = longreason_postprocess(pred)
            else:
                pred_answer = ''

            # Get reference answer
            if isinstance(ref, dict):
                ref_answer = ref.get('answer', '')
            else:
                ref_answer = str(ref)

            # Compare
            is_correct = pred_answer.upper() == ref_answer.upper()
            if is_correct:
                correct += 1

            details.append({
                'prediction': pred_answer,
                'reference': ref_answer,
                'correct': is_correct
            })

        accuracy = correct / total * 100 if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'details': details,  # 保存所有 details（完整结果）
            'details_summary': details[:10],  # 保留前 10 个作为预览
        }

    def leaderboard(self, predictions: List, references: List) -> Dict:
        """Generate leaderboard metrics.

        Args:
            predictions: List of model predictions
            references: List of ground truth answers

        Returns:
            Dictionary with leaderboard metrics
        """
        scores = self.score(predictions, references)
        return {
            'accuracy': scores['accuracy'],
        }
