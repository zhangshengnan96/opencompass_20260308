"""LongReason Dataset Configuration

LongReason is a long-context reasoning benchmark from ByteDance that tests
models' ability to perform complex reasoning over long documents (8k-128k tokens).

Reference: https://huggingface.co/datasets/lz1bytedance/LongReason

Author: Based on eval_llama3_longreason.py
Date: 2025.11.10
"""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongReasonDataset, LongReasonEvaluator

# ============================================================
# LongReason Task Configuration
# ============================================================

# Reader configuration
# 注意：使用数据集自带的完整 prompt
LongReason_reader_cfg = dict(
    input_columns=['prompt'],  # 数据集提供的完整 prompt
    output_column='answer',
)

# Inference configuration
# 数据集提供的 prompt 格式已包含：
# ### Background Information
# {context}
# 
# ### Question about the Background Information
# {final_question}
# Please answer the above question based on the background information!
# 
# ### Answer
# Please analyze step by step, and provide the final answer in the last line using "The answer is" + option (represented by ABCDE)!
LongReason_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}',  # 直接使用数据集提供的完整 prompt
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048),
)

# Evaluation configuration
LongReason_eval_cfg = dict(
    evaluator=dict(type=LongReasonEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type='longreason'),
)


def _create_longreason_dataset(
    abbr,
    split='128k',
    num_samples=None,
    seed=42,
    cache_dir=None,
    max_out_len=2048,
    **kwargs
):
    """Create a LongReason dataset configuration.
    
    Args:
        abbr: Dataset abbreviation
        split: Dataset split to load. Options: 'original', 'expanded', 
               '8k', '16k', '32k', '64k', '128k' (default: '128k')
        num_samples: Number of samples to load (None for all)
        seed: Random seed for sampling
        cache_dir: Directory to cache the dataset
        max_out_len: Maximum output length for generation
        **kwargs: Additional arguments
    
    Available Splits:
        - original: Original questions without expanded context
        - expanded: Questions with expanded context
        - 8k, 16k, 32k, 64k, 128k: Questions with specific context lengths
    
    Evaluation Metric:
        - accuracy: Percentage of correct answers (A-E)
    """
    import copy
    
    # Update max_out_len in infer_cfg
    infer_cfg = copy.deepcopy(LongReason_infer_cfg)
    infer_cfg['inferencer']['max_out_len'] = max_out_len
    
    config = dict(
        type=LongReasonDataset,
        abbr=abbr,
        path=None,  # Not used, dataset is loaded from HuggingFace
        split=split,
        num_samples=num_samples,
        seed=seed,
        cache_dir=cache_dir,
        reader_cfg=LongReason_reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=LongReason_eval_cfg,
        **kwargs
    )
    return config


# ============================================================
# LongReason Datasets - Different Context Lengths
# ============================================================

# 128k context (most challenging) - 默认使用这个
longreason_128k_dataset = _create_longreason_dataset(
    abbr='longreason_128k',
    split='128k',
    num_samples=None,
    max_out_len=2048,
)

# 只使用 128k context - 其他长度已注释（如需其他长度请取消注释）
LongReason_all_datasets = [
    # 128k context (most challenging) - 只推理这个
    longreason_128k_dataset,
]

# 其他长度的配置（已禁用）
# 如需评估其他长度，请取消下面的注释
"""
LongReason_all_datasets = [
    # Original questions (shortest context)
    _create_longreason_dataset(
        abbr='longreason_original',
        split='original',
        num_samples=None,
        max_out_len=2048,
    ),
    
    # Expanded context
    _create_longreason_dataset(
        abbr='longreason_expanded',
        split='expanded',
        num_samples=None,
        max_out_len=2048,
    ),
    
    # 8k context
    _create_longreason_dataset(
        abbr='longreason_8k',
        split='8k',
        num_samples=None,
        max_out_len=2048,
    ),
    
    # 16k context
    _create_longreason_dataset(
        abbr='longreason_16k',
        split='16k',
        num_samples=None,
        max_out_len=2048,
    ),
    
    # 32k context
    _create_longreason_dataset(
        abbr='longreason_32k',
        split='32k',
        num_samples=None,
        max_out_len=2048,
    ),
    
    # 64k context
    _create_longreason_dataset(
        abbr='longreason_64k',
        split='64k',
        num_samples=None,
        max_out_len=2048,
    ),
    
    # 128k context (most challenging)
    longreason_128k_dataset,
]
"""

# ============================================================
# Dataset Exports
# ============================================================

# 128k (默认配置 - 只推理这个)
longreason_128k = [longreason_128k_dataset]

# Default export: 只使用 128k split
longreason_gen = longreason_128k

# 其他长度的导出（已禁用）
"""
# Original
longreason_original = [ds for ds in LongReason_all_datasets if ds['abbr'] == 'longreason_original']

# Expanded
longreason_expanded = [ds for ds in LongReason_all_datasets if ds['abbr'] == 'longreason_expanded']

# 8k
longreason_8k = [ds for ds in LongReason_all_datasets if ds['abbr'] == 'longreason_8k']

# 16k
longreason_16k = [ds for ds in LongReason_all_datasets if ds['abbr'] == 'longreason_16k']

# 32k
longreason_32k = [ds for ds in LongReason_all_datasets if ds['abbr'] == 'longreason_32k']

# 64k
longreason_64k = [ds for ds in LongReason_all_datasets if ds['abbr'] == 'longreason_64k']
"""


# ============================================================
# Quick Start Examples
# ============================================================
# 
# 默认评估（只推理 128k - 当前配置）:
#   python run.py --models <model_config> --datasets longreason_gen
#
# 示例：使用 Qwen2.5-7B 推理 LongReason 128k
#   python run.py \
#       --models configs/mb_internal/models/general/vllm_qwen2_5_7b_instruct_longreason.py \
#       --datasets longreason_gen
#
# 注意：
#   - 当前配置只会推理 128k 的数据
#   - 如需评估其他长度（8k, 16k, 32k, 64k, original, expanded），
#     请取消文件中相应部分的注释

