import os
from opencompass.models import VLLM

### author: Based on vllm_qwen2_5_7b_instruct_longreason.py
### Date: 2025.11.12
### 说明：LongReason 任务 适配 Qwen2.5-72B-Instruct（vLLM）
### 特点：长答案生成（max_out_len=2048），128K上下文，测试长文本推理能力，启用YARN
### 硬件要求：8 x A100 40GB

_batch_size = int(os.environ.get('BATCH_SIZE', '1'))  # 72B模型使用batch_size=1（显存限制）

models = [
    dict(
        type=VLLM,
        abbr='qwen2.5-72b-instruct-vllm-longreason-yarn',
        path='/home/linbiyuan/models/Qwen2.5-72B-Instruct',  # 请根据实际路径修改
        model_kwargs=dict(
            enforce_eager=True,
            tensor_parallel_size=8,  # 72B模型需要8张A100 40GB
            gpu_memory_utilization=0.90,  # 降低到0.90以预留显存空间
            max_model_len=131072,  # 128K context for LongReason
            # 启用长上下文优化
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,  # 降低到8192以适应72B模型
            # 启用YARN进行RoPE扩展
            rope_scaling={
                "type": "yarn",
                "factor": 4.0,  # 扩展因子，从32K扩展到128K
                "original_max_position_embeddings": 32768,  # Qwen2.5的原始上下文长度
            },
        ),
        max_out_len=2048,  # LongReason needs longer output for reasoning
        max_seq_len=131072,  # 128K context window
        batch_size=_batch_size,  # Use batch_size=1 for 72B model
        generation_kwargs=dict(
            temperature=0.1,  # Low temperature for reasoning consistency
            do_sample=True,  # Enable sampling as in reference
            top_p=0.9,
        ),
        run_cfg=dict(num_gpus=8),  # 需要8张GPU
    )
]

