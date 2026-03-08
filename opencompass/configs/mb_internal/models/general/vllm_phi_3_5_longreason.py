import os
from opencompass.models import VLLM


_batch_size = int(os.environ.get('BATCH_SIZE', '1'))  

models = [
    dict(
        type=VLLM,
        abbr='Phi-3.5-mini-instruct-longreason',
        path='/home/linbiyuan/models/Phi-3.5-mini-instruct',  # 请根据实际路径修改
        model_kwargs=dict(
            enforce_eager=True,
            tensor_parallel_size=8,  # 72B模型需要8张A100 40GB
            gpu_memory_utilization=0.90,  # 降低到0.90以预留显存空间
            max_model_len=131072,  # 128K context for LongReason
            # 启用长上下文优化
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,  # 降低到8192以适应72B模型
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

