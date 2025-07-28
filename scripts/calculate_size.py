import math

def calculate_batch_size(
    model_name: str,
    model_params: int,  # 单位：百万参数 (e.g., 500 for 0.5B)
    seq_len: int,
    dataset_size: int,  # 数据集样本数量
    gpu_memory_gb: float,
    use_gradient_checkpointing: bool = True,
    use_flash_attention: bool = True,
    use_mixed_precision: bool = True,
    use_lora: bool = False,
    lora_rank: int = 64,
    sample_packing: bool = False,
    sample_packing_efficiency: float = 0.5,
    optimizer_type: str = "adamw"
) -> int:
    """
    计算推荐的最大 batch_size
    返回: (safe_batch_size, recommended_batch_size)
    """
    print(f"all params: model_name: {model_name} , model_params: {model_params} , seq_len: {seq_len} , dataset_size: {dataset_size} , gpu_memory_gb: {gpu_memory_gb} , use_gradient_checkpointing: {use_gradient_checkpointing} , use_flash_attention: {use_flash_attention} , use_mixed_precision: {use_mixed_precision} , use_lora: {use_lora} , lora_rank: {lora_rank} , sample_packing: {sample_packing} , sample_packing_efficiency: {sample_packing_efficiency} , optimizer_type: {optimizer_type}")
    # 基础模型显存需求 (GB)
    base_model_mem = model_params * 0.004  # 每百万参数基础需求
    
    # 精度因子
    precision_factor = 0.5 if use_mixed_precision else 1.0
    
    # 优化器因子
    optimizer_factors = {
        "adamw": 2.0,
        "adafactor": 1.5,
        "sgd": 1.2,
        "paged_adamw_8bit": 1.8
    }
    optimizer_factor = optimizer_factors.get(optimizer_type.lower(), 2.0)
    
    # 注意力机制因子
    attention_factor = 0.2 if use_flash_attention else 1.0
    
    # 梯度检查点因子
    checkpoint_factor = 0.3 if use_gradient_checkpointing else 1.0
    
    # LoRA 额外开销
    lora_factor = 1.0
    if use_lora:
        lora_factor = 1.0 + (0.0002 * lora_rank * model_params)
    
    # 样本打包因子
    packing_factor = 2.0 if sample_packing else 1.0
    if sample_packing:
        packing_factor = 1.0 / sample_packing_efficiency
    
    # 序列长度因子 (二次方关系)
    seq_len_factor = (seq_len / 1024) ** 2
    
    # 计算每个token的显存开销 (GB/token)
    token_mem_cost = (
        base_model_mem * 
        precision_factor * 
        optimizer_factor * 
        attention_factor * 
        checkpoint_factor * 
        lora_factor * 
        packing_factor * 
        seq_len_factor * 
        0.0000001  # 校准因子
    )
    
    # 计算安全阈值 (保留10%显存)
    safe_memory_gb = gpu_memory_gb * 0.9
    
    # 计算最大可能的token数量
    max_tokens = safe_memory_gb / token_mem_cost
    
    # 计算理论batch_size
    theoretical_batch_size = max_tokens / (seq_len * packing_factor)
    
    # 基于数据集大小的启发式限制
    dataset_limit = max(4, min(256, math.sqrt(dataset_size / 100)))
    
    # 最终推荐值 (考虑实际约束)
    safe_batch_size = max(1, min(
        int(theoretical_batch_size * 0.3),  # 保守估计
        int(gpu_memory_gb * 2),            # 经验上限
        int(dataset_limit)                 # 数据集限制
    ))
    
    # 推荐值 (更激进但安全的估计)
    recommended_batch_size = max(1, min(
        int(theoretical_batch_size * 0.7),
        int(gpu_memory_gb * 4),
        int(dataset_limit * 2)
    ))
    
    return recommended_batch_size

# 使用示例
if __name__ == "__main__":
    # 输入参数
    model_name = "Qwen2-0.5B"
    model_params = 500  # 0.5B 参数
    seq_len = 1024
    dataset_size = 6101
    #avg_tokens_per_sample = 512
    gpu_memory_gb = 80  # 80GB GPU
    
    # 训练配置
    config = {
        "use_gradient_checkpointing": True,
        "use_flash_attention": True,
        "use_mixed_precision": True,
        "use_lora": True,
        "lora_rank": 128,
        "sample_packing": True,
        "sample_packing_efficiency": 0.5,
        "optimizer_type": "paged_adamw_8bit"
    }
    
    # 计算batch_size
    #safe_bs,
    recommended_bs = calculate_batch_size(
        model_name=model_name,
        model_params=model_params,
        seq_len=seq_len,
        dataset_size=dataset_size,
        #avg_tokens_per_sample=avg_tokens_per_sample,
        gpu_memory_gb=gpu_memory_gb,
        **config
    )
    
    print(f"模型: {model_name}, 序列长度: {seq_len}, GPU显存: {gpu_memory_gb}GB")
    #print(f"安全 batch_size: {safe_bs}")
    print(f"推荐 batch_size: {recommended_bs}")
    #print(f"梯度累积步数建议: {max(1, int(64 / safe_bs))} (目标等效batch_size=64)")