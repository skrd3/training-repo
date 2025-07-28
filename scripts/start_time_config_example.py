#!/usr/bin/env python3
"""
开始时间后保存配置示例
展示如何自定义训练开始后多少分钟保存模型
"""

from datetime import datetime, timezone, timedelta
from customized_trainer import WhenToEvalHandler

def create_early_save_strategy(end_time: str, start_time: str):
    """创建早期保存策略：训练开始后很快保存"""
    return WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=5,
        start_time=start_time,
        save_after_start_minutes=5  # 开始后5分钟就保存
    )

def create_mid_save_strategy(end_time: str, start_time: str):
    """创建中期保存策略：训练开始后一段时间保存"""
    return WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=10,
        start_time=start_time,
        save_after_start_minutes=30  # 开始后30分钟保存
    )

def create_late_save_strategy(end_time: str, start_time: str):
    """创建晚期保存策略：训练开始后较长时间保存"""
    return WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=15,
        start_time=start_time,
        save_after_start_minutes=60  # 开始后60分钟保存
    )

def create_dual_save_strategy(end_time: str, start_time: str):
    """创建双重保存策略：开始后和结束前都保存"""
    return WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=10,
        start_time=start_time,
        save_after_start_minutes=45  # 开始后45分钟保存
    )

def create_grpo_strategy(end_time: str, start_time: str):
    """创建GRPO任务专用策略"""
    return WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=15,
        start_time=start_time,
        save_after_start_minutes=30  # GRPO任务开始后30分钟保存
    )

def create_dpo_strategy(end_time: str, start_time: str):
    """创建DPO任务专用策略"""
    return WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=8,
        start_time=start_time,
        save_after_start_minutes=20  # DPO任务开始后20分钟保存
    )

def create_instruct_strategy(end_time: str, start_time: str):
    """创建Instruct任务专用策略"""
    return WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=5,
        start_time=start_time,
        save_after_start_minutes=15  # Instruct任务开始后15分钟保存
    )

# 根据训练时间长度选择策略
def get_strategy_by_training_time(hours_to_complete: float, end_time: str, start_time: str):
    """根据训练时间长度选择合适的策略"""
    
    if hours_to_complete <= 0.5:  # 30分钟以内
        return create_early_save_strategy(end_time, start_time)
    elif hours_to_complete <= 1.0:  # 1小时以内
        return create_mid_save_strategy(end_time, start_time)
    elif hours_to_complete <= 2.0:  # 2小时以内
        return create_late_save_strategy(end_time, start_time)
    else:  # 2小时以上
        return create_dual_save_strategy(end_time, start_time)

# 根据任务类型选择策略
def get_strategy_by_task_type(task_type: str, end_time: str, start_time: str):
    """根据任务类型选择合适的策略"""
    
    if task_type == "GrpoTask":
        return create_grpo_strategy(end_time, start_time)
    elif task_type == "DpoTask":
        return create_dpo_strategy(end_time, start_time)
    elif task_type == "InstructTextTask":
        return create_instruct_strategy(end_time, start_time)
    else:
        # 默认策略
        return create_mid_save_strategy(end_time, start_time)

# 使用示例
if __name__ == "__main__":
    # 模拟时间
    start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end_time = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    
    print("=== 开始时间后保存策略示例 ===")
    
    # 1. 早期保存策略
    early_handler = create_early_save_strategy(end_time, start_time)
    print(f"早期保存策略: 开始后{early_handler.save_after_start_minutes}分钟保存")
    
    # 2. 中期保存策略
    mid_handler = create_mid_save_strategy(end_time, start_time)
    print(f"中期保存策略: 开始后{mid_handler.save_after_start_minutes}分钟保存")
    
    # 3. 晚期保存策略
    late_handler = create_late_save_strategy(end_time, start_time)
    print(f"晚期保存策略: 开始后{late_handler.save_after_start_minutes}分钟保存")
    
    # 4. 双重保存策略
    dual_handler = create_dual_save_strategy(end_time, start_time)
    print(f"双重保存策略: 开始后{dual_handler.save_after_start_minutes}分钟保存，结束前{dual_handler.save_before_remaining_time}分钟保存")
    
    # 5. 根据训练时间选择策略
    print("\n=== 根据训练时间选择策略 ===")
    for hours in [0.25, 0.5, 1.0, 2.0, 4.0]:
        strategy = get_strategy_by_training_time(hours, end_time, start_time)
        print(f"训练时间 {hours} 小时: 开始后{strategy.save_after_start_minutes}分钟保存")
    
    # 6. 根据任务类型选择策略
    print("\n=== 根据任务类型选择策略 ===")
    task_types = ["GrpoTask", "DpoTask", "InstructTextTask"]
    for task_type in task_types:
        strategy = get_strategy_by_task_type(task_type, end_time, start_time)
        print(f"{task_type}: 开始后{strategy.save_after_start_minutes}分钟保存")
    
    print("\n=== 策略选择建议 ===")
    print("1. 短时间训练 (< 30分钟): 使用早期保存策略")
    print("2. 中等时间训练 (30分钟-1小时): 使用中期保存策略")
    print("3. 长时间训练 (1-2小时): 使用晚期保存策略")
    print("4. 超长时间训练 (> 2小时): 使用双重保存策略")
    print("5. GRPO任务: 使用专用GRPO策略")
    print("6. DPO任务: 使用专用DPO策略")
    print("7. Instruct任务: 使用专用Instruct策略") 