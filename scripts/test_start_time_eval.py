#!/usr/bin/env python3
"""
测试开始时间后保存功能的脚本
"""

import datetime
from datetime import timezone, timedelta
from customized_trainer import WhenToEvalHandler, check_time_since_start_greater_than_minutes

def test_start_time_eval():
    """测试开始时间后评估功能"""
    
    # 模拟开始时间（5分钟前）
    start_time = (datetime.datetime.now(timezone.utc) - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"模拟开始时间: {start_time}")
    
    # 模拟结束时间（2小时后）
    end_time = (datetime.datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"模拟结束时间: {end_time}")
    
    # 测试不同的时间间隔
    test_cases = [
        {"minutes": 3, "expected": True, "description": "3分钟前开始，应该触发"},
        {"minutes": 7, "expected": False, "description": "7分钟前开始，不应该触发"},
        {"minutes": 10, "expected": True, "description": "10分钟前开始，应该触发"}
    ]
    
    print("\n=== 测试开始时间检查 ===")
    for case in test_cases:
        # 调整开始时间
        adjusted_start_time = (datetime.datetime.now(timezone.utc) - timedelta(minutes=case["minutes"])).strftime("%Y-%m-%d %H:%M:%S")
        result = check_time_since_start_greater_than_minutes(adjusted_start_time, 5)
        status = "✅" if result == case["expected"] else "❌"
        print(f"{status} {case['description']}: {result} (期望: {case['expected']})")
    
    # 测试 WhenToEvalHandler
    print("\n=== 测试 WhenToEvalHandler ===")
    
    # 测试1：开始后3分钟触发
    handler1 = WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=15,
        start_time=start_time,
        save_after_start_minutes=3
    )
    
    result1 = handler1(global_step=100)
    print(f"开始后3分钟触发测试: {result1}")
    
    # 测试2：开始后10分钟触发
    handler2 = WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=15,
        start_time=start_time,
        save_after_start_minutes=10
    )
    
    result2 = handler2(global_step=100)
    print(f"开始后10分钟触发测试: {result2}")
    
    # 测试3：只基于结束时间触发
    handler3 = WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=15
        # 不设置 start_time 和 save_after_start_minutes
    )
    
    result3 = handler3(global_step=100)
    print(f"只基于结束时间触发测试: {result3}")
    
    # 测试重复触发
    print("\n=== 测试重复触发 ===")
    handler4 = WhenToEvalHandler(
        end_time=end_time,
        save_before_remaining_time=15,
        start_time=start_time,
        save_after_start_minutes=3
    )
    
    # 第一次调用
    result4_1 = handler4(global_step=100)
    print(f"第一次调用: {result4_1}")
    
    # 第二次调用（应该不会触发，因为已经触发过了）
    result4_2 = handler4(global_step=200)
    print(f"第二次调用: {result4_2}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_start_time_eval() 