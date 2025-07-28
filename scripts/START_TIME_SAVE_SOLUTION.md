# 训练开始时间后保存功能解决方案

## 功能概述

在原有的"结束时间前5分钟保存"的基础上，新增了"训练开始时间后若干分钟保存"的逻辑，实现了双重保存机制。

## 主要改进

### 1. 新增时间检查函数

```python
def check_time_since_start_greater_than_minutes(start_time: str, minutes: int) -> bool:
    """检查从开始时间到现在是否超过了指定分钟数"""
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    start_time = start_time.replace(tzinfo=timezone.utc)
    now = datetime.datetime.now(timezone.utc)
    time_diff = now - start_time
    result = time_diff.total_seconds() > minutes * 60
    return result
```

### 2. 增强 WhenToEvalHandler

```python
class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, 
                 start_time: str = None, save_after_start_minutes: int = None):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.start_time = start_time
        self.save_after_start_minutes = save_after_start_minutes
        self.start_eval_triggered = False  # 防止重复触发

    def __call__(self, global_step: int) -> dict:
        # 1. 检查开始时间触发
        if (self.start_time and self.save_after_start_minutes and 
            not self.start_eval_triggered and 
            check_time_since_start_greater_than_minutes(self.start_time, self.save_after_start_minutes)):
            self.start_eval_triggered = True
            return {"eval": True, "reason": "start_time"}
        
        # 2. 检查结束时间触发
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}

        return {"eval": False, "reason": "none"}
```

### 3. 修改 patched_init 函数

```python
def patched_init(self, *trainer_args, **trainer_kwargs):
    # ... 配置代码 ...
    
    # 记录训练开始时间
    start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if original_task_type == TaskType.GRPOTASK.value:
        # GRPO 任务：开始后30分钟保存，结束前15分钟保存
        when_to_eval_handler = WhenToEvalHandler(
            end_time=end_time, 
            save_before_remaining_time=15,
            start_time=start_time,
            save_after_start_minutes=30
        )
    else:
        # 其他任务：开始后15分钟保存，结束前5分钟保存
        when_to_eval_handler = WhenToEvalHandler(
            end_time=end_time, 
            save_before_remaining_time=5,
            start_time=start_time,
            save_after_start_minutes=15
        )
```

## 保存策略配置

### 默认策略

| 任务类型 | 开始后保存时间 | 结束前保存时间 | 说明 |
|---------|---------------|---------------|------|
| GRPO任务 | 30分钟 | 15分钟 | 适合长时间训练 |
| DPO任务 | 15分钟 | 5分钟 | 适合中等时间训练 |
| Instruct任务 | 15分钟 | 5分钟 | 适合中等时间训练 |

### 自定义策略示例

#### 1. 早期保存策略
```python
WhenToEvalHandler(
    end_time=end_time,
    save_before_remaining_time=5,
    start_time=start_time,
    save_after_start_minutes=5  # 开始后5分钟就保存
)
```

#### 2. 中期保存策略
```python
WhenToEvalHandler(
    end_time=end_time,
    save_before_remaining_time=10,
    start_time=start_time,
    save_after_start_minutes=30  # 开始后30分钟保存
)
```

#### 3. 晚期保存策略
```python
WhenToEvalHandler(
    end_time=end_time,
    save_before_remaining_time=15,
    start_time=start_time,
    save_after_start_minutes=60  # 开始后60分钟保存
)
```

#### 4. 双重保存策略
```python
WhenToEvalHandler(
    end_time=end_time,
    save_before_remaining_time=10,
    start_time=start_time,
    save_after_start_minutes=45  # 开始后45分钟保存
)
```

## 策略选择建议

### 根据训练时间长度

- **短时间训练 (< 30分钟)**：使用早期保存策略
- **中等时间训练 (30分钟-1小时)**：使用中期保存策略
- **长时间训练 (1-2小时)**：使用晚期保存策略
- **超长时间训练 (> 2小时)**：使用双重保存策略

### 根据任务类型

- **GRPO任务**：使用专用GRPO策略（开始后30分钟）
- **DPO任务**：使用专用DPO策略（开始后20分钟）
- **Instruct任务**：使用专用Instruct策略（开始后15分钟）

### 根据资源限制

- **计算资源充足**：可以设置更频繁的保存
- **计算资源有限**：减少保存频率
- **存储空间充足**：增加保存次数
- **存储空间有限**：减少保存次数

## 日志输出示例

### 开始时间触发
```
***ALERT: Training has been running for 15 minutes, triggering eval & save
*** current time: 2024-01-15 10:15:00+00:00 start_time: 2024-01-15 10:00:00+00:00 time_diff: 0:15:00
```

### 结束时间触发
```
***ALERT: The time is about to run out need to eval & save the model
*** current time: 2024-01-15 12:55:00+00:00 end_time: 2024-01-15 13:00:00+00:00 time_diff: 0:05:00
```

## 测试验证

### 运行测试脚本
```bash
python scripts/test_start_time_eval.py
```

### 测试内容
1. **开始时间检查**：验证时间计算是否正确
2. **触发逻辑**：验证是否在正确时间触发
3. **重复触发**：验证是否防止重复触发
4. **策略配置**：验证不同策略的配置

## 优势特点

### 1. 双重保护
- **开始时间触发**：确保训练早期就有检查点
- **结束时间触发**：确保训练结束前有最终检查点

### 2. 灵活配置
- 可根据任务类型调整保存时间
- 可根据训练时间长度选择策略
- 可根据资源限制优化配置

### 3. 防止重复
- 每个触发条件只会执行一次
- 避免不必要的重复保存
- 节省计算和存储资源

### 4. 详细日志
- 提供详细的触发日志
- 便于调试和监控
- 记录保存原因和时间

## 预期效果

修改后，系统将支持：

1. ✅ **双重保存机制**：开始时间和结束时间双重触发
2. ✅ **灵活时间配置**：可根据需要调整保存时间
3. ✅ **任务类型适配**：不同任务类型使用不同策略
4. ✅ **防止重复触发**：确保每个条件只触发一次
5. ✅ **详细日志记录**：提供完整的触发和保存日志
6. ✅ **资源优化**：避免过度保存，节省资源

这样就实现了在训练开始时间后若干分钟保存的功能，与原有的结束时间前保存形成了完整的双重保护机制！ 