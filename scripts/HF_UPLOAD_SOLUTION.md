# Hugging Face 上传问题解决方案

## 问题分析

### 当前问题
模型训练完成后保存到本地，但没有自动上传到 Hugging Face Hub。

### 根本原因
1. **CustomEvalSaveCallback 缺少上传逻辑**：当前的 `on_save` 方法只复制文件到本地目录，没有上传到 HF
2. **配置参数传递问题**：`hub_model_id` 和 `hub_token` 没有正确传递给回调函数
3. **上传时机问题**：需要在最佳模型保存时立即上传

## 解决方案

### 1. 修改 CustomEvalSaveCallback

在 `scripts/customized_trainer.py` 中：

```python
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        hub_model_id: str = None,  # 新增
        hub_token: str = None,     # 新增
    ):
        # ... 现有代码 ...
        self.hub_model_id = hub_model_id
        self.hub_token = hub_token or os.environ.get("HUGGINGFACE_TOKEN")

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # ... 现有保存逻辑 ...
        
        # 新增：上传到 Hugging Face
        if self.hub_model_id and self.hub_token:
            try:
                print(f"Uploading model to Hugging Face: {self.hub_model_id}", flush=True)
                self._upload_to_huggingface()
            except Exception as e:
                print(f"Failed to upload to Hugging Face: {e}", flush=True)
        else:
            print("Skipping Hugging Face upload: missing hub_model_id or hub_token", flush=True)

    def _upload_to_huggingface(self):
        """上传模型到 Hugging Face Hub"""
        try:
            hf_api = HfApi(token=self.hub_token)
            
            # 检查仓库是否存在，如果不存在则创建
            try:
                hf_api.repo_info(repo_id=self.hub_model_id, repo_type="model")
                print(f"Repository {self.hub_model_id} already exists", flush=True)
            except:
                print(f"Creating repository {self.hub_model_id}", flush=True)
                create_repo(
                    repo_id=self.hub_model_id,
                    repo_type="model",
                    token=self.hub_token,
                    private=False
                )
            
            # 上传模型文件
            print(f"Uploading model files from {self.submission_dir} to {self.hub_model_id}", flush=True)
            upload_folder(
                folder_path=self.submission_dir,
                repo_id=self.hub_model_id,
                repo_type="model",
                token=self.hub_token,
                commit_message=f"Upload best model checkpoint (step {self.best_checkpoint_info['step']}, loss {self.best_checkpoint_info['loss']:.4f})"
            )
            print(f"Successfully uploaded model to {self.hub_model_id}", flush=True)
            
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}", flush=True)
            raise
```

### 2. 修改 patched_init 函数

在 `scripts/text_trainer.py` 中：

```python
def patched_init(self, *trainer_args, **trainer_kwargs):
    print("************* patching Trainer.__init__", flush=True)
    callbacks = trainer_kwargs.get("callbacks", [])
    
    # 获取 Hugging Face 配置
    hub_model_id = None
    hub_token = None
    
    # 从外部变量中获取配置
    hf_username = args.huggingface_username or os.environ.get("HUGGINGFACE_USERNAME", "rayonlabs")
    repo_name = args.expected_repo_name or str(uuid.uuid4())
    hub_model_id = f"{hf_username}/{repo_name}"
    hub_token = args.huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")
    
    if original_task_type == TaskType.GRPOTASK.value:
        when_to_eval_handler = WhenToEvalHandler(end_time, save_before_remaining_time=15)
        callbacks.append(GRPOCustomEvalSaveCallback(when_to_eval_handler, submission_dir, output_dir, original_model_name, hub_model_id, hub_token))
    else:
        when_to_eval_handler = WhenToEvalHandler(end_time, save_before_remaining_time=5)
        callbacks.append(CustomEvalSaveCallback(when_to_eval_handler, submission_dir, output_dir, original_model_name, hub_model_id, hub_token))
    trainer_kwargs["callbacks"] = callbacks
    original_init(self, *trainer_args, **trainer_kwargs)
```

### 3. 添加必要的导入

在 `scripts/customized_trainer.py` 顶部添加：

```python
from huggingface_hub import HfApi, create_repo, upload_folder
```

## 配置要求

### 环境变量
确保设置以下环境变量：
```bash
export HUGGINGFACE_TOKEN="your_hf_token_here"
export HUGGINGFACE_USERNAME="your_username_here"
```

### 命令行参数
确保传递以下参数：
```bash
--huggingface_token "your_token"
--huggingface_username "your_username"
--expected_repo_name "your_repo_name"  # 可选
```

## 测试验证

### 1. 运行测试脚本
```bash
python scripts/test_upload.py
```

### 2. 检查上传日志
训练过程中应该看到类似日志：
```
Uploading model to Hugging Face: username/repo_name
Creating repository username/repo_name
Uploading model files from /path/to/submission_dir to username/repo_name
Successfully uploaded model to username/repo_name
```

## 故障排除

### 常见问题

1. **Token 无效**
   - 检查 `HUGGINGFACE_TOKEN` 是否正确设置
   - 确保 token 有写入权限

2. **仓库已存在**
   - 系统会自动检测并使用现有仓库
   - 确保有仓库的写入权限

3. **网络问题**
   - 检查网络连接
   - 确保可以访问 huggingface.co

4. **权限问题**
   - 确保 token 有创建仓库的权限
   - 检查用户名是否正确

### 调试步骤

1. **检查环境变量**
   ```bash
   echo $HUGGINGFACE_TOKEN
   echo $HUGGINGFACE_USERNAME
   ```

2. **测试连接**
   ```bash
   python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"
   ```

3. **查看详细日志**
   在训练脚本中添加更多调试信息

## 预期效果

修改后，当模型训练完成并保存最佳检查点时，系统将：

1. ✅ 保存模型到本地目录
2. ✅ 自动上传到 Hugging Face Hub
3. ✅ 创建公开的模型仓库
4. ✅ 包含模型元数据和性能信息
5. ✅ 提供可访问的模型链接

这样就解决了模型保存但不上传的问题。 