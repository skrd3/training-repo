#!/usr/bin/env python3
"""
测试 Hugging Face 上传功能的脚本
"""

import os
import tempfile
import shutil
from huggingface_hub import HfApi, create_repo, upload_folder

def test_huggingface_upload():
    """测试 Hugging Face 上传功能"""
    
    # 获取环境变量
    hub_token = os.environ.get("HUGGINGFACE_TOKEN")
    hf_username = os.environ.get("HUGGINGFACE_USERNAME", "rayonlabs")
    
    if not hub_token:
        print("错误: 未设置 HUGGINGFACE_TOKEN 环境变量")
        return False
    
    # 创建测试目录和文件
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试文件
        test_file_path = os.path.join(temp_dir, "test_model.bin")
        with open(test_file_path, "w") as f:
            f.write("This is a test model file")
        
        # 创建配置文件
        config_file_path = os.path.join(temp_dir, "config.json")
        with open(config_file_path, "w") as f:
            f.write('{"model_type": "test", "test": true}')
        
        # 创建 README
        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Test Model\n\nThis is a test model for upload verification.")
        
        # 生成仓库名称
        import uuid
        repo_name = f"test-upload-{uuid.uuid4().hex[:8]}"
        hub_model_id = f"{hf_username}/{repo_name}"
        
        try:
            print(f"开始上传测试模型到: {hub_model_id}")
            
            # 创建仓库
            create_repo(
                repo_id=hub_model_id,
                repo_type="model",
                token=hub_token,
                private=False
            )
            print(f"成功创建仓库: {hub_model_id}")
            
            # 上传文件
            upload_folder(
                folder_path=temp_dir,
                repo_id=hub_model_id,
                repo_type="model",
                token=hub_token,
                commit_message="Test upload from script"
            )
            print(f"成功上传文件到: {hub_model_id}")
            
            # 验证上传
            hf_api = HfApi(token=hub_token)
            repo_info = hf_api.repo_info(repo_id=hub_model_id, repo_type="model")
            print(f"仓库信息: {repo_info}")
            
            return True
            
        except Exception as e:
            print(f"上传失败: {e}")
            return False

if __name__ == "__main__":
    success = test_huggingface_upload()
    if success:
        print("✅ 上传测试成功!")
    else:
        print("❌ 上传测试失败!") 