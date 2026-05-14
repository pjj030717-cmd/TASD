import os
import subprocess
import argparse


def download_model(model_name, model_id, save_dir):
    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"模型 ID: {model_id}")
    print(f"保存路径: {save_dir}")
    print(f"{'='*60}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = [
        "hf", "download",
        model_id,
        "--local-dir", save_dir,
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, env={**os.environ, "HF_ENDPOINT": "https://hf-mirror.com"})
    
    if result.returncode == 0:
        print(f"\n✓ {model_name} 下载完成!")
    else:
        print(f"\n✗ {model_name} 下载失败!")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="下载量化模型")
    parser.add_argument("--base_dir", type=str, default="./models",
                        help="模型保存的基础目录")
    args = parser.parse_args()
    
    os.makedirs(args.base_dir, exist_ok=True)
    
    models = [
        {
            "name": "目标模型: Qwen2.5-72B-Instruct-AWQ (4-bit)",
            "id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "dir": os.path.join(args.base_dir, "Qwen2.5-72B-Instruct-AWQ"),
        },
        {
            "name": "草稿模型: Qwen2.5-7B-Instruct-GPTQ-Int4 (4-bit)",
            "id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "dir": os.path.join(args.base_dir, "Qwen2.5-7B-Instruct-GPTQ-Int4"),
        },
    ]
    
    success_count = 0
    for model_info in models:
        if download_model(model_info["name"], model_info["id"], model_info["dir"]):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"下载完成! 成功: {success_count}/{len(models)}")
    print(f"{'='*60}")
    print(f"\n模型路径:")
    for model_info in models:
        print(f"  {model_info['name']}: {model_info['dir']}")


if __name__ == "__main__":
    main()
