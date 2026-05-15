import subprocess
import time
import torch


def get_gpu_processes(gpu_id: int) -> list:
    """获取指定 GPU 上正在运行的进程信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", str(gpu_id)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        processes = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    processes.append({
                        "pid": parts[0],
                        "name": parts[1],
                        "memory": parts[2]
                    })
        return processes
    except Exception as e:
        print(f"警告: 无法获取 GPU 进程信息: {e}")
        return []


def is_gpu_available(gpu_id: int) -> bool:
    """检查指定 GPU 是否可用（无其他进程占用）"""
    processes = get_gpu_processes(gpu_id)
    return len(processes) == 0


def get_gpu_usage(gpu_id: int) -> dict:
    """获取指定 GPU 的使用情况"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return None
        
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 4:
            return {
                "memory_total": float(parts[0]),
                "memory_used": float(parts[1]),
                "memory_free": float(parts[2]),
                "utilization": float(parts[3])
            }
        return None
    except Exception as e:
        print(f"警告: 无法获取 GPU 使用情况: {e}")
        return None


def wait_for_gpu(gpu_id: int, max_wait_minutes: int = 60, check_interval: int = 30) -> bool:
    """等待 GPU 变为可用状态
    
    Args:
        gpu_id: GPU ID
        max_wait_minutes: 最大等待时间（分钟）
        check_interval: 检查间隔（秒）
    
    Returns:
        bool: GPU 是否变为可用
    """
    max_wait_seconds = max_wait_minutes * 60
    elapsed = 0
    
    while elapsed < max_wait_seconds:
        processes = get_gpu_processes(gpu_id)
        
        if len(processes) == 0:
            print(f"GPU {gpu_id} 现在可用!")
            return True
        
        gpu_usage = get_gpu_usage(gpu_id)
        usage_info = ""
        if gpu_usage:
            usage_info = f" [显存: {gpu_usage['memory_used']:.0f}/{gpu_usage['memory_total']:.0f} MB, 利用率: {gpu_usage['utilization']:.0f}%]"
        
        print(f"GPU {gpu_id} 被占用，等待中...{usage_info}")
        print(f"  占用进程:")
        for proc in processes:
            print(f"    - PID: {proc['pid']}, 名称: {proc['name']}, 显存: {proc['memory']}")
        
        print(f"  等待 {check_interval} 秒后重新检查...")
        time.sleep(check_interval)
        elapsed += check_interval
    
    print(f"警告: 等待 GPU {gpu_id} 超时 ({max_wait_minutes} 分钟)")
    return False


def select_free_gpu(min_free_memory_mb: int = 1000) -> int:
    """自动选择一个空闲的 GPU
    
    Args:
        min_free_memory_mb: 最小可用显存（MB）
    
    Returns:
        int: 可用的 GPU ID，如果没有可用 GPU 则返回 -1
    """
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA 设备")
        return -1
    
    num_gpus = torch.cuda.device_count()
    print(f"\n检测到 {num_gpus} 个 GPU，正在检查可用性...")
    
    for gpu_id in range(num_gpus):
        processes = get_gpu_processes(gpu_id)
        gpu_usage = get_gpu_usage(gpu_id)
        
        if gpu_usage and gpu_usage["memory_free"] >= min_free_memory_mb:
            if len(processes) == 0:
                print(f"GPU {gpu_id}: 空闲 ✓ [显存: {gpu_usage['memory_free']:.0f} MB 可用]")
                return gpu_id
            else:
                print(f"GPU {gpu_id}: 被占用 (PID: {', '.join([p['pid'] for p in processes])})")
        else:
            print(f"GPU {gpu_id}: 显存不足或不可用")
    
    return -1


def check_gpu_before_use(gpu_id: int, wait: bool = True, max_wait_minutes: int = 2) -> bool:
    """在使用 GPU 前进行检查
    
    Args:
        gpu_id: 要检查的 GPU ID
        wait: 是否等待 GPU 变为可用
        max_wait_minutes: 最大等待时间（分钟）
    
    Returns:
        bool: GPU 是否可用
    """
    print(f"\n检查 GPU {gpu_id} 可用性...")
    
    processes = get_gpu_processes(gpu_id)
    
    if len(processes) == 0:
        print(f"GPU {gpu_id} 空闲，可以使用")
        return True
    
    print(f"警告: GPU {gpu_id} 正在被其他进程使用!")
    print(f"占用进程:")
    for proc in processes:
        print(f"  - PID: {proc['pid']}, 名称: {proc['name']}, 显存: {proc['memory']}")
    
    if not wait:
        print("请手动选择其他 GPU 或等待当前进程释放 GPU")
        return False
    
    print(f"\n将等待 GPU 变为可用（最多 {max_wait_minutes} 分钟）...")
    return wait_for_gpu(gpu_id, max_wait_minutes=max_wait_minutes)
