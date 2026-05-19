import math
from typing import Tuple, Callable, List


# ============================================================
#  TASD 核心模块：质量预算驱动的宽松验证求解器
# ============================================================

def quality_bound(epsilon: float, k: int, alpha: float = 1.0) -> float:
    """
    质量损失上界 (Theorem 1)
    
    参数:
        epsilon : KL 散度宽容度阈值
        k       : 草稿长度
        alpha   : 缩放系数（Pinsker 不等式上界）
    
    返回:
        总变差距离上界 (TVD)
    """
    if epsilon < 0:
        raise ValueError("epsilon 必须非负")
    return alpha * k * math.sqrt(epsilon / 2.0)


def estimate_rollback_prob(epsilon: float, k: int) -> float:
    """
    回滚概率估计 (独立假设近似)
    
    参数:
        epsilon : KL 散度宽容度
        k       : 草稿长度
    
    返回:
        回滚概率估计值
    """
    safe_eps = min(epsilon, 0.99)
    return 1.0 - (1.0 - safe_eps) ** k


def solve_optimal(
    delta_max: float,
    K_max: int,
    C_D: float = 0.1,
    C_T: float = 1.0,
    C_R: float = 1.5,
    alpha: float = 1.0,
    epsilon_max: float = 0.5,
) -> Tuple[float, int, float]:
    """
    自动求解最优 (ε*, k*) (Theorem 2-4)
    
    参数:
        delta_max   : 质量预算（TVD 上界）
        K_max       : 最大草稿长度
        C_D         : 草稿模型前向代价（默认 0.1，归一化值）
        C_T         : 目标模型前向代价（归一化为 1.0）
        C_R         : 回滚代价（默认 1.5，包含重采样和 KV cache 重建开销）
        alpha       : 质量上界缩放系数
        epsilon_max : 宽容度上限
    
    返回:
        (epsilon_star, k_star, best_cost)
    """
    best_cost = float('inf')
    epsilon_star, k_star = 0.0, 1
    
    for k in range(1, K_max + 1):
        epsilon_k = 2.0 * (delta_max / (alpha * k)) ** 2
        epsilon_k = min(epsilon_k, epsilon_max)
        
        p_roll = estimate_rollback_prob(epsilon_k, k)
        cost = k * C_D + C_T + p_roll * C_R
        
        if cost < best_cost:
            best_cost = cost
            epsilon_star = epsilon_k
            k_star = k
    
    return epsilon_star, k_star, best_cost


# ============================================================
#  TASD 宽松验证逻辑：替代标准推测解码的逐 token 判断
# ============================================================

def tasd_relaxed_accept(
    target_dist: List[float],
    draft_dist: List[float],
    epsilon: float,
    smoothing: float = 1e-10,
) -> bool:
    """
    TASD 宽松接受判断：D_KL(target || draft) ≤ ε 则接受
    
    参数:
        target_dist : 目标模型概率分布 (|V| 维)
        draft_dist  : 草稿模型概率分布 (|V| 维)
        epsilon     : 最优宽容度
        smoothing   : 平滑项（防止 log(0)）
    
    返回:
        True 表示接受 draft token，False 表示拒绝
    """
    def kl_divergence(p: List[float], q: List[float]) -> float:
        p_smooth = [(x + smoothing) for x in p]
        q_smooth = [(x + smoothing) for x in q]
        
        p_sum = sum(p_smooth)
        q_sum = sum(q_smooth)
        
        return sum(
            p_smooth[i] / p_sum * math.log((p_smooth[i] / p_sum) / (q_smooth[i] / q_sum))
            for i in range(len(p))
        )
    
    return kl_divergence(target_dist, draft_dist) <= epsilon


# ============================================================
#  示例用法
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("  TASD 质量预算驱动求解器")
    print("="*60)
    
    delta = 0.05
    K_max = 10
    
    eps_star, k_star, cost = solve_optimal(delta_max=delta, K_max=K_max)
    q_bound = quality_bound(eps_star, k_star)
    
    print(f"\n输入:")
    print(f"  质量预算 δ = {delta}")
    print(f"  最大草稿长度 K_max = {K_max}")
    
    print(f"\n输出:")
    print(f"  最优宽容度 ε* = {eps_star:.4f}")
    print(f"  最优草稿长度 k* = {k_star}")
    print(f"  最小期望代价 = {cost:.4f}")
    print(f"  质量损失上界 = {q_bound:.4f}")
    
    print(f"\n宽松验证测试:")
    target_p = [0.7, 0.2, 0.05, 0.05]
    draft_p = [0.65, 0.25, 0.05, 0.05]
    
    accepted = tasd_relaxed_accept(target_p, draft_p, eps_star)
    print(f"  Target 分布: {target_p}")
    print(f"  Draft 分布:  {draft_p}")
    print(f"  宽容度 ε:    {eps_star:.4f}")
    print(f"  接受结果:    {'✓ 接受' if accepted else '✗ 拒绝'}")
