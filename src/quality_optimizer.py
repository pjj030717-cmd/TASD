import math 
from typing import List, Tuple, Optional, Callable 

# ============================================================ 
#  算法 1：质量损失上界计算 
# ============================================================ 
def quality_bound(epsilon: float, k: int, alpha: float = 1.0) -> float: 
    """ 
    计算质量损失的理论上界（总变差距离 TV）。 
    
    参数: 
        epsilon : KL 散度宽容度阈值 
        k       : 草稿长度 
        alpha   : 缩放系数，默认 1.0（Pinsker 不等式上界），可经验校准 
        
    返回: 
        总变差距离上界 
    """ 
    if epsilon < 0: 
        raise ValueError("epsilon 必须非负") 
    return alpha * k * math.sqrt(epsilon / 2.0) 


# ============================================================ 
#  算法 2：约束优化求解器 
# ============================================================ 
# 默认代价参数（示例，需根据实际硬件校准） 
DEFAULT_COST_DRAFT = 0.1      # C_D：草稿模型一次前向的归一化代价 
DEFAULT_COST_TARGET = 1.0     # C_T：目标模型并行验证一次的代价 
DEFAULT_COST_ROLLBACK = 0.2   # C_R：回滚重采样的额外代价 

# 最大宽容度（防止数值不稳定） 
EPSILON_MAX = 0.5 

# 回滚概率估计器类型 
RollbackEstimator = Callable[[float, int], float] 


def estimate_rollback_prob_independent(epsilon: float, k: int) -> float: 
    """ 
    基于独立假设的简单回滚概率估计： 
    每个位置接受概率 ≈ 1 - epsilon，则整轮不被拒绝的概率 ≈ (1 - epsilon)^k。 
    回滚概率 = 1 - (1 - epsilon)^k。 
    注意：这是非常粗略的近似，实际使用时建议用离线校准数据替换。 
    """ 
    safe_eps = min(epsilon, 0.99)  # 防止数值溢出 
    accept_all_prob = (1.0 - safe_eps) ** k 
    return 1.0 - accept_all_prob 


def solve_optimal( 
    delta_max: float, 
    K_max: int, 
    C_D: float = DEFAULT_COST_DRAFT, 
    C_T: float = DEFAULT_COST_TARGET, 
    C_R: float = DEFAULT_COST_ROLLBACK, 
    alpha: float = 1.0, 
    rollback_estimator: RollbackEstimator = estimate_rollback_prob_independent, 
    epsilon_max: float = EPSILON_MAX, 
) -> Tuple[float, int, float]: 
    """ 
    自动求解最优宽容度 ε* 和草稿长度 k*。 
    
    参数: 
        delta_max          : 用户指定的质量预算（TVD 上界） 
        K_max              : 最大草稿长度 
        C_D, C_T, C_R      : 代价参数（需校准） 
        alpha              : 质量上界缩放系数 
        rollback_estimator : 回滚概率估计函数 
        epsilon_max        : 宽容度上限 
        
    返回: 
        (epsilon_star, k_star, best_cost) 
    """ 
    best_cost = float('inf') 
    epsilon_star = 0.0 
    k_star = 1 

    for k in range(1, K_max + 1): 
        # 1. 由质量预算边界显式计算 ε_k 
        epsilon_k = 2.0 * (delta_max / (alpha * k)) ** 2 
        
        # 2. 裁剪到允许范围 
        epsilon_k = min(epsilon_k, epsilon_max) 
        
        # 3. 计算期望代价 
        p_roll = rollback_estimator(epsilon_k, k) 
        cost = k * C_D + C_T + p_roll * C_R 
        
        # 4. 更新最优 
        if cost < best_cost: 
            best_cost = cost 
            epsilon_star = epsilon_k 
            k_star = k 

    return epsilon_star, k_star, best_cost 


# ============================================================ 
#  辅助：基于校准数据的回滚概率查表（推荐方法） 
# ============================================================ 
class CalibratedRollbackEstimator: 
    """ 
    使用离线校准数据构建回滚概率 p_roll(ε, k) 的估计器。 
    校准时，针对不同 (ε, k) 组合运行多轮推测解码，统计回滚频率。 
    实际使用时进行插值或最近邻查找。 
    """ 
    def __init__(self, calibration_data: dict): 
        """ 
        calibration_data: dict, key = (epsilon, k), value = p_roll 
        """ 
        self.data = calibration_data 

    def estimate(self, epsilon: float, k: int) -> float: 
        # 简单实现：找最近邻 
        # 生产环境可替换为更精细的插值 
        key = (round(epsilon, 3), k) 
        if key in self.data: 
            return self.data[key] 
        # 粗略近似 
        return estimate_rollback_prob_independent(epsilon, k) 


# ============================================================ 
#  示例用法 
# ============================================================ 
if __name__ == "__main__": 
    # 用户要求质量预算（例如 TVD ≤ 0.03） 
    delta = 0.03 
    K_max = 10 

    eps_star, k_star, cost = solve_optimal(delta_max=delta, K_max=K_max) 
    print(f"最优宽容度 ε* : {eps_star:.4f}") 
    print(f"最优草稿长度 k* : {k_star}") 
    print(f"最小期望代价 : {cost:.4f}") 
    print(f"对应质量损失上界 : {quality_bound(eps_star, k_star):.4f}")
