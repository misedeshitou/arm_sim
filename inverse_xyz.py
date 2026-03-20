import numpy as np
import math

# ==========================================
# 1. 机械臂物理参数配置 (MCU 上的 const 数组)
# ==========================================
A_PARAMS     = [0.0,       0.290, 0.0,      0.03,      0.0,       0.0]
D_PARAMS     = [0.0,       0.0,  -0.10375,  0.410147,  0.0,       0.211]
ALPHA_PARAMS = [math.pi/2, 0.0,  -math.pi/2, math.pi/2, -math.pi/2, 0.0]
THETA_OFFSET = [math.pi,   0.0,   0.0,      math.pi,   math.pi/2, math.pi]

# ==========================================
# 2. 基础数学工具
# ==========================================
def calc_dh_matrix(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])

# ==========================================
# 3. 核心逆解算法 (带自适应误差停止)
# ==========================================
def calculate_ik_adaptive(x, y, z, roll, pitch, yaw, config=(1, -1, 1), max_iters=50, tol=1e-4):
    """
    输入:
        x, y, z, roll, pitch, yaw: 目标位姿
        max_iters: 单片机最大允许迭代次数 (防止死循环)
        tol: 两次迭代间的角度差阈值 (弧度)
    """
    sign_shoulder, sign_elbow, sign_wrist = config
    
    # 1. 目标矩阵构建
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    
    R_end = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    P_end = np.array([x, y, z])
    Z_dir = R_end[:, 2]
    P_wc = P_end - D_PARAMS[5] * Z_dir
    
    # 2. 开始自适应迭代求解
    theta4_math = 0.0  
    theta = [0.0] * 6
    converged = False
    
    for iteration in range(max_iters):
        theta4_old = theta4_math # 记录上一次的 theta4
        
        # --- 核心计算逻辑 ---
        L = math.sqrt(D_PARAMS[3]**2 + (A_PARAMS[3] * math.cos(theta4_math))**2)
        # psi = math.atan2(A_PARAMS[3] * math.cos(theta4_math), D_PARAMS[3])
        psi = math.atan2(D_PARAMS[3], A_PARAMS[3] * math.cos(theta4_math))
        D_offset = D_PARAMS[2] - A_PARAMS[3] * math.sin(theta4_math)
        
        R_xy = math.sqrt(P_wc[0]**2 + P_wc[1]**2)
        if R_xy < abs(D_offset):
            raise ValueError(f"迭代第 {iteration} 次时进入死区！目标点过近。")
        
        phi = math.atan2(P_wc[1], P_wc[0])
        beta = math.asin(D_offset / R_xy) if sign_shoulder > 0 else math.pi - math.asin(D_offset / R_xy)
        # theta[0] = phi - beta
        theta[0] = phi + beta
        
        X = P_wc[0] * math.cos(theta[0]) + P_wc[1] * math.sin(theta[0])
        Y = P_wc[2]
        
        a2 = A_PARAMS[1]
        cos_gamma = (X**2 + Y**2 - a2**2 - L**2) / (2 * a2 * L)
        cos_gamma = max(-1.0, min(1.0, cos_gamma)) # 钳位保护，防止 acos 报错
        
        gamma = sign_elbow * math.acos(cos_gamma)
        theta[2] = gamma - psi
        
        alpha_angle = math.atan2(Y, X)
        beta_angle = math.atan2(L * math.sin(gamma), a2 + L * math.cos(gamma))
        theta[1] = alpha_angle - beta_angle
        
        # 计算前三轴矩阵
        T03 = np.eye(4)
        for i in range(3):
            T03 = T03 @ calc_dh_matrix(theta[i], D_PARAMS[i], A_PARAMS[i], ALPHA_PARAMS[i])
            
        R03 = T03[0:3, 0:3]
        R36 = R03.T @ R_end
        
        r13, r23, r33 = R36[0, 2], R36[1, 2], R36[2, 2]
        r31, r32 = R36[2, 0], R36[2, 1]
        
        sin_t5 = sign_wrist * math.sqrt(max(0.0, 1.0 - r33**2))
        theta[4] = math.atan2(sin_t5, r33)
        
        if abs(sin_t5) > 1e-6:
            theta[3] = math.atan2(-r23, -r13)
            theta[5] = math.atan2(-r32, r31)
        else:
            theta[3] = theta4_math 
            theta[5] = math.atan2(R36[1, 0], R36[0, 0])
            
        theta4_math = theta[3]
        
        # --- 🌟 收敛判断 🌟 ---
        # 考虑到角度在 -pi 和 pi 之间可能会跳跃，用 atan2(sin, cos) 计算真实的角度差
        diff = abs(math.atan2(math.sin(theta4_math - theta4_old), math.cos(theta4_math - theta4_old)))
        
        if diff < tol:
            converged = True
            print(f"✅ 成功：在第 {iteration + 1} 次迭代后收敛！误差: {diff:.6e}")
            break

    if not converged:
        print(f"⚠️ 警告：达到了最大迭代次数 ({max_iters})，算法未收敛！")

    # 3. 减去 DH 表偏置并归一化
    q_out = [0.0] * 6
    for i in range(6):
        q_out[i] = theta[i] - THETA_OFFSET[i]
        q_out[i] = (q_out[i] + math.pi) % (2 * math.pi) - math.pi
        
    return q_out

def find_best_ik_solution(x, y, z, roll, pitch, yaw, current_angles):
    """
    遍历 8 种构型，寻找距离当前机械臂姿态最近的最优解
    输入:
        x, y, z, roll, pitch, yaw: 目标位姿
        current_angles: 机械臂当前 6 个电机的实际角度 (弧度列表)
    输出:
        最优电机角度列表, 选用的构型, 最小移动代价
    """
    # 机械臂的 8 种全部可能的姿态组合 (肩, 肘, 腕)
    configs = [
        ( 1,  1,  1), ( 1,  1, -1), 
        ( 1, -1,  1), ( 1, -1, -1),
        (-1,  1,  1), (-1,  1, -1), 
        (-1, -1,  1), (-1, -1, -1)
    ]
    
    best_angles = None
    best_config = None
    min_cost = float('inf') # 初始代价设为无穷大
    
    valid_solutions = 0 # 记录算出了几个有效解
    
    for cfg in configs:
        try:
            # 尝试当前构型求解 (如果你之前的代码里加了 print，这里可能会打印很多行)
            calc_angles = calculate_ik_adaptive(
                x, y, z, roll, pitch, yaw, 
                config=cfg, max_iters=50, tol=1e-4
            )
            valid_solutions += 1
            
            # --- 计算这个解与当前姿态的“移动代价” ---
            cost = 0.0
            for i in range(6):
                # 1. 算出目标角度与当前角度的差值
                diff = calc_angles[i] - current_angles[i]
                
                # 2. 🌟 核心：处理角度越界问题，走最短圆弧！ 
                # 例如: 179度 走到 -179度，算出来的 diff 是 -358度
                # 经过这行代码处理后，diff 会变成 2度
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                
                # 3. 累加绝对值作为代价 (你也可以在这里加权重，比如底座电机重，就乘以 2.0)
                cost += abs(diff)
                
            # 如果这个解的代价比之前记录的都小，就更新“最佳榜单”
            if cost < min_cost:
                min_cost = cost
                best_angles = calc_angles
                best_config = cfg
                
        except ValueError:
            # 说明这种构型在这个坐标点无解 (够不到或进入死区)，直接跳过，试下一种
            continue
            
    # 遍历完 8 种情况后，检查是否至少找到了一个解
    if best_angles is None:
        raise ValueError("严重错误：目标点超出了工作空间，所有 8 种姿态均无法到达！")
        
    print(f"\n[寻优完成] 共找到 {valid_solutions} 个有效解。")
    print(f"最优构型为: {best_config}, 总移动代价(弧度): {min_cost:.4f}")
    
    return best_angles, best_config, min_cost
# ==========================================
# 4. 用户调用示例
# ==========================================
if __name__ == "__main__":
    # 我们测试一个姿态相对常规的目标点
    # target_x, target_y, target_z = 0.3, 0.1, 0.3
    target_x, target_y, target_z = 0.471, 0.1037, 0.4101
    target_roll, target_pitch, target_yaw = 0.0, np.pi/2, 0.0 
    # 假设机械臂现在全处于 0 度 (或者某个你读取到的当前电机真实角度)
    current_motor_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     位置 - X:   0.4710 m, Y:   0.1037 m, Z:   0.4101 m
# 姿态 - Roll(X):   0.0000°, Pitch(Y):  90.0000°, Yaw(Z):   0.0000°
    print(f"目标 XYZ: [{target_x}, {target_y}, {target_z}]")
    print(f"目标 RPY: [{target_roll}, {target_pitch}, {target_yaw}]\n")
    
    # try:
    #     # 这里默认最多迭代 50 次，误差小于 0.0001 弧度时停止
    #     motor_angles = calculate_ik_adaptive(
    #         target_x, target_y, target_z, 
    #         target_roll, target_pitch, target_yaw,
    #         config=(1, -1, 1),
    #         max_iters=50,
    #         tol=1e-4
    #     )
    #     print(f"\n=> 输出电机角度 (弧度): {[round(q, 4) for q in motor_angles]}")
        
    # except ValueError as e:
    #     print(f"求解失败: {e}")
    try:
        best_angles, best_cfg, cost = find_best_ik_solution(
            target_x, target_y, target_z, 
            target_roll, target_pitch, target_yaw,
            current_angles=current_motor_angles
        )
        print(f"\n=> 最终决定发送给电机的角度 (弧度): {[round(q, 4) for q in best_angles]}")
        
    except ValueError as e:
        print(e)