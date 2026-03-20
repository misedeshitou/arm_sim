import numpy as np
import math

# ==========================================
# 1. 机械臂物理参数配置 (匹配你的标准DH)
# ==========================================
A_PARAMS = [0.0, 0.290, 0.0, 0.03, 0.0, 0.0]
D_PARAMS = [0.0, 0.0, -0.10375, 0.40147, 0.0, 0.211]
ALPHA_PARAMS = [math.pi/2, 0.0, -math.pi/2, math.pi/2, -math.pi/2, 0.0]
THETA_OFFSET = [math.pi, 0.0, 0.0, math.pi, math.pi/2, 0.0]

def calc_dh_matrix(theta, d, a, alpha):
    """计算单节标准 DH 齐次变换矩阵"""
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])

def forward_kinematics(q):
    """正运动学：输入关节角(不含offset)，输出末端位姿 T06"""
    T = np.eye(4)
    for i in range(6):
        theta_i = q[i] + THETA_OFFSET[i]
        T_i = calc_dh_matrix(theta_i, D_PARAMS[i], A_PARAMS[i], ALPHA_PARAMS[i])
        T = T @ T_i
    return T

# ==========================================
# 2. 核心：逆运动学 (1次迭代版)
# ==========================================
def inverse_kinematics_1_iter(T_target, config=(1, 1, 1)):
    """
    输入: 
        T_target: 4x4 末端目标位姿矩阵
        config: (肩部正负, 肘部正负, 手腕正负) 决定 8 组解中的哪一组
    输出: 
        q: 6个电机的关节角度 (已扣除 offset)
    """
    sign_shoulder, sign_elbow, sign_wrist = config
    
    # 提取目标位置和 Z 轴接近方向
    P_end = T_target[0:3, 3]
    R_end = T_target[0:3, 0:3]
    Z_dir = R_end[:, 2]
    
    # 第一步：计算初始手腕中心 P_wc
    P_wc = P_end - D_PARAMS[5] * Z_dir
    
    # 初始化：假设手腕未旋转 (数学角 theta4)
    theta4_math = 0.0  
    
    # 保存结果的数组 (数学角)
    theta = [0.0] * 6
    
    # === 魔法开始：执行 2 次循环 (0为初始猜测，1为修正迭代) ===
    for iteration in range(2):
        
        # 1. 动态更新小臂参数 (由于 a4=0.03 的偏置)
        # 等效小臂面内投影长度 L
        L = math.sqrt(D_PARAMS[3]**2 + (A_PARAMS[3] * math.cos(theta4_math))**2)
        # 小臂内置偏角 psi
        psi = math.atan2(A_PARAMS[3] * math.cos(theta4_math), D_PARAMS[3])
        # 肩部面外总偏置 (静态 d3 + a4旋转引起的动态偏移)
        D_offset = D_PARAMS[2] - A_PARAMS[3] * math.sin(theta4_math)
        
        # 2. 求解 theta1 (俯视图补偿)
        R_xy = math.sqrt(P_wc[0]**2 + P_wc[1]**2)
        if R_xy < abs(D_offset):
            raise ValueError("目标点过近，进入工作空间死区！")
        
        phi = math.atan2(P_wc[1], P_wc[0])
        beta = math.asin(D_offset / R_xy) if sign_shoulder > 0 else math.pi - math.asin(D_offset / R_xy)
        theta[0] = phi - beta
        
        # 3. 求解 theta2, theta3 (侧视图余弦定理)
        X = P_wc[0] * math.cos(theta[0]) + P_wc[1] * math.sin(theta[0])
        Y = P_wc[2]
        
        a2 = A_PARAMS[1]
        cos_gamma = (X**2 + Y**2 - a2**2 - L**2) / (2 * a2 * L)
        
        # 防止超工作空间引起的浮点误差
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        
        gamma = sign_elbow * math.acos(cos_gamma)
        
        # 真实的 theta3 需要扣除小臂内置偏角
        theta[2] = gamma - psi
        
        # 求解 theta2
        alpha_angle = math.atan2(Y, X)
        beta_angle = math.atan2(L * math.sin(gamma), a2 + L * math.cos(gamma))
        theta[1] = alpha_angle - beta_angle
        
        # 4. 前三轴完成，构建 T03，准备提取姿态
        T03 = np.eye(4)
        for i in range(3):
            T03 = T03 @ calc_dh_matrix(theta[i], D_PARAMS[i], A_PARAMS[i], ALPHA_PARAMS[i])
            
        R03 = T03[0:3, 0:3]
        
        # 5. 计算手腕相对姿态 R36 = (R03)^T * R_target
        R36 = R03.T @ R_end
        
        # 6. 欧拉角矩阵拆解 (提取 theta4, theta5, theta6)
        r13, r23, r33 = R36[0, 2], R36[1, 2], R36[2, 2]
        r31, r32 = R36[2, 0], R36[2, 1]
        
        # 求解 theta5 (带正负号选择)
        sin_t5 = sign_wrist * math.sqrt(max(0.0, 1.0 - r33**2))
        theta[4] = math.atan2(sin_t5, r33)
        
        # 求解 theta4 和 theta6 (加入万向节锁保护)
        if abs(sin_t5) > 1e-6:
            theta[3] = math.atan2(-r23, -r13)
            theta[5] = math.atan2(-r32, r31)
        else:
            # 奇异点：仅用 theta6 完成剩余旋转
            theta[3] = theta4_math # 保持原样
            theta[5] = math.atan2(R36[1, 0], R36[0, 0])
            
        # 迭代核心：将算出的 theta4 喂给下一轮，更新动态偏置！
        theta4_math = theta[3]

    # === 循环结束 ===
    
    # 扣除配置表中的 offset，返回真实的电机执行角度 q
    q_out = [0.0] * 6
    for i in range(6):
        q_out[i] = theta[i] - THETA_OFFSET[i]
        # 规范化到 [-pi, pi]
        q_out[i] = (q_out[i] + math.pi) % (2 * math.pi) - math.pi
        
    return q_out

# ==========================================
# 3. 验证测试主函数
# ==========================================
if __name__ == "__main__":
    # 设定一组随机的电机真实角度 (作为 Ground Truth)
    q_true = [0.5, -0.3, 0.8, -1.2, 0.6, 0.4]
    print(f"设定真实关节角: {[round(q, 4) for q in q_true]}")
    
    # 用正运动学生成目标位姿矩阵
    T_target = forward_kinematics(q_true)
    print("\n生成目标位姿矩阵 T_target:\n", np.round(T_target, 4))
    
    # 用 1-Iteration IK 算法反推关节角
    # 注: 这里选用 config=(1, 1, 1) 具体取决于 q_true 处于 8组解的哪一个半区
    try:
        q_ik = inverse_kinematics_1_iter(T_target, config=(1, 1, 1))
        print(f"\nIK 求解关节角: {[round(q, 4) for q in q_ik]}")
        
        # 计算误差
        error = np.max(np.abs(np.array(q_true) - np.array(q_ik)))
        print(f"\n最大关节角误差: {error:.2e} rad")
        if error < 1e-4:
            print("=> 成功！1次迭代精度已满足要求！")
            
    except ValueError as e:
        print(f"求解失败: {e}")