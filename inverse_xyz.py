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
# 2. 新增：位姿格式转换 (XYZ + Yaw/Pitch/Roll -> 4x4矩阵)
# ==========================================
def rpy_to_matrix(x, y, z, yaw, pitch, roll):
    """
    将位置和欧拉角转换为 4x4 齐次变换矩阵
    采用机器人学标准：内旋 Z-Y-X 顺序 (对应 Yaw-Pitch-Roll)
    """
    # 绕 Z 轴旋转 (Yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    
    # 绕 Y 轴旋转 (Pitch)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([
        [ cp,  0, sp],
        [  0,  1,  0],
        [-sp,  0, cp]
    ])
    
    # 绕 X 轴旋转 (Roll)
    cr, sr = math.cos(roll), math.sin(roll)
    Rx = np.array([
        [1,   0,   0],
        [0,  cr, -sr],
        [0,  sr,  cr]
    ])
    
    # 总旋转矩阵 R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    
    # 构建 4x4 矩阵
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    
    return T

# ==========================================
# 3. 核心：逆运动学 (1次迭代版)
# ==========================================
def inverse_kinematics_1_iter(T_target, config=(1, 1, 1)):
    """底层核心求解器 (输入 4x4 矩阵)"""
    sign_shoulder, sign_elbow, sign_wrist = config
    
    P_end = T_target[0:3, 3]
    R_end = T_target[0:3, 0:3]
    Z_dir = R_end[:, 2]
    
    P_wc = P_end - D_PARAMS[5] * Z_dir
    theta4_math = 0.0  
    theta = [0.0] * 6
    
    for iteration in range(2):
        L = math.sqrt(D_PARAMS[3]**2 + (A_PARAMS[3] * math.cos(theta4_math))**2)
        psi = math.atan2(A_PARAMS[3] * math.cos(theta4_math), D_PARAMS[3])
        D_offset = D_PARAMS[2] - A_PARAMS[3] * math.sin(theta4_math)
        
        R_xy = math.sqrt(P_wc[0]**2 + P_wc[1]**2)
        if R_xy < abs(D_offset):
            raise ValueError("目标点过近，进入工作空间死区！")
        
        phi = math.atan2(P_wc[1], P_wc[0])
        beta = math.asin(D_offset / R_xy) if sign_shoulder > 0 else math.pi - math.asin(D_offset / R_xy)
        theta[0] = phi - beta
        
        X = P_wc[0] * math.cos(theta[0]) + P_wc[1] * math.sin(theta[0])
        Y = P_wc[2]
        
        a2 = A_PARAMS[1]
        cos_gamma = (X**2 + Y**2 - a2**2 - L**2) / (2 * a2 * L)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        
        gamma = sign_elbow * math.acos(cos_gamma)
        theta[2] = gamma - psi
        
        alpha_angle = math.atan2(Y, X)
        beta_angle = math.atan2(L * math.sin(gamma), a2 + L * math.cos(gamma))
        theta[1] = alpha_angle - beta_angle
        
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

    q_out = [0.0] * 6
    for i in range(6):
        q_out[i] = theta[i] - THETA_OFFSET[i]
        q_out[i] = (q_out[i] + math.pi) % (2 * math.pi) - math.pi
        
    return q_out

# ==========================================
# 4. 新增：用户直接调用的 API
# ==========================================
def inverse_kinematics_xyz_rpy(x, y, z, yaw, pitch, roll, config=(1, 1, 1)):
    """
    外层封装函数：直接接收位置和姿态角进行求解
    """
    T_target = rpy_to_matrix(x, y, z, yaw, pitch, roll)
    return inverse_kinematics_1_iter(T_target, config)

# ==========================================
# 5. 验证测试主函数
# ==========================================
if __name__ == "__main__":
    # 设定一个你在空间中想要抓取的目标位姿
    # 例如: 前方0.4米，左侧0.1米，高度0.3米
    target_x = 0.417
    target_y = 0.1037
    target_z = 0.4101
    
    # 设定目标姿态：夹爪垂直向下 (常见抓取姿态)
    # 假设基座原点时，机械臂向前伸直。要让夹爪垂直向下，Pitch需要低头90度
    target_yaw = 0.0
    target_pitch = math.pi / 2
    target_roll = 0.0
    
    print(f"目标输入 [X, Y, Z]: [{target_x}, {target_y}, {target_z}]")
    print(f"目标输入 [Yaw, Pitch, Roll]: [{round(target_yaw,4)}, {round(target_pitch,4)}, {round(target_roll,4)}]")
    
    try:
        # 直接调用新封装的 xyz_rpy 接口
        q_ik = inverse_kinematics_xyz_rpy(
            target_x, target_y, target_z, 
            target_yaw, target_pitch, target_roll, 
            config=(1, 1, 1) # 选择 Righty, Elbow Down, Wrist Flip 的位姿
        )
        print(f"\n=> 算出的电机角度 (rad): {[round(q, 4) for q in q_ik]}")
        
        # 验证算出来的角度能不能回到目标点 (使用正运动学自证清白)
        T_verify = forward_kinematics(q_ik)
        P_verify = T_verify[0:3, 3]
        
        print("\n[验算结果]")
        print(f"FK 反推的实际位置 [X, Y, Z]: {[round(p, 4) for p in P_verify]}")
        print(f"位置误差: {np.linalg.norm(np.array([target_x, target_y, target_z]) - P_verify):.2e} m")
        
    except ValueError as e:
        print(f"求解失败: {e} (可能是目标点太远或进入了死区)")