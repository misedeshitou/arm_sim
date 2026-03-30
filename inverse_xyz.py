import numpy as np
import math

# ==========================================
# 1. 机械臂物理参数配置 (MCU 上的 const 数组)
# ==========================================
A_PARAMS     = [0.0,       0.116, 0.0,      0.012,      0.0,       0.0]
D_PARAMS     = [0.0,       0.0,  -0.0415,  0.16086,  0.0,       0.0844]
ALPHA_PARAMS = [math.pi/2, 0.0,  -math.pi/2, math.pi/2, -math.pi/2, 0.0]
THETA_OFFSET = [math.pi,   0.0,   0.0,      math.pi,   math.pi/2, 0]

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

def calculate_ik_adaptive(x, y, z, roll, pitch, yaw, current_j4_physical=0.0, config=(1, -1, 1), max_iters=50, tol=1e-4):
    sign_shoulder, sign_elbow, sign_wrist = config

    # ... (前面的 1 和 2 矩阵构建、迭代前置逻辑保持不变) ...
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    R_end = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    # print(f"目标末端旋转矩阵 R_end:\n{R_end}")
    P_end = np.array([x, y, z])
    # print(f"目标末端位置 P_end: {P_end}")
    Z_dir = R_end[:, 2]
    # print(f"末端 Z 轴方向 Z_dir: {Z_dir}")
    P_wc = P_end - D_PARAMS[5] * Z_dir
    # print(f"计算得到的腕心位置 P_wc: {P_wc}")

    theta4_math = 0.0  
    theta = [0.0] * 6
    converged = False

    for iteration in range(max_iters):
        theta4_old = theta4_math 

        # ... (中间的逆解推导保持不变，直到计算 theta[3] 和 theta[5] 的地方) ...
        L = math.sqrt(D_PARAMS[3]**2 + (A_PARAMS[3] * math.cos(theta4_math))**2)
        psi = math.atan2(D_PARAMS[3], A_PARAMS[3] * math.cos(theta4_math))
        D_offset = D_PARAMS[2] - A_PARAMS[3] * math.sin(theta4_math)
        # print(f"迭代 {iteration + 1}: 当前 J4 数学角度: {theta4_math:.4f} rad,L: {L:.4f} ,psi: {psi:.4f} rad, D_offset: {D_offset:.4f} m")

        R_xy = math.sqrt(P_wc[0]**2 + P_wc[1]**2)
        # if R_xy < abs(D_offset):
        #     # raise ValueError(f"迭代第 {iteration} 次时进入死区！目标点过近。")
        #     beta = math.pi / 2 if D_offset > 0 else -math.pi / 2


        if R_xy < abs(D_offset):
            # 这通常发生在腕心 P_wc 距离 Z 轴太近时
            # 强制让 beta 变为 pi/2 或 -pi/2，即让手臂尽力向中心靠拢
            beta = math.pi / 2 if D_offset > 0 else -math.pi / 2
        else:
            beta = math.asin(D_offset / R_xy)

        if sign_shoulder < 0:
            beta = math.pi - beta

        phi = math.atan2(P_wc[1], P_wc[0])
        theta[0] = phi + beta


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

        # 🌟 这里的 IF-ELSE 是核心修改点 🌟
        if abs(sin_t5) > 1e-6:
            theta[3] = math.atan2(-r23, -r13)
            theta[5] = math.atan2(-r32, r31)
        else:
            # 【万向锁触发】
            print(f"⚠️ 迭代第 {iteration + 1} 次触发万向锁！强制保持 J4 的物理角度: {current_j4_physical:.4f} rad")
            current_j4_dh = current_j4_physical + THETA_OFFSET[3]
            theta[3] = current_j4_dh

            # 2. 计算被 J4 和 J6 共享的总偏航角
            sum_angle = math.atan2(R36[1, 0], R36[0, 0])

            # 3. 让 J6 单独吃掉所有旋转误差
            if r33 > 0:
                theta[5] = sum_angle - theta[3]
            else:
                theta[5] = theta[3] - sum_angle

            # 🔥 修复说明：这里绝对不能 return！
            # 必须让代码往下走，执行 theta4_math = theta[3]
            # 并在下一次迭代中用强制锁死的 J4 重新补偿 J1/J2/J3 的偏差！

            # 这个赋值至关重要：在奇异点时，它能保证迭代公式立刻稳定下来不再震荡
        theta4_math = theta[3]

        # --- 收敛判断 ---
        diff = abs(math.atan2(math.sin(theta4_math - theta4_old), math.cos(theta4_math - theta4_old)))
        # ... 后续保持不变
        if diff < tol:
            converged = True
            print(f"✅ 成功：在第 {iteration + 1} 次迭代后收敛！误差: {diff:.6e}")
            break

    if not converged:
        raise ValueError(f"⚠️ 达到最大迭代次数 ({max_iters})，算法发散！")

    q_out = [0.0] * 6
    for i in range(6):
        q_out[i] = theta[i] - THETA_OFFSET[i]
        q_out[i] = (q_out[i] + math.pi) % (2 * math.pi) - math.pi

    # 🌟 新增：严格计算末端误差，拦截“伪收敛” 🌟
    # 注意：为了避免循环导入或依赖，你可能需要将 forward_kinematics 的数学部分提取出来，或者直接在这里验证 P_wc
    # 这里我们验证生成的 q_out 是否能让腕心 P_wc 达到目标，或者直接验证末端 XYZ
    x_calc, y_calc, z_calc, _, _, _ = forward_kinematics(q_out)
    pos_error = math.sqrt((x_calc - x)**2 + (y_calc - y)**2 + (z_calc - z)**2)
    
    if pos_error > 0.005: # 设定 5mm 为无法接受的误差阈值
        raise ValueError(f"⚠️ 算法伪收敛！坐标物理不可达或进入死区 (位置误差: {pos_error*1000:.1f} mm)。")

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
        (1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
        (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)
    ]

    best_angles = None
    best_config = None
    min_cost = float('inf') # 初始代价设为无穷大

    valid_solutions = 0 # 记录算出了几个有效解

    for cfg in configs:
        try:
            calc_angles = calculate_ik_adaptive(
                x, y, z, roll, pitch, yaw, 
                current_j4_physical=current_angles[3], # 🔥 把当前的真实 J4 传进去
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

def forward_kinematics(joint_angles_rad):
    """
    基于全局 DH 参数的 6 轴机械臂正运动学求解
    输入: 
        joint_angles_rad: 长度为 6 的列表/数组，电机当前的物理弧度角 (q)
    输出:
        x, y, z: 末端位置 (米)
        roll, pitch, yaw: 末端姿态 (弧度, ZYX 欧拉角)
    """
    
    # 1. 初始化 4x4 单位矩阵
    T = np.eye(4)
    
    # 2. 迭代计算每一级关节的变换矩阵 (T_01, T_12, ..., T_56)
    # 逻辑：theta = q_motor + theta_offset
    for i in range(6):
        theta = joint_angles_rad[i] + THETA_OFFSET[i]
        d = D_PARAMS[i]
        a = A_PARAMS[i]
        alpha = ALPHA_PARAMS[i]
        
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        
        # 标准 DH 变换矩阵公式
        T_step = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [ 0,       sa,       ca,      d],
            [ 0,        0,        0,      1]
        ])
        
        # 矩阵累乘
        T = T @ T_step

    # 3. 提取末端位置 (x, y, z)
    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    # 4. 提取末端姿态 (Roll, Pitch, Yaw)
    # 采用 ZYX 欧拉角约定 (r31 = -sin(pitch))
    # T[0,0]=cos(y)cos(p), T[1,0]=sin(y)cos(p), T[2,0]=-sin(p)
    
    sy = math.sqrt(T[0, 0]**2 + T[1, 0]**2)
    singular = sy < 1e-6  # 万向锁判定阈值

    if not singular:
        # 正常情况
        roll  = math.atan2(T[2, 1], T[2, 2])
        pitch = math.atan2(-T[2, 0], sy)
        yaw   = math.atan2(T[1, 0], T[0, 0])
    else:
        # 发生万向锁 (Pitch = ±90°)
        # 此时只能求出 Roll 和 Yaw 的和或差，通常设 Yaw 为 0
        roll  = math.atan2(-T[1, 2], T[1, 1])
        pitch = math.atan2(-T[2, 0], sy)
        yaw   = 0.0

    return x, y, z, roll, pitch, yaw
# ==========================================
# 4. 用户调用示例
# ==========================================
if __name__ == "__main__":
    current_motor_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_gimbal_lock_angles = [1.0, 0, -0.4, 3.0, -np.pi/2, 3.0]

    # 2. 用正运动学算出这个状态下的绝对 XYZ 和 RPY
    # target_x, target_y, target_z, target_roll, target_pitch, target_yaw = forward_kinematics(test_gimbal_lock_angles)
    target_x=-0.1884
    target_y=-0.0415
    target_z=0.16086
    # target_roll=2.35619
    # target_pitch=-np.pi/2
    # target_yaw=-2.35619
    target_roll=-np.pi
    target_pitch=np.pi/2
    target_yaw=0
    print(f"构造的万向锁点 XYZ: [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]")
    print(f"构造的万向锁点 RPY: [{target_roll:.4f}, {target_pitch:.4f}, {target_yaw:.4f}]\n")
    # sx, sy, sz, sroll, spitch, syaw = extract_pose_from_matrix(T_singular) # 假设你有这个提取函数，或者直接用 T 矩阵验证

    # 3. 假装我们不知道 J4 和 J6 是多少，把这个奇异点丢给 IK 去逆解
    # 并且告诉 IK：我现在物理上 J4 就是 1.0 弧度！
    test_current_angles = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    try:
        best_angles, best_cfg, cost = find_best_ik_solution(
            target_x, target_y, target_z, 
            target_roll, target_pitch, target_yaw,
            current_angles=current_motor_angles
        )
        print(f"\n=> 最终决定发送给电机的角度 (弧度): {[round(q, 4) for q in best_angles]}")
        x, y, z, roll, pitch, yaw = forward_kinematics(best_angles) # 先看看当前姿态是什么样的
        print(f"反推 XYZ: [{x}, {y}, {z}]")
        print(f"反推 RPY: [{roll}, {pitch}, {yaw}]\n")

    except ValueError as e:
        print(e)