import numpy as np
import matplotlib.pyplot as plt

class StandardDHArm:
    def __init__(self, dh_table):
        """
        dh_table 每行顺序: [theta_offset, d, alpha, a]
        """
        self.dh_table = dh_table

    def get_matrix(self, theta, d, alpha, a):
        """
        标准 DH 变换矩阵计算:
        T = Rotz(theta) * Transz(d) * Transx(a) * Rotx(alpha)
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def extract_pose(self, T):
        """
        从 4x4 齐次变换矩阵中提取平移(x, y, z) 和 旋转(roll, pitch, yaw)
        采用标准的 ZYX 欧拉角约定
        """
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        
        # 计算 Roll (X), Pitch (Y), Yaw (Z)
        sy = np.sqrt(T[0, 0] * T[0, 0] + T[1, 0] * T[1, 0])
        singular = sy < 1e-6 # 检查是否处于万向锁奇异点

        if not singular:
            roll = np.arctan2(T[2, 1], T[2, 2])
            pitch = np.arctan2(-T[2, 0], sy)
            yaw = np.arctan2(T[1, 0], T[0, 0])
        else:
            roll = np.arctan2(-T[1, 2], T[1, 1])
            pitch = np.arctan2(-T[2, 0], sy)
            yaw = 0
        return x, y, z, roll, pitch, yaw

    def plot(self, joint_angles_rad):
        # 1. 计算所有坐标系矩阵
        transforms = [np.eye(4)]
        current_T = np.eye(4)
        
        for i, row in enumerate(self.dh_table):
            theta_total = joint_angles_rad[i] + row[0]
            d, alpha, a = row[1], row[2], row[3]
            
            T_step = self.get_matrix(theta_total, d, alpha, a)
            current_T = current_T @ T_step
            transforms.append(current_T)
        
        # 🌟 提取并输出末端位姿 🌟
        EE_T = transforms[-1]
        x, y, z, roll, pitch, yaw = self.extract_pose(EE_T)
        
        print("=" * 45)
        print("🎯 末端执行器 (End-Effector) 实时位姿")
        print("=" * 45)
        print(f"位置 - X: {x:8.4f} m, Y: {y:8.4f} m, Z: {z:8.4f} m")
        print(f"姿态 - Roll(X): {roll:8.4f}, Pitch(Y): {pitch:8.4f}, Yaw(Z): {yaw:8.4f}")
        print("=" * 45)

        # 2. 绘图准备
        positions = np.array([T[:3, 3] for T in transforms])
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制连杆
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'o-', 
                color='#2c3e50', linewidth=4, markersize=8, label="Robot Links")

        # 3. 绘制坐标系函数
        def draw_axis(T, length=0.15, label=""):
            origin = T[:3, 3]
            ax.quiver(*origin, *(T[:3, 0]*length), color='r', linewidth=1.5) # X
            ax.quiver(*origin, *(T[:3, 1]*length), color='g', linewidth=1.5) # Y
            ax.quiver(*origin, *(T[:3, 2]*length), color='b', linewidth=1.5) # Z
            if label: ax.text(*origin, label)

        # 绘制基座 (Initial) 和 末端 (End-Effector)
        draw_axis(transforms[0], length=0.3, label="Base")
        draw_axis(transforms[-1], length=0.3, label="EE")

        # 图表修饰
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title("6-DOF Arm ($\theta, d, \\alpha, a$ Order) - Radian Input")
        
        # 保持等比例
        max_range = np.array([positions[:,0].max()-positions[:,0].min(), 
                              positions[:,1].max()-positions[:,1].min(), 
                              positions[:,2].max()-positions[:,2].min()]).max() / 2.0
        mid = positions.mean(axis=0)
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
        
        plt.show()


def main():
    #  [theta_offset, d, alpha, a]
    dh_params = np.array([
        # [theta_off,  d,         alpha,     a  ]
        [np.pi,       0,         np.pi/2,   0   ],   # J1
        [0,           0,         0,         0.29],   # J2
        [0,          -0.10375,  -np.pi/2,   0   ],   # J3
        [np.pi,       0.410147,  np.pi/2,   0.03],   # J4
        [np.pi/2,     0,        -np.pi/2,   0   ],   # J5
        [np.pi,       0.211,     0,         0   ]    # J6
    ])

    # 在这里调节 6 个关节的实时转动角度 (从初始姿态开始转) 单位：弧度 (Radian)
    target_angles = [0, 0, 0, 0, -np.pi/2, 0]
    # target_angles = [0.5867, 1.241, 0.0665, -3.1416, 0.2633, 0.5867] 
    # target_angles = [3.1415, 2.0117, 2.9954, 0.0005, -1.2761, -0.0005]
    # target_angles = [3.1415, 2.0117, 2.9954, 0.0005, -1.2761, -0.0005]
    # target_angles = [3.124, 2.3506, 2.2107, 3.1326, -0.3489, -3.1231]

    arm = StandardDHArm(dh_params)
    arm.plot(target_angles)

if __name__ == "__main__":
    main()