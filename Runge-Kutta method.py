import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理参数定义
params = {
    'm': 1.0,  # 质量 [kg]
    'R_coil': 2.0,  # 线圈电阻 [Ω]
    'k': 15.0,  # 弹簧常数 [N/m]
    'N': 150,  # 线圈匝数
    'l_c': 0.08,  # 线圈长度 [m]
    'z_c': 0.0,  # 线圈中心位置 [m]
    'a': 0.03,  # 线圈半径 [m]
    'mu0': 4 * np.pi * 1e-7,  # 真空磁导率 [H/m]
    'M': 8e5,  # 磁化强度 [A/m]
    'L_magnet': 0.15,  # 磁体长度 [m]
    'R_magnet': 0.08  # 磁体半径 [m]
}


def B_z(z):
    """计算轴向磁场分量"""
    eps = 1e-12  # 避免分母为零

    term1 = (z + params['L_magnet'] / 2) / (np.sqrt((z + params['L_magnet'] / 2) ** 2 +
                                                    params['R_magnet'] ** 2 + eps) ** 3)
    term2 = (z - params['L_magnet'] / 2) / (np.sqrt((z - params['L_magnet'] / 2) ** 2 +
                                                    params['R_magnet'] ** 2 + eps) ** 3)

    return (params['mu0'] * params['M'] / (4 * np.pi)) * (term1 - term2)


def Phi_prime(x):
    """计算磁通量导数 Φ'(x)"""
    z1 = params['z_c'] + params['l_c'] / 2 - x
    z2 = params['z_c'] - params['l_c'] / 2 - x

    area = np.pi * params['a'] ** 2  # 线圈截面积

    B1 = B_z(z1)
    B2 = B_z(z2)

    return -(params['N'] / params['l_c']) * area * (B1 - B2)


def system(t, y):
    """定义微分方程系统"""
    x, v = y

    phi_prime_val = Phi_prime(x)
    damping_coeff = (phi_prime_val ** 2 / params['R_coil'])

    dxdt = v
    dvdt = -(1 / params['m']) * (damping_coeff * v + params['k'] * x)

    return [dxdt, dvdt]


def solve_and_plot():
    """求解并绘制结果图"""

    # 初始条件
    x0 = 0.06  # 初始位移 [m]
    v0 = 0.0  # 初始速度 [m/s]
    y0 = [x0, v0]

    # 时间范围
    t_span = [0, 8]  # 8秒模拟时间
    t_eval = np.linspace(t_span[0], t_span[1], 2000)

    print("正在求解非线性振动系统...")

    # 使用Scipy的RK45方法求解
    solution = solve_ivp(system, t_span, y0, method='RK45',
                         t_eval=t_eval, rtol=1e-8, atol=1e-10)

    t = solution.t
    x = solution.y[0, :]
    v = solution.y[1, :]

    # 计算能量
    kinetic_energy = 0.5 * params['m'] * v ** 2
    potential_energy = 0.5 * params['k'] * x ** 2
    total_energy = kinetic_energy + potential_energy

    # 计算磁通量导数
    phi_prime_vals = np.array([Phi_prime(xi) for xi in x])

    print("求解完成!")

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('非线性电磁阻尼振动系统分析', fontsize=20, fontweight='bold')

    # 颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    # 1. 位移时间序列
    axes[0, 0].plot(t, x * 1000, color=colors[0], linewidth=2.5)
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('位移 (mm)')
    axes[0, 0].set_title('位移时间历程', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, f"初始条件: x₀ = {x0 * 1000:.1f} mm, v₀ = {v0} m/s",
                    transform=axes[0, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. 速度时间序列
    axes[0, 1].plot(t, v, color=colors[1], linewidth=2.5)
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('速度 (m/s)')
    axes[0, 1].set_title('速度时间历程', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, f"质量: m = {params['m']} kg",
                    transform=axes[0, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. 相图
    axes[0, 2].plot(x * 1000, v, color=colors[2], linewidth=2.0)
    axes[0, 2].set_xlabel('位移 (mm)')
    axes[0, 2].set_ylabel('速度 (m/s)')
    axes[0, 2].set_title('相平面图', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].text(0.05, 0.95, f"弹簧常数: k = {params['k']} N/m",
                    transform=axes[0, 2].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. 能量变化
    axes[1, 0].plot(t, kinetic_energy, color='#F18F01', linewidth=2.5, label='动能')
    axes[1, 0].plot(t, potential_energy, color='#2E86AB', linewidth=2.5, label='势能')
    axes[1, 0].plot(t, total_energy, color='#A23B72', linewidth=2.5, label='总能量')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('能量 (J)')
    axes[1, 0].set_title('系统能量变化', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    energy_loss = (total_energy[0] - total_energy[-1]) / total_energy[0] * 100
    axes[1, 0].text(0.05, 0.95, f"能量耗散: {energy_loss:.1f}%",
                    transform=axes[1, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. 磁场分布
    z_range = np.linspace(-0.4, 0.4, 1000)
    B_field = [B_z(z) * 1000 for z in z_range]  # 转换为mT
    axes[1, 1].plot(z_range * 1000, B_field, color=colors[3], linewidth=2.5)
    axes[1, 1].set_xlabel('轴向位置 (mm)')
    axes[1, 1].set_ylabel('磁场强度 (mT)')
    axes[1, 1].set_title('轴向磁场分布', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95,
                    f"磁体: L = {params['L_magnet'] * 1000:.1f} mm, R_m = {params['R_magnet'] * 1000:.1f} mm",
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 6. 磁通量导数
    x_range = np.linspace(-0.1, 0.1, 500)
    phi_prime_range = [Phi_prime(xi) for xi in x_range]
    axes[1, 2].plot(x_range * 1000, phi_prime_range, color=colors[4], linewidth=2.5)
    axes[1, 2].set_xlabel('位移 (mm)')
    axes[1, 2].set_ylabel("Φ'(x)")
    axes[1, 2].set_title('磁通量导数', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].text(0.05, 0.95, f"线圈: N = {params['N']}, l_c = {params['l_c'] * 1000:.1f} mm",
                    transform=axes[1, 2].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # 保存图片
    plt.savefig('nonlinear_vibration_system.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印系统参数和结果
    print("\n系统参数:")
    print(f"质量: m = {params['m']} kg")
    print(f"弹簧常数: k = {params['k']} N/m")
    print(f"线圈电阻: R = {params['R_coil']} Ω")
    print(f"线圈匝数: N = {params['N']}")
    print(f"线圈长度: l_c = {params['l_c'] * 1000:.1f} mm")
    print(f"线圈半径: a = {params['a'] * 1000:.1f} mm")
    print(f"磁体长度: L = {params['L_magnet'] * 1000:.1f} mm")
    print(f"磁体半径: R_m = {params['R_magnet'] * 1000:.1f} mm")
    print(f"磁化强度: M = {params['M'] / 1e6:.1f}×10^6 A/m")

    print(f"\n初始条件:")
    print(f"初始位移: x₀ = {x0 * 1000:.1f} mm")
    print(f"初始速度: v₀ = {v0} m/s")

    max_displacement = np.max(np.abs(x)) * 1000
    max_velocity = np.max(np.abs(v))
    print(f"\n系统响应特性:")
    print(f"最大位移: {max_displacement:.2f} mm")
    print(f"最大速度: {max_velocity:.3f} m/s")
    print(f"能量耗散率: {energy_loss:.1f}%")

    return t, x, v, total_energy


# 运行分析
if __name__ == "__main__":
    t, x, v, energy = solve_and_plot()