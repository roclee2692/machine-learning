"""
梯度下降可视化 - 交互式动画演示
看到梯度下降的每一步如何工作！
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 示例1：单变量梯度下降动画 ====================

def example1_single_variable_animation():
    """示例1：单变量梯度下降的完整过程动画"""
    print("="*60)
    print("示例1：单变量线性回归梯度下降动画")
    print("="*60)

    # 生成数据 y = 3x + 2
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = 3 * X + 2 + np.random.randn(10) * 2

    # 初始化参数
    w_init = 0.0
    b_init = 0.0
    learning_rate = 0.1
    n_iterations = 50

    # 记录每一步的参数
    w_history = [w_init]
    b_history = [b_init]
    cost_history = []

    # 梯度下降
    w, b = w_init, b_init
    for i in range(n_iterations):
        learning_rate = 0.01
        y_pred = w * X + b

        # 损失
        m = len(X)
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        cost_history.append(cost)

        # 梯度
        dw = (1/m) * np.sum((y_pred - y) * X)
        db = (1/m) * np.sum(y_pred - y)

        # 更新
        w = w - learning_rate * dw
        b = b - learning_rate * db

        w_history.append(w)
        b_history.append(b)

    print(f"初始参数: w={w_init:.2f}, b={b_init:.2f}")
    print(f"最终参数: w={w_history[-1]:.2f}, b={b_history[-1]:.2f}")
    print(f"初始损失: {cost_history[0]:.2f}")
    print(f"最终损失: {cost_history[-1]:.2f}")

    # 创建动画
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：数据拟合过程
    ax1 = axes[0]
    ax1.scatter(X, y, color='blue', s=50, alpha=0.6, label='训练数据')
    # 设置更醒目的初始拟合线，确保第一帧即可见
    line, = ax1.plot([], [], color='crimson', linewidth=3, zorder=5, label='拟合直线')
    text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                    verticalalignment='top', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('梯度下降拟合过程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 11)
    ax1.set_ylim(min(min(y)-5, -5), max(y)+5)

    # 初始化直线位置，避免部分后端首帧不可见
    X_line_init = np.array([0, 11])
    y_line_init = w_history[0] * X_line_init + b_history[0]
    line.set_data(X_line_init, y_line_init)

    # 右图：损失函数下降
    ax2 = axes[1]
    cost_line, = ax2.plot([], [], 'b-', linewidth=2)
    cost_point, = ax2.plot([], [], 'ro', markersize=10)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('损失函数 J(w,b)')
    ax2.set_title('损失函数变化')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_iterations)
    ax2.set_ylim(0, max(cost_history)*1.1)

    def animate(frame):
        # 更新拟合直线
        if frame < len(w_history):
            w_current = w_history[frame]
            b_current = b_history[frame]
            X_line = np.array([0, 11])
            y_line = w_current * X_line + b_current
            line.set_data(X_line, y_line)

        # 更新文本
        if frame < len(cost_history):
            text.set_text(f'迭代 {frame}/{n_iterations}\n'
                         f'w = {w_history[frame]:.4f}\n'
                         f'b = {b_history[frame]:.4f}\n'
                         f'损失 = {cost_history[frame]:.4f}')
        else:
            text.set_text(f'迭代 {n_iterations}/{n_iterations}\n'
                         f'w = {w_history[-1]:.4f}\n'
                         f'b = {b_history[-1]:.4f}\n'
                         f'损失 = {cost_history[-1]:.4f}')

        # 更新损失曲线
        if frame <= len(cost_history):
            cost_line.set_data(range(min(frame+1, len(cost_history))), 
                             cost_history[:min(frame+1, len(cost_history))])
        if frame < len(cost_history):
            cost_point.set_data([frame], [cost_history[frame]])

        return line, text, cost_line, cost_point

    anim = FuncAnimation(fig, animate, frames=len(w_history),
                        interval=200, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()

    print("\n观察：")
    print("- 左图：红线逐渐拟合蓝色数据点")
    print("- 右图：损失函数持续下降直到收敛")


# ==================== 示例2：3D损失函数等高线 ====================

def example2_3d_loss_surface():
    """示例2：3D损失函数表面和梯度下降路径"""
    print("\n" + "="*60)
    print("示例2：3D损失函数表面可视化")
    print("="*60)

    # 简单数据
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    # 创建参数网格
    w_range = np.linspace(-2, 5, 100)
    b_range = np.linspace(-5, 5, 100)
    W, B = np.meshgrid(w_range, b_range)

    # 计算每个点的损失
    m = len(X)
    J = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_val = W[i, j]
            b_val = B[i, j]
            y_pred = w_val * X + b_val
            J[i, j] = (1/(2*m)) * np.sum((y_pred - y)**2)

    # 梯度下降路径（保证 w、b、J 的长度一致）
    w, b = 0.0, 0.0
    learning_rate = 0.1
    init_pred = w * X + b
    init_cost = (1/(2*m)) * np.sum((init_pred - y)**2)
    path_w, path_b, path_j = [w], [b], [init_cost]

    for _ in range(30):
        y_pred = w * X + b
        dw = (1/m) * np.sum((y_pred - y) * X)
        db = (1/m) * np.sum(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        path_w.append(w)
        path_b.append(b)

        y_pred_new = w * X + b
        cost_new = (1/(2*m)) * np.sum((y_pred_new - y)**2)
        path_j.append(cost_new)

    # 创建图形
    fig = plt.figure(figsize=(18, 6))

    # 3D表面图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(W, B, J, alpha=0.6, cmap='viridis')
    ax1.plot(path_w, path_b, path_j, 'ro-', linewidth=2, markersize=5, label='梯度下降路径')
    ax1.set_xlabel('w')
    ax1.set_ylabel('b')
    ax1.set_zlabel('J(w,b)')
    ax1.set_title('3D损失函数表面')
    ax1.legend()

    # 等高线图
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(W, B, J, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(path_w, path_b, 'ro-', linewidth=2, markersize=5, label='梯度下降路径')
    ax2.scatter([path_w[0]], [path_b[0]], color='green', s=100, marker='*',
                label='起点', zorder=5)
    ax2.scatter([path_w[-1]], [path_b[-1]], color='red', s=100, marker='*',
                label='终点', zorder=5)
    ax2.set_xlabel('w')
    ax2.set_ylabel('b')
    ax2.set_title('损失函数等高线图')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    

    # 参数变化
    ax3 = fig.add_subplot(133)
    iterations = range(len(path_w))
    ax3.plot(iterations, path_w, 'b-o', label='w的变化', linewidth=2)
    ax3.plot(iterations, path_b, 'r-s', label='b的变化', linewidth=2)
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('参数值')
    ax3.set_title('参数收敛过程')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n最优参数: w={path_w[-1]:.4f}, b={path_b[-1]:.4f}")
    print(f"真实关系: y = 2x (所以w应该≈2, b应该≈0)")


# ==================== 示例3：不同学习率对比 ====================

def example3_learning_rate_comparison():
    """示例3：对比不同学习率的影响"""
    print("\n" + "="*60)
    print("示例3：学习率影响对比")
    print("="*60)

    # 数据
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 7, 9, 11])

    learning_rates = [0.01, 0.1, 0.5, 0.9]
    colors = ['blue', 'green', 'orange', 'red']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, (lr, color) in enumerate(zip(learning_rates, colors)):
        # 梯度下降
        w, b = 0.0, 0.0
        w_history, b_history, cost_history = [w], [b], []

        for _ in range(50):
            m = len(X)
            y_pred = w * X + b
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            cost_history.append(cost)

            dw = (1/m) * np.sum((y_pred - y) * X)
            db = (1/m) * np.sum(y_pred - y)

            w -= lr * dw
            b -= lr * db

            w_history.append(w)
            b_history.append(b)

        # 绘图
        ax = axes[idx]
        ax.plot(cost_history, color=color, linewidth=2)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失函数')
        ax.set_title(f'学习率 = {lr}')
        ax.grid(True, alpha=0.3)

        # 添加收敛信息
        if len(cost_history) > 1:
            if cost_history[-1] < 1:
                status = "收敛"
                text_color = 'green'
            elif cost_history[-1] > cost_history[0]:
                status = "发散"
                text_color = 'red'
            else:
                status = "收敛慢"
                text_color = 'orange'

            ax.text(0.7, 0.9, status, transform=ax.transAxes,
                   fontsize=14, color=text_color, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        print(f"\n学习率 {lr}:")
        print(f"  最终w = {w_history[-1]:.4f}, b = {b_history[-1]:.4f}")
        print(f"  最终损失 = {cost_history[-1]:.4f}")

    plt.tight_layout()
    plt.show()

    print("\n总结：")
    print("- 学习率太小(0.01)：收敛太慢")
    print("- 学习率适中(0.1)：收敛快且稳定")
    print("- 学习率较大(0.5)：可能震荡")
    print("- 学习率太大(0.9)：可能发散")


# ==================== 示例4：批量 vs 随机梯度下降 ====================

def example4_batch_vs_stochastic():
    """示例4：批量梯度下降 vs 随机梯度下降对比"""
    print("\n" + "="*60)
    print("示例4：BGD vs SGD 可视化对比")
    print("="*60)

    # 生成数据
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 5 + np.random.randn(100, 1) * 2
    y = y.ravel()
    X = X.ravel()

    # 批量梯度下降
    def batch_gd(X, y, lr=0.01, n_iter=50):
        m = len(X)
        w, b = 0.0, 0.0
        cost_history = []

        for _ in range(n_iter):
            y_pred = w * X + b
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            cost_history.append(cost)

            dw = (1/m) * np.sum((y_pred - y) * X)
            db = (1/m) * np.sum(y_pred - y)

            w -= lr * dw
            b -= lr * db

        return w, b, cost_history

    # 随机梯度下降
    def stochastic_gd(X, y, lr=0.01, n_epochs=50):
        m = len(X)
        w, b = 0.0, 0.0
        cost_history = []

        for epoch in range(n_epochs):
            for i in range(m):
                xi = X[i]
                yi = y[i]

                y_pred = w * xi + b

                dw = (y_pred - yi) * xi
                db = y_pred - yi

                w -= lr * dw
                b -= lr * db

            # 每个epoch后计算总损失
            y_pred_all = w * X + b
            cost = (1/(2*m)) * np.sum((y_pred_all - y)**2)
            cost_history.append(cost)

        return w, b, cost_history

    # 训练两个模型（对当前数据尺度，0.1 容易发散）
    w_bgd, b_bgd, cost_bgd = batch_gd(X, y, lr=0.01, n_iter=50)
    w_sgd, b_sgd, cost_sgd = stochastic_gd(X, y, lr=0.01, n_epochs=50)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 损失函数对比
    ax1 = axes[0]
    ax1.plot(cost_bgd, 'b-', linewidth=2, label='批量梯度下降(BGD)')
    ax1.plot(cost_sgd, 'r-', linewidth=2, alpha=0.7, label='随机梯度下降(SGD)')
    ax1.set_xlabel('迭代/Epoch')
    ax1.set_ylabel('损失函数')
    ax1.set_title('损失函数下降对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 拟合结果对比
    ax2 = axes[1]
    ax2.scatter(X, y, alpha=0.3, s=20, label='训练数据')
    X_line = np.array([0, 10])
    ax2.plot(X_line, w_bgd * X_line + b_bgd, 'b-', linewidth=2, label=f'BGD: y={w_bgd:.2f}x+{b_bgd:.2f}')
    ax2.plot(X_line, w_sgd * X_line + b_sgd, 'r--', linewidth=2, label=f'SGD: y={w_sgd:.2f}x+{b_sgd:.2f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.set_title('最终拟合结果对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n批量梯度下降: w={w_bgd:.4f}, b={b_bgd:.4f}")
    print(f"随机梯度下降: w={w_sgd:.4f}, b={b_sgd:.4f}")
    print(f"\n特点对比:")
    print("BGD: 收敛平滑，但每次迭代慢")
    print("SGD: 收敛波动大，但每次迭代快，适合大数据")


# ==================== 示例5：动量优化 ====================

def example5_momentum_visualization():
    """示例5：带动量的梯度下降"""
    print("\n" + "="*60)
    print("示例5：梯度下降 vs 动量梯度下降")
    print("="*60)

    # 数据
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    # 普通梯度下降
    def normal_gd(X, y, lr=0.1, n_iter=30):
        m = len(X)
        w, b = 0.0, 0.0
        path_w, path_b, costs = [w], [b], []

        for _ in range(n_iter):
            y_pred = w * X + b
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            costs.append(cost)

            dw = (1/m) * np.sum((y_pred - y) * X)
            db = (1/m) * np.sum(y_pred - y)

            w -= lr * dw
            b -= lr * db

            path_w.append(w)
            path_b.append(b)

        return path_w, path_b, costs

    # 带动量的梯度下降
    def momentum_gd(X, y, lr=0.1, momentum=0.9, n_iter=30):
        m = len(X)
        w, b = 0.0, 0.0
        vw, vb = 0.0, 0.0  # 速度
        path_w, path_b, costs = [w], [b], []

        for _ in range(n_iter):
            y_pred = w * X + b
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            costs.append(cost)

            dw = (1/m) * np.sum((y_pred - y) * X)
            db = (1/m) * np.sum(y_pred - y)

            # 更新速度
            vw = momentum * vw - lr * dw
            vb = momentum * vb - lr * db

            # 更新参数
            w += vw
            b += vb

            path_w.append(w)
            path_b.append(b)

        return path_w, path_b, costs

    # 运行两种方法
    path_w_normal, path_b_normal, cost_normal = normal_gd(X, y)
    path_w_momentum, path_b_momentum, cost_momentum = momentum_gd(X, y)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 损失对比
    ax1 = axes[0]
    ax1.plot(cost_normal, 'b-o', label='普通梯度下降', linewidth=2)
    ax1.plot(cost_momentum, 'r-s', label='动量梯度下降', linewidth=2)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('损失函数')
    ax1.set_title('收敛速度对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 参数空间路径
    ax2 = axes[1]
    ax2.plot(path_w_normal, path_b_normal, 'b-o', label='普通GD', linewidth=2, markersize=4)
    ax2.plot(path_w_momentum, path_b_momentum, 'r-s', label='动量GD', linewidth=2, markersize=4)
    ax2.scatter([path_w_normal[0]], [path_b_normal[0]], color='green', s=100, marker='*', label='起点', zorder=5)
    ax2.set_xlabel('w')
    ax2.set_ylabel('b')
    ax2.set_title('参数空间中的优化路径')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n动量优势：")
    print("- 加速收敛")
    print("- 减少震荡")
    print("- 更容易跳出局部最优（在复杂函数中）")


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("\n" + "🎯"*30)
    print("梯度下降可视化演示")
    print("🎯"*30 + "\n")

    print("这个程序会依次展示5个可视化示例：")
    print("1. 单变量梯度下降动画（实时看到拟合过程）")
    print("2. 3D损失函数表面（理解参数空间）")
    print("3. 不同学习率对比（理解超参数影响）")
    print("4. BGD vs SGD对比（理解优化方法差异）")
    print("5. 动量优化（理解高级优化技巧）")
    print("\n" + "="*60)

    # 运行示例
    example1_single_variable_animation()

    input("\n按回车键继续下一个示例...")
    example2_3d_loss_surface()

    input("\n按回车键继续下一个示例...")
    example3_learning_rate_comparison()

    input("\n按回车键继续下一个示例...")
    example4_batch_vs_stochastic()

    input("\n按回车键继续下一个示例...")
    example5_momentum_visualization()

    print("\n" + "="*60)
    print("✅ 所有演示完成！")
    print("="*60)
    print("\n关键要点：")
    print("1. 梯度下降通过迭代逐步接近最优解")
    print("2. 学习率太大会发散，太小会收敛慢")
    print("3. 损失函数在参数空间中形成一个曲面")
    print("4. 梯度指向损失增加最快的方向，我们朝反方向走")
    print("5. 动量等技巧可以加速收敛")
