"""
交互式线性回归演示脚本
让你可以调整参数，观察对训练过程的影响
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InteractiveLinearRegression:
    """交互式线性回归 - 用于学习和实验"""
    
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.loss_history = []
        self.w_history = []
        self.b_history = []
    
    def fit(self, X, y, show_detail=False):
        """训练模型并记录每一步"""
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        self.loss_history = []
        self.w_history = []
        self.b_history = []
        
        for iteration in range(self.iterations):
            # 【第1步】前向传播
            y_pred = np.dot(X, self.w) + self.b
            
            # 【第2步】计算损失
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # 【第3步】计算梯度
            error = y_pred - y
            dw = (2 / m) * np.dot(X.T, error)
            db = (2 / m) * np.sum(error)
            
            # 【第4步】更新参数
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            # 记录历史（用于可视化）
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b.copy())
            
            if show_detail and iteration % max(1, self.iterations // 10) == 0:
                print(f"迭代 {iteration:3d}: Loss={loss:.4f}, w={self.w[0]:.4f}, b={self.b:.4f}")
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b
    
    def get_r2_score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def create_sample_data(n_samples=50, noise=10):
    """创建示例数据集"""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 100
    # 真实关系：y = 3*x + 50 + 噪声
    y = X.flatten() * 3 + 50 + np.random.randn(n_samples) * noise
    
    X_normalized = (X - X.min()) / (X.max() - X.min())
    return X_normalized, y


def plot_training_results(model, X, y, title=""):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图1：损失函数曲线
    ax = axes[0, 0]
    ax.plot(model.loss_history, 'b-', linewidth=2)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('损失函数 (MSE)')
    ax.set_title('损失函数变化曲线')
    ax.grid(True, alpha=0.3)
    
    # 图2：权重变化
    ax = axes[0, 1]
    ax.plot(model.w_history, 'r-', linewidth=2)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('权重 w')
    ax.set_title('权重参数变化')
    ax.grid(True, alpha=0.3)
    
    # 图3：偏置变化
    ax = axes[1, 0]
    ax.plot(model.b_history, 'g-', linewidth=2)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('偏置 b')
    ax.set_title('偏置参数变化')
    ax.grid(True, alpha=0.3)
    
    # 图4：拟合效果
    ax = axes[1, 1]
    y_pred = model.predict(X)
    ax.scatter(X, y, alpha=0.5, s=50, label='实际值')
    ax.plot(X, y_pred, 'r-', linewidth=2, label='预测值')
    ax.set_xlabel('输入特征 X')
    ax.set_ylabel('输出 y')
    ax.set_title(f'拟合结果 (R²={model.get_r2_score(X, y):.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ==================== 演示 1: 基础演示 ====================
def demo_basic():
    """演示1：基础训练过程"""
    print("\n" + "="*70)
    print("演示1：基础训练过程")
    print("="*70)
    
    X, y = create_sample_data()
    
    model = InteractiveLinearRegression(learning_rate=0.01, iterations=200)
    model.fit(X, y, show_detail=True)
    
    print(f"\n最终参数:")
    print(f"  w = {model.w[0]:.6f}")
    print(f"  b = {model.b:.6f}")
    print(f"  R² = {model.get_r2_score(X, y):.6f}")
    
    fig = plot_training_results(model, X, y, "演示1: 基础训练")
    plt.savefig('demo1_basic.png', dpi=100, bbox_inches='tight')
    print("图表已保存为 demo1_basic.png")


# ==================== 演示 2: 学习率对比 ====================
def demo_learning_rates():
    """演示2：不同学习率的影响"""
    print("\n" + "="*70)
    print("演示2：不同学习率的影响")
    print("="*70)
    
    X, y = create_sample_data()
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        model = InteractiveLinearRegression(learning_rate=lr, iterations=200)
        model.fit(X, y)
        
        ax = axes[idx]
        ax.plot(model.loss_history, linewidth=2)
        ax.set_title(f'学习率 = {lr}')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失函数')
        ax.grid(True, alpha=0.3)
        
        final_loss = model.loss_history[-1]
        final_r2 = model.get_r2_score(X, y)
        print(f"学习率 {lr:5.3f} -> 最终Loss: {final_loss:10.4f}, R²: {final_r2:.6f}")
    
    plt.suptitle('学习率对训练的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo2_learning_rates.png', dpi=100, bbox_inches='tight')
    print("图表已保存为 demo2_learning_rates.png")


# ==================== 演示 3: 迭代次数对比 ====================
def demo_iterations():
    """演示3：迭代次数的影响"""
    print("\n" + "="*70)
    print("演示3：迭代次数的影响")
    print("="*70)
    
    X, y = create_sample_data()
    iterations_list = [10, 50, 100, 500]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, iters in enumerate(iterations_list):
        model = InteractiveLinearRegression(learning_rate=0.01, iterations=iters)
        model.fit(X, y)
        
        ax = axes[idx]
        ax.plot(model.loss_history, linewidth=2)
        ax.set_title(f'迭代次数 = {iters}')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失函数')
        ax.grid(True, alpha=0.3)
        
        final_loss = model.loss_history[-1]
        final_r2 = model.get_r2_score(X, y)
        print(f"迭代次数 {iters:3d} -> 最终Loss: {final_loss:10.4f}, R²: {final_r2:.6f}")
    
    plt.suptitle('迭代次数对训练的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo3_iterations.png', dpi=100, bbox_inches='tight')
    print("图表已保存为 demo3_iterations.png")


# ==================== 演示 4: 噪声影响 ====================
def demo_noise():
    """演示4：数据噪声的影响"""
    print("\n" + "="*70)
    print("演示4：数据噪声的影响")
    print("="*70)
    
    noise_levels = [0, 5, 10, 20]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, noise in enumerate(noise_levels):
        X, y = create_sample_data(noise=noise)
        model = InteractiveLinearRegression(learning_rate=0.01, iterations=200)
        model.fit(X, y)
        
        ax = axes[idx]
        y_pred = model.predict(X)
        ax.scatter(X, y, alpha=0.5, s=30, label='实际值')
        ax.plot(X, y_pred, 'r-', linewidth=2, label='拟合线')
        ax.set_title(f'噪声标准差 = {noise}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        final_r2 = model.get_r2_score(X, y)
        print(f"噪声 {noise:2d} -> R² = {final_r2:.6f}")
    
    plt.suptitle('数据噪声对模型的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo4_noise.png', dpi=100, bbox_inches='tight')
    print("图表已保存为 demo4_noise.png")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 交互式线性回归学习演示")
    print("="*70)
    
    # 运行所有演示
    demo_basic()
    demo_learning_rates()
    demo_iterations()
    demo_noise()
    
    print("\n" + "="*70)
    print("✅ 所有演示完成！")
    print("="*70)
    print("\n📊 生成的图表：")
    print("  1. demo1_basic.png - 基础训练过程")
    print("  2. demo2_learning_rates.png - 学习率对比")
    print("  3. demo3_iterations.png - 迭代次数对比")
    print("  4. demo4_noise.png - 数据噪声影响")
