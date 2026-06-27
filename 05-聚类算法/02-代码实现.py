"""
聚类算法 - 完整实现
包含K-Means、层次聚类和DBSCAN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_blobs, make_moons
from sklearn.metrics import (silhouette_score, silhouette_samples, 
                             davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score)
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 数据生成 ==========

def generate_clustering_data(n_samples=300, n_clusters=3, random_state=42):
    """生成聚类数据"""
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, 
                          cluster_std=0.6, random_state=random_state)
    return X, y_true


# ========== 2. 可视化函数 ==========

def plot_clusters(X, labels, centers=None, title="聚类结果"):
    """绘制聚类结果"""
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # 异常点用黑色
            color = [0, 0, 0, 1]
        
        class_member_mask = (labels == label)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], label=f'簇{label}', alpha=0.6, s=50)
    
    # 绘制中心
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', 
                   s=300, edgecolors='black', linewidth=2, label='中心')
    
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ========== 3. 示例1：K-Means基础 ==========

def example1_kmeans_basic():
    """示例1：K-Means基础"""
    print("="*50)
    print("示例1：K-Means基础")
    print("="*50)

    # 生成数据
    X, y_true = generate_clustering_data(n_samples=300, n_clusters=3)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    print("\n训练K-Means (K=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 评估
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    
    print(f"惯性: {inertia:.4f}")
    print(f"轮廓系数: {silhouette:.4f}")

    # 绘制
    plot_clusters(X_scaled, labels, kmeans.cluster_centers_, "K-Means聚类结果 (K=3)")


# ========== 4. 示例2：肘部法则 ==========

def example2_elbow_method():
    """示例2：肘部法则选择K"""
    print("\n" + "="*50)
    print("示例2：肘部法则")
    print("="*50)

    # 生成数据
    X, y_true = generate_clustering_data(n_samples=300, n_clusters=3)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 测试不同的K值
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 惯性
    axes[0].plot(k_range, inertias, 'bo-')
    axes[0].set_xlabel('K值')
    axes[0].set_ylabel('惯性')
    axes[0].set_title('肘部法则')
    axes[0].grid(True)

    # 轮廓系数
    axes[1].plot(k_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('K值')
    axes[1].set_ylabel('轮廓系数')
    axes[1].set_title('轮廓系数')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\n最优K值（基于轮廓系数）: {k_range[np.argmax(silhouette_scores)]}")


# ========== 5. 示例3：层次聚类 ==========

def example3_hierarchical_clustering():
    """示例3：层次聚类"""
    print("\n" + "="*50)
    print("示例3：层次聚类")
    print("="*50)

    # 生成数据
    X, y_true = generate_clustering_data(n_samples=100, n_clusters=3)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 层次聚类
    print("\n训练层次聚类...")
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = hierarchical.fit_predict(X_scaled)

    # 评估
    silhouette = silhouette_score(X_scaled, labels)
    print(f"轮廓系数: {silhouette:.4f}")

    # 绘制树状图
    print("\n生成树状图...")
    linkage_matrix = linkage(X_scaled, method='ward')
    
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix)
    plt.title('层次聚类树状图')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.axhline(y=10, color='r', linestyle='--', label='K=3')
    plt.legend()
    plt.show()

    # 聚类结果
    plot_clusters(X_scaled, labels, title="层次聚类结果")


# ========== 6. 示例4：DBSCAN ==========

def example4_dbscan():
    """示例4：DBSCAN聚类"""
    print("\n" + "="*50)
    print("示例4：DBSCAN聚类")
    print("="*50)

    # 生成非凸数据
    X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN
    print("\n训练DBSCAN (eps=0.3, min_samples=5)...")
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # 统计
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"簇的数量: {n_clusters}")
    print(f"异常点数: {n_noise}")

    # 绘制
    plot_clusters(X_scaled, labels, title="DBSCAN聚类结果")


# ========== 7. 示例5：参数对DBSCAN的影响 ==========

def example5_dbscan_parameters():
    """示例5：DBSCAN参数的影响"""
    print("\n" + "="*50)
    print("示例5：DBSCAN参数的影响")
    print("="*50)

    # 生成数据
    X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 测试不同的eps值
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, eps in enumerate(eps_values):
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        ax = axes[idx]
        plot_clusters_in_subplot(X_scaled, labels, ax, f"eps={eps}, 簇数={n_clusters}")

    # 隐藏最后一个子图
    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_clusters_in_subplot(X, labels, ax, title):
    """在子图中绘制聚类结果"""
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0, 1]
        
        class_member_mask = (labels == label)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[color], alpha=0.6, s=50)
    
    ax.set_title(title)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.grid(True, alpha=0.3)


# ========== 8. 示例6：三种算法对比 ==========

def example6_algorithm_comparison():
    """示例6：三种聚类算法对比"""
    print("\n" + "="*50)
    print("示例6：三种聚类算法对比")
    print("="*50)

    # 生成数据
    X, y_true = generate_clustering_data(n_samples=300, n_clusters=3)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)

    # 层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels_hierarchical = hierarchical.fit_predict(X_scaled)
    silhouette_hierarchical = silhouette_score(X_scaled, labels_hierarchical)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    silhouette_dbscan = silhouette_score(X_scaled, labels_dbscan) if len(set(labels_dbscan)) > 1 else -1

    # 绘制对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_clusters_in_subplot(X_scaled, labels_kmeans, axes[0], 
                            f"K-Means\n轮廓系数={silhouette_kmeans:.3f}")
    plot_clusters_in_subplot(X_scaled, labels_hierarchical, axes[1], 
                            f"层次聚类\n轮廓系数={silhouette_hierarchical:.3f}")
    plot_clusters_in_subplot(X_scaled, labels_dbscan, axes[2], 
                            f"DBSCAN\n轮廓系数={silhouette_dbscan:.3f}")

    plt.tight_layout()
    plt.show()

    print(f"\nK-Means轮廓系数: {silhouette_kmeans:.4f}")
    print(f"层次聚类轮廓系数: {silhouette_hierarchical:.4f}")
    print(f"DBSCAN轮廓系数: {silhouette_dbscan:.4f}")


# ========== 7. 示例7：轮廓系数分析 ==========

def example7_silhouette_analysis():
    """示例7：轮廓系数分析"""
    print("\n" + "="*50)
    print("示例7：轮廓系数分析")
    print("="*50)

    # 生成数据
    X, y_true = generate_clustering_data(n_samples=300, n_clusters=3)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分析不同K值
    k_values = [2, 3, 4, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, labels)
        sample_silhouette_values = silhouette_samples(X_scaled, labels)

        ax = axes[idx]
        y_lower = 10

        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            alpha=0.7, label=f'簇{i}')
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                  label=f'平均={silhouette_avg:.2f}')
        ax.set_title(f'K={k}')
        ax.set_ylabel('簇')
        ax.set_xlabel('轮廓系数')
        ax.set_ylim([0, len(X_scaled) + (k + 1) * 10])

    plt.tight_layout()
    plt.show()


# ========== 主函数 ==========

if __name__ == "__main__":
    # 运行所有示例
    example1_kmeans_basic()
    example2_elbow_method()
    example3_hierarchical_clustering()
    example4_dbscan()
    example5_dbscan_parameters()
    example6_algorithm_comparison()
    example7_silhouette_analysis()

    print("\n" + "="*50)
    print("所有示例运行完成！")
    print("="*50)

