"""
集成算法复杂非线性回归 - 完整实操代码
数据：California Housing（加州房价原始数据集）
模型：线性回归、SVR、随机森林、GBDT、XGBoost

流程：数据获取→预处理(异常值/缺失值/特征工程/标准化)→模型训练→评估→可视化
注意：California Housing 数据集目标变量在 5.0 处有截断（capping）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================= 第1步：导入库 =============================
print("="*80)
print("集成算法复杂非线性回归实验 - 完整施工")
print("="*80)

print("\n[步骤1] 导入必要的库...")
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

print("✓ 所有库导入成功")

# ============================= 第2步：数据获取与加载 =============================
print("\n[步骤2] 数据获取与加载...")

# 固定随机种子 - 保证实验可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 获取加州房价数据集
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print(f"✓ 数据加载成功")
print(f"  数据集形状: {X.shape}")
print(f"  样本数: {X.shape[0]}")
print(f"  特征维度: {X.shape[1]}")
print(f"  特征列表: {list(X.columns)}")
print(f"  标签统计:")
print(f"    最小值: {y.min():.4f}")
print(f"    最大值: {y.max():.4f}")
print(f"    均值: {y.mean():.4f}")
print(f"    标准差: {y.std():.4f}")

# ============================= 第3步：数据预处理（完整版） =============================
print("\n[步骤3] 数据预处理（异常值处理 + 缺失值处理 + 特征工程 + 标准化）...")

# 3.1 数据集划分：训练集80% 测试集20%（先划分，避免数据泄露）
print("  3.1 数据集划分 (80% 训练 / 20% 测试)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"    训练集大小: {X_train.shape[0]}")
print(f"    测试集大小: {X_test.shape[0]}")

# 3.2 异常值处理：IQR 方法裁剪（在训练集上计算边界，应用到训练集和测试集）
print("  3.2 异常值处理 (IQR 裁剪)...")
total_clipped = 0

for col in X_train.columns:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clipped_train = ((X_train[col] < lower) | (X_train[col] > upper)).sum()
    X_train[col] = X_train[col].clip(lower, upper)
    clipped_test = ((X_test[col] < lower) | (X_test[col] > upper)).sum()
    X_test[col] = X_test[col].clip(lower, upper)

    if clipped_train + clipped_test > 0:
        print(f"    {col:20s}: 裁剪 {clipped_train} 训练 + {clipped_test} 测试 异常值 "
              f"[{lower:.2f}, {upper:.2f}]")
        total_clipped += clipped_train + clipped_test

if total_clipped == 0:
    print("    ✓ 无明显异常值")
else:
    print(f"    ✓ 共裁剪 {total_clipped} 个异常值")

# 3.3 缺失值处理
print("  3.3 缺失值处理...")
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
print(f"    训练集缺失值: {train_missing}")
print(f"    测试集缺失值: {test_missing}")

if train_missing > 0 or test_missing > 0:
    fill_values = X_train.median()
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)
    print("    ✓ 已用训练集中位数填充缺失值")
else:
    print("    ✓ 无缺失值")

# 3.4 偏态特征对数变换（右偏特征取 log1p，降低极端值影响）
print("  3.4 偏态特征对数变换...")
skewness = X_train.skew()
skewed_features = skewness[abs(skewness) > 1.0].index.tolist()
print(f"    偏态特征 (|skew| > 1): {skewed_features}")

for feat in skewed_features:
    upper_bound = X_train[feat].max()
    X_train[feat] = np.log1p(X_train[feat].clip(upper=upper_bound))
    X_test[feat] = np.log1p(X_test[feat].clip(upper=upper_bound))
    print(f"    ✓ {feat}: log1p 变换完成 (训练集上界={upper_bound:.2f})")

# 3.5 领域特征工程（基于业务含义构造新特征）
print("  3.5 领域特征工程...")
original_features = X_train.columns.tolist()

X_train['RoomsPerHousehold'] = X_train['AveRooms'] / X_train['AveOccup']
X_test['RoomsPerHousehold'] = X_test['AveRooms'] / X_test['AveOccup']

X_train['RoomsPerPerson'] = X_train['AveRooms'] / X_train['Population']
X_test['RoomsPerPerson'] = X_test['AveRooms'] / X_test['Population']

X_train['BedroomRatio'] = X_train['AveBedrms'] / X_train['AveRooms']
X_test['BedroomRatio'] = X_test['AveBedrms'] / X_test['AveRooms']

X_train['PopulationPerHousehold'] = X_train['Population'] / X_train['AveOccup']
X_test['PopulationPerHousehold'] = X_test['Population'] / X_test['AveOccup']

new_features = [f for f in X_train.columns if f not in original_features]
print(f"    新增特征: {new_features}")
print(f"    特征数: {len(original_features)} → {X_train.shape[1]}")

# 保存特征名称（供后续特征重要性分析使用）
feature_names = X_train.columns.tolist()

# 3.6 特征标准化
print("  3.6 特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"    ✓ 标准化完成")
print(f"    训练集: {X_train_scaled.shape[0]} 样本 × {X_train_scaled.shape[1]} 特征")
print(f"    测试集: {X_test_scaled.shape[0]} 样本 × {X_test_scaled.shape[1]} 特征")
print(f"    特征数变化: 8 → {X_train_scaled.shape[1]}")

# ============================= 第4步：搭建对比实验模型 =============================
print("\n[步骤4] 搭建对比实验模型 (5个模型)...")

models_dict = {}
predictions_dict = {}
metrics_dict = {}

print("  4.1 线性回归 (LinearRegression) - 基线模型...")
lr_model = LinearRegression()
models_dict['LinearRegression'] = lr_model
print("      ✓ 构建完成")

print("  4.2 支持向量回归 (LinearSVR)...")
svr_model = LinearSVR(C=1.0, max_iter=5000, random_state=RANDOM_SEED)
models_dict['SVR'] = svr_model
print("      ✓ 构建完成 (LinearSVR, C=1.0)")

print("  4.3 随机森林 (RandomForestRegressor)...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
models_dict['RandomForest'] = rf_model
print("      ✓ 构建完成 (n_estimators=100, max_depth=15)")

print("  4.4 梯度提升树 (GradientBoostingRegressor)...")
gbdt_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_SEED
)
models_dict['GBDT'] = gbdt_model
print("      ✓ 构建完成 (n_estimators=100, learning_rate=0.1, max_depth=5)")

print("  4.5 XGBoost (XGBRegressor) - 核心实验模型...")
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
models_dict['XGBoost'] = xgb_model
print("      ✓ 构建完成")

print("\n✓ 5个模型全部搭建完成")

# ============================= 第5步：模型训练 =============================
print("\n[步骤5] 模型训练...")

print("  5.1 训练 LinearRegression...")
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
predictions_dict['LinearRegression'] = y_pred_lr
print("      ✓ 训练完成")

print("  5.2 训练 SVR...")
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
predictions_dict['SVR'] = y_pred_svr
print("      ✓ 训练完成")

print("  5.3 训练 RandomForest...")
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
predictions_dict['RandomForest'] = y_pred_rf
print("      ✓ 训练完成")

print("  5.4 训练 GBDT...")
gbdt_model.fit(X_train_scaled, y_train)
y_pred_gbdt = gbdt_model.predict(X_test_scaled)
predictions_dict['GBDT'] = y_pred_gbdt
print("      ✓ 训练完成")

print("  5.5 训练 XGBoost (基础参数)...")
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
predictions_dict['XGBoost'] = y_pred_xgb
print("      ✓ 训练完成")

# ============================= 第6步：XGBoost 网格搜索调参 =============================
print("\n[步骤6] XGBoost 网格搜索调参 (重点优化)...")

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}

print("  网格搜索参数:")
print(f"    max_depth: {param_grid['max_depth']}")
print(f"    learning_rate: {param_grid['learning_rate']}")
print(f"    n_estimators: {param_grid['n_estimators']}")
print(f"  总搜索次数: {np.prod([len(v) for v in param_grid.values()])}")

print("\n  执行网格搜索 (这会耗时 1~2 分钟)...")
xgb_grid = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1)
grid_search = GridSearchCV(
    xgb_grid,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n  ✓ 网格搜索完成")
print(f"  最佳参数: {grid_search.best_params_}")
print(f"  最佳交叉验证 R²: {grid_search.best_score_:.4f}")

xgb_best = grid_search.best_estimator_
y_pred_xgb_tuned = xgb_best.predict(X_test_scaled)
predictions_dict['XGBoost-Tuned'] = y_pred_xgb_tuned

print(f"  ✓ 最优 XGBoost 模型已生成")

# ============================= 第7步：指标计算 =============================
print("\n[步骤7] 计算评估指标...")

def calculate_metrics(y_true, y_pred, model_name):
    """计算 MSE、RMSE、R²"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
    }

print("  计算各模型指标...")
for model_name, y_pred in predictions_dict.items():
    metrics = calculate_metrics(y_test, y_pred, model_name)
    metrics_dict[model_name] = metrics
    print(f"    {model_name:20s}: R² = {metrics['R²']:.4f}, RMSE = {metrics['RMSE']:.4f}")

# ============================= 第8步：结果对比表 =============================
print("\n[步骤8] 生成对比表格...")

metrics_df = pd.DataFrame(metrics_dict).T
metrics_df = metrics_df.round(4)
metrics_df = metrics_df.sort_values('R²', ascending=False)

print("\n" + "="*80)
print("所有模型回归指标对比表")
print("="*80)
print(metrics_df.to_string())

best_model_name = metrics_df.index[0]
best_r2 = metrics_df.loc[best_model_name, 'R²']
print(f"\n✓ 最优模型: {best_model_name}")
print(f"  R² 分数: {best_r2:.4f}")

# ============================= 第9步：可视化 - 真实值 vs 预测值 =============================
print("\n[步骤9] 生成可视化图表...")

print("  9.1 绘制真实值 vs 预测值对比图...")

fig, axes = plt.subplots(2, 3, figsize=(16, 12))
fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.9, bottom=0.08)
axes = axes.flatten()

model_names = list(predictions_dict.keys())
for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    y_pred = predictions_dict[model_name]
    r2 = metrics_dict[model_name]['R²']
    rmse = metrics_dict[model_name]['RMSE']

    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')

    ax.set_xlabel('真实房价', fontsize=11)
    ax.set_ylabel('预测房价', fontsize=11)
    ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.savefig('01_predictions_comparison.png', dpi=300, bbox_inches='tight')
print("      ✓ 已保存: 01_predictions_comparison.png")
plt.close()

print("  9.2 绘制指标对比柱状图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 13))
fig.subplots_adjust(hspace=0.45, wspace=0.3, top=0.94, bottom=0.11, left=0.1)
axes = axes.flatten()

ax = axes[0]
model_names_sorted = metrics_df.index.tolist()
r2_values = metrics_df['R²'].values
colors = ['#FF6B6B' if r2 < 0.5 else '#4ECDC4' if r2 < 0.7 else '#45B7D1' for r2 in r2_values]
ax.bar(range(len(model_names_sorted)), r2_values, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(model_names_sorted)))
ax.set_xticklabels(model_names_sorted, rotation=40, ha='right')
ax.set_ylabel('R² 分数', fontsize=11, fontweight='bold')
ax.set_title('R² 对比 (越高越好)', fontsize=12, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5基准线')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper left', fontsize=9)

ax = axes[1]
rmse_values = metrics_df['RMSE'].values
ax.bar(range(len(model_names_sorted)), rmse_values, color='#FFA07A', alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(model_names_sorted)))
ax.set_xticklabels(model_names_sorted, rotation=40, ha='right')
ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax.set_title('RMSE 对比 (越低越好)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[2]
mae_values = metrics_df['MAE'].values
ax.bar(range(len(model_names_sorted)), mae_values, color='#98D8C8', alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(model_names_sorted)))
ax.set_xticklabels(model_names_sorted, rotation=40, ha='right')
ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax.set_title('MAE 对比 (越低越好)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[3]
mse_values = metrics_df['MSE'].values
ax.bar(range(len(model_names_sorted)), mse_values, color='#F7DC6F', alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(model_names_sorted)))
ax.set_xticklabels(model_names_sorted, rotation=40, ha='right')
ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
ax.set_title('MSE 对比 (越低越好)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('02_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("      ✓ 已保存: 02_metrics_comparison.png")
plt.close()

print("  9.3 绘制特征重要性分析 (Top-15)...")

TOP_N = 15
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

ax = axes[0]
rf_importance = rf_model.feature_importances_
sorted_idx = np.argsort(rf_importance)[::-1][:TOP_N]
ax.barh(range(len(sorted_idx)), rf_importance[sorted_idx], color='#4ECDC4', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('特征重要性', fontsize=11, fontweight='bold')
ax.set_title(f'RandomForest Top-{TOP_N} 特征重要性', fontsize=12, fontweight='bold')
ax.invert_yaxis()

ax = axes[1]
xgb_importance = xgb_best.feature_importances_
sorted_idx_xgb = np.argsort(xgb_importance)[::-1][:TOP_N]
ax.barh(range(len(sorted_idx_xgb)), xgb_importance[sorted_idx_xgb], color='#FF6B6B', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(sorted_idx_xgb)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx_xgb])
ax.set_xlabel('特征重要性', fontsize=11, fontweight='bold')
ax.set_title(f'XGBoost-Tuned Top-{TOP_N} 特征重要性', fontsize=12, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('03_feature_importance.png', dpi=300, bbox_inches='tight')
print("      ✓ 已保存: 03_feature_importance.png")
plt.close()

print("  9.4 绘制预测误差分布...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.94, bottom=0.08)
axes = axes.flatten()

top_models = ['LinearRegression', 'SVR', 'RandomForest', 'XGBoost-Tuned']
for idx, model_name in enumerate(top_models):
    ax = axes[idx]
    y_pred = predictions_dict[model_name]
    errors = y_test.values - y_pred

    ax.hist(errors, bins=30, color='#45B7D1', alpha=0.7, edgecolor='black')
    ax.set_xlabel('预测误差', fontsize=11)
    ax.set_ylabel('频率', fontsize=11)
    ax.set_title(f'{model_name} 误差分布', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
plt.savefig('04_error_distribution.png', dpi=300, bbox_inches='tight')
print("      ✓ 已保存: 04_error_distribution.png")
plt.close()

# ============================= 第10步：实验总结 =============================
print("\n" + "="*80)
print("实验总结")
print("="*80)

print("\n【模型非线性拟合能力排序】")
for idx, (model_name, metrics) in enumerate(metrics_df.iterrows(), 1):
    print(f"  {idx}. {model_name:20s}: R² = {metrics['R²']:.4f}, RMSE = {metrics['RMSE']:.4f}")

print("\n【关键发现】")
print(f"  ✓ 最优模型: {best_model_name} (R² = {best_r2:.4f})")

lr_r2 = metrics_dict['LinearRegression']['R²']
print(f"\n  ✓ 非线性提升幅度:")
print(f"    相比线性回归: R² 提升 {(best_r2 - lr_r2)/abs(lr_r2)*100:.2f}%")

if 'XGBoost' in metrics_dict and 'XGBoost-Tuned' in metrics_dict:
    xgb_r2_before = metrics_dict['XGBoost']['R²']
    xgb_r2_after = metrics_dict['XGBoost-Tuned']['R²']
    print(f"\n  ✓ XGBoost 调参前后对比:")
    print(f"    调参前 R²: {xgb_r2_before:.4f}")
    print(f"    调参后 R²: {xgb_r2_after:.4f}")
    print(f"    提升: {(xgb_r2_after - xgb_r2_before):.4f}")

print("\n【特征重要性Top-3】")
rf_top3_idx = np.argsort(rf_model.feature_importances_)[::-1][:3]
print("  随机森林:")
for i, idx in enumerate(rf_top3_idx, 1):
    print(f"    {i}. {feature_names[idx]:30s}: {rf_model.feature_importances_[idx]:.4f}")

xgb_top3_idx = np.argsort(xgb_best.feature_importances_)[::-1][:3]
print("  XGBoost:")
for i, idx in enumerate(xgb_top3_idx, 1):
    print(f"    {i}. {feature_names[idx]:30s}: {xgb_best.feature_importances_[idx]:.4f}")

print("\n【结论】")
if 'RandomForest' in metrics_df.index and metrics_df.loc['RandomForest', 'R²'] > 0.5:
    print("  ✓ 集成算法（RandomForest、GBDT、XGBoost）在非线性拟合上显著优于线性模型")
if best_r2 > 0.7:
    print(f"  ✓ 最优模型 {best_model_name} 的 R² 达到 {best_r2:.4f}，拟合效果优秀")
else:
    print(f"  ✓ 最优模型 {best_model_name} 的 R² 为 {best_r2:.4f}，还有优化空间")

print("\n" + "="*80)
print("所有输出已保存至当前文件夹")
print("  - 01_predictions_comparison.png    : 真实值vs预测值对比")
print("  - 02_metrics_comparison.png        : 模型指标对比")
print("  - 03_feature_importance.png        : 特征重要性分析")
print("  - 04_error_distribution.png        : 误差分布分析")
print("="*80)
print("\n实验完成！🎉\n")
