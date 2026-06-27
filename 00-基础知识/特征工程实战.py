"""
特征工程完整实战案例
从原始数据到模型就绪的完整流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 创建模拟数据集 ====================

def create_sample_dataset():
    """创建一个模拟的客户流失数据集"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        # 数值特征
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'months_subscribed': np.random.randint(1, 120, n_samples),
        'num_purchases': np.random.poisson(10, n_samples),
        'satisfaction_score': np.random.uniform(1, 10, n_samples),

        # 类别特征
        'gender': np.random.choice(['男', '女'], n_samples),
        'city': np.random.choice(['北京', '上海', '广州', '深圳', '其他'], n_samples),
        'membership_level': np.random.choice(['青铜', '白银', '黄金', '钻石'], n_samples),
        'payment_method': np.random.choice(['信用卡', '支付宝', '微信', '银行卡'], n_samples),

        # 时间特征
        'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
    }

    df = pd.DataFrame(data)

    # 生成目标变量（流失与否）
    # 简化的规则：年轻、收入低、订阅时间短、满意度低 → 更容易流失
    churn_prob = (
        (df['age'] < 30) * 0.2 +
        (df['income'] < 40000) * 0.2 +
        (df['months_subscribed'] < 12) * 0.3 +
        (df['satisfaction_score'] < 5) * 0.3
    )
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)

    # 引入缺失值
    df.loc[df.sample(frac=0.1).index, 'income'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'satisfaction_score'] = np.nan

    # 引入异常值
    df.loc[df.sample(frac=0.02).index, 'income'] = df['income'].max() * 10

    return df


# ==================== 1. 数据探索 ====================

def explore_data(df):
    """数据探索"""
    print("="*60)
    print("1. 数据探索")
    print("="*60)

    print(f"\n数据维度: {df.shape}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n前5行:\n{df.head()}")
    print(f"\n统计描述:\n{df.describe()}")

    # 缺失值
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n缺失值统计:\n{missing[missing > 0]}")

    # 目标变量分布
    print(f"\n目标变量分布:\n{df['churn'].value_counts()}")
    print(f"流失率: {df['churn'].mean():.2%}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 数值特征分布
    df['age'].hist(ax=axes[0, 0], bins=30, edgecolor='black')
    axes[0, 0].set_title('年龄分布')

    df['income'].hist(ax=axes[0, 1], bins=30, edgecolor='black')
    axes[0, 1].set_title('收入分布')

    df['satisfaction_score'].hist(ax=axes[0, 2], bins=20, edgecolor='black')
    axes[0, 2].set_title('满意度分布')

    # 类别特征
    df['gender'].value_counts().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('性别分布')

    df['city'].value_counts().plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('城市分布')

    df['churn'].value_counts().plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_title('是否流失')

    plt.tight_layout()
    plt.show()

    # 相关性分析
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.show()


# ==================== 2. 数据清洗 ====================

def clean_data(df):
    """数据清洗"""
    print("\n" + "="*60)
    print("2. 数据清洗")
    print("="*60)

    df_clean = df.copy()

    # 处理异常值（使用IQR方法）
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers_count = ((df[column] < lower) | (df[column] > upper)).sum()
        print(f"{column} 异常值数量: {outliers_count}")

        # 截断异常值
        df[column] = df[column].clip(lower=lower, upper=upper)
        return df

    # 处理收入异常值
    df_clean = remove_outliers(df_clean, 'income')

    # 处理缺失值
    print(f"\n缺失值处理前: {df_clean.isnull().sum().sum()}")

    # 数值特征用中位数填充
    df_clean['income'].fillna(df_clean['income'].median(), inplace=True)
    df_clean['satisfaction_score'].fillna(df_clean['satisfaction_score'].median(), inplace=True)

    print(f"缺失值处理后: {df_clean.isnull().sum().sum()}")

    return df_clean


# ==================== 3. 特征工程 ====================

def feature_engineering(df):
    """特征工程"""
    print("\n" + "="*60)
    print("3. 特征工程")
    print("="*60)

    df_fe = df.copy()

    # 3.1 时间特征提取
    print("\n3.1 时间特征提取")
    df_fe['signup_year'] = df_fe['signup_date'].dt.year
    df_fe['signup_month'] = df_fe['signup_date'].dt.month
    df_fe['signup_dayofweek'] = df_fe['signup_date'].dt.dayofweek
    df_fe['is_weekend_signup'] = df_fe['signup_dayofweek'].isin([5, 6]).astype(int)

    # 3.2 数值特征变换
    print("\n3.2 数值特征变换")

    # 对数变换（收入）
    df_fe['income_log'] = np.log1p(df_fe['income'])

    # 平方根变换（购买次数）
    df_fe['num_purchases_sqrt'] = np.sqrt(df_fe['num_purchases'])

    # 3.3 交互特征
    print("\n3.3 创建交互特征")

    # 每月平均购买次数
    df_fe['purchases_per_month'] = df_fe['num_purchases'] / (df_fe['months_subscribed'] + 1)

    # 收入-年龄比
    df_fe['income_age_ratio'] = df_fe['income'] / df_fe['age']

    # 满意度分组
    df_fe['satisfaction_level'] = pd.cut(
        df_fe['satisfaction_score'],
        bins=[0, 3, 6, 10],
        labels=['低', '中', '高']
    )

    # 3.4 年龄分组
    df_fe['age_group'] = pd.cut(
        df_fe['age'],
        bins=[0, 25, 35, 50, 100],
        labels=['年轻', '青年', '中年', '老年']
    )

    # 3.5 类别特征编码
    print("\n3.5 类别特征编码")

    # 二值特征 - Label Encoding
    le = LabelEncoder()
    df_fe['gender_encoded'] = le.fit_transform(df_fe['gender'])

    # 有序类别 - Ordinal Encoding
    membership_order = {'青铜': 0, '白银': 1, '黄金': 2, '钻石': 3}
    df_fe['membership_encoded'] = df_fe['membership_level'].map(membership_order)

    # 无序类别 - One-Hot Encoding
    df_fe = pd.get_dummies(df_fe, columns=['city', 'payment_method'], prefix=['city', 'payment'])

    print(f"\n特征工程后的特征数量: {df_fe.shape[1]}")

    return df_fe


# ==================== 4. 特征选择 ====================

def feature_selection(X_train, y_train, X_test, feature_names):
    """特征选择"""
    print("\n" + "="*60)
    print("4. 特征选择")
    print("="*60)

    # 4.1 方差阈值
    print("\n4.1 删除低方差特征")
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_train)
    removed_features = [f for f, v in zip(feature_names, selector.get_support()) if not v]
    if removed_features:
        print(f"删除的低方差特征: {removed_features}")

    # 4.2 单变量特征选择
    print("\n4.2 单变量特征选择（SelectKBest）")

    selector = SelectKBest(f_classif, k=15)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # 获取得分
    scores = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)

    print("\n特征重要性得分（Top 10）:")
    print(scores.head(10))

    # 4.3 树模型特征重要性
    print("\n4.3 基于树模型的特征重要性")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n特征重要性（Top 10）:")
    print(importances.head(10))

    # 可视化
    plt.figure(figsize=(12, 6))
    importances.head(15).plot(x='Feature', y='Importance', kind='barh')
    plt.xlabel('重要性')
    plt.title('特征重要性排名（Top 15）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # 选择重要性>0.01的特征
    important_features = importances[importances['Importance'] > 0.01]['Feature'].tolist()
    print(f"\n选择的重要特征数量: {len(important_features)}")

    return important_features


# ==================== 5. 特征缩放 ====================

def feature_scaling(X_train, X_test):
    """特征缩放"""
    print("\n" + "="*60)
    print("5. 特征缩放")
    print("="*60)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"缩放前 - 训练集均值: {X_train.mean():.2f}, 标准差: {X_train.std():.2f}")
    print(f"缩放后 - 训练集均值: {X_train_scaled.mean():.2f}, 标准差: {X_train_scaled.std():.2f}")

    return X_train_scaled, X_test_scaled, scaler


# ==================== 6. 模型训练与评估 ====================

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练和评估模型"""
    print("\n" + "="*60)
    print("6. 模型训练与评估")
    print("="*60)

    # 训练模型
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 评估
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    print("\n测试集详细报告:")
    print(classification_report(y_test, y_pred_test,
                                target_names=['未流失', '流失']))

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5折交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model


# ==================== 完整流程 ====================

def complete_pipeline():
    """完整的特征工程流程"""
    print("\n" + "🎯"*30)
    print("特征工程完整实战流程")
    print("🎯"*30 + "\n")

    # 1. 创建数据
    print("创建模拟数据集...")
    df = create_sample_dataset()

    # 2. 数据探索
    explore_data(df)

    input("\n按回车键继续数据清洗...")

    # 3. 数据清洗
    df_clean = clean_data(df)

    # 4. 特征工程
    df_fe = feature_engineering(df_clean)

    # 5. 准备建模数据
    print("\n" + "="*60)
    print("5. 准备建模数据")
    print("="*60)

    # 删除不需要的列
    drop_cols = ['churn', 'signup_date', 'gender', 'membership_level',
                 'satisfaction_level', 'age_group']
    feature_cols = [col for col in df_fe.columns if col not in drop_cols]

    X = df_fe[feature_cols]
    y = df_fe['churn']

    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    input("\n按回车键继续特征选择...")

    # 6. 特征选择
    important_features = feature_selection(
        X_train.values, y_train.values, X_test.values, feature_cols
    )

    # 只保留重要特征
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]

    # 7. 特征缩放
    X_train_scaled, X_test_scaled, scaler = feature_scaling(
        X_train_selected.values, X_test_selected.values
    )

    input("\n按回车键继续模型训练...")

    # 8. 模型训练
    model = train_and_evaluate(X_train_scaled, y_train.values,
                               X_test_scaled, y_test.values)

    # 9. 对比：不做特征工程
    print("\n" + "="*60)
    print("对比实验：不做特征工程的效果")
    print("="*60)

    # 只用原始数值特征
    basic_features = ['age', 'income', 'months_subscribed', 'num_purchases', 'satisfaction_score']
    X_basic = df_clean[basic_features]

    X_train_basic, X_test_basic, _, _ = train_test_split(
        X_basic, y, test_size=0.2, random_state=42, stratify=y
    )

    # 缩放
    scaler_basic = StandardScaler()
    X_train_basic_scaled = scaler_basic.fit_transform(X_train_basic)
    X_test_basic_scaled = scaler_basic.transform(X_test_basic)

    # 训练
    model_basic = LogisticRegression(max_iter=1000, random_state=42)
    model_basic.fit(X_train_basic_scaled, y_train)

    # 评估
    y_pred_basic = model_basic.predict(X_test_basic_scaled)
    acc_basic = accuracy_score(y_test, y_pred_basic)

    print(f"\n基础模型（无特征工程）准确率: {acc_basic:.4f}")
    print(f"完整特征工程后准确率: {accuracy_score(y_test, model.predict(X_test_scaled)):.4f}")
    print(f"提升: {(accuracy_score(y_test, model.predict(X_test_scaled)) - acc_basic):.4f}")

    print("\n" + "="*60)
    print("✅ 完整流程演示完成！")
    print("="*60)

    print("\n关键收获：")
    print("1. 数据探索帮助理解数据特点")
    print("2. 数据清洗提高数据质量")
    print("3. 特征工程创造更多有价值的特征")
    print("4. 特征选择减少冗余，提高效率")
    print("5. 特征缩放帮助模型更好收敛")
    print("6. 完整流程显著提升模型性能")


# ==================== 主函数 ====================

if __name__ == "__main__":
    complete_pipeline()
