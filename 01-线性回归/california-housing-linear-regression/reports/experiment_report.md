# Linear Regression Experiment Report / 线性回归实验报告

## 1. Task / 实验任务

English:
The task is to predict median house value using the California Housing dataset.

中文：
本实验任务是使用 California Housing 数据集预测房屋价值中位数。

## 2. Dataset / 数据集

English:
The dataset contains 20,640 samples and 8 housing-related features, including
median income, house age, average rooms, average bedrooms, population,
average occupancy, latitude, and longitude.

中文：
该数据集包含 20,640 条样本和 8 个房屋相关特征，包括收入中位数、房龄、
平均房间数、平均卧室数、人口、平均居住人数、纬度和经度。

English:
The target value is median house value in units of 100,000 dollars. The target
is capped near 5.0, so many expensive houses share the same maximum target
value.

中文：
目标值是房屋价值中位数，单位为 10 万美元。该目标值在约 5.0 处被截断，
因此许多高价房屋会共享相同的最大目标值。

## 3. Method / 实验方法

English:
A Linear Regression model was used as a simple baseline model. The data was
split into training and testing sets with an 80:20 ratio. StandardScaler was
applied to normalize feature scales.

中文：
本实验使用线性回归作为简单基线模型。数据按照 80:20 的比例划分为训练集
和测试集，并使用 StandardScaler 对特征进行标准化。

## 4. Evaluation Metrics / 评估指标

English:

- MAE: Mean Absolute Error, easy to interpret because it has the same unit as
  the target.
- MSE: Mean Squared Error, penalizes large errors more strongly.
- RMSE: Root Mean Squared Error, also in the same unit as the target.
- R2: Coefficient of Determination, measures how much target variance the model
  explains.

中文：

- MAE：平均绝对误差，单位与目标值相同，较容易解释。
- MSE：均方误差，对大误差惩罚更强。
- RMSE：均方根误差，单位也与目标值相同。
- R2：决定系数，用于衡量模型解释了多少目标变量方差。

## 5. Results / 实验结果

English:
The experiment produced the following test-set metrics:

中文：
测试集上的实验指标如下：

| Metric / 指标 | Value / 数值 |
| ------------- | -----------: |
| MAE           |       0.5332 |
| MSE           |       0.5559 |
| RMSE          |       0.7456 |
| R2            |       0.5758 |

English:
Basic data checks:

中文：
基础数据检查：

| Item / 项目                                 |       Value / 数值 |
| ------------------------------------------- | -----------------: |
| Dataset shape / 数据集形状                  |          20640 x 8 |
| Train shape / 训练集形状                    |          16512 x 8 |
| Test shape / 测试集形状                     |           4128 x 8 |
| Test target range / 测试目标范围            |   0.1500 to 5.0000 |
| Prediction range / 预测范围                 | -1.0138 to 11.5003 |
| Targets capped near 5 / 约为 5 的目标值数量 |                184 |
| Predictions below 0 / 小于 0 的预测数量     |                 15 |
| Predictions above 5 / 大于 5 的预测数量     |                 39 |

English:
The learned feature coefficients sorted by absolute value were:

中文：
按绝对值排序后的模型特征系数如下：

| Feature / 特征 | Coefficient / 系数 |
| -------------- | -----------------: |
| Latitude       |          -0.896929 |
| Longitude      |          -0.869842 |
| MedInc         |           0.854383 |
| AveBedrms      |           0.339259 |
| AveRooms       |          -0.294410 |
| HouseAge       |           0.122546 |
| AveOccup       |          -0.040829 |
| Population     |          -0.002308 |

## 6. Visualization Analysis / 可视化分析

English:
The true-vs-predicted plot compares each real target value with the model
prediction. Points close to the dashed ideal line are accurate predictions.
The vertical band around true value 5.0 is expected because the California
Housing target is capped near 5.0.

中文：
真实值与预测值图用于比较每个样本的真实目标值和模型预测值。点越接近虚线
理想预测线，说明预测越准确。真实值约为 5.0 附近的竖直带状分布是正常现象，
因为 California Housing 的目标值在约 5.0 处被截断。

English:
The residual plot shows prediction errors. Ideally, residuals should be roughly
centered around zero without a strong pattern. In this experiment, the residual
plot shows visible structure, which suggests that a simple linear model cannot
fully capture all patterns in the housing data.

中文：
残差图展示模型预测误差。理想情况下，残差应该大致围绕 0 随机分布，不应有明显
结构。本实验中残差图存在可见结构，说明简单线性模型无法完全捕捉房价数据中的
所有模式。

## 7. Analysis / 结果分析

English:
Linear Regression provides a useful baseline model. The R2 score is 0.5758,
which means the model explains part of the target variance but still leaves
substantial error. Latitude, longitude, and median income have large
coefficients after standardization, indicating that location and income are
important predictors in this baseline model.

中文：
线性回归提供了一个有用的基线模型。R2 为 0.5758，说明模型解释了一部分目标值
方差，但仍然存在较明显误差。标准化后，纬度、经度和收入中位数的系数较大，
说明在当前基线模型中，地理位置和收入是重要预测因素。

English:
Linear Regression is unconstrained, so it can predict values below 0 or above
the dataset target cap. This explains the few negative predictions and the
large prediction outlier around 11.5.

中文：
线性回归没有输出范围约束，因此可能预测出小于 0 或大于数据集上限的值。这解释了
少量负预测，以及约 11.5 的极端预测值。

Features / 特征:
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude


**加州房价数据集**的特征：

| 英文                 | 中文       | 说明                                      |
| -------------------- | ---------- | ----------------------------------------- |
| **MedInc**     | 中位数收入 | 该地区的中位数家庭收入（以$10,000为单位） |
| **HouseAge**   | 房屋年龄   | 该地区房屋的中位年龄（年）                |
| **AveRooms**   | 平均房间数 | 每户平均房间数                            |
| **AveBedrms**  | 平均卧室数 | 每户平均卧室数                            |
| **Population** | 人口       | 该地区的总人口                            |
| **AveOccup**   | 平均住户数 | 每户平均人数                              |
| **Latitude**   | 纬度       | 地理位置（纬度坐标）                      |
| **Longitude**  | 经度       | 地理位置（经度坐标）                      |

 **目标变量** ：

* **MedHouseVal** = 中位房价（以$100,000为单位


## 8. Conclusion / 结论

English:
This experiment completed a full regression pipeline: data loading,
preprocessing, model training, evaluation, visualization, and analysis. The
model is suitable as a baseline, but more advanced models may be needed for
higher accuracy.

中文：
本实验完成了完整的回归流程：数据加载、预处理、模型训练、评估、可视化和分析。
该模型适合作为基线模型；如果需要更高精度，可以进一步尝试更复杂的模型。
