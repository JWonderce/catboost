import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 加载数据
X_test = pd.read_csv("C:/Users/孔莱熙/Desktop/X_test.csv")
y_test = pd.read_csv("C:/Users/孔莱熙/Desktop/y_test.csv")
X_train = pd.read_csv("C:/Users/孔莱熙/Desktop/X_train.csv")
y_train = pd.read_csv("C:/Users/孔莱熙/Desktop/y_train.csv")

from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold  # 导入KFold
from catboost import CatBoostRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
# CatBoost模型参数
params_cat = {
    'learning_rate': 0.0086,  # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'iterations': 136,  # 弱学习器（决策树）的数量
    'depth': 5,  # 决策树的深度，控制模型复杂度
    'eval_metric': 'RMSE',  # 评估指标，这里使用均方根误差（Root Mean Squared Error，简称RMSE）
    'random_seed': 42,  # 随机种子，用于重现模型的结果
    'verbose': 500  # 控制CatBoost输出信息的详细程度，每100次迭代输出一次
}

# 准备k折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

# 交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = CatBoostRegressor(**params_cat)
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)

    # 预测并计算得分
    y_val_pred = model.predict(X_val_fold)
    score = mean_squared_error(y_val_fold, y_val_pred)  # RMSE
    scores.append(score)
    print(f'第 {fold + 1} 折 RMSE: {score}')

    # 保存得分最好的模型
    if score < best_score:
        best_score = score
        best_model = model

print(f'最佳 RMSE: {best_score}')
from sklearn import metrics

# 使用最佳模型进行测试集预测
y_pred_four = best_model.predict(X_test)
y_pred_list = y_pred_four.tolist()

#保存模型以pkl格式
import pickle
# 保存最佳 CatBoost 模型为 pkl 文件
with open('catboost_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("最佳模型已保存为 best_catboost_model.pkl")
# 计算评估指标
mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)

# 输出评估结果
print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("平均绝对误差 (MAE):", mae)
print("拟合优度 (R-squared):", r2)

import shap
import matplotlib.pyplot as plt


# 设置matplotlib的中文字体
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 或者使用 'SimHei'（黑体）
plt.rcParams['font.size'] = 5  # 设置字体大小

# 构建 shap 解释器
explainer = shap.TreeExplainer(best_model)

# 计算测试集的 shap 值
shap_values = explainer.shap_values(X_test)

# 如果是分类模型，shap_values 可能是一个列表，转换为 Explanation 对象时要传递正确的列名
shap_values = shap.Explanation(values=shap_values,
                               base_values=explainer.expected_value,
                               data=X_test,
                               feature_names=X_test.columns)
#绘制特征重要性图
shap.plots.bar(shap_values, max_display=13)  # 设置为显示所有 13 个特征
# 特征标签
labels = X_test.columns

# 设置中文字体
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 或 'SimHei'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 13

# 绘制 SHAP summary plot解释摘要图
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=labels, plot_type="dot")




import shap
import numpy as np

# 选择一个样本的索引
sample_index = 7  # 替换为你想选择的样本索引

# 保留 SHAP 值的小数三位
shap_values_rounded = np.round(shap_values, 3)

# 绘制单个样本的 Force Plot
shap.force_plot(explainer.expected_value, shap_values_rounded[sample_index], X_test.iloc[sample_index], matplotlib=True)

#热图
# 创建 shap.Explanation 对象
shap_explanation = shap.Explanation(values=shap_values[0:500,:],
                                    base_values=explainer.expected_value,
                                    data=X_test.iloc[0:500,:], feature_names=X_test.columns)
# 绘制热图
# shap.plots.heatmap(shap_explanation)