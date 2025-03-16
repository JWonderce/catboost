import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径
import shap  # 导入 SHAP 库，用于解释模型

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'catboost_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("2型糖尿病血糖控制预测模型")



# 使用滑动条接收输入特征，设置合适的范围和默认值
diabetes_duration = st.sidebar.slider("糖尿病病程 (1=小于5年, 2=5-10年,3=大于10年)", min_value=1, max_value=3, value=1, step=1)
cvd = st.sidebar.slider("心血管病变 (0 = 无, 1 = 有)", min_value=0, max_value=1, value=0, step=1)
comorbidities = st.sidebar.slider("慢性合并症数量", min_value=0, max_value=5, value=1, step=1)
neuropathy = st.sidebar.slider("糖尿病周围神经病变 (0 = 无, 1 = 有)", min_value=0, max_value=1, value=0, step=1)
sbp = st.sidebar.slider("收缩压 (SBP, mmHg)", min_value=80, max_value=200, value=120, step=5)
bmi = st.sidebar.slider("体重指数 (BMI, kg/m²)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
# 使用 mmol/L 单位的合理范围
ldl = st.sidebar.slider("低密度脂蛋白 (LDL-C, mmol/L)", min_value=1.0, max_value=5.2, value=2.6, step=0.1)
fpg = st.sidebar.slider("空腹血糖 (FPG, mmol/L)", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
diet_score = st.sidebar.slider("饮食标准分", min_value=0, max_value=10, value=5, step=1)
exercise_score = st.sidebar.slider("运动标准分", min_value=0, max_value=10, value=5, step=1)
medication_score = st.sidebar.slider("服药标准分", min_value=0, max_value=10, value=5, step=1)
blood_sugar_monitoring_score = st.sidebar.slider("血糖监测标准分", min_value=0, max_value=10, value=5, step=1)
monthly_blood_sugar_checks = st.sidebar.slider("每月血糖检测次数", min_value=0, max_value=30, value=5, step=1)

# 创建输入数据框，将输入的特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    '糖尿病病程': [diabetes_duration],
    '心血管病变': [cvd],
    '慢性合并症数量': [comorbidities],
    '糖尿病周围神经病变': [neuropathy],
    'SBP': [sbp],
    'BMI': [bmi],
    'LDL-C': [ldl],
    'FPG': [fpg],
    '饮食标准分': [diet_score],
    '运动标准分': [exercise_score],
    '服药标准分': [medication_score],
    '血糖监测标准分': [blood_sugar_monitoring_score],
    '每月血糖检测次数': [monthly_blood_sugar_checks]
})



 
# 添加预测按钮
if st.button("预测"):
    # 使用模型进行预测
    prediction = model.predict(input_data)
    st.write(f"血糖控制达标可能性: {prediction[0]}")  # 仅显示第一个样本的预测结果

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)  # 计算树模型的 SHAP 值
    shap_values = explainer(input_data)

   

    # 选择第一个样本进行 SHAP 解释
    sample_shap_values = shap_values[0]  # 确保索引 0 有效
    expected_value = explainer.expected_value  # 获取 SHAP 期望值

    # 创建 SHAP Explanation 对象
    explanation = shap.Explanation(
        values=sample_shap_values.values,  # SHAP 值
        base_values=expected_value,  # 期望值
        data=input_data.iloc[0].values,  # 该样本的输入特征
        feature_names=input_data.columns.tolist()  # 特征名称
    )

    # 生成 SHAP 力图并保存为 HTML
    shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))

    # 在 Streamlit 中显示 SHAP 力图
    st.subheader("模型预测的 SHAP 力图")
    with open("shap_force_plot.html") as f:
        st.components.v1.html(f.read(), height=600)
