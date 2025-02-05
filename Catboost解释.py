
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

# 在侧边栏中输入特征
st.sidebar.header("输入特征")  # 侧边栏的标题

# 使用滑动条接收输入特征，设置合适的范围和默认值
diabetes_duration = st.sidebar.slider("糖尿病病程 (年)", min_value=0, max_value=20, value=5, step=1)
cvd = st.sidebar.slider("心血管病变 (0 = 无, 1 = 有)", min_value=0, max_value=1, value=0, step=1)
comorbidities = st.sidebar.slider("慢性合并症数量", min_value=0, max_value=5, value=1, step=1)
neuropathy = st.sidebar.slider("糖尿病周围神经病变 (0 = 无, 1 = 有)", min_value=0, max_value=1, value=0, step=1)
sbp = st.sidebar.slider("收缩压 (SBP, mmHg)", min_value=80, max_value=200, value=120, step=5)
bmi = st.sidebar.slider("体重指数 (BMI, kg/m²)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
ldl = st.sidebar.slider("低密度脂蛋白 (LDL-C, mg/dL)", min_value=50, max_value=200, value=100, step=5)
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

# 添加预测按钮，用户点击后进行模型预测
if st.button("预测"):
    prediction = model.predict(input_data)  # 使用加载的模型进行预测
    if prediction[0] == 0:
        st.write("预测结果: 血糖控制良好 (HbA1c < 7%)")
    else:
        st.write("预测结果: 血糖未控制 (HbA1c ≥ 7%)")

    # 计算 SHAP 值
    explainer = shap.Explainer(model)  # 或者使用 shap.TreeExplainer(model) 来计算树模型的 SHAP 值
    shap_values = explainer(input_data)

    # 提取单个样本的 SHAP 值和期望值
    sample_shap_values = shap_values[0]  # 提取第一个样本的 SHAP 值
    expected_value = explainer.expected_value[0]  # 获取对应输出的期望值

    # 创建 Explanation 对象
    explanation = shap.Explanation(
        values=sample_shap_values[:, 0],  # 选择特定输出的 SHAP 值
        base_values=expected_value,
        data=input_data.iloc[0].values,
        feature_names=input_data.columns.tolist()
    )

    # 保存为 HTML 文件
    shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))

    # 在 Streamlit 中显示 HTML
    st.subheader("模型预测的 SHAP 力图")
    with open("shap_force_plot.html") as f:
        st.components.v1.html(f.read(), height=600)
