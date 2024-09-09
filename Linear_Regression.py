import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Step 1: 读取数据
file_path = "相关性信息_类别1.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Step 2: 数据预处理，选择相关的列
X = df[["种植成本/(元/亩)", "销售单价均值/(元/斤)"]].copy()
y = df["预期销售量/斤"]

# Step 3: 填充缺失值（如有）
X = X.fillna(0)
y = y.fillna(0)

# Step 4: 线性回归模型拟合
linear_reg = LinearRegression()
linear_reg.fit(X, y)
print("回归系数:", linear_reg.coef_)  # 输出系数
print("截距:", linear_reg.intercept_)  # 输出截距
# Step 5: 计算 R^2 值
y_pred = linear_reg.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² 值: {r2}")

# Step 6: 计算 VIF（方差膨胀因子）
X_with_constant = sm.add_constant(X)  # 为 statsmodels 添加常数项
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_constant.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_with_constant.values, i)
    for i in range(X_with_constant.shape[1])
]
print("VIF 数据:\n", vif_data)
