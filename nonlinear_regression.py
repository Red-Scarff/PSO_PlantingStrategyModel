import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

# Step 1: 读取数据
file_path = "相关性信息_类别3.xlsx"  # 类别1：4，类别2：4，类别3：4
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Step 2: 数据预处理，选择相关的列
X = df[["种植成本/(元/亩)", "销售单价均值/(元/斤)"]].copy()
y = df["预期销售量/斤"]

# Step 3: 填充缺失值（如有）
X = X.fillna(0)
y = y.fillna(0)

# 转换为二次多项式特征
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# 拟合多项式回归模型
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly, y)

print("回归系数:", poly_reg_model.coef_)  # 输出系数
print("截距:", poly_reg_model.intercept_)  # 输出截距

# 预测值和R^2
y_poly_pred = poly_reg_model.predict(X_poly)
r2_poly = r2_score(y, y_poly_pred)
print(f"多项式回归的 R² 值: {r2_poly}")


# def nonlinear_model(X, a, b, c, d):
#     # X[0] 是种植成本，X[1] 是销售单价均值
#     return a * np.exp(b * X[0] + c * X[1]) + d


# # Step 4: 拟合非线性回归模型
# # 提取 X1 (种植成本) 和 X2 (销售单价均值)
# X_data = [X["种植成本/(元/亩)"], X["销售单价均值/(元/斤)"]]

# # 初始猜测的参数值
# initial_guess = [1, 0.01, 0.01, 0.01]

# # 使用 curve_fit 拟合模型
# params, covariance = curve_fit(nonlinear_model, X_data, y, p0=initial_guess)

# # Step 5: 获取拟合参数 a, b, c
# a, b, c, d = params
# print(f"拟合的参数: a = {a}, b = {b}, c = {c},d={d}")

# y_pred = nonlinear_model(X_data, a, b, c, d)
# r2 = r2_score(y, y_pred)
# print(f"非线性回归的 R² 值: {r2}")
