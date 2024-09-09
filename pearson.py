import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, pearsonr

file_path_new = "相关性信息_类别1.xlsx"

data_new = pd.read_excel(file_path_new)
# 选择相关列
columns_of_interest = ["预期销售量/斤", "销售单价均值/(元/斤)", "种植成本/(元/亩)"]
data_filtered = data_new[columns_of_interest].dropna()

# 绘制散点图，查看线性关系
sns.pairplot(data_filtered)
plt.show()

# 检查正态性 - 使用Shapiro-Wilk检验
for column in columns_of_interest:
    stat, p = shapiro(data_filtered[column])
    print(f"{column}的P值: {p}")
    if p > 0.05:
        print(f"{column} 满足正态分布")
    else:
        print(f"{column} 不满足正态分布")

# 计算皮尔森相关系数
for col1 in columns_of_interest:
    for col2 in columns_of_interest:
        if col1 != col2:
            corr, p_value = pearsonr(data_filtered[col1], data_filtered[col2])
            print(f"{col1} 和 {col2} 的皮尔森相关系数: {corr}, P值: {p_value}")
