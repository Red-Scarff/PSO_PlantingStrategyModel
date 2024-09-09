本题中所有用到的表格和程序均已置入此目录

所有运行得到的种植方案结果在“运行结果”目录下

nonlinear_regression.py用于问题三的预期销售量非线性回归
Linear_Regression.py用于问题三的预期销售量线性回归
pearson.py用于问题三判断能否使用皮尔森并输出皮尔森系数
Q1_1.py用于求解问题一第一问
Q1_2.py用于求解问题一第二问
Q2.py用于求解问题二
Q3_cluster.py用于求解第三问（kmeans聚类）
Q3_ABCD_EF.py用于求解第三问（手动二分类地块-大棚）

相关性信息记录了所有作物的预期销售量，种植成本和销售单价均值
相关性信息_类别1.xlsx，相关性信息_类别2.xlsx，相关性信息_类别3.xlsx是问题三kmeans聚类得到的三个类别信息，经过了数据处理
地块信息和作物信息是我们整理的地块和作物信息数据
汇总补充信息记录了2023全年的所有可用数据


要求解问题，请安装python3.11并相应的选择运行Q1_1，Q1_2，Q2
需要安装的库：
pandas,numpy,sko,matplotlib,sklearn,statsmodels,scipy