# %%
import pandas as pd
import numpy as np
from sko.PSO import PSO
import matplotlib.pyplot as plt

MIN_AREA_PENG = 0.2
MIN_AREA = 6
# %%
# 读取数据
land_data = pd.read_excel("附件1.xlsx")  # 读取地块信息
land_supplement = pd.read_excel("附件1-补充.xlsx")  # 读取地块补充信息
crop_data = pd.read_excel("2023总产量.xlsx")  # 读取作物基本信息
crop_supplement = pd.read_excel("牛逼文件.xlsx")  # 读取作物补充信息
# %%
# 转换数据类型并清理数据
land_data["地块面积/亩"] = pd.to_numeric(
    land_data["地块面积/亩"], errors="coerce"
).fillna(0)
crop_supplement["亩产量/斤"] = pd.to_numeric(
    crop_supplement["亩产量/斤"], errors="coerce"
).fillna(0)
crop_supplement["种植成本/(元/亩)"] = pd.to_numeric(
    crop_supplement["种植成本/(元/亩)"], errors="coerce"
).fillna(0)
crop_supplement["销售单价均值/(元/斤)"] = pd.to_numeric(
    crop_supplement["销售单价均值/(元/斤)"], errors="coerce"
).fillna(0)

# %%
# 提取相关信息
areas = land_data["地块面积/亩"].tolist()  # 地块面积列表
land_names = land_data["地块名称"].tolist()  # 地块名称列表
crop_ids = crop_supplement["作物编号"].tolist()  # 作物编号列表
crop_names = crop_data["作物名称"].tolist()  # 作物名称列表
crop_yield = crop_supplement["亩产量/斤"].tolist()  # 每亩产量
crop_cost = crop_supplement["种植成本/(元/亩)"].tolist()  # 种植成本
crop_price = crop_supplement["销售单价均值/(元/斤)"].tolist()  # 销售单价
expected_demand = crop_supplement["总产量/斤"].tolist()  # 预期销售量
marks = crop_supplement["标记"].tolist()  # 标记
crop_ids = [x - 1 for x in crop_ids]
# %%
# 增加第二期地块
areas += areas[26:54]
land_names += land_names[26:54]
# %%
# 定义问题参数
num_blocks = len(areas)  # 地块数量
num_crops = len(crop_ids)  # 作物数量
num_rows = len(marks)


# %%
# def water_crop(x):  # 合法
#     for k in range(7):
#         for i in range(26, 34):
#             for j in range(0, 41):
#                 if j == 15:
#                     continue
#                 else:
#                     if x[i][j][k] != 0:
#                         return False
#         for i in range(54, 62):
#             for j in range(41):
#                 x[i][j][k] = 0
#     return True


def water_crop_vege(x):
    flag_crop = False
    flag_vege = False
    for k in range(7):
        for i in range(26, 34):  # 第一季
            for j in range(41):
                if (j < 15 or j >= 34) and x[i][j][k] > 0:
                    x[i][j][k] = 0
                if j == 15 and x[i][j][k] > 0:
                    flag_crop = True
                elif j >= 16 and j < 34 and x[i][j][k] > 0:
                    flag_vege = True
        if flag_vege:
            for i in range(54, 62):  # 第二季
                for j in range(41):
                    if j >= 34 and j < 37:
                        continue  # 特殊蔬菜
                    else:
                        x[i][j][k] = 0
        if flag_crop:
            for i in range(54, 62):
                for j in range(41):
                    x[i][j][k] = 0
    return True


def ordipeng(x):
    for k in range(7):
        for i in range(34, 50):
            for j in range(41):
                if j >= 16 and j < 34:
                    continue  # 正常蔬菜
                else:
                    x[i][j][k] = 0
        for i in range(62, 78):
            for j in range(41):
                if j >= 37:
                    continue  # 菌类
                else:
                    x[i][j][k] = 0

    return True


def smartpeng(x):
    for k in range(7):
        for i in range(50, 54):
            for j in range(41):
                if j >= 16 and j < 34:
                    continue  # 正常蔬菜
                else:
                    x[i][j][k] = 0
        for i in range(78, 82):
            for j in range(41):
                if j >= 16 and j < 34:
                    continue  # 正常蔬菜
                else:
                    x[i][j][k] = 0
    return True


def seperate(x):
    for k in range(7):
        for i in range(82):
            for j in range(41):
                if (i >= 34 and i < 54) or (i >= 62):  # 大棚
                    if x[i][j][k] > 0 and x[i][j][k] < 0.2:
                        x[i][j][k] = 0
                else:  # 普通地
                    if x[i][j][k] > 0 and x[i][j][k] < 6:
                        x[i][j][k] = 0
    return True


def is_bean(x): #j编号在0-4，16-18为豆
    for i in range(82):
        for j in range(0, 5):
            for k in range(0, 5):
                if x[i][j][k] + x[i][j][k + 1] + x[i][j][k + 2] == 0:
                    if i < 34 or (i >= 54 and i < 62):
                        x[i][j][k] = 6
                    else:
                        x[i][j][k] = 0.2
        for j in range(16, 19):
            for k in range(0, 5):
                if x[i][j][k] + x[i][j][k + 1] + x[i][j][k + 2] == 0:
                    if i < 34 or (i >= 54 and i < 62):
                        x[i][j][k] = 6
                    else:
                        x[i][j][k] = 0.2
    return True
    


def again(x):
    for i in range(54):
        for j in range(41):
            for k in range(6):
                if i < 26 and x[i][j][k] * x[i][j][k + 1] != 0:
                    x[i][j][k] = 0
                elif i >= 50 and (
                    x[i][j][k] * x[i + 28][j][k] != 0
                    or x[i + 28][j][k] * x[i][j][k + 1] != 0
                ):
                    x[i + 28][j][k] = 0
    return True


# %%


def select_range(x, s, j, crop_areas, k):
    if s == "平旱地单季":
        for i in range(6):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "梯田单季":
        for i in range(6, 20):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "山坡地单季":
        for i in range(20, 26):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "水浇地单季":
        for i in range(26, 34):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "水浇地第一季":
        for i in range(26, 34):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "水浇地第二季":
        for i in range(54, 62):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "普通大棚第一季":
        for i in range(34, 50):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "智慧大棚第一季":
        for i in range(50, 54):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "普通大棚第二季":
        for i in range(62, 78):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    elif s == "智慧大棚第二季":
        for i in range(78, 82):
            if x[i][j][k] != 0:
                crop_areas += x[i][j][k]
    return crop_areas


# %%
# 定义目标函数
def profit_function(x):
    x = np.reshape(x, (82, 41, 7))  # 变三维数组
    total_profit = 0
     # 多个约束条件
    water_crop_vege(x)
    ordipeng(x)
    smartpeng(x)
    is_bean(x)
    again(x)
    for k in range(7):
        for block in range(0, num_blocks):
            # ABC约束
            if block < 26:
                for j in range(16, 41):
                    x[block][j][k] = 0
            # 第7项约束
            block_total_area = sum(x[block][0:41][k])  # 当前地块所有作物种植面积的总和
            if block_total_area > areas[block]:
                # over_area = block_total_area - areas[block] #多的面积
                # block_total_area -= over_area
                # cnt = sum(1 for j in range(41) if x[block][j][k] != 0)
                # over_area_average =over_area / cnt
                for j in range(41):
                    x[block][j][k] *= (areas[block] / block_total_area)
    #鲁棒优化
    # 调整预期销售量
    num_scenarios = 40
    scenarios_profit = []

    for _ in range(num_scenarios):        
        for k in range(7):
            for row in range(num_rows):
                crop_areas = 0
                j = crop_ids[row]
                crop_areas = select_range(x, marks[row], j, crop_areas, k)
                # 鲁棒优化
                # 预期销售量
                if j == 6 or j == 7: # 小麦，玉米
                    demand_change = np.random.uniform(1.05, 1.10) ** k #按年增长
                else:
                    demand_change = np.random.uniform(0.95, 1.05) ** k
                # 亩产
                yield_change = np.random.uniform(0.90, 1.10)
                # 成本
                cost_change = np.random.uniform(1.04, 1.06) ** k
                # 蔬菜售价
                if j >=16 and j < 37:
                    price_change = np.random.uniform(1.04, 1.06) ** k
                #食用菌
                if j >= 37 and j < 40: #菌类（除羊肚菌）
                    price_change = np.random.uniform(0.95, 0.99) ** k
                elif j == 40: #羊肚菌
                    price_change = 0.95 ** k
                production = crop_yield[row] * yield_change * crop_areas  # 计算产量
                cost = crop_cost[row] * cost_change * crop_areas  # 
                expected_demand[row] *= demand_change
                crop_price[row] *= price_change
                revenue = (
                    min(production, expected_demand[row]) * crop_price[row]
                )  # 计算有效收入(考虑滞销)
                total_profit += revenue - cost + max(0, production - expected_demand[row]) * crop_price[row] / 2 # 累加净收益
        scenarios_profit.append(total_profit)
    total_profit = min(scenarios_profit)
    return -total_profit  # 目标是最大化收益，因此返回负的净收益
#%%

# 定义种植面积的上下界
lb = [0] * (num_blocks * 41 * 7)  # 下界为0
ub = [
    area for block_area in areas for area in [block_area] * 41 * 7
]  # 上界为各地块面积
# 使用PSO算法进行优化
pso = PSO(func=profit_function, dim= 82*41*7 ,pop=50, max_iter=50,lb=lb, ub=ub, w=0.8,c1=2,c2=2)
pso.run()
best_pos = pso.gbest_x
best_profit = pso.gbest_y
# %%
# 处理优化结果
best_pos = [round(p, 2) for p in best_pos]  # 将结果四舍五入到小数点后两位
len(best_pos)
# 绘制全局最优目标函数值随迭代次数的变化
plt.plot(pso.gbest_y_hist, 'b-', label='Best Objective Value')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('PSO Optimization Convergence')
plt.legend()
plt.show()
# %%
# 打印最终的最大化的利润
final_profit = -best_profit  # 由于目标函数返回的是负的净收益，这里取相反数
print(f"2024年到2030年累计的最终最大利润: {final_profit}元")

# 读取结果模板
result_df = pd.read_excel("result1_1.xlsx")
best_pos_3d = np.reshape(best_pos, (82, 41, 7))
for block_index in range(82):
    for crop_index in range(41):
        for season_index in range(7):
            crop_area = best_pos_3d[block_index][crop_index][season_index]
            if crop_area > 0:  # 只填充大于0的值
                crop_name = crop_names[crop_index]  # 从作物名称列表中获取作物名称
                block_name = land_names[block_index]  # 从地块名称列表中获取地块名称

                # 检查地块名称和作物名称是否在DataFrame的列中
                if (
                    block_name in result_df["地块名"].values
                    and crop_name in result_df.columns
                ):
                    # 填充数据
                    result_df.loc[result_df["地块名"] == block_name, crop_name] = (
                        crop_area
                    )

# 保存结果到Excel
result_df.to_excel("result_filled.xlsx", index=False)
print("最优种植方案已保存到'result_filled.xlsx'.")
