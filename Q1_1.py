# %%
import pandas as pd
import numpy as np
from sko.PSO import PSO
import matplotlib.pyplot as plt

MIN_AREA_PENG = 0.2
MIN_AREA = 6
# %%
# 读取数据
land_data = pd.read_excel("地块信息.xlsx")  # 读取地块信息
crop_data = pd.read_excel("作物信息.xlsx")  # 读取作物基本信息
crop_supplement = pd.read_excel("汇总补充信息.xlsx")  # 读取作物补充信息
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


# 约束函数


def ABC(x):  # 平旱地，梯田，山坡地
    for k in range(7):
        for block in range(0, num_blocks):
            # ABC约束
            if block < 26:
                for j in range(15, 41):
                    x[block][j][k] = 0
    return True


def water_crop_vege(x, areas):  # 水浇地约束
    flag_crop = [False] * 8
    for k in range(7):
        for i in range(26, 34):  # 第一季
            for j in range(41):
                if j < 15 or j >= 34:
                    x[i][j][k] = 0  # 不能种水稻和蔬菜之外的
                if j == 15 and x[i][j][k] > 0:
                    for j_vege in range(16, 34):
                        x[i][j_vege][k] = 0  # 种水稻种不了蔬菜
                    flag_crop[i - 26] = True
                elif j >= 16 and j < 34 and x[i][j][k] > 0:
                    x[i][15][k] = 0
        for cnt in range(8):  # 第二季
            i = cnt + 54
            if flag_crop[cnt]:
                for j in range(41):
                    x[i][j][k] = 0
            else:
                for j in range(41):
                    if j >= 34 and j < 37:
                        continue  # 特殊蔬菜
                    else:
                        x[i][j][k] = 0
    return True


def ordipeng(x):  # 普通大棚
    for k in range(7):
        for i in range(34, 50):  # 第一季
            for j in range(41):
                if j >= 16 and j < 34:
                    continue  # 正常蔬菜
                else:
                    x[i][j][k] = 0
        for i in range(62, 78):  # 第二季
            for j in range(41):
                if j >= 37:
                    continue  # 菌类
                else:
                    x[i][j][k] = 0

    return True


def smartpeng(x):  # 智慧大棚
    for k in range(7):
        for i in range(50, 54):  # 第一季
            for j in range(41):
                if j >= 16 and j < 34:
                    continue  # 正常蔬菜
                else:
                    x[i][j][k] = 0
        for i in range(78, 82):  # 第二季
            for j in range(41):
                if j >= 16 and j < 34:
                    continue  # 正常蔬菜
                else:
                    x[i][j][k] = 0
    return True


# def seperate(x):
#     for k in range(7):
#         for i in range(82):
#             for j in range(41):
#                 if (i >= 34 and i < 54) or (i >= 62):  # 大棚
#                     if x[i][j][k] > 0 and x[i][j][k] < 0.2:
#                         x[i][j][k] = 0
#                 else:  # 普通地
#                     if x[i][j][k] > 0 and x[i][j][k] < 6:
#                         x[i][j][k] = 0
#     return True


def is_bean(x):  # 豆类约束，j编号在0-4，16-18为豆
    for i in range(26):  # 粮食豆
        for j in range(0, 5):
            for k in range(0, 5):
                if x[i][j][k] + x[i][j][k + 1] + x[i][j][k + 2] == 0:
                    x[i][j][k] = 6
    for i in range(26, 54):  # 菜豆
        for j in range(16, 19):
            for k in range(0, 5):
                if x[i][j][k] + x[i][j][k + 1] + x[i][j][k + 2] == 0:
                    if i < 34:
                        x[i][j][k] = 6
                    else:
                        x[i][j][k] = 0.2
    for i in range(62, 82):
        for j in range(16, 19):
            for k in range(0, 5):
                if x[i][j][k] + x[i][j][k + 1] + x[i][j][k + 2] == 0:
                    x[i][j][k] = 0.2
    return True


def again(x):  # 重茬
    for i in range(54):
        for j in range(41):
            for k in range(6):
                if i < 26 and x[i][j][k] * x[i][j][k + 1] != 0:
                    x[i][j][k] = 0
                elif i >= 50 and (
                    x[i][j][k] * x[i + 28][j][k] != 0
                    or x[i + 28][j][k] * x[i][j][k + 1] != 0
                ):  # 含第二季的情况
                    x[i + 28][j][k] = 0
    return True


def arealimit(x):  # 种地面积约束
    for k in range(7):
        for i in range(0, 82):
            block_total_area = sum(
                x[i][j][k] for j in range(41)
            )  # 当前地块所有作物种植面积的总和
            if block_total_area > areas[i]:
                # 记录种的第一多和第二多的植物
                one = 0
                two = 0
                more = 0  # 不合格的累及面积
                for j in range(41):
                    x[i][j][k] *= areas[i] / block_total_area
                    if i < 34 or (i >= 54 and i < 62):  # 普通地块
                        if j > 5:
                            if x[i][j][k] > x[i][one][k]:
                                one = j
                            elif x[i][j][k] > x[i][two][k]:
                                two = j
                            if x[i][j][k] < 6:
                                more += x[i][j][k]  # 不满足条件的多余的加起来
                                x[i][j][k] = 0
                        else:
                            continue
                    else:  # 大棚
                        if j < 16 or j >= 19:  # 非豆类
                            if x[i][j][k] > x[i][one][k]:
                                one = j
                            elif x[i][j][k] > x[i][two][k]:
                                two = j
                            if x[i][j][k] < 0.2:
                                more += x[i][j][k]
                                x[i][j][k] = 0
                x[i][one][k] += more / 2
                x[i][two][k] += more / 2
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
    is_bean(x)
    again(x)
    ABC(x)
    water_crop_vege(x, areas)
    ordipeng(x)
    smartpeng(x)
    # seperate(x)
    arealimit(x)
    water_crop_vege(x, areas)
    ordipeng(x)
    smartpeng(x)

    for k in range(7):
        for row in range(num_rows):
            crop_areas = 0
            j = crop_ids[row]
            crop_areas = select_range(x, marks[row], j, crop_areas, k)
            production = crop_yield[row] * crop_areas  # 计算产量
            cost = crop_cost[row] * crop_areas  # 计算种植成本
            revenue = (
                min(production, expected_demand[row]) * crop_price[row]
            )  # 计算有效收入(考虑滞销)
            total_profit += revenue - cost  # 累加净收益
    return -total_profit  # 目标是最大化收益，因此返回负的净收益


# %%

# 定义种植面积的上下界
lb = [0] * (num_blocks * 41 * 7)  # 下界为0
ub = [
    area for block_area in areas for area in [block_area] * 41 * 7
]  # 上界为各地块面积
# 使用PSO算法进行优化
pso = PSO(
    func=profit_function,
    dim=82 * 41 * 7,
    pop=50,
    max_iter=200,
    lb=lb,
    ub=ub,
    w=0.9,
    c1=2,
    c2=2,
)
pso.run()
best_pos = pso.gbest_x
best_profit = pso.gbest_y
# %%
# 处理优化结果
best_pos = [round(p, 2) for p in best_pos]  # 将结果四舍五入到小数点后两位
len(best_pos)
# 绘制全局最优目标函数值随迭代次数的变化
plt.plot(pso.gbest_y_hist, "b-", label="Best Objective Value")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("PSO Optimization Convergence")
plt.legend()
plt.show()
# %%
# 打印最终的最大化的利润
final_profit = -best_profit  # 由于目标函数返回的是负的净收益，这里取相反数
print(f"2024年到2030年累计的最终最大利润: {final_profit}元")
# %%
# 读取结果模板
xls = pd.ExcelFile("result1_1.xlsx")
best_pos_3d = np.reshape(best_pos, (82, 41, 7))
# 读sheet
dfs = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}

for idx, sheet_name in enumerate(xls.sheet_names):
    # 第三维度是表的个数
    sheet_data = best_pos_3d[:, :, idx]

    # 逐行
    for row in range(sheet_data.shape[0]):
        for col in range(sheet_data.shape[1]):
            dfs[sheet_name].iloc[row, col] = sheet_data[row, col]

# 保存
output_path = "updated_result1_1.xlsx"
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, df in dfs.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
