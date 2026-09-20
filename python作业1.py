"""
Python 实操题：Carseats 数据集多元线性回归分析
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------
# 1. 加载数据与预处理
# ---------------------------------------------------------
# 注意：Carseats 数据集通常包含在 ISLR 相关的包中。
# 这里假设你有一个 'Carseats.csv' 文件在同目录下。
# 如果没有，可以使用以下代码生成一个模拟的或者从网上下载。
# 这里为了代码可运行，我们尝试直接读取本地文件。
try:
    df = pd.read_csv('Carseats.csv')
except FileNotFoundError:
    print("错误：未找到 Carseats.csv 文件。请确保文件在当前目录下。")
    print("你可以从以下地址下载：https://www.statlearning.com/s/Carseats.csv")
    exit()

# 检查数据前几行
print("数据预览：")
print(df.head())
print("-" * 50)

# 定义响应变量和自变量
# Sales: 响应变量
# Price, Income, Advertising: 数值型自变量
# ShelveLoc: 分类型自变量
y = df['Sales']
X = df[['Price', 'Income', 'Advertising', 'ShelveLoc']]

# 将分类变量 ShelveLoc 转换为哑变量 (Dummy Variables)
# drop_first=True 表示丢弃第一个类别作为基准组，避免多重共线性
X = pd.get_dummies(X, columns=['ShelveLoc'], drop_first=True, dtype=int)

# 添加常数项 (截距项)
X = sm.add_constant(X)

# ---------------------------------------------------------
# 2. 建立多元线性回归模型并拟合
# ---------------------------------------------------------
model = sm.OLS(y, X).fit()

# 提取模型拟合报告
print("模型拟合报告 (OLS Regression Results)：")
print(model.summary())
print("-" * 50)

# ---------------------------------------------------------
# 3. 回答问题
# ---------------------------------------------------------

# 问题 1: 指出 ShelveLoc 的基准组是什么？
# 根据 pd.get_dummies(drop_first=True) 的逻辑，第一个类别会被丢弃作为基准。
# 查看原始数据的 ShelveLoc 类别顺序（通常是字母顺序：Bad, Good, Medium）。
# 因此，基准组应该是 'Bad'。
# 在报告中，你会看到 ShelveLoc_Good 和 ShelveLoc_Medium 两个变量。
print("【问题1解答】")
print("ShelveLoc 的基准组是：Bad (因为 drop_first=True 丢弃了第一个类别，通常是字母序第一个)")
print("-" * 50)

# 问题 2: 解读 ShelveLoc[Good] 系数的实际商业含义。
# 系数值表示：在保持 Price, Income, Advertising 不变的情况下，
# 货架位置为 Good 相比 货架位置为 Bad (基准组) 时，Sales 的平均变化量。
coef_good = model.params.get('ShelveLoc_Good', 0)
print("【问题2解答】")
print(f"ShelveLoc_Good 的系数为: {coef_good:.4f}")
print(f"商业含义：在价格、收入和广告支出不变的情况下，")
print(f"将货架位置从 'Bad' 改善为 'Good'，预计销售额 (Sales) 平均会增加 {coef_good:.4f} 个单位。")
print("-" * 50)

# 问题 3: 计算各变量的 VIF，评估是否存在多重共线性风险。
# 注意：VIF 计算时通常不包含常数项（const），或者包含但解释不同。
# 标准做法是计算自变量（不含截距）的 VIF。
X_vif = X.drop(columns=['const']) # 移除常数项计算 VIF

vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("【问题3解答】")
print("各变量的方差膨胀因子 (VIF)：")
print(vif_data)

# 评估风险
max_vif = vif_data['VIF'].max()
print(f"\n最大 VIF 值为: {max_vif:.2f}")
if max_vif > 10:
    print("结论：存在严重的多重共线性风险 (VIF > 10)。")
elif max_vif > 5:
    print("结论：存在中等程度的多重共线性风险 (VIF > 5)。")
else:
    print("结论：各变量间多重共线性风险较低 (VIF < 5)。")
