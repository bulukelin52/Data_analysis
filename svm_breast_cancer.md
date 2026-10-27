# 

> 数据集：Kaggle Wisconsin Breast Cancer（乳腺癌二分类，和课件SVM完全对应：RBF核、C/gamma超参、网格搜索交叉验证、支持向量） Kaggle数据集地址：https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

```
# SVM支持向量机分类实验报告
## 1. 实验简介
本实验使用**支持向量机SVM（RBF高斯核）**对威斯康星乳腺癌数据集做二分类任务，预测肿瘤良性/恶性，完全对应课件SVM知识点：最大间隔、支持向量、核技巧、超参数C、γ、网格交叉验证调参。

数据集来源：Kaggle Wisconsin Breast Cancer Dataset
任务：二分类，0=良性，1=恶性，569条样本，30个数值特征。

## 2. 依赖安装
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Python完整代码

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================== 1.加载数据集 =====================
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names
print(f"样本总数：{X.shape[0]}, 特征维度：{X.shape[1]}")
print(f"标签：{target_names}")

# ===================== 2.划分训练集测试集 + 标准化 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 3.网格搜索：交叉验证调参C、gamma（课件两个超参） =====================
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}
svm = SVC(random_state=42)
grid = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

print("===== 网格搜索最优参数 =====")
print(f"最优参数组合：{grid.best_params_}")
print(f"训练集5折交叉验证最优准确率：{grid.best_score_:.4f}")
best_model = grid.best_estimator_

# ===================== 4.模型预测与评估 =====================
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率：{acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=target_names))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("SVM混淆矩阵")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 输出支持向量数量（课件重点：只有α_i>0才是支持向量）
print(f"\n支持向量总数：{best_model.n_support_}")
print(f"良性类支持向量个数：{best_model.n_support_[0]}")
print(f"恶性类支持向量个数：{best_model.n_support_[1]}")
```

## 实验结果分析（结合课件知识点）

### 1 核心参数理解（和课件完全对齐）

1. C：间隔宽度 vs 分类错误的权衡

   - C小：惩罚比较轻，间隔更宽，允许更多样本错分，容易欠拟合
   - C大：对错分惩罚很重，尽量减少训练集错误，间隔变窄，容易过拟合

2. γ（RBF核）：单个样本影响半径，控制边界复杂度

   - γ很小：每个样本影响范围很大，决策边界平滑

   - γ很大：每个样本只影响附近一小片区域，边界扭曲、容易过拟合

     > 本实验使用GridSearchCV网格搜索+5折交叉验证自动选出最优C和γ，就是课件说的：**两者靠交叉验证来调**。

### 2 支持向量理解（课件KKT条件、稀疏性）

SVM解 $w=\sum \alpha_i y_i x_i$，只有支持向量满足 $\boldsymbol{\alpha_i>0}$，其余样本$\alpha_i=0$，不参与超平面计算；

- 落在间隔面上的样本：$0<\alpha_i<C$
- 落在间隔内部、被错分的样本：$\boldsymbol{\alpha_i=C}$（对应Q&A问题：$\alpha_i=C$代表该样本违反间隔约束，是被惩罚的样本）

### 3 核技巧

RBF核不需要显式映射到高维空间，只计算样本内积，解决非线性可分问题，对应课件洞察：**只出现内积 → 核技巧，通往非线性**。

### 4 实验结论

该数据集使用RBF-SVM可以达到很高的分类精度，模型依靠少量支持向量决定分类间隔； 调参上，γ过大极易过拟合，C过大也会过拟合，交叉验证是寻找平衡点的可靠方式。

