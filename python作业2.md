# 课堂快问答案

## 快问1

**答案**：当特征数 $(p>)$样本量$n$时，设计矩阵$X$的秩最大只能是$(n<p)$，$X^TX$不可逆，因此OLS不存在唯一解。

> 一句话背诵版：$(p>)n$时$\text{rank}(X)\le n < p$，$X^TX$是奇异矩阵，没有逆矩阵，OLS无唯一解。

## 快问2

**答案**：拟合值$\hat Y=X\hat\beta$，$\hat Y$是$y$在$X$列空间上的投影，只在列空间内运算；但系数$\hat\beta=(X^TX)^{-1}X^Ty$依赖$X^TX$求逆，当$X$接近共线性时，$X^TX$的特征值极小，求逆会放大噪声，导致系数$\hat\beta$爆炸。

## 快问3

**答案**：OLS会强行用新增噪声特征去拟合训练集里的随机误差，降低训练残差，所以训练$R^2$上升；这些噪声特征对应的真实系数是0。

------

# Python 演示代码（验证这3个现象）

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ========== 1. \(p>\)n：OLS没有唯一解演示 ==========
np.random.seed(42)
n = 10  # 样本量
p = 20 # 特征数 > n
X = np.random.randn(n, p)
y = np.random.randn(n)

\(X^TX\) = X.T @ X
print(f"\(X^TX\)矩阵形状: {\(X^TX\).shape}")
print(f"\(X^TX\)的秩: {np.linalg.matrix_rank(\(X^TX\))}")
try:
    inv_\(X^TX\) = np.linalg.inv(\(X^TX\))
except np.linalg.LinAlgError:
    print("\(X^TX\)奇异，无法求逆，OLS无唯一解")

# ========== 2. 多重共线性，系数爆炸，但拟合值正常 ==========
np.random.seed(42)
n = 100
x1 = np.random.randn(n)
x2 = x1 + 1e-6 * np.random.randn(n) # x2几乎等于x1，高度共线
X = np.c_[x1, x2]
y = 2*x1 + np.random.randn(n)*0.1

beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
y_hat = X @ beta_hat
print(f"\n共线性下估计系数 beta={beta_hat}（系数爆炸）")
print(f"训练集R²: {r2_score(y, y_hat):.4f}（拟合效果依旧很好）")

# ========== 3. 添加无关噪声特征，训练R²上升 ==========
np.random.seed(42)
n = 50
true_x = np.random.randn(n)
y = 3 * true_x + np.random.randn(n)*0.8
X_base = true_x.reshape(-1,1)

r2_list = []
for noise_feature_num in range(0,15):
    # 增加noise_feature_num个纯噪声特征，和y无关，真实系数=0
    noise_X = np.random.randn(n, noise_feature_num)
    X = np.hstack([X_base, noise_X])
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    r2_list.append(r2_score(y,y_pred))

plt.plot(r2_list, marker='o')
plt.xlabel("新增噪声特征数量")
plt.ylabel("训练R²")
plt.title("增加无关噪声特征，训练R²持续上升")
plt.show()
```

运行代码可以直观看到：

1. $(p>)n$时$X^TX$奇异，无法求逆；
2. 高度共线性，系数变得巨大，但$\hat Y$仍然能很好贴合$y$；
3. 不断加入无关噪声，训练集$R^2$持续走高，这就是过拟合。

