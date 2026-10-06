# 作业完整方案：ElasticNet 分析 Kaggle Netflix Prize数据集

> 目标：预测用户对电影的评分（回归任务，目标y=电影评分1~5），使用**ElasticNet（弹性网，L1+L2混合正则）**，对比OLS、Ridge、Lasso，理解正则、多重共线性、特征选择。 数据集：Kaggle Netflix Prize，原始是用户ID、电影ID、评分；**不能直接丢ID进线性模型，需要构造特征矩阵**。

## 一、三道选择题（巩固知识点）

Q1：**B**，右奇异向量$v_i$是$A^\top A$的特征向量。

> SVD：$A=U\Sigma V^\top$，$A^\top A=V\Sigma^2 V^\top$，所以V的列是$A^\top A$特征向量。

Q2：**A** Eckart–Young定理：截断SVD在**Frobenius范数、谱范数(2范数)**下是最优秩k逼近。

Q3：**B**，岭回归在奇异向量方向$\sigma_j$的收缩因子：$\frac{\sigma_j^2}{\sigma_j^2+\lambda}$。

------

## 

### 1. 背景介绍

Netflix Prize数据集：用户对电影评分，预测用户评分。 ElasticNet：融合Lasso(L1)和Ridge(L2)；L2收缩系数，L1可以把不重要特征压缩到0，实现特征筛选；适合特征多、特征之间存在相关性的场景，正好匹配Netflix用户-电影高维数据。 损失函数： $$ \mathcal{L}(\beta)=\frac12|y-X\beta|_2^2+\alpha\left(\rho|\beta|_1+\frac{1-\rho}{2}|\beta|_2^2\right) $$ $\rho$就是`l1_ratio`：$\rho=0$等价岭回归；$\rho=1$等价Lasso。

### 2. 数据获取与预处理（重点！Netflix原始数据是长表）

原始数据格式：`user_id, movie_id, rating, date`

> ❗不能直接把user_id、movie_id当数值特征！要做**独热编码 / 低维嵌入**，但独热后维度爆炸，所以抽样小样本做课程作业。

预处理步骤：

1. 下载Kaggle Netflix Prize数据，**抽样小部分子集**（千万级原始数据跑不动，作业取几千用户+几百电影）
2. 构造用户-电影评分矩阵（行=用户，列=电影，值=评分）
3. 填充缺失值（用户没看过的电影，均值填充）
4. 划分训练集、测试集
5. **标准化特征！！** ElasticNet对特征尺度敏感，必须StandardScaler（L1/L2正则依赖特征量纲）
6. 目标变量y：用户评分

> 注意：高维p>>n场景，正好对应前面课堂快问OLS无解的问题，适合用来对比OLS爆炸、ElasticNet正则稳定。

### 3. 模型训练：ElasticNet + 交叉验证调参

两个超参数：

- `alpha`：正则强度，越大惩罚越强
- `l1_ratio`：L1占比，0~1之间

使用`ElasticNetCV`做5折交叉验证，自动搜索最优`alpha,l1_ratio`。 对比基线模型：OLS、Ridge、Lasso。

### 4. 模型评估

回归指标：

- RMSE（Netflix比赛标准指标）
- $R^2$训练集、测试集
- 查看系数：哪些电影特征被ElasticNet压缩到0（L1特征选择效果） 画图：

1. 不同`l1_ratio`下，测试RMSE变化曲线
2. 最优模型的系数分布，看哪些特征被置零
3. 残差图，观察预测误差分布

### 5. 结果分析 & 讨论（作业报告核心）

1. OLS：p>n时无解；特征共线性时系数爆炸，测试误差很大
2. Ridge(l1_ratio=0)：保留全部特征，只收缩系数，**不能做特征筛选**
3. Lasso(l1_ratio=1)：容易在高度相关的一组特征里随机选一个，不稳定
4. ElasticNet：折中！既能收缩系数，又能自动把无关特征压缩到0；对高度相关特征组表现比Lasso稳定。
5. 缺点：线性模型表达能力有限，不如矩阵分解SVD推荐模型。

### 6. 结论

ElasticNet在高维用户电影评分预测任务中，通过L1+L2正则解决OLS的奇异/系数爆炸问题，同时自动筛选特征，降低过拟合。

------

## 完整Python代码

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ====================== 1. 读取Netflix数据（Kaggle netflix-prize-data）======================
# 这里用抽样子集演示，原始数据要读取combined_data_*.txt
# 下面代码模拟Netflix用户电影评分矩阵，如果你本地下载了真实数据替换这部分即可
np.random.seed(42)
n_users = 800    # 样本：用户数量
n_movies = 120   # 特征：电影数量，p=120 > n=800不成立，你可以改成1200制造p>n场景
# 模拟用户电影评分矩阵
rating_mat = np.random.normal(loc=3.2, scale=0.8, size=(n_users, n_movies))
rating_mat = np.clip(rating_mat, 1,5) # 限制评分1~5

df = pd.DataFrame(rating_mat)
X = df.iloc[:, 1:]
y = df.iloc[:, 0] # 预测第一个电影的用户评分

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ElasticNet必须标准化！
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== 2. ElasticNet CV 交叉验证调参 ======================
# 搜索超参数：alpha 和 l1_ratio
enet_cv = ElasticNetCV(
    l1_ratio = [0.1,0.3,0.5,0.7,0.9],
    alphas = np.logspace(-4, 2, 50),
    cv=5,
    max_iter=10000,
    random_state=42
)
enet_cv.fit(X_train_scaled, y_train)
print(f"最优 alpha: {enet_cv.alpha_:.4f}")
print(f"最优 l1_ratio: {enet_cv.l1_ratio_:.2f}")

# 预测
y_pred_enet = enet_cv.predict(X_test_scaled)
rmse_enet = np.sqrt(mean_squared_error(y_test, y_pred_enet))
r2_enet = r2_score(y_test, y_pred_enet)
print(f"ElasticNet Test RMSE: {rmse_enet:.4f}, Test \(R^2\): {r2_enet:.4f}")

# ====================== 3. 对比 OLS / Ridge / Lasso ======================
def evaluate_model(model, Xtr, Xte, ytr, yte, name):
    model.fit(Xtr,ytr)
    yhat = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte,yhat))
    r2 = r2_score(yte,yhat)
    print(f"{name:10s} | RMSE:{rmse:.4f} | \(R^2\):{r2:.4f} | 非零系数数量: {np.sum(model.coef_ !=0)}")

evaluate_model(LinearRegression(), X_train_scaled,X_test_scaled,y_train,y_test,"OLS")
evaluate_model(Ridge(alpha=enet_cv.alpha_), X_train_scaled,X_test_scaled,y_train,y_test,"Ridge")
evaluate_model(Lasso(alpha=enet_cv.alpha_), X_train_scaled,X_test_scaled,y_train,y_test,"Lasso")
evaluate_model(enet_cv, X_train_scaled,X_test_scaled,y_train,y_test,"ElasticNet")

# ====================== 4. 可视化：系数分布 ======================
plt.figure(figsize=(8,4))
plt.plot(enet_cv.coef_, marker='o',linestyle='')
plt.axhline(y=0, color="red", ls="--")
plt.title("ElasticNet 各特征系数")
plt.xlabel("电影特征序号")
plt.ylabel("系数β")
plt.show()

# ====================== 5. 画图：不同l1_ratio下测试RMSE ======================
l1_list = [0,0.2,0.4,0.6,0.8,1.0]
rmse_list = []
for r in l1_list:
    model = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=r, max_iter=10000)
    model.fit(X_train_scaled,y_train)
    yhat = model.predict(X_test_scaled)
    rmse_list.append(np.sqrt(mean_squared_error(y_test,yhat)))

plt.plot(l1_list, rmse_list, marker='o')
plt.xlabel("l1_ratio (0=Ridge,1=Lasso)")
plt.ylabel("Test RMSE")
plt.title("不同L1占比下模型误差")
plt.grid(True)
plt.show()
```



1. ❗**千万不要直接拿原始user_id movie_id当特征**，ID是分类变量，不是数值；要么构造用户/电影特征矩阵，要么做隐因子，否则模型无意义。
2. ❗ElasticNet一定要标准化，因为L1正则对特征scale敏感。
3. 把前面课堂快问知识点串进报告：对比OLS在高维p>n时奇异无解，ElasticNet靠正则解决这个问题，呼应课程SVD、矩阵秩、正则收缩因子。
4. 讨论部分写清楚：Lasso在相关特征组不稳定，ElasticNet的优势就是处理**特征组相关性**，这个是ElasticNet提出的核心动机。

## 

可以用SVD分解用户-电影矩阵，提取隐特征，再喂给ElasticNet，把SVD（刚才选择题知识点）和弹性网结合，分数更高。

如果你需要，我可以帮你直接生成完整作业报告（markdown / word文字版），或者精简成Kaggle notebook完整文字报告。