
# In[]:
from IPython.display import Image
import pandas as pd

data_dir = 'E:/python-workspace/mici-learning/1.4-1.5/'
df = pd.read_csv(data_dir + 'housing.data.txt',
                 header=None,
                 sep='\s+')  # '\s+'表示匹配任意的空白字符/空格

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# In[]:
# # Evaluating the performance of linear regression models
from sklearn.model_selection import train_test_split
# 除最后一列数据
X = df.iloc[:, :-1].values
# 使用列名索引
y = df['MEDV'].values
# 分割数据集-30%测试数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X.shape,len(y_train),len(y_test)

# In[]:
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# 残差图-预测值的残差（真实值与预测值之间的差异或者垂直距离）
# 通过将预测结果减去对应目标变量真实值，便可获得残差的值
# 对于一个好的回归模型，期望误差是随机分布的，同时残差也随机分布于中心线附近。
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()

# In[]:
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 度量
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# In[]:
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.2)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
# lasso的系数值
print(lasso.coef_)
# 正常线性回归系数值
print(slr.coef_)

# In[]:
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# Ridge regression:

# In[]:
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
print(ridge.coef_)
print(slr.coef_)


# Elastic Net regression:

# In[]:
from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
elanet.fit(X_train, y_train)
y_train_pred = elanet.predict(X_train)
y_test_pred = elanet.predict(X_test)
print(elanet.coef_)
print(slr.coef_)

#%%
