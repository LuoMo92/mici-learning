
# In[]:
from IPython.display import Image
import pandas as pd


data_dir = 'E:/python-workspace/mici-learning/1.4-1.5/'
df = pd.read_csv(data_dir+'housing.data.txt',
                 header=None,
                 sep='\s+')  # '\s+'表示匹配任意的空白字符/空格

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


# ## Visualizing the important characteristics of a dataset

# In[]:
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('img.png', dpi=300)
plt.show()


# In[]:
import numpy as np

cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.show()


# In[]:
class LinearRegressionGD(object):
    # eta,表示学习率，控制每次更新的大小
    # n_iter表示训练的次数，每一次对一批的所有样本进行同时训练
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    # 训练函数
    # 输入样本特征数据X和每个样本特征对应的真实标签值y
    def fit(self, X, y):
        # 将所有w_初始化为0
        self.w_ = np.zeros(1 + X.shape[1])
        # 用来保存每一次的预测误差
        self.cost_ = []

        for i in range(self.n_iter):
            # 处理的都是一组数据(向量)
            # 线性函数的预测结果
            output = self.net_input(X)
            # 真实值减去预测值就是误差
            errors = (y - output)
            # 梯度下降
            self.w_[1:] += self.eta * X.T.dot(errors)
            # w0单独处理
            self.w_[0] += self.eta * errors.sum()
            # 代价函数-最小二乘法
            cost = (errors**2).sum() / 2.0
            # 记录每一次训练后的代价值
            self.cost_.append(cost)
        return self

    # 输入n个样本特征，并根据线性公式计算出对于的y值，即线性回归的预测值。
    # X是输入的样本特征值，可以是一组特征向量，也可以是多组特征向量
    def net_input(self, X):
        # w_用来存储特征值的权重，通过随机梯度下降的优化方法求出。
        # w_是一个向量，长度是特征个数加1 w0
        # np.dot用于向量点乘或矩阵乘法(本质也是行与列的点乘)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 预测函数
    # 必须在fit训练后进行使用-使用训练后w的计算输出结果
    def predict(self, X):
        return self.net_input(X)

# In[]:
from sklearn.preprocessing import StandardScaler

# 取X值使用两个[]为了将一维数据转为二维数据，转为n行1列
X = df[['RM']].values
y = df['MEDV'].values
X.shape

# In[]:
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
# 转为二维数据进行标准化，然后调用flatten再转为一维
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()


# In[]:
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# In[]:
# 显示每次训练和对应的cost值
plt.plot(range(1, lr.n_iter+1), lr.cost_)
# 差值平方和
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# In[]:
# 散点图是原始数据，直线式拟合后的结果
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

# In[]:
# 由于只有一个变量，下面参数分别对应直线的斜率slope和截距Intercept
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[]:
# 应用-有5个房间的房子价格
# 输入一个特征值5，并进行标准化
num_rooms_std = sc_x.transform(np.array([[5.0]]))
# 使用训练好的模型进行预测,预测结果也是标准化的数据
price_std = lr.predict(num_rooms_std)
print(price_std)
# 通过inverse_transform反向操作，将标准化值转为真实的价格数据
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))


# 使用python模型库的线性回归模型做一样的事情
# ## Estimating the coefficient of a regression model via scikit-learn

# In[]:
from sklearn.linear_model import LinearRegression

slr = LinearRegression()

# 直接使用未标准化的数据
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

# 使用标准化的数据
slr.fit(X_std,y_std)
y_pred = slr.predict(X_std)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

# In[]:
lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()
