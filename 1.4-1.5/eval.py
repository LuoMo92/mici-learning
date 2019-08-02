# 留出法   交叉验证法    自助法

# 留出法
# In[]
import numpy as np
import pandas as pd

# 读入数据，为dataframe格式
data_dir = 'E:/python-workspace/mici-learning/1.4-1.5/'
df = pd.read_csv(data_dir + 'iris.data', header=None)
# 显示前面行
df.head()

# In[]
# 显示后面行
df.tail()

# In[]
# 取dataframe中的数据到数组array
y = df.iloc[:, 4].values
# 长度和类别
len(y), np.unique(y)

# In[]
# 把类别转为整数
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_i = le.fit_transform(y)
np.unique(y_i)
y_i

# In[]
# 前50个数据和最后50个数据
y_i[:50], y_i[-50:]

# In[]
# 获取两列特征数据
x = df.iloc[:,[2,3]].values
# shape和前10行
x.shape,x[:10]

# In[]
# 分割训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 测试次数 
print('留出法')
n = 10
scores = 0.0

# In[]:
for i in range(n):

    # stratify=y 分层采样
    x_train, x_test, y_train, y_test = train_test_split(
                        x, y_i, test_size=0.3, stratify=y)

    # 数据标准化
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # 逻辑回归进行训练
    lr = LogisticRegression(C=100.0, random_state=1)
    lr.fit(x_train_std, y_train)
    # 准确率
    s = lr.score(x_test_std, y_test)
    scores = scores + s
    print('%d -- Accuracy: %.2f' % (i+1, s))

# In[]
print('Average accuracy is %.2f' % (scores/n))

########################################
# 交叉验证
# 注意n折交叉验证与n次重复留出法的区别
# 交叉验证中是平分没有交集的n份，n-1份训练，1份验证
# 留出法的n次，只是重复的次数，每次随机的取一定比例进行训练，剩下的进行验证


# In[]:
# 10折/倍交叉验证
import numpy as np
from sklearn.model_selection import StratifiedKFold

print('\n\n交叉验证')
# 数字化后的标签值
y = y_i

# n_splits 倍数
kfold = StratifiedKFold(n_splits=10, random_state=1).split(x,y)

# In[]
scores = []
lr = LogisticRegression(C=100.0, random_state=1)

# In[]
for k, (train,test) in enumerate(kfold):
    lr.fit(x[train], y[train])
    score = lr.score(x[test], y[test])
    scores.append(score)
    print('Fold: %2d, Acc: %.3f' % (k+1, score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


#########################################
# 自助法

# In[]:
import random
from sklearn.linear_model import LogisticRegression

print('\n\n自助法\n')

# In[]
def BootStrap(num):
    slice=[]

    while len(slice) < num:
        p=random.randrange(0, num) #[0,num)
        slice.append(p)

    return slice

def GetDataByIndex(x, y, index):
    new_x = []
    new_y = []

    for i in index:
        new_x.append(x[i])
        new_y.append(y[i])
    
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    return new_x, new_y

# In[]
# 随机处理次数
times = 10

# 样本数量
y = y_i
num = len(y)
# 取标签的所有索引
all_index = set(np.arange(num))

scores = []
# 创建逻辑回归模型
lr = LogisticRegression(C=100.0, random_state=1)

# In[]
for i in range(times):
    print()
    # 自助采样
    train_index = BootStrap(num)
    print(train_index)

    # 找出未出现的索引号
    
    # 首先去除重复
    print('去除重复')
    unique_index = list(set(train_index))
    print(unique_index)

    # 未出现索引数据
    print('补集，未出现数据索引')
    test_index = list(all_index - set(unique_index))
    print(test_index)

    # 根据索引获取训练数据和测试数据
    x_train, y_train = GetDataByIndex(x, y, train_index)
    x_test, y_test = GetDataByIndex(x, y, test_index)

    # 训练
    lr.fit(x_train, y_train)
    # 打分
    score = lr.score(x_test, y_test)
    scores.append(score)
    print('次数: %2d, 准确率: %.3f' % (i+1, score))
    
print('\n平均准确率: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))