# 性能度量方法
# In[]
import numpy as np
import pandas as pd

# 读入数据，为dataframe格式
data_dir = 'E:/pycharm-workspace/mici-learning/1.6/'
df = pd.read_csv(data_dir+'iris.data', header=None)
# 显示前面行和后面行
df.head()

# In[]
# 取dataframe中的数据到数组array，展示二分类性能度量，取后100个数据
y_org = df.iloc[-100:, 4].values 
# 长度和类别
len(y_org), np.unique(y_org)

# In[]
# 取特征数据
x = df.iloc[-100:,[2,3]].values
x.shape,x[:10]


# In[]
# 把类别转为整数
from sklearn.preprocessing import LabelEncoder
# 将两类'Iris-versicolor', 'Iris-virginica'转为整数
# 关心的分类为positive类,注意在python包中，缺省的认为1是positive类
le = LabelEncoder()
# fit是设置参数，这里找到所有类别，按照字母顺序排序后，以0为索引给类别分配数字
# 所以'Iris-versicolor', 'Iris-virginica'分别是0和1类
le.fit(y_org)
le.classes_


# In[]
# transform是根据fit的参数进行转换，将y_org中的所有类别值根据设置好的整数类别值进行转换
# 'Iris-virginica'是1值，是二分类中的positive类
y_i =  le.transform(y_org)
le.classes_

# In[]
np.unique(y_i)

# In[]
y_org

# In[]
y_i

# In[]
# 由于1表示positive类，如果关注的是'Iris-versicolor'类，需要把该类设置为1即可
# 思路很简单，将y_i中的1设置0，0设置为1即可,当然方法有很多种，编程实现即可
y_t = y_i.copy()
y_t[:50] = 1
y_t[50:] = 0
y_t

# In[]
y_i

# In[]
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

print('\n\n交叉验证\n')
y = y_i
kfold = StratifiedKFold(n_splits=10, random_state=1).split(x, y)

scores = []
lr = LogisticRegression(C=100.0, random_state=1)

for k, (train, test) in enumerate(kfold):
    lr.fit(x[train], y[train])
    print('Fold: %2d' % (k+1))
    # 注意！！在python中，认为1是positive，所以关注的类，要赋值为1
    # 混淆矩阵
    y_pred = lr.predict(x[test])
    confmat = confusion_matrix(y_true=y[test], y_pred=y_pred)
    print(confmat)
    print()

    # 以Fold 2为例，混淆矩阵如下，1表示positive
    #    0  1 
    # 0  5  0
    # 1  1  4
    # TP = 4, TN = 5, FP = 0, FN = 1, 下面套公式计算即可

    # 可以看出混淆矩阵的缺省排序是从0开始，由于1是positive，从1开始和所讲的就保持一致了
    confmat2 = confusion_matrix(y_true=y[test], y_pred=y_pred, labels=[1, 0])
    print(confmat2)

    # 仍以Fold 2为例，混淆矩阵如下，1表示positive
    #    1  0 
    # 1  4  1
    # 0  0  5
    # TP = 4, TN = 5, FP = 0, FN = 1, 下面套公式计算即可

    # 准确率度量
    score = lr.score(x[test], y[test])
    scores.append(score)
    print('Acc: %.3f' % (score))

    # 精确度
    print('Precision: %.3f' % precision_score(y_true=y[test], y_pred=y_pred))
    # 召回率/敏感度/真正率/真阳性率
    print('Recall: %.3f' % recall_score(y_true=y[test], y_pred=y_pred))
    # F1-score
    print('F1: %.3f' % f1_score(y_true=y[test], y_pred=y_pred))

    # 以Fold 2为例，混淆矩阵如下，1表示positive
    #    0  1 
    # 0  5  0
    # 1  1  4
    # TP = 4, TN = 5, FP = 0, FN = 1, 下面套公式计算即可

    # 特异度/真负率/真阴性率，可以通过混淆矩阵直接运算出来,事实上，其他的性能指标都可以通过混淆矩阵计算
    tn = confmat[0,0]
    fp = confmat[0,1]
    spec = float(tn)/(float(tn)+float(fp))
    print('Spec: %.3f' % spec)

    print('\n')
    


#%%
