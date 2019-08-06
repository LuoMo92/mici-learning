# In[]
import pandas as pd
df = pd.read_csv('E:/python-workspace/mici-learning/1.4-1.5/housing.data.txt',
                 header=None,
                 sep='\s+')   # '\s+'表示匹配任意多个空白字符/空格
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# In[]
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('img.png', dpi=300)
plt.show()

# In[]
import numpy as np

# 皮尔逊积矩相关系数--用来衡量两两特征间的线性依赖关系
cm = np.corrcoef(df[cols].values.T)
print(cm)
sns.set(font_scale=1.5) #设置字体大小
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


#%%
