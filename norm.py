import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 生成一个均值是5，标准差是3的正态分布对象
norm_dis = stats.norm(5,3)
# 在区间[-5,15]上均匀的取101个点
x = np.linspace(-5,15,101)

# 计算该分布在x个点的概率密度分布函数值(PDF)
pdf = norm_dis.pdf(x)

# 下面是利用matplotlib画图
plt.figure(1)

# plot pdf
plt.plot(x,pdf,'b-',label='pdf')
plt.ylabel('Prob Density')
plt.title('PDF of normal distribution')
plt.text(-5.0,.12,r'$\mu=5,\ \sigma=3$')
# （-5.0, .12）是文本显示的起始坐标值 
# Matplotlib中支持LaTex语法，后面字符串是LaTex显示语法，更好的显示希腊字母
# 'r'是防止字符转义的，为了保证正确的LaTex语法
"""
例：

s=r'\tt'
print(s)
Output:
'\tt'
 
s='\tt'
print(s)
Output:
'        t'
"""
plt.legend(loc='best', frameon=False)
plt.show()