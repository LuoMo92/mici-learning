from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

# 二项分布

# 事件总的次数
n = 20
# 事件发生的概率
p = 0.3
# 求事件发生k次的概率
k = np.arange(1,21)
binomial = binom.pmf(k,n,p)
print(binomial)

plt.plot(k,binomial,'o-')
plt.title('Binomial:n = %i, p = %0.2f' % (n,p),fontsize=15)
plt.xlabel('Number of successes')
plt.ylabel('Probability of successes',fontsize=15)
plt.show()