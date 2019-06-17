"""
小的匿名函数
"""

# In[1]:
# 这里返回的是一个匿名函数，匿名函数输入的是x，
# 即输入的参数，输出是x+n
def make_incrementor(n):
    return lambda x: x + n

f = make_incrementor(42)
print(f(0))

print(f(1))


#%%
