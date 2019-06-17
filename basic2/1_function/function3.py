"""
函数缺省值：缺省值如果是一组数据的类型，比如list，是可以变化的
"""

# In[]:
def f1(a, L=[]):
    L.append(a)
    return L

print(f1(1))
print(f1(2))
print(f1(3))

# In[]:
"""
如果你不想参数每次调用的时候被共享和修改，可以使用下面的方法
"""

# In[]:    
def f2(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L

print(f2(1))
print(f2(2))
print(f2(3))



#%%
