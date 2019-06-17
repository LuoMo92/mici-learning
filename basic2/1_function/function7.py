"""
展开参数列表：当使用的参数已经在一个列表中的时候
"""
# In[1]
# 正常调用
print(list(range(3, 6)))

# In[2]
# 通过list传入一组参数，使用一个*符号
args = [3, 6,2]
print(list(range(*args)))

# In[3]
# 类似的，在函数中使用字典的方式传入key-value参数
def parrot(voltage, state='a stiff', action='voom'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")

d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)

#%%
