"""
函数：使用关键字作为参数
"""

# In[]:
def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")

# In[]:
parrot(1000) # 一个按照顺序位置设置参数，没有写关键字就是按照参数的位置顺序设置参数
# In[]:
parrot(voltage=1000) # 一个关键字参数
# In[]:
parrot(voltage=1000000, action='VOOOOOM') # 2个关键字参数
# In[]:
# 2个关键字参数，可以没有顺序关系
parrot(action='VOOOOOM', voltage=1000000) 
# In[]:
# 按照顺序的三个参数
parrot('a million', 'bereft of life', 'jump') 
# In[]:
# 一个顺序位置参数，一个关键字参数
parrot('a thousand', state='pushing up the daisies') 


# In[]:
# 下面调用都是错误的

"""
parrot() # voltage没有缺省值，需要参数值
parrot(voltage=5.0, 'dead') # 关键字参数不能在非关键字参数的前面
parrot(110, voltage=220) # 相当于给voltage重复赋值
parrot(actor='John Cleese') # 使用了没有设置的关键字参数

"""