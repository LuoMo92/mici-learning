"""
可以多次抛出raise异常，即接受到异常，继续raise
"""
# In[]:
try:
    raise NameError('异常出现')
except NameError:
    print('出现了一个异常!')
    raise

#%%
